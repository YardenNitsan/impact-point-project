"""Train the Multi-Task Multi-Head MLP Regressor weather model.

This is the main neural-network training script. It uses TensorFlow/Keras for
Dense layers, BatchNormalization, Dropout, the Adam optimizer, callbacks, and
GPU support. It does NOT use the legacy tree backend or the legacy NumPy MLP.

Important features:
    * Float32 everywhere — important for ~32 GB RAM machines training on a
      full month of ERA5 data (around 31 daily files, each 2-4 GB).
    * Aggressive freeing of intermediate arrays (we explicitly ``del`` raw
      buffers once normalized copies exist, then call ``gc.collect()``).
    * Dry-run mode (``TRAIN_DRY_RUN=1``) that lists the files to be used,
      estimates RAM, and exits without training.
    * All settings exposed via env vars with sensible defaults for 32 GB RAM.
    * Robust callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
      and CSVLogger (so a long run that crashes still leaves a per-epoch log).
    * Plotting is wrapped in try/except so a Matplotlib hiccup never destroys
      a long training run; set ``TRAIN_SAVE_PLOTS=0`` to skip plotting.
"""

from __future__ import annotations

print("RUNNING UPDATED MULTI_HEAD_MLP TRAINER VERSION 2026-04-27")

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_gam_weather.config import SamplingConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file
from era5_gam_weather.multi_head_mlp_model import (
    ERA5_TO_OUTPUT,
    FEATURE_METADATA_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    NORMALIZATION_FILENAME,
    TARGET_OUTPUTS,
    TARGET_TRANSFORMS,
    MultiHeadMLPWeatherModel,
    build_multi_head_mlp,
)
from era5_gam_weather.weather_features import WeatherFeatureBuilder, WeatherFeatureConfig

# ---------------------------------------------------------------------------
# Optional psutil for accurate RSS reporting; fall back to a small approximation.
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore
    _PROCESS = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil is optional
    _PROCESS = None


def _rss_gb() -> float:
    if _PROCESS is None:
        return float("nan")
    try:
        return float(_PROCESS.memory_info().rss) / (1024 ** 3)
    except Exception:
        return float("nan")


def _log(msg: str) -> None:
    rss = _rss_gb()
    if not np.isnan(rss):
        print(f"[mem {rss:5.2f} GB] {msg}", flush=True)
    else:
        print(msg, flush=True)


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for this training script. Install it with: pip install tensorflow"
        ) from exc
    return tf, keras


def _build_warmup_cosine_schedule(tf, base_lr: float, total_steps: int, warmup_steps: int, min_lr: float):
    """Linear warmup → cosine decay LR schedule, returned as a Keras schedule.

    Implemented as a small subclass so it serializes cleanly in the .keras
    artifact. We avoid relying on the (newer) ``CosineDecay(warmup_steps=...)``
    keyword so this works on TF 2.10+.
    """
    keras = tf.keras
    schedules = keras.optimizers.schedules

    class WarmupCosineSchedule(schedules.LearningRateSchedule):
        def __init__(self, base_lr_, total_steps_, warmup_steps_, min_lr_):
            super().__init__()
            self.base_lr = float(base_lr_)
            self.total_steps = int(max(1, total_steps_))
            self.warmup_steps = int(max(0, warmup_steps_))
            self.min_lr = float(min_lr_)

        def __call__(self, step):
            step_f = tf.cast(step, tf.float32)
            warmup_f = tf.cast(self.warmup_steps, tf.float32)
            total_f = tf.cast(self.total_steps, tf.float32)

            warm_lr = self.base_lr * (step_f / tf.maximum(1.0, warmup_f))
            decay_progress = (step_f - warmup_f) / tf.maximum(1.0, total_f - warmup_f)
            decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
            cos_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * decay_progress)
            )
            return tf.where(step_f < warmup_f, warm_lr, cos_lr)

        def get_config(self):
            return {
                "base_lr_": self.base_lr,
                "total_steps_": self.total_steps,
                "warmup_steps_": self.warmup_steps,
                "min_lr_": self.min_lr,
            }

    return WarmupCosineSchedule(base_lr, total_steps, warmup_steps, min_lr)


def _seasonal_harmonics_for_months(months: List[int]) -> int:
    """Return a sane annual-harmonics count given the training month coverage.

    With one or two adjacent months, sin/cos(2π·doy/365.25) is essentially
    constant and adds no signal — we strip those features so the network
    isn't forced to memorize their narrow constant range.
    """
    if not months:
        return 0
    unique = sorted(set(int(m) for m in months))
    if len(unique) <= 1:
        return 0
    span = unique[-1] - unique[0]
    if len(unique) <= 2 and span <= 1:
        return 0
    return 2


# ---------------------------------------------------------------------------
# Environment variables (single source of truth for run configuration).
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid integer for {name}: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid float for {name}: {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_months(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


YEAR = _env_int("TRAIN_YEAR", 2025)
MONTHS = _parse_months(os.getenv("TRAIN_MONTHS", "5"))  # default: May only
TRAIN_SAMPLES_PER_FILE = _env_int("TRAIN_SAMPLES_PER_FILE", 50000)
SEED = _env_int("TRAIN_SEED", 42)
BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 8192)
MAX_EPOCHS = _env_int("TRAIN_MAX_EPOCHS", 250)
LEARNING_RATE = _env_float("TRAIN_LEARNING_RATE", 2e-3)
DROPOUT_RATE = _env_float("TRAIN_DROPOUT_RATE", 0.0)
EARLY_STOPPING_PATIENCE = _env_int("TRAIN_EARLY_STOPPING_PATIENCE", 50)
REDUCE_LR_PATIENCE = _env_int("TRAIN_REDUCE_LR_PATIENCE", 10)
TRAIN_RATIO = _env_float("TRAIN_SPLIT_TRAIN_RATIO", 0.8)
VAL_RATIO = _env_float("TRAIN_SPLIT_VAL_RATIO", 0.1)
TRAIN_MAX_TOTAL_SAMPLES = _env_int("TRAIN_MAX_TOTAL_SAMPLES", 0)  # 0 = no cap
TRAIN_SAVE_PLOTS = _env_bool("TRAIN_SAVE_PLOTS", True)
TRAIN_DRY_RUN = _env_bool("TRAIN_DRY_RUN", False)

# New knobs (suggestions #5, #6, #7).
TRAIN_LR_SCHEDULE = (os.getenv("TRAIN_LR_SCHEDULE") or "cosine").strip().lower()
TRAIN_WARMUP_EPOCHS = _env_int("TRAIN_WARMUP_EPOCHS", 5)
TRAIN_MIN_LR = _env_float("TRAIN_MIN_LR", 1e-6)
TRAIN_HUBER_DELTA = _env_float("TRAIN_HUBER_DELTA", 0.5)
TRAIN_WEIGHT_DECAY = _env_float("TRAIN_WEIGHT_DECAY", 1e-5)
TRAIN_USE_ADAMW = _env_bool("TRAIN_USE_ADAMW", True)
TRAIN_N_RESIDUAL_BLOCKS = _env_int("TRAIN_N_RESIDUAL_BLOCKS", 4)
TRAIN_BLOCK_WIDTH = _env_int("TRAIN_BLOCK_WIDTH", 512)

TRAIN_LOSS_W_TEMP = _env_float("TRAIN_LOSS_W_TEMPERATURE", 0.5)
TRAIN_LOSS_W_PRESSURE = _env_float("TRAIN_LOSS_W_PRESSURE", 0.5)
TRAIN_LOSS_W_WIND_U = _env_float("TRAIN_LOSS_W_WIND_U", 1.5)
TRAIN_LOSS_W_WIND_V = _env_float("TRAIN_LOSS_W_WIND_V", 1.5)

# Sampling knobs (#9, #10).
TRAIN_AREA_WEIGHTED_LAT = _env_bool("TRAIN_AREA_WEIGHTED_LAT", True)
_RAW_LEVEL_WEIGHTS = os.getenv("TRAIN_LEVEL_WEIGHTS", "").strip()


def _parse_level_weights(value: str):
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"Invalid TRAIN_LEVEL_WEIGHTS: {value!r}") from exc


TRAIN_LEVEL_WEIGHTS = _parse_level_weights(_RAW_LEVEL_WEIGHTS)

TARGET_NAMES = ["T", "P", "U", "V"]
RAW_FEATURE_NAMES = ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for base in [start] + list(start.parents):
        if (base / "data" / "era5").exists():
            return base
    return start.parent


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(THIS_DIR)
DATA_ROOT = Path(os.getenv("ERA5_DATA_ROOT") or str(PROJECT_ROOT / "data" / "era5")).expanduser().resolve()
ARTIFACT_DIR = Path(
    os.getenv("WEATHER_ARTIFACT_DIR") or str(THIS_DIR / "artifacts" / "multi_head_mlp_weather")
).expanduser().resolve()
PLOTS_DIR = ARTIFACT_DIR / "training_plots"


# ---------------------------------------------------------------------------
# Sample collection (memory-conscious).
# ---------------------------------------------------------------------------
def _empty_feature_dict() -> Dict[str, list]:
    return {k: [] for k in RAW_FEATURE_NAMES}


def _empty_target_dict() -> Dict[str, list]:
    return {k: [] for k in TARGET_NAMES}


def _finalize(storage_features: Dict[str, list], storage_targets: Dict[str, list]):
    """Concatenate per-file lists into single contiguous float32 arrays.

    We concatenate one key at a time and clear the source list immediately so
    the per-file chunks can be garbage-collected before we move to the next
    feature; this halves peak memory during the concat step.
    """
    features: Dict[str, np.ndarray] = {}
    for k in RAW_FEATURE_NAMES:
        chunks = storage_features[k]
        if chunks:
            features[k] = np.concatenate(chunks).astype(np.float32, copy=False)
        else:
            features[k] = np.empty(0, dtype=np.float32)
        chunks.clear()
    targets: Dict[str, np.ndarray] = {}
    for k in TARGET_NAMES:
        chunks = storage_targets[k]
        if chunks:
            targets[k] = np.concatenate(chunks).astype(np.float32, copy=False)
        else:
            targets[k] = np.empty(0, dtype=np.float32)
        chunks.clear()
    gc.collect()
    return features, targets


def _collect_all_splits(
    files: Iterable[str],
    config: SamplingConfig,
    train_ratio: float,
    val_ratio: float,
    max_total_train: int = 0,
):
    """Sample from every file and split each file's rows into train/val/test.

    Every day contributes to all three splits so the model sees the full
    calendar during training.  Split is deterministic: per-file shuffle uses
    config.seed + file_index so results are reproducible.
    """
    train_sf, train_st = _empty_feature_dict(), _empty_target_dict()
    val_sf, val_st = _empty_feature_dict(), _empty_target_dict()
    test_sf, test_st = _empty_feature_dict(), _empty_target_dict()

    n_train_total = n_val_total = n_test_total = 0

    for idx, path in enumerate(list(files), start=1):
        if max_total_train > 0 and n_train_total >= max_total_train:
            print(f"[COLLECT] reached TRAIN_MAX_TOTAL_SAMPLES={max_total_train}, stopping after {idx-1} files.")
            break
        print(f"[COLLECT {idx}] {path}")
        batch = sample_from_file(path, config)
        n = len(batch.features["lat"])
        if n == 0:
            del batch
            continue

        rng = np.random.default_rng(config.seed + idx * 1000)
        perm = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

        for k in RAW_FEATURE_NAMES:
            arr = np.asarray(batch.features[k], dtype=np.float32)
            train_sf[k].append(arr[idx_train])
            val_sf[k].append(arr[idx_val])
            test_sf[k].append(arr[idx_test])
        for k in TARGET_NAMES:
            arr = np.asarray(batch.targets[k], dtype=np.float32)
            train_st[k].append(arr[idx_train])
            val_st[k].append(arr[idx_val])
            test_st[k].append(arr[idx_test])

        n_train_total += len(idx_train)
        n_val_total += len(idx_val)
        n_test_total += len(idx_test)
        del batch, perm, idx_train, idx_val, idx_test

    print(f"[COLLECT] split complete — train={n_train_total}, val={n_val_total}, test={n_test_total}")

    train_features, train_targets = _finalize(train_sf, train_st)
    _log("after train finalize")
    val_features, val_targets = _finalize(val_sf, val_st)
    _log("after val finalize")
    test_features, test_targets = _finalize(test_sf, test_st)
    _log("after test finalize")

    return train_features, train_targets, val_features, val_targets, test_features, test_targets


def _normalise_X_inplace(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Normalize in place so we don't double the memory footprint of X."""
    np.subtract(X, x_mean.reshape(1, -1), out=X)
    np.divide(X, x_std.reshape(1, -1), out=X)
    return X


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def _evaluate_split(
    wrapper: MultiHeadMLPWeatherModel,
    features: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    pred = wrapper.predict_features(features)
    return {
        output_name: _metrics(targets[era5_key], pred[output_name])
        for era5_key, output_name in ERA5_TO_OUTPUT.items()
    }


def _print_metrics_table(metrics: Dict[str, Dict[str, float]], title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"{title}")
    print(f"{'=' * 72}")
    print(f"{'Target':<18} {'MAE':>14} {'RMSE':>14} {'Max Abs':>14}")
    print("-" * 72)
    for name in TARGET_OUTPUTS:
        m = metrics[name]
        print(f"{name:<18} {m['mae']:>14.6f} {m['rmse']:>14.6f} {m['max_abs']:>14.6f}")
    print("=" * 72)


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_save_plot(fn, *args, **kwargs) -> None:
    """Run a plotting function; warn but don't crash on failure."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - matplotlib is environment-dependent
        print(f"WARNING: plotting step failed: {exc}", file=sys.stderr, flush=True)


def _save_training_plots(history: Dict[str, list]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history.get("loss", []), label="Training loss")
    plt.plot(history.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss on normalized targets")
    plt.title("Multi-Head MLP - Overall Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_training_validation_loss.png", dpi=200)
    plt.close()

    for target in TARGET_OUTPUTS:
        train_key = f"{target}_loss"
        val_key = f"val_{target}_loss"
        if train_key not in history:
            continue
        plt.figure(figsize=(10, 6))
        plt.plot(history.get(train_key, []), label=f"{target} train loss")
        plt.plot(history.get(val_key, []), label=f"{target} validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Huber loss on normalized target")
        plt.title(f"Multi-Head MLP - {target} Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{target}_training_validation_loss.png", dpi=200)
        plt.close()


def _save_metric_plots(test_metrics: Dict[str, Dict[str, float]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("mae", "MAE", "test_mae_by_target.png"),
        ("rmse", "RMSE", "test_rmse_by_target.png"),
        ("max_abs", "Max Absolute Error", "test_max_abs_by_target.png"),
    ]
    targets = list(TARGET_OUTPUTS)
    for key, ylabel, filename in metric_specs:
        values = [float(test_metrics[target][key]) for target in targets]
        plt.figure(figsize=(10, 6))
        plt.bar(targets, values)
        plt.xlabel("Target")
        plt.ylabel(ylabel)
        plt.title(f"Held-out Test {ylabel} by Target")
        ymax = max(values) if values else 1.0
        for i, value in enumerate(values):
            plt.text(i, value + ymax * 0.02, f"{value:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Configuration printing & sanity checks.
# ---------------------------------------------------------------------------
def _print_config_block() -> None:
    print("=" * 72)
    print("Multi-Task Multi-Head MLP Weather Trainer")
    print("=" * 72)
    print(f"  ERA5_DATA_ROOT             = {DATA_ROOT}")
    print(f"  WEATHER_ARTIFACT_DIR       = {ARTIFACT_DIR}")
    print(f"  TRAIN_YEAR                 = {YEAR}")
    print(f"  TRAIN_MONTHS               = {MONTHS}")
    print(f"  TRAIN_SAMPLES_PER_FILE     = {TRAIN_SAMPLES_PER_FILE}")
    print(f"  TRAIN_MAX_TOTAL_SAMPLES    = {TRAIN_MAX_TOTAL_SAMPLES} (0 = unlimited)")
    print(f"  TRAIN_AREA_WEIGHTED_LAT    = {TRAIN_AREA_WEIGHTED_LAT}")
    print(f"  TRAIN_LEVEL_WEIGHTS        = {TRAIN_LEVEL_WEIGHTS}")
    print(f"  TRAIN_BATCH_SIZE           = {BATCH_SIZE}")
    print(f"  TRAIN_MAX_EPOCHS           = {MAX_EPOCHS}")
    print(f"  TRAIN_LEARNING_RATE        = {LEARNING_RATE}")
    print(f"  TRAIN_LR_SCHEDULE          = {TRAIN_LR_SCHEDULE}")
    print(f"  TRAIN_WARMUP_EPOCHS        = {TRAIN_WARMUP_EPOCHS}")
    print(f"  TRAIN_MIN_LR               = {TRAIN_MIN_LR}")
    print(f"  TRAIN_DROPOUT_RATE         = {DROPOUT_RATE}")
    print(f"  TRAIN_HUBER_DELTA          = {TRAIN_HUBER_DELTA}")
    print(f"  TRAIN_USE_ADAMW            = {TRAIN_USE_ADAMW}")
    print(f"  TRAIN_WEIGHT_DECAY         = {TRAIN_WEIGHT_DECAY}")
    print(f"  TRAIN_N_RESIDUAL_BLOCKS    = {TRAIN_N_RESIDUAL_BLOCKS}")
    print(f"  TRAIN_BLOCK_WIDTH          = {TRAIN_BLOCK_WIDTH}")
    print(f"  TRAIN_LOSS_W (T,P,U,V)     = {TRAIN_LOSS_W_TEMP},{TRAIN_LOSS_W_PRESSURE},{TRAIN_LOSS_W_WIND_U},{TRAIN_LOSS_W_WIND_V}")
    print(f"  TRAIN_EARLY_STOPPING_PAT.  = {EARLY_STOPPING_PATIENCE}")
    print(f"  TRAIN_REDUCE_LR_PATIENCE   = {REDUCE_LR_PATIENCE}")
    print(f"  TRAIN_SPLIT_TRAIN_RATIO    = {TRAIN_RATIO}")
    print(f"  TRAIN_SPLIT_VAL_RATIO      = {VAL_RATIO}")
    print(f"  TRAIN_SAVE_PLOTS           = {TRAIN_SAVE_PLOTS}")
    print(f"  TRAIN_DRY_RUN              = {TRAIN_DRY_RUN}")
    print(f"  TRAIN_SEED                 = {SEED}")
    print("=" * 72)


def _estimate_peak_gb(n_train: int, n_val: int, n_test: int, n_features: int) -> float:
    """Rough peak-RAM estimate in GB for the in-memory training arrays.

    We charge: raw float32 train/val/test features + targets, the engineered
    train/val feature matrix, and the normalized copy that briefly coexists
    during in-place normalization (we use ``np.subtract(out=X)`` so we charge
    a single matrix, plus a small overhead). The estimate is intentionally
    conservative.
    """
    bytes_per = 4
    raw_targets = (n_train + n_val + n_test) * 4 * bytes_per                 # 4 ERA5 targets
    raw_features = (n_train + n_val + n_test) * len(RAW_FEATURE_NAMES) * bytes_per
    eng_features = (n_train + n_val) * n_features * bytes_per                # X_train, X_val
    norm_targets = (n_train + n_val) * 4 * bytes_per
    overhead_gb = 0.5  # TF + xarray + Python overhead headroom
    total = raw_targets + raw_features + eng_features + norm_targets
    return total / (1024 ** 3) + overhead_gb


def _check_data_root_or_exit() -> None:
    if not DATA_ROOT.exists():
        print(
            f"\nERROR: ERA5 data directory not found: {DATA_ROOT}\n"
            f"  Set ERA5_DATA_ROOT to the directory containing era5_YYYY_MM_DD.nc files,\n"
            f"  e.g.  export ERA5_DATA_ROOT=~/impact-data/era5\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_training() -> None:
    _print_config_block()
    _check_data_root_or_exit()

    files = discover_era5_files(str(DATA_ROOT), YEAR, MONTHS)
    if not files:
        print(
            f"\nERROR: No ERA5 files found under {DATA_ROOT} for year={YEAR}, months={MONTHS}.\n"
            f"  Expected file pattern: era5_{YEAR}_MM_DD.nc\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nFile discovery: {len(files)} ERA5 file(s) found.")
    for path in files:
        print(f"  {path}")

    # Up-front estimate using a temporary builder (we can't yet know exact n_features
    # for non-default configs without instantiating one).
    annual_harmonics = _seasonal_harmonics_for_months(MONTHS)
    feature_config = WeatherFeatureConfig(annual_harmonics=annual_harmonics)
    feature_builder_preview = WeatherFeatureBuilder(feature_config)
    n_features_preview = len(feature_builder_preview.feature_names)
    print(f"  annual_harmonics (auto)    = {annual_harmonics} (months={MONTHS})")
    n_per_file = TRAIN_SAMPLES_PER_FILE
    n_train_est = int(len(files) * n_per_file * TRAIN_RATIO)
    n_val_est = int(len(files) * n_per_file * VAL_RATIO)
    n_test_est = len(files) * n_per_file - n_train_est - n_val_est
    cap = TRAIN_MAX_TOTAL_SAMPLES
    if cap > 0:
        n_train_est = min(n_train_est, cap)
    est_gb = _estimate_peak_gb(n_train_est, n_val_est, n_test_est, n_features_preview)
    print(
        f"\nEstimated dataset sizes: train≈{n_train_est}, val≈{n_val_est}, test≈{n_test_est}\n"
        f"Estimated peak in-memory array footprint: ~{est_gb:.2f} GB\n"
        f"  (excludes TensorFlow runtime, GPU memory, and OS file caches)"
    )
    if est_gb > 22.0:
        print(
            "WARNING: estimated peak > 22 GB. On a 32 GB machine you should reduce\n"
            "  TRAIN_SAMPLES_PER_FILE, drop a month from TRAIN_MONTHS, or set\n"
            "  TRAIN_MAX_TOTAL_SAMPLES to a smaller value.",
            flush=True,
        )

    if TRAIN_DRY_RUN:
        print("\nTRAIN_DRY_RUN=1 — exiting without training.\n")
        return

    # ---------------------------------------------------------------------
    # TF setup (deferred until after dry-run gate).
    # ---------------------------------------------------------------------
    tf, keras = _import_tensorflow()
    print("TensorFlow version:", tf.__version__)
    physical_devices = tf.config.list_physical_devices()
    print("TF physical devices:", [(d.device_type, d.name) for d in physical_devices])
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Available GPU(s):", [gpu.name for gpu in gpus])
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:  # pragma: no cover
                print(f"  (could not enable memory growth on {gpu.name}: {exc})")
    else:
        print("No GPU detected; training will run on CPU.")

    # Determinism.
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sampling = SamplingConfig(
        samples_per_file=TRAIN_SAMPLES_PER_FILE,
        seed=SEED,
        stratified_time_level=True,
        area_weighted_lat=TRAIN_AREA_WEIGHTED_LAT,
        level_weights=TRAIN_LEVEL_WEIGHTS,
    )

    print("\n--- Collecting and splitting data (per-file 80/10/10) ---")
    (
        train_features, train_targets,
        val_features, val_targets,
        test_features, test_targets,
    ) = _collect_all_splits(files, sampling, TRAIN_RATIO, VAL_RATIO, max_total_train=TRAIN_MAX_TOTAL_SAMPLES)
    _log("after data collection and split")

    n_train = len(train_features["lat"])
    n_val = len(val_features["lat"])
    n_test = len(test_features["lat"])
    print(f"\nDataset sizes: train={n_train}, val={n_val}, test={n_test}")

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise RuntimeError("Train/validation/test split produced an empty split. Check available ERA5 files.")

    feature_builder = WeatherFeatureBuilder(feature_config)
    print(f"Feature count: {len(feature_builder.feature_names)} (annual_harmonics={annual_harmonics})")

    print("\n--- Building engineered features ---")
    X_train_raw = feature_builder.transform(train_features)
    _log(f"X_train_raw shape={X_train_raw.shape} dtype={X_train_raw.dtype}")
    X_val_raw = feature_builder.transform(val_features)
    _log(f"X_val_raw   shape={X_val_raw.shape} dtype={X_val_raw.dtype}")

    x_mean, x_std, y_mean, y_std, _ = MultiHeadMLPWeatherModel.compute_normalization(
        X_train_raw,
        train_targets,
    )

    # In-place normalization: do NOT keep a copy of X_train_raw afterwards.
    X_train = _normalise_X_inplace(X_train_raw, x_mean, x_std)
    X_val = _normalise_X_inplace(X_val_raw, x_mean, x_std)
    del X_train_raw, X_val_raw  # they ARE X_train / X_val now (in-place), but be explicit.
    gc.collect()
    _log("after normalization (raw X arrays freed)")

    y_train = MultiHeadMLPWeatherModel.normalize_targets(train_targets, y_mean, y_std)
    y_val = MultiHeadMLPWeatherModel.normalize_targets(val_targets, y_mean, y_std)
    _log("after target normalization")

    loss_weights = {
        "temperature_k": TRAIN_LOSS_W_TEMP,
        "pressure_pa": TRAIN_LOSS_W_PRESSURE,
        "wind_u": TRAIN_LOSS_W_WIND_U,
        "wind_v": TRAIN_LOSS_W_WIND_V,
    }

    steps_per_epoch = max(1, (n_train + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = steps_per_epoch * max(0, TRAIN_WARMUP_EPOCHS)

    use_cosine = TRAIN_LR_SCHEDULE == "cosine"
    if use_cosine:
        lr_arg = _build_warmup_cosine_schedule(
            tf,
            base_lr=LEARNING_RATE,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr=TRAIN_MIN_LR,
        )
        print(
            f"Using WarmupCosineSchedule: base_lr={LEARNING_RATE}, "
            f"warmup_epochs={TRAIN_WARMUP_EPOCHS} ({warmup_steps} steps), "
            f"total_steps={total_steps}, min_lr={TRAIN_MIN_LR}",
            flush=True,
        )
    else:
        lr_arg = LEARNING_RATE
        print(f"Using constant LR={LEARNING_RATE} with ReduceLROnPlateau", flush=True)

    model = build_multi_head_mlp(
        n_features=X_train.shape[1],
        dropout_rate=DROPOUT_RATE,
        learning_rate=lr_arg,
        loss_weights=loss_weights,
        n_residual_blocks=TRAIN_N_RESIDUAL_BLOCKS,
        block_width=TRAIN_BLOCK_WIDTH,
        huber_delta=TRAIN_HUBER_DELTA,
        weight_decay=TRAIN_WEIGHT_DECAY,
        use_adamw=TRAIN_USE_ADAMW,
    )
    model.summary(print_fn=lambda line: print(line, flush=True))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ARTIFACT_DIR / MODEL_FILENAME),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # CSVLogger keeps a per-epoch record on disk, so a crashed long run
        # still leaves usable progress data.
        keras.callbacks.CSVLogger(str(ARTIFACT_DIR / "training_log.csv"), append=False),
    ]
    if not use_cosine:
        # ReduceLROnPlateau only makes sense with a plain (non-scheduled) LR.
        callbacks.insert(
            1,
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=REDUCE_LR_PATIENCE,
                min_lr=TRAIN_MIN_LR,
                verbose=1,
            ),
        )

    print("\n--- Starting model.fit ---")
    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    _log("after model.fit")

    # Re-save the in-memory model: EarlyStopping with restore_best_weights=True
    # ensures these are the best weights regardless of whether ModelCheckpoint
    # already wrote them.
    model.save(ARTIFACT_DIR / MODEL_FILENAME)

    metadata = {
        "model_type": "Multi-Task Multi-Head MLP Regressor",
        "backend": "multi_head_mlp",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": YEAR,
        "months": MONTHS,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": SEED,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout_rate": DROPOUT_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "loss": f"Huber(delta={TRAIN_HUBER_DELTA}) on normalized targets per head",
            "huber_delta": TRAIN_HUBER_DELTA,
            "optimizer": "AdamW" if TRAIN_USE_ADAMW else "Adam",
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "lr_schedule": TRAIN_LR_SCHEDULE,
            "warmup_epochs": TRAIN_WARMUP_EPOCHS,
            "min_lr": TRAIN_MIN_LR,
            "loss_weights": loss_weights,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "max_total_train_samples": TRAIN_MAX_TOTAL_SAMPLES,
        },
        "architecture": {
            "type": "residual_mlp_multi_head",
            "stem": f"Dense({TRAIN_BLOCK_WIDTH}, linear)",
            "n_residual_blocks": TRAIN_N_RESIDUAL_BLOCKS,
            "block_width": TRAIN_BLOCK_WIDTH,
            "block": [
                "BN", "ReLU",
                f"Dense({TRAIN_BLOCK_WIDTH}, linear)",
                "BN", "ReLU",
                f"Dropout({DROPOUT_RATE})" if DROPOUT_RATE > 0.0 else "Dropout(off)",
                f"Dense({TRAIN_BLOCK_WIDTH}, linear)",
                "Add(skip)",
            ],
            "heads": {
                "temperature_k": ["Dense(128, relu)", "Dense(1, linear)"],
                "pressure_pa": ["Dense(128, relu)", "Dense(1, linear)"],
                "wind_u": ["Dense(256, relu)", "Dense(128, relu)", "Dense(1, linear)"],
                "wind_v": ["Dense(256, relu)", "Dense(128, relu)", "Dense(1, linear)"],
            },
        },
        "target_transforms": dict(TARGET_TRANSFORMS),
        "feature_count": int(X_train.shape[1]),
        "feature_names": feature_builder.feature_names,
        "data_root": str(DATA_ROOT),
    }

    normalization_payload = {
        "x_mean": x_mean.astype(float).tolist(),
        "x_std": x_std.astype(float).tolist(),
        "targets": {
            output_name: {
                "source_key": era5_key,
                "transform": TARGET_TRANSFORMS[output_name],
                "mean": float(y_mean[output_name]),
                "std": float(y_std[output_name]),
            }
            for era5_key, output_name in ERA5_TO_OUTPUT.items()
        },
    }

    _save_json(ARTIFACT_DIR / NORMALIZATION_FILENAME, normalization_payload)
    _save_json(ARTIFACT_DIR / FEATURE_METADATA_FILENAME, feature_builder.metadata())
    _save_json(ARTIFACT_DIR / METADATA_FILENAME, metadata)

    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    _save_json(ARTIFACT_DIR / "training_history.json", history)

    if TRAIN_SAVE_PLOTS:
        _safe_save_plot(_save_training_plots, history)

    # We can drop the giant normalized X arrays now that fit is done.
    del X_train, X_val, y_train, y_val
    gc.collect()
    _log("after dropping X_train/X_val")

    wrapper = MultiHeadMLPWeatherModel(
        keras_model=model,
        feature_builder=feature_builder,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        metadata=metadata,
    )

    print("\n--- Evaluating model in physical units ---")
    train_metrics = _evaluate_split(wrapper, train_features, train_targets)
    val_metrics = _evaluate_split(wrapper, val_features, val_targets)
    test_metrics = _evaluate_split(wrapper, test_features, test_targets)

    _print_metrics_table(train_metrics, "TRAIN METRICS - PHYSICAL UNITS")
    _print_metrics_table(val_metrics, "VALIDATION METRICS - PHYSICAL UNITS")
    _print_metrics_table(test_metrics, "HELD-OUT TEST METRICS - PHYSICAL UNITS")

    if TRAIN_SAVE_PLOTS:
        _safe_save_plot(_save_metric_plots, test_metrics)

    metrics_payload = {
        "model_type": "Multi-Task Multi-Head MLP Regressor",
        "backend": "multi_head_mlp",
        "artifact_dir": str(ARTIFACT_DIR),
        "model_path": str(ARTIFACT_DIR / MODEL_FILENAME),
        "normalization_path": str(ARTIFACT_DIR / NORMALIZATION_FILENAME),
        "feature_metadata_path": str(ARTIFACT_DIR / FEATURE_METADATA_FILENAME),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    _save_json(ARTIFACT_DIR / "training_metrics.json", metrics_payload)

    print("\nSaved artifacts to:", ARTIFACT_DIR)
    print("Main files:")
    for filename in [
        MODEL_FILENAME,
        NORMALIZATION_FILENAME,
        FEATURE_METADATA_FILENAME,
        METADATA_FILENAME,
        "training_history.json",
        "training_metrics.json",
        "training_log.csv",
    ]:
        print(" -", ARTIFACT_DIR / filename)
    if TRAIN_SAVE_PLOTS:
        print("Plots:", PLOTS_DIR)
    _log("training script complete")


if __name__ == "__main__":
    run_training()

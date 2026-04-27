"""Train the Multi-Task Multi-Head MLP Regressor weather model.

Designed by hand. TensorFlow only provides:
    * Dense / BatchNormalization / Dropout layer math
    * ReLU and linear activations
    * Adam optimizer
    * Huber loss
    * autodiff (backprop)
    * model.fit + callbacks (EarlyStopping, ReduceLROnPlateau, ...)

KEY CHANGE (this version): the default split is now ``random`` rather than
``temporal_day``. Why: the previous day-based split (train May 1-23, val 24-27,
test 28-31) was effectively measuring "future-day forecasting", which made
validation loss start high and drift up — see the previous run's wind plots.
For an interpolation/lookup service used by a ballistic simulator, a random
split across all collected samples is the honest evaluation: it asks "given an
unseen (lat, lon, altitude, time) point, can the model recover the ERA5 value
there?" Set TRAIN_SPLIT_MODE=temporal_day to recover the old behavior.

Important features:
    * Float32 throughout. ~32 GB RAM target.
    * Aggressive freeing of intermediate arrays.
    * Dry-run mode (TRAIN_DRY_RUN=1).
    * All settings exposed via env vars.
    * EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + CSVLogger.
    * Plotting wrapped in try/except so a crash never destroys a long run.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_gam_weather.config import SamplingConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day
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

print("RUNNING UPDATED MULTI_HEAD_MLP TRAINER VERSION 2026-04-27-v2")

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


# ---------------------------------------------------------------------------
# Environment-variable parsing.
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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _parse_months(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_loss_weights(raw: str) -> Optional[Dict[str, float]]:
    """Parse ``T:1.0,P:0.5,U:1.5,V:1.5`` into a dict keyed by TARGET_OUTPUTS.

    Empty / ``None`` returns None which means "use uniform weights".
    """
    if not raw or not raw.strip():
        return None
    short_to_long = {short: ERA5_TO_OUTPUT[short] for short in ERA5_TO_OUTPUT}
    out: Dict[str, float] = {}
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise SystemExit(
                f"Invalid TRAIN_LOSS_WEIGHTS entry {piece!r}; expected format 'T:1.0,P:0.5,U:1.5,V:1.5'."
            )
        key, val = piece.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in short_to_long:
            output_name = short_to_long[key]
        elif key in TARGET_OUTPUTS:
            output_name = key
        else:
            raise SystemExit(
                f"Unknown target {key!r} in TRAIN_LOSS_WEIGHTS. Use one of T/P/U/V "
                f"or {list(TARGET_OUTPUTS)}."
            )
        try:
            out[output_name] = float(val)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight value for {key}: {val!r}") from exc
    return out if out else None


YEAR = _env_int("TRAIN_YEAR", 2025)
MONTHS = _parse_months(os.getenv("TRAIN_MONTHS", "5"))
TRAIN_SAMPLES_PER_FILE = _env_int("TRAIN_SAMPLES_PER_FILE", 50000)
EVAL_SAMPLES_PER_FILE = _env_int("EVAL_SAMPLES_PER_FILE", 10000)
SEED = _env_int("TRAIN_SEED", 42)
BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 1024)
MAX_EPOCHS = _env_int("TRAIN_MAX_EPOCHS", 250)
LEARNING_RATE = _env_float("TRAIN_LEARNING_RATE", 1e-3)
DROPOUT_RATE = _env_float("TRAIN_DROPOUT_RATE", 0.04)
EARLY_STOPPING_PATIENCE = _env_int("TRAIN_EARLY_STOPPING_PATIENCE", 25)
REDUCE_LR_PATIENCE = _env_int("TRAIN_REDUCE_LR_PATIENCE", 10)
TRAIN_MAX_TOTAL_SAMPLES = _env_int("TRAIN_MAX_TOTAL_SAMPLES", 0)        # 0 = no cap
TRAIN_MAX_TOTAL_EVAL_SAMPLES = _env_int("TRAIN_MAX_TOTAL_EVAL_SAMPLES", 0)
TRAIN_SAVE_PLOTS = _env_bool("TRAIN_SAVE_PLOTS", True)
TRAIN_DRY_RUN = _env_bool("TRAIN_DRY_RUN", False)
L2_WEIGHT_DECAY = _env_float("TRAIN_L2_WEIGHT_DECAY", 0.0)
LOSS_WEIGHTS = _parse_loss_weights(os.getenv("TRAIN_LOSS_WEIGHTS", ""))

# Split control.
SPLIT_MODE = _env_str("TRAIN_SPLIT_MODE", "random").lower()
if SPLIT_MODE not in {"random", "temporal_day"}:
    raise SystemExit(
        f"Invalid TRAIN_SPLIT_MODE={SPLIT_MODE!r}. Use 'random' (default) or 'temporal_day'."
    )

# Random-mode split fractions (sum should be 1.0).
SPLIT_TRAIN_FRACTION = _env_float("TRAIN_SPLIT_TRAIN_FRACTION", 0.8)
SPLIT_VAL_FRACTION = _env_float("TRAIN_SPLIT_VAL_FRACTION", 0.1)

# Temporal-mode day boundaries (kept for reproducibility of the legacy split).
VALIDATION_SPLIT_TRAIN_END = _env_int("TRAIN_SPLIT_TRAIN_END_DAY", 23)
VALIDATION_SPLIT_VAL_END = _env_int("TRAIN_SPLIT_VAL_END_DAY", 27)

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
# Sample collection.
# ---------------------------------------------------------------------------
def _empty_feature_dict() -> Dict[str, list]:
    return {k: [] for k in RAW_FEATURE_NAMES}


def _empty_target_dict() -> Dict[str, list]:
    return {k: [] for k in TARGET_NAMES}


def _append_batch(storage_features: Dict[str, list], storage_targets: Dict[str, list], batch) -> int:
    n = len(batch.features["lat"])
    if n == 0:
        return 0
    for k in RAW_FEATURE_NAMES:
        storage_features[k].append(np.asarray(batch.features[k], dtype=np.float32))
    for k in TARGET_NAMES:
        storage_targets[k].append(np.asarray(batch.targets[k], dtype=np.float32))
    return n


def _finalize(storage_features: Dict[str, list], storage_targets: Dict[str, list]):
    """Concatenate per-file lists into single contiguous float32 arrays."""
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


def _collect(files: Iterable[str], config: SamplingConfig, tag: str, max_total: int = 0):
    sf = _empty_feature_dict()
    st = _empty_target_dict()
    total = 0
    file_list = list(files)
    for idx, path in enumerate(file_list, start=1):
        if max_total > 0 and total >= max_total:
            print(f"[{tag}] reached cap={max_total}, stopping after {idx-1}/{len(file_list)} files.")
            break
        print(f"[{tag} {idx}/{len(file_list)}] {path}")
        batch = sample_from_file(path, config)
        added = _append_batch(sf, st, batch)
        total += added
        del batch
    print(f"[{tag}] collected {total} rows from {min(idx, len(file_list))} files")
    return _finalize(sf, st)


def _index_subset(arr_dict: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in arr_dict.items()}


def _normalise_X_inplace(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """Normalize in place so we don't double the memory footprint of X."""
    np.subtract(X, x_mean.reshape(1, -1), out=X)
    np.divide(X, x_std.reshape(1, -1), out=X)
    return X


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
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
    out: Dict[str, Dict[str, float]] = {}
    for era5_key, output_name in ERA5_TO_OUTPUT.items():
        m = _metrics(targets[era5_key], pred[output_name])
        if output_name == "pressure_pa":
            # Add hPa convenience copies so the supervisor-facing output is readable.
            m["mae_hpa"] = m["mae"] / 100.0
            m["rmse_hpa"] = m["rmse"] / 100.0
            m["max_abs_hpa"] = m["max_abs"] / 100.0
        out[output_name] = m
    return out


def _print_metrics_table(metrics: Dict[str, Dict[str, float]], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"{'Target':<18} {'MAE':>14} {'RMSE':>14} {'Max Abs':>14}  {'unit':<8}")
    print("-" * 80)
    units = {"temperature_k": "K", "pressure_pa": "Pa", "wind_u": "m/s", "wind_v": "m/s"}
    for name in TARGET_OUTPUTS:
        m = metrics[name]
        print(f"{name:<18} {m['mae']:>14.6f} {m['rmse']:>14.6f} {m['max_abs']:>14.6f}  {units[name]:<8}")
        if name == "pressure_pa":
            print(f"{'  (in hPa)':<18} {m['mae_hpa']:>14.4f} {m['rmse_hpa']:>14.4f} {m['max_abs_hpa']:>14.4f}  {'hPa':<8}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def _safe_save_plot(fn, *args, **kwargs) -> None:
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
    plt.ylabel("Weighted Huber loss on normalized targets")
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


def _save_metric_plots(test_metrics: Dict[str, Dict[str, float]],
                       y_std_physical: Dict[str, float]) -> None:
    """Bar charts of test metrics, plus a normalized-error chart for fair cross-target comparison.

    Pressure values in the original chart dwarf temperature/wind because Pa is a
    huge unit. We add a normalized-error chart where each metric is divided by
    that target's physical-units standard deviation, so all four targets share
    a comparable y-scale.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = list(TARGET_OUTPUTS)
    units = {"temperature_k": "K", "pressure_pa": "Pa", "wind_u": "m/s", "wind_v": "m/s"}

    metric_specs = [
        ("mae", "MAE (physical units)", "test_mae_by_target.png"),
        ("rmse", "RMSE (physical units)", "test_rmse_by_target.png"),
        ("max_abs", "Max Absolute Error (physical units)", "test_max_abs_by_target.png"),
    ]
    for key, ylabel, filename in metric_specs:
        values = [float(test_metrics[t][key]) for t in targets]
        labels = [f"{t}\n[{units[t]}]" for t in targets]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.ylabel(ylabel)
        plt.title(f"Held-out Test {ylabel} by Target")
        ymax = max(values) if values else 1.0
        for i, value in enumerate(values):
            plt.text(i, value + ymax * 0.02, f"{value:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close()

    # Pressure-only plot in hPa for readability.
    p = test_metrics["pressure_pa"]
    plt.figure(figsize=(8, 5))
    plt.bar(["MAE", "RMSE", "Max Abs"], [p["mae_hpa"], p["rmse_hpa"], p["max_abs_hpa"]])
    plt.ylabel("Error (hPa)")
    plt.title("Held-out Test Pressure Error (hPa)")
    for i, v in enumerate([p["mae_hpa"], p["rmse_hpa"], p["max_abs_hpa"]]):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_pressure_error_hpa.png", dpi=200)
    plt.close()

    # Normalized error: MAE / std(y_physical) per target — cross-target fair comparison.
    norm_mae = []
    for t in targets:
        sd = float(y_std_physical.get(t, 1.0)) or 1.0
        norm_mae.append(test_metrics[t]["mae"] / sd)
    plt.figure(figsize=(10, 6))
    plt.bar(targets, norm_mae)
    plt.ylabel("MAE / std(y)  (unitless)")
    plt.title("Held-out Test Normalized MAE — fair cross-target comparison")
    ymax = max(norm_mae) if norm_mae else 1.0
    for i, v in enumerate(norm_mae):
        plt.text(i, v + ymax * 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_mae_normalized_by_target.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Configuration printing & sanity checks.
# ---------------------------------------------------------------------------
def _print_config_block() -> None:
    print("=" * 80)
    print("Multi-Task Multi-Head MLP Weather Trainer")
    print("=" * 80)
    print(f"  ERA5_DATA_ROOT                = {DATA_ROOT}")
    print(f"  WEATHER_ARTIFACT_DIR          = {ARTIFACT_DIR}")
    print(f"  TRAIN_YEAR                    = {YEAR}")
    print(f"  TRAIN_MONTHS                  = {MONTHS}")
    print(f"  TRAIN_SPLIT_MODE              = {SPLIT_MODE}")
    if SPLIT_MODE == "random":
        print(f"    TRAIN_SPLIT_TRAIN_FRACTION  = {SPLIT_TRAIN_FRACTION}")
        print(f"    TRAIN_SPLIT_VAL_FRACTION    = {SPLIT_VAL_FRACTION}")
        print(f"    (test fraction implied)     = {1.0 - SPLIT_TRAIN_FRACTION - SPLIT_VAL_FRACTION:.3f}")
    else:
        print(f"    TRAIN_SPLIT_TRAIN_END_DAY   = {VALIDATION_SPLIT_TRAIN_END}")
        print(f"    TRAIN_SPLIT_VAL_END_DAY     = {VALIDATION_SPLIT_VAL_END}")
    print(f"  TRAIN_SAMPLES_PER_FILE        = {TRAIN_SAMPLES_PER_FILE}")
    print(f"  EVAL_SAMPLES_PER_FILE         = {EVAL_SAMPLES_PER_FILE}  (used only in temporal_day mode)")
    print(f"  TRAIN_MAX_TOTAL_SAMPLES       = {TRAIN_MAX_TOTAL_SAMPLES} (0 = unlimited)")
    print(f"  TRAIN_MAX_TOTAL_EVAL_SAMPLES  = {TRAIN_MAX_TOTAL_EVAL_SAMPLES} (0 = unlimited)")
    print(f"  TRAIN_BATCH_SIZE              = {BATCH_SIZE}")
    print(f"  TRAIN_MAX_EPOCHS              = {MAX_EPOCHS}")
    print(f"  TRAIN_LEARNING_RATE           = {LEARNING_RATE}")
    print(f"  TRAIN_DROPOUT_RATE            = {DROPOUT_RATE}")
    print(f"  TRAIN_L2_WEIGHT_DECAY         = {L2_WEIGHT_DECAY}")
    print(f"  TRAIN_LOSS_WEIGHTS            = {LOSS_WEIGHTS or '(uniform 1.0 each)'}")
    print(f"  TRAIN_EARLY_STOPPING_PAT.     = {EARLY_STOPPING_PATIENCE}")
    print(f"  TRAIN_REDUCE_LR_PATIENCE      = {REDUCE_LR_PATIENCE}")
    print(f"  TRAIN_SAVE_PLOTS              = {TRAIN_SAVE_PLOTS}")
    print(f"  TRAIN_DRY_RUN                 = {TRAIN_DRY_RUN}")
    print(f"  TRAIN_SEED                    = {SEED}")
    print("=" * 80)


def _estimate_peak_gb(n_total: int, n_features: int) -> float:
    """Rough peak-RAM estimate in GB for the in-memory training arrays."""
    bytes_per = 4
    raw_targets = n_total * 4 * bytes_per
    raw_features = n_total * len(RAW_FEATURE_NAMES) * bytes_per
    eng_features = n_total * n_features * bytes_per
    norm_targets = n_total * 4 * bytes_per
    overhead_gb = 0.5
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
# Split builders.
# ---------------------------------------------------------------------------
def _make_random_split(
    pool_features: Dict[str, np.ndarray],
    pool_targets: Dict[str, np.ndarray],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray],
]:
    """Pool all collected samples and split deterministically by random permutation."""
    n_total = len(pool_features["lat"])
    if n_total == 0:
        raise RuntimeError("Pool is empty — nothing to split.")
    test_fraction = 1.0 - train_fraction - val_fraction
    if train_fraction <= 0 or val_fraction <= 0 or test_fraction <= 0:
        raise SystemExit(
            f"Bad split fractions: train={train_fraction}, val={val_fraction}, "
            f"test={test_fraction}. All three must be positive."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).astype(np.int64, copy=False)

    n_train = int(round(train_fraction * n_total))
    n_val = int(round(val_fraction * n_total))
    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    print(f"random split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_features = _index_subset(pool_features, train_idx)
    train_targets = _index_subset(pool_targets, train_idx)
    val_features = _index_subset(pool_features, val_idx)
    val_targets = _index_subset(pool_targets, val_idx)
    test_features = _index_subset(pool_features, test_idx)
    test_targets = _index_subset(pool_targets, test_idx)
    return train_features, train_targets, val_features, val_targets, test_features, test_targets


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
    for path in files[:5]:
        print("  -", path)
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    for path in files[max(5, len(files) - 5):]:
        print("  -", path)

    # ---- estimate ---------------------------------------------------------
    feature_builder_preview = WeatherFeatureBuilder(WeatherFeatureConfig())
    n_features_preview = len(feature_builder_preview.feature_names)
    n_total_est = len(files) * TRAIN_SAMPLES_PER_FILE
    if TRAIN_MAX_TOTAL_SAMPLES > 0:
        n_total_est = min(n_total_est, TRAIN_MAX_TOTAL_SAMPLES)
    est_gb = _estimate_peak_gb(n_total_est, n_features_preview)
    print(
        f"\nEstimated total pool size:        ~{n_total_est} rows\n"
        f"Estimated peak in-memory footprint: ~{est_gb:.2f} GB\n"
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

    # ---- TF setup ---------------------------------------------------------
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

    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Sample collection ------------------------------------------------
    train_sampling = SamplingConfig(
        samples_per_file=TRAIN_SAMPLES_PER_FILE,
        seed=SEED,
        stratified_time_level=True,
    )
    eval_sampling = SamplingConfig(
        samples_per_file=EVAL_SAMPLES_PER_FILE,
        seed=SEED + 100,
        stratified_time_level=True,
    )

    if SPLIT_MODE == "random":
        # One pool, then split by random permutation.
        print("\n--- Collecting pool (TRAIN_SPLIT_MODE=random) ---")
        pool_features, pool_targets = _collect(
            files, train_sampling, "POOL", max_total=TRAIN_MAX_TOTAL_SAMPLES,
        )
        _log("after pool collection")
        (train_features, train_targets,
         val_features, val_targets,
         test_features, test_targets) = _make_random_split(
            pool_features, pool_targets,
            train_fraction=SPLIT_TRAIN_FRACTION,
            val_fraction=SPLIT_VAL_FRACTION,
            seed=SEED,
        )
        del pool_features, pool_targets
        gc.collect()
        _log("after random split & pool freed")

    else:  # temporal_day
        print(f"\n--- Splitting files by day (train_end={VALIDATION_SPLIT_TRAIN_END}, val_end={VALIDATION_SPLIT_VAL_END}) ---")
        splits = split_files_by_day(
            files, train_end=VALIDATION_SPLIT_TRAIN_END, val_end=VALIDATION_SPLIT_VAL_END,
        )
        n_train_files = len(splits["train"])
        n_val_files = len(splits["val"])
        n_test_files = len(splits["test"])
        print(f"  Train files: {n_train_files}, Val files: {n_val_files}, Test files: {n_test_files}")
        if n_train_files == 0 or n_val_files == 0 or n_test_files == 0:
            raise RuntimeError(
                "Train/validation/test split produced an empty split. "
                "Adjust TRAIN_SPLIT_TRAIN_END_DAY / TRAIN_SPLIT_VAL_END_DAY."
            )
        print("\n--- Collecting train ---")
        train_features, train_targets = _collect(splits["train"], train_sampling, "TRAIN", max_total=TRAIN_MAX_TOTAL_SAMPLES)
        _log("after train collection")
        print("\n--- Collecting val ---")
        val_features, val_targets = _collect(splits["val"], eval_sampling, "VAL", max_total=TRAIN_MAX_TOTAL_EVAL_SAMPLES)
        _log("after val collection")
        print("\n--- Collecting test ---")
        test_features, test_targets = _collect(splits["test"], eval_sampling, "TEST", max_total=TRAIN_MAX_TOTAL_EVAL_SAMPLES)
        _log("after test collection")

    n_train = len(train_features["lat"])
    n_val = len(val_features["lat"])
    n_test = len(test_features["lat"])
    print(f"\nDataset sizes after split: train={n_train}, val={n_val}, test={n_test}")
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise RuntimeError("Train/validation/test split produced an empty split.")

    # ---- Feature engineering ---------------------------------------------
    feature_builder = WeatherFeatureBuilder(WeatherFeatureConfig())
    print(f"Feature count: {len(feature_builder.feature_names)}")

    print("\n--- Building engineered features ---")
    X_train_raw = feature_builder.transform(train_features)
    _log(f"X_train shape={X_train_raw.shape} dtype={X_train_raw.dtype}")
    X_val_raw = feature_builder.transform(val_features)
    _log(f"X_val   shape={X_val_raw.shape} dtype={X_val_raw.dtype}")

    x_mean, x_std, y_mean, y_std, _ = MultiHeadMLPWeatherModel.compute_normalization(
        X_train_raw, train_targets,
    )

    # In-place normalization.
    X_train = _normalise_X_inplace(X_train_raw, x_mean, x_std)
    X_val = _normalise_X_inplace(X_val_raw, x_mean, x_std)
    del X_train_raw, X_val_raw
    gc.collect()
    _log("after normalization (raw X freed)")

    y_train = MultiHeadMLPWeatherModel.normalize_targets(train_targets, y_mean, y_std)
    y_val = MultiHeadMLPWeatherModel.normalize_targets(val_targets, y_mean, y_std)
    _log("after target normalization")

    # ---- Build model ------------------------------------------------------
    model = build_multi_head_mlp(
        n_features=X_train.shape[1],
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        l2_weight_decay=L2_WEIGHT_DECAY,
        loss_weights=LOSS_WEIGHTS,
    )
    model.summary(print_fn=lambda line: print(line, flush=True))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ARTIFACT_DIR / MODEL_FILENAME),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(ARTIFACT_DIR / "training_log.csv"), append=False),
    ]

    print("\n--- Starting model.fit ---")
    history_obj = model.fit(
        X_train, y_train,
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

    # ---- Save metadata & artifacts ---------------------------------------
    metadata = {
        "model_type": "Multi-Task Multi-Head MLP Regressor",
        "backend": "multi_head_mlp",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "year": YEAR,
        "months": MONTHS,
        "split_mode": SPLIT_MODE,
        "split_config": (
            {"train_fraction": SPLIT_TRAIN_FRACTION, "val_fraction": SPLIT_VAL_FRACTION,
             "test_fraction": 1.0 - SPLIT_TRAIN_FRACTION - SPLIT_VAL_FRACTION}
            if SPLIT_MODE == "random"
            else {"train_end_day": VALIDATION_SPLIT_TRAIN_END, "val_end_day": VALIDATION_SPLIT_VAL_END}
        ),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": SEED,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout_rate": DROPOUT_RATE,
            "l2_weight_decay": L2_WEIGHT_DECAY,
            "loss_weights": LOSS_WEIGHTS or {n: 1.0 for n in TARGET_OUTPUTS},
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "reduce_lr_patience": REDUCE_LR_PATIENCE,
            "loss": "Huber(delta=1.0) on normalized targets per head",
            "optimizer": "Adam",
            "max_total_train_samples": TRAIN_MAX_TOTAL_SAMPLES,
        },
        "architecture": {
            "shared_trunk": [
                "Dense(256, relu)",
                "BatchNormalization",
                f"Dropout({DROPOUT_RATE})",
                "Dense(256, relu)",
                "BatchNormalization",
                f"Dropout({DROPOUT_RATE})",
                "Dense(128, relu)",
                "BatchNormalization",
            ],
            "heads": {
                "temperature_k": ["Dense(64, relu)", "Dense(1, linear)"],
                "pressure_pa": ["Dense(64, relu)", "Dense(1, linear)"],
                "wind_u": ["Dense(128, relu)", "Dense(64, relu)", "Dense(1, linear)"],
                "wind_v": ["Dense(128, relu)", "Dense(64, relu)", "Dense(1, linear)"],
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

    def _save_json(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    _save_json(ARTIFACT_DIR / NORMALIZATION_FILENAME, normalization_payload)
    _save_json(ARTIFACT_DIR / FEATURE_METADATA_FILENAME, feature_builder.metadata())
    _save_json(ARTIFACT_DIR / METADATA_FILENAME, metadata)

    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    _save_json(ARTIFACT_DIR / "training_history.json", history)

    if TRAIN_SAVE_PLOTS:
        _safe_save_plot(_save_training_plots, history)

    # Drop normalized X arrays — done with model.fit.
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

    # ---- Physical-units evaluation ---------------------------------------
    print("\n--- Evaluating model in physical units ---")
    train_metrics = _evaluate_split(wrapper, train_features, train_targets)
    val_metrics = _evaluate_split(wrapper, val_features, val_targets)
    test_metrics = _evaluate_split(wrapper, test_features, test_targets)

    _print_metrics_table(train_metrics, "TRAIN METRICS — PHYSICAL UNITS")
    _print_metrics_table(val_metrics, "VALIDATION METRICS — PHYSICAL UNITS")
    _print_metrics_table(test_metrics, "HELD-OUT TEST METRICS — PHYSICAL UNITS")

    # Compute physical-unit standard deviations of TEST targets to express
    # "normalized error" per target — useful because pressure (in Pa) dwarfs
    # temperature/wind otherwise.
    y_std_physical: Dict[str, float] = {}
    for era5_key, output_name in ERA5_TO_OUTPUT.items():
        y_std_physical[output_name] = float(np.std(np.asarray(test_targets[era5_key], dtype=np.float64)))

    if TRAIN_SAVE_PLOTS:
        _safe_save_plot(_save_metric_plots, test_metrics, y_std_physical)

    metrics_payload = {
        "model_type": "Multi-Task Multi-Head MLP Regressor",
        "backend": "multi_head_mlp",
        "split_mode": SPLIT_MODE,
        "artifact_dir": str(ARTIFACT_DIR),
        "model_path": str(ARTIFACT_DIR / MODEL_FILENAME),
        "normalization_path": str(ARTIFACT_DIR / NORMALIZATION_FILENAME),
        "feature_metadata_path": str(ARTIFACT_DIR / FEATURE_METADATA_FILENAME),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "y_std_physical_test": y_std_physical,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    _save_json(ARTIFACT_DIR / "training_metrics.json", metrics_payload)

    # ---- Quick health check on the val/train gap -------------------------
    val_loss_last = history.get("val_loss", [None])[-1]
    train_loss_last = history.get("loss", [None])[-1]
    if val_loss_last is not None and train_loss_last is not None and train_loss_last > 0:
        ratio = val_loss_last / train_loss_last
        print(f"\nFinal train_loss={train_loss_last:.6f}, val_loss={val_loss_last:.6f}, ratio={ratio:.2f}x")
        if ratio > 2.5:
            print(
                "NOTE: validation loss is more than 2.5× training loss.\n"
                "  In TRAIN_SPLIT_MODE=random this signals real overfitting → try:\n"
                f"    TRAIN_L2_WEIGHT_DECAY=1e-5  (current={L2_WEIGHT_DECAY})\n"
                f"    TRAIN_DROPOUT_RATE=0.10     (current={DROPOUT_RATE})\n"
                f"    TRAIN_LEARNING_RATE=5e-4    (current={LEARNING_RATE})\n"
                "  In TRAIN_SPLIT_MODE=temporal_day this is expected for wind."
            )

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

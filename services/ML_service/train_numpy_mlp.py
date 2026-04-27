"""
Training script for the NumPy MLP weather model.

This script:
  1. Loads ERA5 data using the existing project sampler.
  2. Splits data deterministically into train / val / test by calendar day.
  3. Trains 4 independent NumPy MLP networks (one per weather target).
  4. Evaluates on train / val / test splits.
  5. Saves the model artefact (.npz), metrics (JSON), and training curves (JSON).

Usage:
    python train_numpy_mlp.py

No sklearn models, no PyTorch, no TensorFlow – only NumPy for the ML algorithm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt

from era5_gam_weather.config import SamplingConfig, SplitConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day
from era5_gam_weather.numpy_mlp_model import NumpyMLPWeatherModel, DEFAULT_HPARAMS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEAR = 2025
MONTHS = [4, 5]

TRAIN_SAMPLES_PER_FILE = 60000
EVAL_SAMPLES_PER_FILE = 12000

TARGET_NAMES = ["T", "P", "U", "V"]
FEATURE_NAMES = ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]

SEED = 42

# Architecture: Input → 128 → 128 → 64 → 1
LAYER_SIZES = (128, 128, 64, 1)

# Per-target hyper-parameters (tuned for weather data)
HPARAMS = {
    "T": {"lr": 5e-4, "epochs": 400, "batch_size": 512, "patience": 40, "weight_decay": 1e-5},
    "P": {"lr": 5e-4, "epochs": 400, "batch_size": 512, "patience": 40, "weight_decay": 1e-5},
    "U": {"lr": 3e-4, "epochs": 600, "batch_size": 512, "patience": 60, "weight_decay": 1e-5},
    "V": {"lr": 3e-4, "epochs": 600, "batch_size": 512, "patience": 60, "weight_decay": 1e-5},
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for base in [start] + list(start.parents):
        if (base / "data" / "era5").exists():
            return base
    return start.parent


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(THIS_DIR)
DATA_ROOT = PROJECT_ROOT / "data" / "era5"
ARTIFACT_DIR = THIS_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "numpy_mlp_weather_model.npz"
METRICS_PATH = ARTIFACT_DIR / "eval_metrics_numpy_mlp.json"
CURVES_PATH = ARTIFACT_DIR / "training_curves_numpy_mlp.json"

# Plot output directories (separate from sklearn plots so nothing is overwritten)
PLOTS_DIR = ARTIFACT_DIR / "numpy_mlp_training_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data collection helpers (same logic as train_may_tree.py)
# ---------------------------------------------------------------------------
def _empty_feature_dict() -> Dict[str, list]:
    return {k: [] for k in FEATURE_NAMES}


def _empty_target_dict() -> Dict[str, list]:
    return {k: [] for k in TARGET_NAMES}


def _append_batch(sf: Dict[str, list], st: Dict[str, list], batch) -> int:
    n = len(batch.features["lat"])
    if n == 0:
        return 0
    for k in FEATURE_NAMES:
        sf[k].append(np.asarray(batch.features[k], dtype=np.float64))
    for k in TARGET_NAMES:
        st[k].append(np.asarray(batch.targets[k], dtype=np.float64))
    return n


def _finalize(sf: Dict[str, list], st: Dict[str, list]):
    features = {k: np.concatenate(v) if v else np.empty(0, dtype=np.float64) for k, v in sf.items()}
    targets = {k: np.concatenate(v) if v else np.empty(0, dtype=np.float64) for k, v in st.items()}
    return features, targets


def _collect(files: Iterable[str], config: SamplingConfig, tag: str):
    sf = _empty_feature_dict()
    st = _empty_target_dict()
    total = 0
    for idx, path in enumerate(files, 1):
        print(f"[{tag} {idx}] {path}")
        batch = sample_from_file(path, config)
        total += _append_batch(sf, st, batch)
    print(f"[{tag}] collected {total} rows")
    return _finalize(sf, st)


# ---------------------------------------------------------------------------
# Plot generation (mirrors the sklearn training script's outputs)
# ---------------------------------------------------------------------------
def _save_training_curves(model) -> None:
    """One PNG per target showing training & validation loss across epochs."""
    for tgt in TARGET_NAMES:
        history = model.histories[tgt]
        train = np.asarray(history.train_loss, dtype=np.float64)
        val = np.asarray(history.val_loss, dtype=np.float64)
        if train.size == 0:
            continue

        epochs = np.arange(1, len(train) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train, marker="o", markersize=3, linewidth=2, label="Training loss")
        if val.size > 0:
            plt.plot(epochs, val, marker="o", markersize=3, linewidth=2, label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (normalised MSE)")
        plt.title(f"{tgt} - Training and Validation Loss (NumPy MLP)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{tgt.lower()}_training_validation_loss.png", dpi=300)
        plt.close()


def _save_test_metric_plots(test_metrics: Dict[str, Dict[str, float]]) -> None:
    """Bar charts for held-out test MAE / RMSE / max-abs error."""
    targets = TARGET_NAMES
    mae_vals = [test_metrics[t]["mae"] for t in targets]
    rmse_vals = [test_metrics[t]["rmse"] for t in targets]
    max_abs_vals = [test_metrics[t]["max_abs"] for t in targets]

    for values, ylabel, fname, title in [
        (mae_vals, "MAE", "test_mae_by_target.png", "Held-out Test MAE by Target (NumPy MLP)"),
        (rmse_vals, "RMSE", "test_rmse_by_target.png", "Held-out Test RMSE by Target (NumPy MLP)"),
        (max_abs_vals, "Max Absolute Error", "test_max_abs_by_target.png",
         "Held-out Test Max Absolute Error by Target (NumPy MLP)"),
    ]:
        plt.figure(figsize=(10, 6))
        plt.bar(targets, values)
        plt.xlabel("Target")
        plt.ylabel(ylabel)
        plt.title(title)
        ymax = max(values) if max(values) > 0 else 1.0
        for i, v in enumerate(values):
            plt.text(i, v + ymax * 0.02, f"{v:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=300)
        plt.close()

    with open(PLOTS_DIR / "plots_summary.txt", "w", encoding="utf-8") as f:
        f.write("Generated plots (NumPy MLP):\n")
        f.write("- One training/validation loss curve per target: T, P, U, V\n")
        f.write("- Held-out test MAE by target\n")
        f.write("- Held-out test RMSE by target\n")
        f.write("- Held-out test Max Absolute Error by target\n")


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
def _print_metrics_table(metrics: Dict[str, Dict[str, float]], title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Target':<8} {'MAE':>12} {'RMSE':>12} {'Max Abs':>12}")
    print(f"  {'-'*44}")
    for tgt in TARGET_NAMES:
        m = metrics[tgt]
        print(f"  {tgt:<8} {m['mae']:>12.4f} {m['rmse']:>12.4f} {m['max_abs']:>12.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------------
def run_training() -> None:
    if not DATA_ROOT.exists():
        print(f"ERROR: ERA5 data directory not found: {DATA_ROOT}", file=sys.stderr)
        print("Set ERA5 data path or ensure 'data/era5/' exists relative to project root.")
        sys.exit(1)

    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)
    train_sampling = SamplingConfig(samples_per_file=TRAIN_SAMPLES_PER_FILE, seed=SEED,
                                    stratified_time_level=True)
    eval_sampling = SamplingConfig(samples_per_file=EVAL_SAMPLES_PER_FILE, seed=SEED + 100,
                                   stratified_time_level=True)

    files = discover_era5_files(str(DATA_ROOT), YEAR, MONTHS)
    if not files:
        print(f"ERROR: No ERA5 files found under {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    splits = split_files_by_day(files,
                                 train_end=split_config.train_end_day_inclusive,
                                 val_end=split_config.val_end_day_inclusive)

    print("--- Collecting training data ---")
    train_features, train_targets = _collect(splits["train"], train_sampling, "TRAIN")

    print("\n--- Collecting validation data ---")
    val_features, val_targets = _collect(splits["val"], eval_sampling, "VAL")

    print("\n--- Collecting test data ---")
    test_features, test_targets = _collect(splits["test"], eval_sampling, "TEST")

    n_train = len(train_features["lat"])
    n_val = len(val_features["lat"])
    n_test = len(test_features["lat"])
    print(f"\nDataset sizes:  train={n_train}  val={n_val}  test={n_test}")

    if n_train == 0:
        print("ERROR: No training rows!", file=sys.stderr)
        sys.exit(1)

    # ---- Train ----
    print("\n" + "=" * 60)
    print("  TRAINING NUMPY MLP WEATHER MODEL")
    print("=" * 60)

    model = NumpyMLPWeatherModel.train_from_data(
        train_features=train_features,
        train_targets=train_targets,
        val_features=val_features,
        val_targets=val_targets,
        layer_sizes=LAYER_SIZES,
        hparams=HPARAMS,
        seed=SEED,
        verbose=True,
    )

    # ---- Evaluate ----
    print("\n--- Evaluating on train split ---")
    train_metrics = model.evaluate(train_features, train_targets, "train")
    _print_metrics_table(train_metrics, "TRAIN METRICS")

    print("--- Evaluating on validation split ---")
    val_metrics = model.evaluate(val_features, val_targets, "val")
    _print_metrics_table(val_metrics, "VALIDATION METRICS")

    print("--- Evaluating on test split ---")
    test_metrics = model.evaluate(test_features, test_targets, "test")
    _print_metrics_table(test_metrics, "TEST METRICS")

    # ---- Save model ----
    model.save(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")

    # ---- Save plots ----
    _save_training_curves(model)
    _save_test_metric_plots(test_metrics)
    print(f"Plots saved to:  {PLOTS_DIR}")

    # ---- Save metrics ----
    payload = {
        "model_type": "NumpyMLPWeatherModel",
        "year": YEAR,
        "months": MONTHS,
        "model_path": str(MODEL_PATH),
        "architecture": {
            "layer_sizes": list(LAYER_SIZES),
            "n_input_features": model._metadata.get("n_input_features", 0),
        },
        "hparams": HPARAMS,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": SEED,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    # ---- Save training curves ----
    curves = {}
    for tgt in TARGET_NAMES:
        curves[tgt] = {
            "train_loss": model.histories[tgt].train_loss,
            "val_loss": model.histories[tgt].val_loss,
        }
    with open(CURVES_PATH, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    print(f"Training curves saved to: {CURVES_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    run_training()

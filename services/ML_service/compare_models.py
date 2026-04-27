"""
Model comparison script – sklearn tree model vs. NumPy MLP.

Loads both models and evaluates them on the same test data, printing a
side-by-side comparison table.

Usage:
    python compare_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from era5_gam_weather.config import SamplingConfig, SplitConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day

TARGET_NAMES = ["T", "P", "U", "V"]
TARGET_LABELS = {"T": "temperature_k", "P": "pressure_pa", "U": "wind_u", "V": "wind_v"}
FEATURE_NAMES = ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]

YEAR = 2025
MONTHS = [4, 5]
EVAL_SAMPLES = 12000
SEED = 42


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

TREE_MODEL_PATH = ARTIFACT_DIR / "weather_tree_bundle_2025_04_05.joblib"
NUMPY_MODEL_PATH = ARTIFACT_DIR / "numpy_mlp_weather_model.npz"


def _collect_test_data():
    """Collect test split using the same method as training scripts."""
    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)
    sampling = SamplingConfig(samples_per_file=EVAL_SAMPLES, seed=SEED + 100,
                               stratified_time_level=True)

    files = discover_era5_files(str(DATA_ROOT), YEAR, MONTHS)
    if not files:
        print(f"ERROR: No ERA5 files found under {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    splits = split_files_by_day(files,
                                 train_end=split_config.train_end_day_inclusive,
                                 val_end=split_config.val_end_day_inclusive)

    sf = {k: [] for k in FEATURE_NAMES}
    st = {k: [] for k in TARGET_NAMES}
    for path in splits["test"]:
        batch = sample_from_file(path, sampling)
        for k in FEATURE_NAMES:
            sf[k].append(np.asarray(batch.features[k], dtype=np.float64))
        for k in TARGET_NAMES:
            st[k].append(np.asarray(batch.targets[k], dtype=np.float64))

    features = {k: np.concatenate(v) for k, v in sf.items()}
    targets = {k: np.concatenate(v) for k, v in st.items()}
    return features, targets


def _manual_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def main() -> None:
    print("Collecting test data...")
    features, targets = _collect_test_data()
    n = len(features["lat"])
    print(f"Test samples: {n}")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # --- sklearn tree model ---
    if TREE_MODEL_PATH.exists():
        try:
            from era5_gam_weather.tree_model import WeatherTreeBundle
            bundle = WeatherTreeBundle.load(str(TREE_MODEL_PATH))
            tree_metrics = bundle.evaluate(features, targets)
            results["sklearn_tree"] = {
                tgt: {"mae": tree_metrics.mae[tgt], "rmse": tree_metrics.rmse[tgt],
                      "max_abs": tree_metrics.max_abs[tgt]}
                for tgt in TARGET_NAMES
            }
            print("sklearn tree model loaded and evaluated.")
        except Exception as e:
            print(f"Could not load sklearn tree model: {e}")
    else:
        print(f"sklearn tree model not found at {TREE_MODEL_PATH}")

    # --- NumPy MLP model ---
    if NUMPY_MODEL_PATH.exists():
        try:
            from era5_gam_weather.numpy_mlp_model import NumpyMLPWeatherModel
            mlp_model = NumpyMLPWeatherModel.load(str(NUMPY_MODEL_PATH))
            mlp_metrics = mlp_model.evaluate(features, targets, "test")
            results["numpy_mlp"] = mlp_metrics
            print("NumPy MLP model loaded and evaluated.")
        except Exception as e:
            print(f"Could not load NumPy MLP model: {e}")
    else:
        print(f"NumPy MLP model not found at {NUMPY_MODEL_PATH}")

    if not results:
        print("\nNo models available for comparison.")
        sys.exit(1)

    # --- Print comparison table ---
    model_names = list(results.keys())

    print("\n" + "=" * 80)
    print("  MODEL COMPARISON ON TEST SET")
    print("=" * 80)

    for metric_name in ("mae", "rmse", "max_abs"):
        print(f"\n  {metric_name.upper()}")
        header = f"  {'Target':<14}"
        for mn in model_names:
            header += f" {mn:>16}"
        if len(model_names) == 2:
            header += f" {'Δ (MLP-Tree)':>16}"
        print(header)
        print(f"  {'-' * (14 + 17 * len(model_names) + (17 if len(model_names)==2 else 0))}")

        for tgt in TARGET_NAMES:
            label = f"{tgt} ({TARGET_LABELS[tgt]})"
            row = f"  {label:<14}"
            vals = []
            for mn in model_names:
                v = results[mn][tgt][metric_name]
                vals.append(v)
                row += f" {v:>16.4f}"
            if len(vals) == 2:
                delta = vals[1] - vals[0]
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:>15.4f}"
            print(row)

    print("\n" + "=" * 80)

    # Save comparison
    out_path = ARTIFACT_DIR / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved to: {out_path}")


if __name__ == "__main__":
    main()

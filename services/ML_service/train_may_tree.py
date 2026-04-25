from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from era5_gam_weather.config import SamplingConfig, SplitConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day
from era5_gam_weather.tree_features import TreeFeatureBuilder, TreeFeatureConfig
from era5_gam_weather.tree_model import WeatherTreeTrainer

YEAR = 2025
MONTHS = [4, 5]

TRAIN_SAMPLES_PER_FILE = 60000
EVAL_SAMPLES_PER_FILE = 12000

LAT_RANGE = None
LON_RANGE = None
ALTITUDE_RANGE_M = None

WEIGHT_MODE = "ballistics"

TARGET_NAMES = ["T", "P", "U", "V"]
FEATURE_NAMES = ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]

TARGET_CONFIG_GRID = {
    "T": [
        {"learning_rate": 0.05, "max_iter": 700, "max_leaf_nodes": 127, "min_samples_leaf": 48, "l2_regularization": 1e-3},
        {"learning_rate": 0.035, "max_iter": 900, "max_leaf_nodes": 127, "min_samples_leaf": 48, "l2_regularization": 1e-3},
    ],
    "P": [
        {"learning_rate": 0.05, "max_iter": 700, "max_leaf_nodes": 127, "min_samples_leaf": 64, "l2_regularization": 1e-3},
        {"learning_rate": 0.035, "max_iter": 900, "max_leaf_nodes": 127, "min_samples_leaf": 64, "l2_regularization": 1e-3},
    ],
    "U": [
    {
        "learning_rate": 0.03,
        "max_iter": 2200,
        "max_leaf_nodes": 255,
        "min_samples_leaf": 16,
        "l2_regularization": 1e-4,
        "n_iter_no_change": 80,
    },
    {
        "learning_rate": 0.02,
        "max_iter": 3200,
        "max_leaf_nodes": 511,
        "min_samples_leaf": 12,
        "l2_regularization": 1e-4,
        "n_iter_no_change": 100,
    },
    ],
    "V": [
        {
            "learning_rate": 0.03,
            "max_iter": 2200,
            "max_leaf_nodes": 255,
            "min_samples_leaf": 16,
            "l2_regularization": 1e-4,
            "n_iter_no_change": 80,
        },
        {
            "learning_rate": 0.02,
            "max_iter": 3200,
            "max_leaf_nodes": 511,
            "min_samples_leaf": 12,
            "l2_regularization": 1e-4,
            "n_iter_no_change": 100,
        },
    ],
}

BASE_CFG = {
    "loss": "squared_error",
    "max_bins": 255,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 45,
    "random_state": 42,
}


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

MODEL_PATH = ARTIFACT_DIR / "weather_tree_bundle_2025_04_05.joblib"
METRICS_PATH = ARTIFACT_DIR / "eval_metrics_tree_2025_04_05.json"
PLOTS_DIR = ARTIFACT_DIR / "training_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _empty_feature_dict() -> Dict[str, list]:
    return {k: [] for k in FEATURE_NAMES}


def _empty_target_dict() -> Dict[str, list]:
    return {k: [] for k in TARGET_NAMES}


def _scope_mask(features: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(features["lat"])
    mask = np.ones(n, dtype=bool)

    if LAT_RANGE is not None:
        mask &= (features["lat"] >= LAT_RANGE[0]) & (features["lat"] <= LAT_RANGE[1])
    if LON_RANGE is not None:
        mask &= (features["lon"] >= LON_RANGE[0]) & (features["lon"] <= LON_RANGE[1])
    if ALTITUDE_RANGE_M is not None:
        mask &= (features["altitude_m"] >= ALTITUDE_RANGE_M[0]) & (features["altitude_m"] <= ALTITUDE_RANGE_M[1])

    return mask


def _append_batch(storage_features: Dict[str, list], storage_targets: Dict[str, list], batch) -> int:
    mask = _scope_mask(batch.features)
    n_keep = int(np.sum(mask))
    if n_keep == 0:
        return 0

    for k in FEATURE_NAMES:
        storage_features[k].append(np.asarray(batch.features[k], dtype=np.float64)[mask])
    for k in TARGET_NAMES:
        storage_targets[k].append(np.asarray(batch.targets[k], dtype=np.float64)[mask])

    return n_keep


def _finalize_dicts(
    storage_features: Dict[str, list],
    storage_targets: Dict[str, list],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    features = {k: np.concatenate(v) if v else np.empty(0, dtype=np.float64) for k, v in storage_features.items()}
    targets = {k: np.concatenate(v) if v else np.empty(0, dtype=np.float64) for k, v in storage_targets.items()}
    return features, targets


def _collect(files: Iterable[str], sampling_config: SamplingConfig, tag: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    storage_features = _empty_feature_dict()
    storage_targets = _empty_target_dict()

    total = 0
    kept = 0
    for idx, path in enumerate(files, start=1):
        print(f"[{tag} {idx}] {path}")
        batch = sample_from_file(path, sampling_config)
        total += len(batch.features["lat"])
        kept += _append_batch(storage_features, storage_targets, batch)

    print(f"[{tag}] kept {kept} / {total} sampled rows after optional scope filtering")
    return _finalize_dicts(storage_features, storage_targets)


def _concat_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def _sample_weight_for_target(features: Dict[str, np.ndarray], target_name: str) -> np.ndarray | None:
    # Leave temperature and pressure untouched.
    if target_name in ("T", "P"):
        return None

    alt_km = np.asarray(features["altitude_m"], dtype=np.float64) / 1000.0
    w = np.ones_like(alt_km)

    # Wind-specific weighting for ballistic relevance without punishing high altitude.
    w[alt_km <= 3.0] = 2.4
    w[(alt_km > 3.0) & (alt_km <= 8.0)] = 2.0
    w[(alt_km > 8.0) & (alt_km <= 16.0)] = 1.5
    w[(alt_km > 16.0) & (alt_km <= 26.0)] = 1.5
    w[alt_km > 26.0] = 1.1

    return w


def _extract_single_target(targets: Dict[str, np.ndarray], name: str) -> Dict[str, np.ndarray]:
    return {name: targets[name]}


def _mae_for_target(bundle, features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray], target: str) -> float:
    metrics = bundle.evaluate(features, targets)
    return float(metrics.mae[target])

def _as_positive_loss_curve(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    # In some sklearn settings scores may be stored as negative loss.
    if np.nanmean(arr) < 0:
        arr = -arr

    # HistGradientBoosting stores an initial baseline point at index 0.
    if arr.size > 1:
        arr = arr[1:]

    return arr


def _save_training_curves(bundle) -> None:
    """
    Saves one graph per target showing how training loss and validation loss
    changed across boosting iterations.
    """
    for target_name, model in bundle.models.items():
        train_curve = _as_positive_loss_curve(getattr(model, "train_score_", []))
        val_curve = _as_positive_loss_curve(getattr(model, "validation_score_", []))

        if train_curve.size == 0:
            continue

        if val_curve.size > 0:
            n = min(len(train_curve), len(val_curve))
            train_curve = train_curve[:n]
            val_curve = val_curve[:n]
        else:
            n = len(train_curve)

        iterations = np.arange(1, n + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_curve, marker="o", markersize=3, linewidth=2, label="Training loss")

        if val_curve.size > 0:
            plt.plot(iterations, val_curve, marker="o", markersize=3, linewidth=2, label="Validation loss")

        plt.xlabel("Boosting iteration")
        plt.ylabel("Loss")
        plt.title(f"{target_name} - Training and Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{target_name.lower()}_training_validation_loss.png", dpi=300)
        plt.close()


def _save_test_metric_plots(test_metrics) -> None:
    """
    Saves final bar charts for held-out test results.
    """
    targets = list(test_metrics.mae.keys())

    mae_values = [float(test_metrics.mae[t]) for t in targets]
    rmse_values = [float(test_metrics.rmse[t]) for t in targets]
    max_abs_values = [float(test_metrics.max_abs[t]) for t in targets]

    # Test MAE
    plt.figure(figsize=(10, 6))
    plt.bar(targets, mae_values)
    plt.xlabel("Target")
    plt.ylabel("MAE")
    plt.title("Held-out Test MAE by Target")
    for i, v in enumerate(mae_values):
        plt.text(i, v + max(mae_values) * 0.02 if max(mae_values) > 0 else 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_mae_by_target.png", dpi=300)
    plt.close()

    # Test RMSE
    plt.figure(figsize=(10, 6))
    plt.bar(targets, rmse_values)
    plt.xlabel("Target")
    plt.ylabel("RMSE")
    plt.title("Held-out Test RMSE by Target")
    for i, v in enumerate(rmse_values):
        plt.text(i, v + max(rmse_values) * 0.02 if max(rmse_values) > 0 else 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_rmse_by_target.png", dpi=300)
    plt.close()

    # Test max absolute error
    plt.figure(figsize=(10, 6))
    plt.bar(targets, max_abs_values)
    plt.xlabel("Target")
    plt.ylabel("Max Absolute Error")
    plt.title("Held-out Test Max Absolute Error by Target")
    for i, v in enumerate(max_abs_values):
        plt.text(i, v + max(max_abs_values) * 0.02 if max(max_abs_values) > 0 else 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_max_abs_by_target.png", dpi=300)
    plt.close()

    with open(PLOTS_DIR / "plots_summary.txt", "w", encoding="utf-8") as f:
        f.write("Generated plots:\n")
        f.write("- One training/validation loss curve for each target: T, P, U, V\n")
        f.write("- Held-out test MAE by target\n")
        f.write("- Held-out test RMSE by target\n")
        f.write("- Held-out test Max Absolute Error by target\n")

def _print_training_summary(bundle, test_metrics) -> None:
    print("\n" + "=" * 70)
    print("FINAL TRAINING SUMMARY")
    print("=" * 70)

    print("\nPer-target training progress:")
    for target_name, model in bundle.models.items():
        train_curve = _as_positive_loss_curve(getattr(model, "train_score_", []))
        val_curve = _as_positive_loss_curve(getattr(model, "validation_score_", []))

        n_iter = getattr(model, "n_iter_", len(train_curve))
        final_train_loss = float(train_curve[-1]) if train_curve.size > 0 else float("nan")
        final_val_loss = float(val_curve[-1]) if val_curve.size > 0 else float("nan")

        print(
            f"{target_name}: "
            f"iterations={n_iter}, "
            f"final_train_loss={final_train_loss:.6f}, "
            f"final_internal_val_loss={final_val_loss:.6f}"
        )

    print("\nHeld-out test metrics:")
    for t in test_metrics.mae.keys():
        print(
            f"{t}: "
            f"MAE={float(test_metrics.mae[t]):.6f} | "
            f"RMSE={float(test_metrics.rmse[t]):.6f} | "
            f"MAX_ABS={float(test_metrics.max_abs[t]):.6f}"
        )

    print("\nSaved plot files in:")
    print(PLOTS_DIR)
    print("=" * 70 + "\n")


def run_training() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"ERA5 data directory not found: {DATA_ROOT}")

    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)
    train_sampling = SamplingConfig(samples_per_file=TRAIN_SAMPLES_PER_FILE, seed=42, stratified_time_level=True)
    eval_sampling = SamplingConfig(samples_per_file=EVAL_SAMPLES_PER_FILE, seed=142, stratified_time_level=True)

    files = discover_era5_files(str(DATA_ROOT), YEAR, MONTHS)
    if not files:
        raise FileNotFoundError(f"No ERA5 files found under {DATA_ROOT}")

    splits = split_files_by_day(
        files,
        train_end=split_config.train_end_day_inclusive,
        val_end=split_config.val_end_day_inclusive,
    )

    print("Collecting training samples...")
    train_features, train_targets = _collect(splits["train"], train_sampling, tag="TRAIN")

    print("Collecting validation samples...")
    val_features, val_targets = _collect(splits["val"], eval_sampling, tag="VAL")

    print("Collecting test samples...")
    test_features, test_targets = _collect(splits["test"], eval_sampling, tag="TEST")

    if len(train_features["lat"]) == 0:
        raise RuntimeError("No training rows remained after filtering")

    feature_builder = TreeFeatureBuilder(TreeFeatureConfig())

    best_configs: Dict[str, Dict] = {}
    tuning_scores: Dict[str, float] = {}

    print("Hyperparameter search per target...")
    for target_name, grid in TARGET_CONFIG_GRID.items():
        best_mae = None
        best_cfg = None

        target_train = _extract_single_target(train_targets, target_name)
        target_val = _extract_single_target(val_targets, target_name)

        for idx, partial_cfg in enumerate(grid, start=1):
            cfg = {target_name: {**BASE_CFG, **partial_cfg}}
            trainer = WeatherTreeTrainer(
                feature_builder=feature_builder,
                model_configs=cfg,
                target_names=(target_name,),
            )
            target_weight = _sample_weight_for_target(train_features, target_name)
            bundle = trainer.fit(train_features, target_train, sample_weight=target_weight)
            mae = _mae_for_target(bundle, val_features, target_val, target_name)

            print(f"[TUNE {target_name} #{idx}] val MAE = {mae:.6f} with {cfg[target_name]}")

            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_cfg = cfg[target_name]

        if best_cfg is None:
            raise RuntimeError(f"Could not determine best config for target {target_name}")

        best_configs[target_name] = best_cfg
        tuning_scores[target_name] = float(best_mae)

    final_train_features = _concat_dicts(train_features, val_features)
    final_train_targets = _concat_dicts(train_targets, val_targets)
    final_train_weights = {
        name: _sample_weight_for_target(final_train_features, name)
        for name in TARGET_NAMES
    }

    print(f"Training final tree bundle on {len(final_train_features['lat'])} rows...")
    final_trainer = WeatherTreeTrainer(
        feature_builder=feature_builder,
        model_configs=best_configs,
        target_names=TARGET_NAMES,
    )
    bundle = final_trainer.fit(
        final_train_features,
        final_train_targets,
        sample_weight=final_train_weights,
    )
    _save_training_curves(bundle)
    bundle.save(str(MODEL_PATH))

    print("Evaluating final model on held-out test split...")
    test_metrics = bundle.evaluate(test_features, test_targets)
    _save_test_metric_plots(test_metrics)
    _print_training_summary(bundle, test_metrics)

    payload = {
        "year": YEAR,
        "months": MONTHS,
        "model_path": str(MODEL_PATH),
        "train_rows": int(len(train_features["lat"])),
        "val_rows": int(len(val_features["lat"])),
        "test_rows": int(len(test_features["lat"])),
        "final_train_rows": int(len(final_train_features["lat"])),
        "feature_metadata": feature_builder.metadata(),
        "train_sampling_config": train_sampling.to_dict(),
        "eval_sampling_config": eval_sampling.to_dict(),
        "split_config": split_config.to_dict(),
        "scope": {
            "lat_range": LAT_RANGE,
            "lon_range": LON_RANGE,
            "altitude_range_m": ALTITUDE_RANGE_M,
            "weight_mode": WEIGHT_MODE,
        },
        "best_model_configs": best_configs,
        "tuning_validation_mae": tuning_scores,
        "test": {
            "mae": test_metrics.mae,
            "rmse": test_metrics.rmse,
            "max_abs": test_metrics.max_abs,
            "n_samples": test_metrics.n_samples,
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved model and metrics.")
    print(json.dumps(payload["test"], indent=2))


if __name__ == "__main__":
    run_training()
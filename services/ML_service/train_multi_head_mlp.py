"""
Train the Multi-Task Multi-Head MLP Regressor weather model.

This is the new main neural-network training script. It uses TensorFlow/Keras
for Dense layers, BatchNormalization, Dropout, Adam, callbacks, and GPU support.
It does not use the old tree backend or a custom NumPy neural net.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from era5_gam_weather.config import SamplingConfig, SplitConfig
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


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for this training script. Install it with: pip install tensorflow"
        ) from exc
    return tf, keras


def _parse_months(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


YEAR = int(os.getenv("TRAIN_YEAR", "2025"))
MONTHS = _parse_months(os.getenv("TRAIN_MONTHS", "4,5"))
TRAIN_SAMPLES_PER_FILE = int(os.getenv("TRAIN_SAMPLES_PER_FILE", "60000"))
EVAL_SAMPLES_PER_FILE = int(os.getenv("EVAL_SAMPLES_PER_FILE", "12000"))
SEED = int(os.getenv("TRAIN_SEED", "42"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1024"))
MAX_EPOCHS = int(os.getenv("TRAIN_MAX_EPOCHS", "300"))
LEARNING_RATE = float(os.getenv("TRAIN_LEARNING_RATE", "0.001"))
DROPOUT_RATE = float(os.getenv("TRAIN_DROPOUT_RATE", "0.04"))
EARLY_STOPPING_PATIENCE = int(os.getenv("TRAIN_EARLY_STOPPING_PATIENCE", "25"))
REDUCE_LR_PATIENCE = int(os.getenv("TRAIN_REDUCE_LR_PATIENCE", "10"))

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
DATA_ROOT = Path(os.getenv("ERA5_DATA_ROOT", str(PROJECT_ROOT / "data" / "era5"))).resolve()
ARTIFACT_DIR = Path(
    os.getenv("WEATHER_ARTIFACT_DIR", str(THIS_DIR / "artifacts" / "multi_head_mlp_weather"))
).resolve()
PLOTS_DIR = ARTIFACT_DIR / "training_plots"


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
    features = {
        k: np.concatenate(v).astype(np.float32) if v else np.empty(0, dtype=np.float32)
        for k, v in storage_features.items()
    }
    targets = {
        k: np.concatenate(v).astype(np.float32) if v else np.empty(0, dtype=np.float32)
        for k, v in storage_targets.items()
    }
    return features, targets


def _collect(files: Iterable[str], config: SamplingConfig, tag: str):
    sf = _empty_feature_dict()
    st = _empty_target_dict()
    total = 0
    for idx, path in enumerate(files, start=1):
        print(f"[{tag} {idx}] {path}")
        batch = sample_from_file(path, config)
        total += _append_batch(sf, st, batch)
    print(f"[{tag}] collected {total} rows")
    return _finalize(sf, st)


def _normalise_X(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return ((X - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)


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


def _save_training_plots(history: Dict[str, list]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Overall loss curve.
    plt.figure(figsize=(10, 6))
    plt.plot(history.get("loss", []), label="Training loss")
    plt.plot(history.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss on normalized targets")
    plt.title("Multi-Head MLP - Overall Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_training_validation_loss.png", dpi=300)
    plt.close()

    # Per-head losses, if present in Keras history.
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
        plt.savefig(PLOTS_DIR / f"{target}_training_validation_loss.png", dpi=300)
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
        plt.savefig(PLOTS_DIR / filename, dpi=300)
        plt.close()


def run_training() -> None:
    tf, keras = _import_tensorflow()

    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Available GPU(s):", [gpu.name for gpu in gpus])
    else:
        print("No GPU detected; training will run on CPU.")

    if not DATA_ROOT.exists():
        print(f"ERROR: ERA5 data directory not found: {DATA_ROOT}", file=sys.stderr)
        print("Set ERA5_DATA_ROOT or place NetCDF files under data/era5/.", file=sys.stderr)
        sys.exit(1)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)
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

    files = discover_era5_files(str(DATA_ROOT), YEAR, MONTHS)
    if not files:
        print(f"ERROR: No ERA5 files found under {DATA_ROOT} for year={YEAR}, months={MONTHS}", file=sys.stderr)
        sys.exit(1)

    splits = split_files_by_day(
        files,
        train_end=split_config.train_end_day_inclusive,
        val_end=split_config.val_end_day_inclusive,
    )

    print("--- Collecting training data ---")
    train_features, train_targets = _collect(splits["train"], train_sampling, "TRAIN")
    print("\n--- Collecting validation data ---")
    val_features, val_targets = _collect(splits["val"], eval_sampling, "VAL")
    print("\n--- Collecting test data ---")
    test_features, test_targets = _collect(splits["test"], eval_sampling, "TEST")

    n_train = len(train_features["lat"])
    n_val = len(val_features["lat"])
    n_test = len(test_features["lat"])
    print(f"Dataset sizes: train={n_train}, val={n_val}, test={n_test}")

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise RuntimeError("Train/validation/test split produced an empty split. Check available ERA5 files.")

    feature_builder = WeatherFeatureBuilder(WeatherFeatureConfig())
    X_train_raw = feature_builder.transform(train_features)
    X_val_raw = feature_builder.transform(val_features)

    x_mean, x_std, y_mean, y_std, _ = MultiHeadMLPWeatherModel.compute_normalization(
        X_train_raw,
        train_targets,
    )
    X_train = _normalise_X(X_train_raw, x_mean, x_std)
    X_val = _normalise_X(X_val_raw, x_mean, x_std)
    y_train = MultiHeadMLPWeatherModel.normalize_targets(train_targets, y_mean, y_std)
    y_val = MultiHeadMLPWeatherModel.normalize_targets(val_targets, y_mean, y_std)

    tf.keras.utils.set_random_seed(SEED)
    model = build_multi_head_mlp(
        n_features=X_train.shape[1],
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

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
    ]

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

    # Save final/restored-best model as the primary artifact.
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
            "loss": "Huber on normalized targets per head",
            "optimizer": "Adam",
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

    _save_json(ARTIFACT_DIR / NORMALIZATION_FILENAME, normalization_payload)
    _save_json(ARTIFACT_DIR / FEATURE_METADATA_FILENAME, feature_builder.metadata())
    _save_json(ARTIFACT_DIR / METADATA_FILENAME, metadata)

    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    _save_json(ARTIFACT_DIR / "training_history.json", history)
    _save_training_plots(history)

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
    _save_metric_plots(test_metrics)

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
    ]:
        print(" -", ARTIFACT_DIR / filename)
    print("Plots:", PLOTS_DIR)


if __name__ == "__main__":
    run_training()

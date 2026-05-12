"""Post-training deep analysis for the ERA5 Multi-Head MLP weather model.

This script DOES NOT retrain the model. It loads an existing artifact directory
(model.keras + normalization_stats.json + feature_metadata.json + metadata.json),
resamples a smaller deterministic evaluation set from the ERA5 NetCDF files,
and creates advanced plots/tables for the project book:

  * train/validation loss with best epoch and overfitting/divergence marker
  * estimated learning-rate curve and LR-vs-loss plot
  * model architecture diagram + model summary + layer parameter table
  * predicted-vs-actual and residual distribution plots per target
  * residual/error analysis by altitude, latitude, and UTC hour
  * engineered-feature permutation importance
  * gradient saliency per engineered feature
  * selected internal activation distributions
  * weight distribution histograms and weight statistics
  * regression-as-binned-class confusion matrices
  * derived high-wind confusion matrix, ROC curve, and Precision-Recall curve

Why confusion/ROC/PR are "derived": the weather model is a REGRESSION model,
not a classifier. ROC/PR/confusion matrices are native to classification tasks.
For this project we create honest, derived classification views:
  1) binned regression confusion matrices (true low/mid/high vs predicted low/mid/high)
  2) high-wind event detection using wind speed >= threshold

Environment variables:
  WEATHER_ARTIFACT_DIR            artifact directory to analyze (required/recommended)
  ERA5_DATA_ROOT                  path to data/era5 (default: project_root/data/era5)
  ANALYSIS_YEAR                   default 2025
  ANALYSIS_MONTHS                 default from metadata months, otherwise 5
  ANALYSIS_SAMPLES_PER_FILE       default 20000 (keep modest; no retraining)
  ANALYSIS_MAX_FILES              default 0 = all matching files
  ANALYSIS_SEED                   default 123
  ANALYSIS_TRAIN_RATIO            default from metadata, otherwise 0.85
  ANALYSIS_VAL_RATIO              default from metadata, otherwise 0.10
  ANALYSIS_OUTPUT_DIR             default: WEATHER_ARTIFACT_DIR/model_analysis
  ANALYSIS_MAX_PLOT_POINTS        default 25000 for scatter plots
  ANALYSIS_PERMUTE_SAMPLES        default 20000
  ANALYSIS_GRADIENT_SAMPLES       default 4096
  ANALYSIS_ACTIVATION_SAMPLES     default 4096
  ANALYSIS_HIGH_WIND_THRESHOLD    default 10.0 m/s
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from era5_gam_weather.config import SamplingConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file
from era5_gam_weather.multi_head_mlp_model import (
    ERA5_TO_OUTPUT,
    MODEL_FILENAME,
    TARGET_OUTPUTS,
    TARGET_TRANSFORMS,
    MultiHeadMLPWeatherModel,
)

RAW_FEATURE_NAMES = ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]
TARGET_NAMES = ["T", "P", "U", "V"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _parse_months(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _history_from_artifact(artifact_dir: Path) -> Dict[str, List[float]]:
    history_path = artifact_dir / "training_history.json"
    if history_path.exists():
        raw = _read_json(history_path)
        return {k: [float(x) for x in v] for k, v in raw.items() if isinstance(v, list)}

    csv_path = artifact_dir / "training_log.csv"
    if not csv_path.exists():
        return {}
    hist: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key == "epoch" or value in (None, ""):
                    continue
                try:
                    hist.setdefault(key, []).append(float(value))
                except ValueError:
                    pass
    return hist


def _metadata_from_artifact(artifact_dir: Path) -> Dict[str, Any]:
    for name in ("metadata.json", "model_metadata.json"):
        path = artifact_dir / name
        if path.exists():
            return _read_json(path)
    return {}


def _project_root_from_here() -> Path:
    here = Path(__file__).resolve().parent
    for base in [here] + list(here.parents):
        if (base / "data" / "era5").exists():
            return base
    return here


def _empty_feature_dict() -> Dict[str, list]:
    return {k: [] for k in RAW_FEATURE_NAMES}


def _empty_target_dict() -> Dict[str, list]:
    return {k: [] for k in TARGET_NAMES}


def _finalize(storage_features: Dict[str, list], storage_targets: Dict[str, list]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    features: Dict[str, np.ndarray] = {}
    for key in RAW_FEATURE_NAMES:
        chunks = storage_features[key]
        features[key] = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty(0, dtype=np.float32)
        chunks.clear()
    targets: Dict[str, np.ndarray] = {}
    for key in TARGET_NAMES:
        chunks = storage_targets[key]
        targets[key] = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty(0, dtype=np.float32)
        chunks.clear()
    gc.collect()
    return features, targets


def _collect_analysis_split(
    files: Sequence[str],
    config: SamplingConfig,
    train_ratio: float,
    val_ratio: float,
    max_files: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Resample the same style of split used during training.

    We collect train/val/test because some plots compare generalization gap. For
    expensive post-hoc plots we mainly use the test split.
    """
    if max_files > 0:
        files = list(files)[:max_files]

    train_sf, train_st = _empty_feature_dict(), _empty_target_dict()
    val_sf, val_st = _empty_feature_dict(), _empty_target_dict()
    test_sf, test_st = _empty_feature_dict(), _empty_target_dict()

    for idx, path in enumerate(files, start=1):
        print(f"[ANALYSIS COLLECT {idx}/{len(files)}] {path}", flush=True)
        batch = sample_from_file(path, config)
        n = len(batch.features["lat"])
        if n == 0:
            continue
        rng = np.random.default_rng(config.seed + idx * 1000)
        perm = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

        for key in RAW_FEATURE_NAMES:
            arr = np.asarray(batch.features[key], dtype=np.float32)
            train_sf[key].append(arr[idx_train])
            val_sf[key].append(arr[idx_val])
            test_sf[key].append(arr[idx_test])
        for key in TARGET_NAMES:
            arr = np.asarray(batch.targets[key], dtype=np.float32)
            train_st[key].append(arr[idx_train])
            val_st[key].append(arr[idx_val])
            test_st[key].append(arr[idx_test])

        del batch, perm, idx_train, idx_val, idx_test
        gc.collect()

    train_features, train_targets = _finalize(train_sf, train_st)
    val_features, val_targets = _finalize(val_sf, val_st)
    test_features, test_targets = _finalize(test_sf, test_st)
    print(
        f"[ANALYSIS] split sizes: train={len(train_features['lat'])}, "
        f"val={len(val_features['lat'])}, test={len(test_features['lat'])}",
        flush=True,
    )
    return train_features, train_targets, val_features, val_targets, test_features, test_targets


def _target_true_dict(targets_era5: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {output_name: np.asarray(targets_era5[era5_key]).reshape(-1) for era5_key, output_name in ERA5_TO_OUTPUT.items()}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64).reshape(-1) - np.asarray(y_true, dtype=np.float64).reshape(-1)
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_error": float(np.median(abs_err)),
        "p90_abs_error": float(np.percentile(abs_err, 90)),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "p99_abs_error": float(np.percentile(abs_err, 99)),
        "max_abs": float(np.max(abs_err)),
        "bias_mean_error": float(np.mean(err)),
        "std_error": float(np.std(err)),
        "r2": float(1.0 - (np.sum(err ** 2) / max(1e-12, np.sum((y_true - np.mean(y_true)) ** 2)))),
    }


def _predict_from_xnorm(wrapper: MultiHeadMLPWeatherModel, X_norm: np.ndarray) -> Dict[str, np.ndarray]:
    raw = wrapper._raw_model_predict(X_norm)  # intentionally uses wrapper internals to avoid rebuilding features.
    out: Dict[str, np.ndarray] = {}
    for name in TARGET_OUTPUTS:
        transformed = raw[name] * wrapper.y_std[name] + wrapper.y_mean[name]
        if TARGET_TRANSFORMS.get(name) == "log":
            physical = np.exp(transformed)
            physical = np.clip(physical, 1.0, None)
        else:
            physical = transformed
        out[name] = np.asarray(physical, dtype=np.float32).reshape(-1)
    return out


def _subset_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _estimate_lr_by_epoch(history: Mapping[str, Sequence[float]], metadata: Mapping[str, Any]) -> List[float]:
    epochs = len(history.get("loss", []))
    if epochs == 0:
        return []
    cfg = metadata.get("training_config", {}) if isinstance(metadata.get("training_config"), dict) else {}
    base_lr = float(cfg.get("learning_rate", 2e-3))
    min_lr = float(cfg.get("min_lr", 1e-6))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    schedule = str(cfg.get("lr_schedule", "cosine")).lower()

    # If CSVLogger recorded lr/learning_rate, prefer it.
    for key in ("lr", "learning_rate"):
        if key in history and len(history[key]) == epochs:
            return [float(x) for x in history[key]]

    if schedule != "cosine":
        return [base_lr] * epochs

    out: List[float] = []
    for epoch_idx in range(epochs):
        e = epoch_idx + 1
        if warmup_epochs > 0 and e <= warmup_epochs:
            lr = base_lr * e / max(1, warmup_epochs)
        else:
            progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        out.append(float(lr))
    return out


def _find_overfit_epoch(train_loss: Sequence[float], val_loss: Sequence[float], patience: int = 8) -> int | None:
    """Find first sustained region where train keeps improving while val degrades.

    Returns 1-based epoch index or None. This is for visualization only, not a
    formal statistical test.
    """
    n = min(len(train_loss), len(val_loss))
    if n < patience + 2:
        return None
    for i in range(1, n - patience):
        train_window = np.asarray(train_loss[i:i + patience], dtype=float)
        val_window = np.asarray(val_loss[i:i + patience], dtype=float)
        train_trend = train_window[-1] - train_window[0]
        val_trend = val_window[-1] - val_window[0]
        if train_trend < 0 and val_trend > 0:
            return i + 1
    return None


def plot_training_diagnostics(history: Mapping[str, Sequence[float]], metadata: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history or "loss" not in history:
        return {"warning": "No training history/log found."}

    train_loss = [float(x) for x in history.get("loss", [])]
    val_loss = [float(x) for x in history.get("val_loss", [])]
    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss) + 1) if val_loss else None
    overfit_epoch = _find_overfit_epoch(train_loss, val_loss) if val_loss else None

    plt.figure(figsize=(11, 6))
    plt.plot(epochs, train_loss, label="Train loss")
    if val_loss:
        plt.plot(epochs, val_loss, label="Validation loss")
    if best_epoch:
        plt.axvline(best_epoch, linestyle="--", label=f"Best validation epoch: {best_epoch}")
    if overfit_epoch:
        plt.axvline(overfit_epoch, linestyle=":", label=f"Possible overfit start: {overfit_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Huber loss on normalized targets")
    plt.title("Training vs Validation Loss with Best Checkpoint / Overfitting Marker")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_loss_with_overfitting_marker.png", dpi=220)
    plt.close()

    if val_loss:
        gap = np.asarray(val_loss[:len(train_loss)]) - np.asarray(train_loss[:len(val_loss)])
        plt.figure(figsize=(11, 5))
        plt.plot(np.arange(1, len(gap) + 1), gap, label="Validation loss - Train loss")
        plt.axhline(0.0, linestyle="--")
        if best_epoch:
            plt.axvline(best_epoch, linestyle="--", label=f"Best validation epoch: {best_epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Generalization gap")
        plt.title("Generalization Gap During Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "02_generalization_gap.png", dpi=220)
        plt.close()

    lr_by_epoch = _estimate_lr_by_epoch(history, metadata)
    if lr_by_epoch:
        plt.figure(figsize=(11, 5))
        plt.plot(epochs, lr_by_epoch[:len(epochs)], label="Learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.yscale("log")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "03_learning_rate_schedule.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(lr_by_epoch[:len(train_loss)], train_loss, marker="o", markersize=2, linewidth=1, label="Train loss")
        if val_loss:
            plt.plot(lr_by_epoch[:len(val_loss)], val_loss, marker="o", markersize=2, linewidth=1, label="Validation loss")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.title("Learning Rate vs Loss (Epoch-Level Approximation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "04_learning_rate_vs_loss.png", dpi=220)
        plt.close()

    summary = {
        "epochs_recorded": len(train_loss),
        "best_validation_epoch": best_epoch,
        "best_validation_loss": float(np.min(val_loss)) if val_loss else None,
        "final_train_loss": float(train_loss[-1]) if train_loss else None,
        "final_validation_loss": float(val_loss[-1]) if val_loss else None,
        "possible_overfit_start_epoch": overfit_epoch,
        "comment": "No sustained overfitting marker found." if overfit_epoch is None else "A possible overfitting region was detected by trend heuristic.",
    }
    _save_json(out_dir / "training_diagnostics_summary.json", summary)
    return summary


def plot_architecture(wrapper: MultiHeadMLPWeatherModel, metadata: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = wrapper.model

    # Text summary
    with open(out_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    rows = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output_shape)
        except Exception:
            out_shape = ""
        rows.append({
            "layer_name": layer.name,
            "layer_type": layer.__class__.__name__,
            "output_shape": out_shape,
            "trainable_params": int(np.sum([np.prod(w.shape) for w in layer.trainable_weights])),
            "non_trainable_params": int(np.sum([np.prod(w.shape) for w in layer.non_trainable_weights])),
        })
    _write_csv(out_dir / "layer_parameter_table.csv", rows)

    # Try official Keras diagram if pydot/graphviz are available.
    try:
        from tensorflow import keras  # type: ignore
        keras.utils.plot_model(
            model,
            to_file=str(out_dir / "model_architecture_keras.png"),
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            dpi=160,
        )
    except Exception as exc:
        with open(out_dir / "model_architecture_keras_FAILED.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))

    # Always create a clean handmade block diagram, no external Graphviz needed.
    feature_count = len(wrapper.feature_names)
    arch = metadata.get("architecture", {}) if isinstance(metadata.get("architecture"), dict) else {}
    n_blocks = arch.get("n_residual_blocks", "?")
    width = arch.get("block_width", "?")

    blocks = [
        ("Input", f"{feature_count} engineered features"),
        ("Stem", f"Dense({width})"),
        ("Residual trunk", f"{n_blocks} × [BN → ReLU → Dense({width}) → BN → ReLU → Dense({width}) + skip]"),
        ("Shared representation", "BatchNorm → ReLU"),
        ("Temperature head", "Dense(128) → Dense(1)"),
        ("Pressure head", "Dense(128) → Dense(1), trained as log(P)"),
        ("Wind-U head", "Dense(256) → Dense(128) → Dense(1)"),
        ("Wind-V head", "Dense(256) → Dense(128) → Dense(1)"),
    ]
    plt.figure(figsize=(15, 7))
    ax = plt.gca()
    ax.axis("off")
    x_positions = [0.03, 0.20, 0.42, 0.67]
    y_main = 0.58
    main_blocks = blocks[:4]
    for idx, (title, text) in enumerate(main_blocks):
        x = x_positions[idx]
        rect = plt.Rectangle((x, y_main), 0.18, 0.22, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.09, y_main + 0.15, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 0.09, y_main + 0.07, text, ha="center", va="center", fontsize=9, wrap=True)
        if idx < len(main_blocks) - 1:
            ax.annotate("", xy=(x_positions[idx + 1], y_main + 0.11), xytext=(x + 0.18, y_main + 0.11), arrowprops=dict(arrowstyle="->"))

    head_positions = [(0.67, 0.26), (0.67, 0.02), (0.67, -0.22), (0.67, -0.46)]
    for (title, text), (x, y) in zip(blocks[4:], head_positions):
        rect = plt.Rectangle((x, y), 0.25, 0.15, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.125, y + 0.10, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + 0.125, y + 0.04, text, ha="center", va="center", fontsize=9, wrap=True)
        ax.annotate("", xy=(x + 0.03, y + 0.15), xytext=(0.76, y_main), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.9)
    plt.title("Multi-Task Multi-Head Residual MLP Architecture")
    plt.tight_layout()
    plt.savefig(out_dir / "model_architecture_block_diagram.png", dpi=220)
    plt.close()


def _feature_group(name: str) -> str:
    if "altitude" in name or name in ("log1p_altitude_km",):
        return "altitude"
    if name.startswith("utc") or name.startswith("solar") or "hour" in name:
        return "time_of_day"
    if name.startswith("doy") or "day_of_year" in name:
        return "day_of_year"
    if "lat" in name or "lon" in name or name.startswith("sphere"):
        return "location"
    return "other"


def evaluate_and_plot_predictions(
    wrapper: MultiHeadMLPWeatherModel,
    features: Mapping[str, np.ndarray],
    targets_era5: Mapping[str, np.ndarray],
    out_dir: Path,
    max_points: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = _target_true_dict(targets_era5)
    y_pred = wrapper.predict_features(features)
    metrics = {name: _metrics(y_true[name], y_pred[name]) for name in TARGET_OUTPUTS}
    _save_json(out_dir / "test_regression_metrics_detailed.json", metrics)

    metric_rows = []
    for target, vals in metrics.items():
        row = {"target": target}
        row.update(vals)
        metric_rows.append(row)
    _write_csv(out_dir / "test_regression_metrics_detailed.csv", metric_rows)

    # Predicted vs actual and residual histograms.
    n = len(next(iter(y_true.values())))
    idx = _subset_indices(n, max_points, seed)
    for target in TARGET_OUTPUTS:
        true = y_true[target][idx]
        pred = y_pred[target][idx]
        err = pred - true

        plt.figure(figsize=(7, 7))
        plt.scatter(true, pred, s=6, alpha=0.25)
        lo = float(min(np.min(true), np.min(pred)))
        hi = float(max(np.max(true), np.max(pred)))
        plt.plot([lo, hi], [lo, hi], linestyle="--", label="ideal y=x")
        plt.xlabel(f"True {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"Predicted vs Actual - {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"predicted_vs_actual_{target}.png", dpi=220)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.hist(err, bins=80)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel("Prediction error (predicted - true)")
        plt.ylabel("Count")
        plt.title(f"Residual Distribution - {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"residual_histogram_{target}.png", dpi=220)
        plt.close()

    # Bar charts for detailed regression metrics.
    for metric_name in ("mae", "rmse", "p95_abs_error", "max_abs", "r2"):
        values = [metrics[target][metric_name] for target in TARGET_OUTPUTS]
        plt.figure(figsize=(9, 5))
        plt.bar(list(TARGET_OUTPUTS), values)
        plt.xlabel("Target")
        plt.ylabel(metric_name)
        plt.title(f"Held-Out Test {metric_name.upper()} by Target")
        ymax = max(values) if values else 1.0
        for i, value in enumerate(values):
            plt.text(i, value + (abs(ymax) * 0.02 if ymax != 0 else 0.01), f"{value:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(out_dir / f"metric_{metric_name}_by_target.png", dpi=220)
        plt.close()

    return y_true, y_pred, metrics


def plot_error_by_physical_axis(
    features: Mapping[str, np.ndarray],
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_specs = [
        ("altitude_m", "Altitude (m)", 12),
        ("lat", "Latitude (deg)", 12),
        ("utc_hour", "UTC hour", 24),
    ]
    rows = []
    for axis_key, axis_label, bins_count in axis_specs:
        axis_values = np.asarray(features[axis_key]).reshape(-1)
        if axis_key == "utc_hour":
            bins = np.linspace(0, 24, bins_count + 1)
        else:
            bins = np.linspace(float(np.min(axis_values)), float(np.max(axis_values)), bins_count + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])

        for target in TARGET_OUTPUTS:
            abs_err = np.abs(np.asarray(y_pred[target]).reshape(-1) - np.asarray(y_true[target]).reshape(-1))
            means = []
            counts = []
            for i in range(len(bins) - 1):
                if i == len(bins) - 2:
                    mask = (axis_values >= bins[i]) & (axis_values <= bins[i + 1])
                else:
                    mask = (axis_values >= bins[i]) & (axis_values < bins[i + 1])
                count = int(np.sum(mask))
                counts.append(count)
                means.append(float(np.mean(abs_err[mask])) if count > 0 else float("nan"))
                rows.append({
                    "axis": axis_key,
                    "target": target,
                    "bin_start": float(bins[i]),
                    "bin_end": float(bins[i + 1]),
                    "bin_center": float(centers[i]),
                    "count": count,
                    "mean_abs_error": means[-1],
                })

            plt.figure(figsize=(10, 5))
            plt.plot(centers, means, marker="o", label=target)
            plt.xlabel(axis_label)
            plt.ylabel("Mean absolute error")
            plt.title(f"Error vs {axis_label} - {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"error_vs_{axis_key}_{target}.png", dpi=220)
            plt.close()
    _write_csv(out_dir / "error_by_physical_axis.csv", rows)


def plot_weight_distributions(wrapper: MultiHeadMLPWeatherModel, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_kernels = []
    all_biases = []
    per_layer_values: List[Tuple[str, np.ndarray]] = []
    for layer in wrapper.model.layers:
        weights = layer.get_weights()
        for idx, arr in enumerate(weights):
            arr = np.asarray(arr).reshape(-1)
            if arr.size == 0:
                continue
            name = f"{layer.name}_w{idx}"
            rows.append({
                "name": name,
                "layer": layer.name,
                "index": idx,
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "p01": float(np.percentile(arr, 1)),
                "p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
                "fraction_abs_lt_1e_minus_3": float(np.mean(np.abs(arr) < 1e-3)),
                "fraction_abs_gt_1": float(np.mean(np.abs(arr) > 1.0)),
            })
            if arr.size >= 100:
                per_layer_values.append((name, arr))
            if "kernel" in name or idx == 0:
                all_kernels.append(arr)
            else:
                all_biases.append(arr)
    _write_csv(out_dir / "weights_distribution_stats.csv", rows)

    if all_kernels:
        merged = np.concatenate(all_kernels)
        plt.figure(figsize=(10, 5))
        plt.hist(merged, bins=120)
        plt.xlabel("Weight value")
        plt.ylabel("Count")
        plt.title("All Kernel Weight Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "weights_distribution_all_kernels.png", dpi=220)
        plt.close()

    # Show the largest few layers so the page stays readable.
    per_layer_values.sort(key=lambda x: x[1].size, reverse=True)
    for name, arr in per_layer_values[:12]:
        plt.figure(figsize=(9, 5))
        plt.hist(arr, bins=100)
        plt.xlabel("Weight value")
        plt.ylabel("Count")
        plt.title(f"Weight Distribution - {name}")
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        plt.savefig(out_dir / f"weights_distribution_{safe}.png", dpi=180)
        plt.close()


def permutation_importance(
    wrapper: MultiHeadMLPWeatherModel,
    features: Mapping[str, np.ndarray],
    y_true: Mapping[str, np.ndarray],
    out_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(next(iter(y_true.values())))
    idx = _subset_indices(n, max_samples, seed)
    sub_features = {key: np.asarray(value)[idx] for key, value in features.items()}
    sub_true = {key: np.asarray(value)[idx] for key, value in y_true.items()}

    X = wrapper._normalize_features(sub_features)
    baseline_pred = _predict_from_xnorm(wrapper, X)
    baseline_mae = {
        target: float(np.mean(np.abs(baseline_pred[target] - sub_true[target])))
        for target in TARGET_OUTPUTS
    }

    rng = np.random.default_rng(seed + 7000)
    rows = []
    for j, feature_name in enumerate(wrapper.feature_names):
        X_perm = X.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        pred = _predict_from_xnorm(wrapper, X_perm)
        for target in TARGET_OUTPUTS:
            mae = float(np.mean(np.abs(pred[target] - sub_true[target])))
            rows.append({
                "feature": feature_name,
                "feature_group": _feature_group(feature_name),
                "target": target,
                "baseline_mae": baseline_mae[target],
                "permuted_mae": mae,
                "delta_mae": mae - baseline_mae[target],
                "relative_delta": (mae - baseline_mae[target]) / max(1e-12, baseline_mae[target]),
            })
        del X_perm, pred
    _write_csv(out_dir / "permutation_importance_by_feature.csv", rows)

    # Aggregated feature group importance.
    group_rows = []
    for target in TARGET_OUTPUTS:
        target_rows = [r for r in rows if r["target"] == target]
        for group in sorted(set(r["feature_group"] for r in target_rows)):
            vals = [float(r["delta_mae"]) for r in target_rows if r["feature_group"] == group]
            group_rows.append({
                "target": target,
                "feature_group": group,
                "sum_delta_mae": float(np.sum(vals)),
                "mean_delta_mae": float(np.mean(vals)),
            })
    _write_csv(out_dir / "permutation_importance_by_feature_group.csv", group_rows)

    for target in TARGET_OUTPUTS:
        target_rows = sorted([r for r in rows if r["target"] == target], key=lambda r: float(r["delta_mae"]), reverse=True)[:15]
        labels = [str(r["feature"]) for r in target_rows][::-1]
        values = [float(r["delta_mae"]) for r in target_rows][::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(labels, values)
        plt.xlabel("Increase in MAE after permutation")
        plt.title(f"Permutation Feature Importance - {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"permutation_importance_top15_{target}.png", dpi=220)
        plt.close()

        group_for_target = sorted([r for r in group_rows if r["target"] == target], key=lambda r: float(r["sum_delta_mae"]), reverse=True)
        plt.figure(figsize=(8, 5))
        plt.bar([str(r["feature_group"]) for r in group_for_target], [float(r["sum_delta_mae"]) for r in group_for_target])
        plt.xlabel("Feature group")
        plt.ylabel("Total delta MAE")
        plt.title(f"Permutation Importance by Feature Group - {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"permutation_importance_groups_{target}.png", dpi=220)
        plt.close()


def gradient_saliency(
    wrapper: MultiHeadMLPWeatherModel,
    features: Mapping[str, np.ndarray],
    out_dir: Path,
    max_samples: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        with open(out_dir / "gradient_saliency_FAILED.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))
        return

    n = len(next(iter(features.values())))
    idx = _subset_indices(n, max_samples, seed)
    sub_features = {key: np.asarray(value)[idx] for key, value in features.items()}
    X_np = wrapper._normalize_features(sub_features)
    X = tf.convert_to_tensor(X_np, dtype=tf.float32)

    rows = []
    for target in TARGET_OUTPUTS:
        with tf.GradientTape() as tape:
            tape.watch(X)
            raw = wrapper.model(X, training=False)
            if isinstance(raw, dict):
                y = raw[target]
            else:
                output_names = list(getattr(wrapper.model, "output_names", TARGET_OUTPUTS))
                y = raw[output_names.index(target)]
            objective = tf.reduce_mean(y)
        grad = tape.gradient(objective, X)
        if grad is None:
            continue
        sal = np.mean(np.abs(grad.numpy()), axis=0)
        for feature_name, value in zip(wrapper.feature_names, sal):
            rows.append({
                "target": target,
                "feature": feature_name,
                "feature_group": _feature_group(feature_name),
                "mean_abs_gradient": float(value),
            })
    _write_csv(out_dir / "gradient_saliency_by_feature.csv", rows)

    for target in TARGET_OUTPUTS:
        target_rows = sorted([r for r in rows if r["target"] == target], key=lambda r: float(r["mean_abs_gradient"]), reverse=True)[:15]
        labels = [str(r["feature"]) for r in target_rows][::-1]
        values = [float(r["mean_abs_gradient"]) for r in target_rows][::-1]
        plt.figure(figsize=(10, 7))
        plt.barh(labels, values)
        plt.xlabel("Mean absolute gradient")
        plt.title(f"Gradient Saliency - {target}")
        plt.tight_layout()
        plt.savefig(out_dir / f"gradient_saliency_top15_{target}.png", dpi=220)
        plt.close()


def activation_analysis(wrapper: MultiHeadMLPWeatherModel, features: Mapping[str, np.ndarray], out_dir: Path, max_samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except Exception as exc:
        with open(out_dir / "activation_analysis_FAILED.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))
        return

    selected_layer_names = [
        "stem_dense",
        "resblock_1_add",
        "resblock_2_add",
        "resblock_3_add",
        "resblock_4_add",
        "trunk_final_relu",
        "temperature_head_dense_1",
        "pressure_head_dense_1",
        "wind_u_head_dense_2",
        "wind_v_head_dense_2",
    ]
    layers = []
    layer_names = []
    existing = {layer.name: layer for layer in wrapper.model.layers}
    for name in selected_layer_names:
        if name in existing:
            layers.append(existing[name].output)
            layer_names.append(name)
    if not layers:
        return

    n = len(next(iter(features.values())))
    idx = _subset_indices(n, max_samples, seed)
    sub_features = {key: np.asarray(value)[idx] for key, value in features.items()}
    X_np = wrapper._normalize_features(sub_features)
    activation_model = keras.Model(inputs=wrapper.model.input, outputs=layers)
    outputs = activation_model.predict(X_np, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    rows = []
    for name, arr in zip(layer_names, outputs):
        flat = np.asarray(arr).reshape(-1)
        rows.append({
            "layer": name,
            "count": int(flat.size),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "p01": float(np.percentile(flat, 1)),
            "p50": float(np.percentile(flat, 50)),
            "p99": float(np.percentile(flat, 99)),
            "max": float(np.max(flat)),
            "fraction_zero_or_negative": float(np.mean(flat <= 0.0)),
        })
        plt.figure(figsize=(9, 5))
        plt.hist(flat, bins=100)
        plt.xlabel("Activation value")
        plt.ylabel("Count")
        plt.title(f"Activation Distribution - {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"activation_distribution_{name}.png", dpi=180)
        plt.close()
    _write_csv(out_dir / "activation_distribution_stats.csv", rows)


def _quantile_bins(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    # Ensure strictly increasing enough for digitize.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def _confusion_matrix(y_true_class: np.ndarray, y_pred_class: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_class, y_pred_class):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1
    return cm


def _plot_cm(cm: np.ndarray, labels: Sequence[str], title: str, path: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, aspect="auto")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted class/bin")
    plt.ylabel("True class/bin")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def binned_confusion_matrices(y_true: Mapping[str, np.ndarray], y_pred: Mapping[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = ["low", "mid", "high"]
    for target in TARGET_OUTPUTS:
        true = np.asarray(y_true[target]).reshape(-1)
        pred = np.asarray(y_pred[target]).reshape(-1)
        edges = _quantile_bins(true, n_bins=3)
        true_cls = np.digitize(true, edges[1:-1], right=False)
        pred_cls = np.digitize(pred, edges[1:-1], right=False)
        cm = _confusion_matrix(true_cls, pred_cls, 3)
        _plot_cm(cm, labels, f"Binned Regression Confusion Matrix - {target}", out_dir / f"binned_confusion_matrix_{target}.png")
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append({
                    "target": target,
                    "true_bin": true_label,
                    "predicted_bin": pred_label,
                    "count": int(cm[i, j]),
                    "bin_edge_low": float(edges[i]),
                    "bin_edge_high": float(edges[i + 1]),
                })
    _write_csv(out_dir / "binned_confusion_matrices.csv", rows)


def _roc_curve_binary(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = y_true[order].astype(np.int64)
    s = scores[order]
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = tps / positives
    fpr = fps / negatives
    thresholds = s
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[np.inf], thresholds])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, thresholds, auc


def _pr_curve_binary(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = y_true[order].astype(np.int64)
    s = scores[order]
    positives = max(1, int(np.sum(y == 1)))
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    precision = tps / np.maximum(1, tps + fps)
    recall = tps / positives
    thresholds = s
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thresholds = np.concatenate([[np.inf], thresholds])
    # Average precision approximation by step integration.
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return recall, precision, thresholds, ap


def high_wind_classification_analysis(
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    threshold: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    true_speed = np.sqrt(np.asarray(y_true["wind_u"]) ** 2 + np.asarray(y_true["wind_v"]) ** 2)
    pred_speed = np.sqrt(np.asarray(y_pred["wind_u"]) ** 2 + np.asarray(y_pred["wind_v"]) ** 2)
    y_event = (true_speed >= threshold).astype(np.int64)
    y_pred_event = (pred_speed >= threshold).astype(np.int64)
    cm = _confusion_matrix(y_event, y_pred_event, 2)
    labels = [f"< {threshold:g} m/s", f">= {threshold:g} m/s"]
    _plot_cm(cm, labels, f"High-Wind Event Confusion Matrix (threshold={threshold:g} m/s)", out_dir / "high_wind_confusion_matrix.png")

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    summary = {
        "threshold_m_per_s": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "positive_rate": float(np.mean(y_event)),
    }
    _save_json(out_dir / "high_wind_classification_metrics.json", summary)

    fpr, tpr, roc_thresholds, auc = _roc_curve_binary(y_event, pred_speed)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curve - Derived High-Wind Event Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "high_wind_roc_curve.png", dpi=220)
    plt.close()
    _write_csv(out_dir / "high_wind_roc_curve.csv", [
        {"fpr": float(a), "tpr": float(b), "threshold": float(c) if np.isfinite(c) else "inf"}
        for a, b, c in zip(fpr, tpr, roc_thresholds)
    ])

    recall_arr, precision_arr, pr_thresholds, ap = _pr_curve_binary(y_event, pred_speed)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_arr, precision_arr, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Derived High-Wind Event Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "high_wind_precision_recall_curve.png", dpi=220)
    plt.close()
    _write_csv(out_dir / "high_wind_precision_recall_curve.csv", [
        {"recall": float(a), "precision": float(b), "threshold": float(c) if np.isfinite(c) else "inf"}
        for a, b, c in zip(recall_arr, precision_arr, pr_thresholds)
    ])


def write_book_notes(out_dir: Path, training_summary: Mapping[str, Any], metrics: Mapping[str, Mapping[str, float]]) -> None:
    lines = []
    lines.append("# Advanced Model Analysis Notes\n")
    lines.append("This folder contains post-training analysis figures for the multi-task multi-head residual MLP weather model.\n")
    lines.append("\n## Important interpretation\n")
    lines.append("The model is a regression model. Therefore, confusion matrices, ROC curves and Precision-Recall curves are not native evaluation metrics for the four atmospheric outputs. To satisfy the requirement honestly, the script creates derived classification views: binned regression confusion matrices and a high-wind event classifier based on predicted wind speed.\n")
    lines.append("\n## Training behavior\n")
    lines.append(f"Best validation epoch: {training_summary.get('best_validation_epoch')}\n")
    lines.append(f"Possible overfitting start: {training_summary.get('possible_overfit_start_epoch')}\n")
    lines.append(f"Comment: {training_summary.get('comment')}\n")
    lines.append("\n## Held-out regression metrics\n")
    for target in TARGET_OUTPUTS:
        if target in metrics:
            m = metrics[target]
            lines.append(
                f"- {target}: MAE={m.get('mae', float('nan')):.4f}, "
                f"RMSE={m.get('rmse', float('nan')):.4f}, "
                f"P95 abs error={m.get('p95_abs_error', float('nan')):.4f}, "
                f"R²={m.get('r2', float('nan')):.4f}\n"
            )
    lines.append("\n## Recommended figures for the project book\n")
    lines.append("1. `architecture/model_architecture_block_diagram.png`\n")
    lines.append("2. `training_diagnostics/01_loss_with_overfitting_marker.png`\n")
    lines.append("3. `training_diagnostics/03_learning_rate_schedule.png` and `04_learning_rate_vs_loss.png`\n")
    lines.append("4. `prediction_quality/predicted_vs_actual_*.png` and `residual_histogram_*.png`\n")
    lines.append("5. `physical_error_analysis/error_vs_altitude_m_*.png`\n")
    lines.append("6. `feature_importance/permutation_importance_top15_*.png`\n")
    lines.append("7. `weights/weights_distribution_all_kernels.png`\n")
    lines.append("8. `classification_views/high_wind_confusion_matrix.png`, `high_wind_roc_curve.png`, and `high_wind_precision_recall_curve.png`\n")
    (out_dir / "BOOK_NOTES.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    default_artifact = Path(__file__).resolve().parent / "artifacts" / "multi_head_mlp_weather"
    artifact_dir = Path(os.getenv("WEATHER_ARTIFACT_DIR") or default_artifact).expanduser().resolve()
    if not artifact_dir.exists():
        raise SystemExit(f"Artifact directory not found: {artifact_dir}\nSet WEATHER_ARTIFACT_DIR to your trained folder.")
    if not (artifact_dir / MODEL_FILENAME).exists():
        raise SystemExit(f"model.keras not found inside: {artifact_dir}")

    metadata = _metadata_from_artifact(artifact_dir)
    train_cfg = metadata.get("training_config", {}) if isinstance(metadata.get("training_config"), dict) else {}
    months_default = ",".join(str(m) for m in metadata.get("months", [5])) if metadata.get("months") else "5"

    project_root = _project_root_from_here()
    data_root = Path(os.getenv("ERA5_DATA_ROOT") or str(project_root / "data" / "era5")).expanduser().resolve()
    output_dir = Path(os.getenv("ANALYSIS_OUTPUT_DIR") or str(artifact_dir / "model_analysis")).expanduser().resolve()

    year = _env_int("ANALYSIS_YEAR", int(metadata.get("year", 2025)))
    months = _parse_months(os.getenv("ANALYSIS_MONTHS", months_default))
    samples_per_file = _env_int("ANALYSIS_SAMPLES_PER_FILE", 20000)
    max_files = _env_int("ANALYSIS_MAX_FILES", 0)
    seed = _env_int("ANALYSIS_SEED", 123)
    train_ratio = _env_float("ANALYSIS_TRAIN_RATIO", float(train_cfg.get("train_ratio", 0.85)))
    val_ratio = _env_float("ANALYSIS_VAL_RATIO", float(train_cfg.get("val_ratio", 0.10)))
    max_plot_points = _env_int("ANALYSIS_MAX_PLOT_POINTS", 25000)
    permute_samples = _env_int("ANALYSIS_PERMUTE_SAMPLES", 20000)
    gradient_samples = _env_int("ANALYSIS_GRADIENT_SAMPLES", 4096)
    activation_samples = _env_int("ANALYSIS_ACTIVATION_SAMPLES", 4096)
    high_wind_threshold = _env_float("ANALYSIS_HIGH_WIND_THRESHOLD", 10.0)

    print("=" * 80)
    print("Advanced post-training analysis")
    print("=" * 80)
    print(f"Artifact dir: {artifact_dir}")
    print(f"Data root:    {data_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Year/months:  {year}/{months}")
    print(f"Samples/file: {samples_per_file}")
    print("This script does NOT retrain the model.")

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = MultiHeadMLPWeatherModel.load(artifact_dir)

    history = _history_from_artifact(artifact_dir)
    print("[1/9] Training diagnostics...")
    training_summary = plot_training_diagnostics(history, metadata, output_dir / "training_diagnostics")

    print("[2/9] Architecture visualization...")
    plot_architecture(wrapper, metadata, output_dir / "architecture")

    files = discover_era5_files(str(data_root), year=year, months=months)
    if not files:
        raise SystemExit(f"No ERA5 .nc files found in {data_root} for year={year}, months={months}")
    sampling = SamplingConfig(
        samples_per_file=samples_per_file,
        seed=seed,
        stratified_time_level=True,
        area_weighted_lat=True,
    )
    print("[3/9] Sampling analysis dataset...")
    train_features, train_targets, val_features, val_targets, test_features, test_targets = _collect_analysis_split(
        files=files,
        config=sampling,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_files=max_files,
    )

    print("[4/9] Prediction quality plots...")
    y_true, y_pred, metrics = evaluate_and_plot_predictions(
        wrapper,
        test_features,
        test_targets,
        output_dir / "prediction_quality",
        max_points=max_plot_points,
        seed=seed,
    )

    print("[5/9] Physical error analysis...")
    plot_error_by_physical_axis(test_features, y_true, y_pred, output_dir / "physical_error_analysis")

    print("[6/9] Weight distributions...")
    plot_weight_distributions(wrapper, output_dir / "weights")

    print("[7/9] Permutation feature importance...")
    permutation_importance(wrapper, test_features, y_true, output_dir / "feature_importance", max_samples=permute_samples, seed=seed)

    print("[8/9] Gradient saliency and activations...")
    gradient_saliency(wrapper, test_features, output_dir / "gradient_saliency", max_samples=gradient_samples, seed=seed)
    activation_analysis(wrapper, test_features, output_dir / "activations", max_samples=activation_samples, seed=seed)

    print("[9/9] Classification-style views...")
    binned_confusion_matrices(y_true, y_pred, output_dir / "classification_views")
    high_wind_classification_analysis(y_true, y_pred, high_wind_threshold, output_dir / "classification_views")

    summary = {
        "artifact_dir": str(artifact_dir),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "year": year,
        "months": months,
        "samples_per_file": samples_per_file,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "training_summary": training_summary,
        "test_metrics": metrics,
        "note": "No retraining was performed. Classification plots are derived views because the base model is a regression model.",
    }
    _save_json(output_dir / "analysis_summary.json", summary)
    write_book_notes(output_dir, training_summary, metrics)

    print("\nDone. Advanced analysis saved to:")
    print(output_dir)
    print("Recommended starting file:", output_dir / "BOOK_NOTES.md")


if __name__ == "__main__":
    main()

"""Multi-Task Multi-Head MLP weather model.

This module:
  * defines the Keras architecture (shared trunk + 4 task-specific heads),
  * computes train-only normalization statistics (no leakage),
  * provides a serving wrapper (``MultiHeadMLPWeatherModel``) that loads a
    saved Keras model + JSON statistics and produces predictions in physical
    units.

It is intentionally free of any sklearn / xgboost / lightgbm / catboost / tree
dependency. TensorFlow is imported lazily so callers that only need feature
engineering or sampling can import this package without paying the TF import
cost.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .weather_features import WeatherFeatureBuilder, WeatherFeatureConfig

TARGET_OUTPUTS: Tuple[str, ...] = ("temperature_k", "pressure_pa", "wind_u", "wind_v")
ERA5_TO_OUTPUT: Dict[str, str] = {
    "T": "temperature_k",
    "P": "pressure_pa",
    "U": "wind_u",
    "V": "wind_v",
}
OUTPUT_TO_ERA5: Dict[str, str] = {v: k for k, v in ERA5_TO_OUTPUT.items()}
TARGET_TRANSFORMS: Dict[str, str] = {
    "temperature_k": "identity",
    "pressure_pa": "log",
    "wind_u": "identity",
    "wind_v": "identity",
}

MODEL_FILENAME = "model.keras"
NORMALIZATION_FILENAME = "normalization_stats.json"
METADATA_FILENAME = "metadata.json"
FEATURE_METADATA_FILENAME = "feature_metadata.json"


def _import_keras():
    """Import TensorFlow lazily so non-neural callers don't pay the TF cost."""
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "TensorFlow is required for WEATHER_MODEL_BACKEND=multi_head_mlp. "
            "Install it with: pip install tensorflow"
        ) from exc
    return tf, keras


def _safe_std(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    return np.where(std < 1e-8, 1.0, std).astype(np.float32)


def _forward_target_transform(output_name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if TARGET_TRANSFORMS.get(output_name) == "log":
        return np.log(np.clip(arr, 1e-6, None)).astype(np.float32)
    return arr


def _inverse_target_transform(output_name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if TARGET_TRANSFORMS.get(output_name) == "log":
        return np.exp(arr).astype(np.float32)
    return arr


def datetime_to_day_hour(sim_datetime: str | datetime | None) -> Tuple[int, float, int]:
    """Convert ISO datetime / datetime / None to (day_of_year, utc_hour, year)."""
    if sim_datetime is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(sim_datetime, datetime):
        dt = sim_datetime
    else:
        text = sim_datetime.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid sim_datetime: {sim_datetime!r}. Expected ISO 8601 (e.g. 2025-05-15T12:00:00Z)."
            ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3_600_000_000.0
    return int(dt.timetuple().tm_yday), float(utc_hour), int(dt.year)


def build_multi_head_mlp(n_features: int, dropout_rate: float = 0.04, learning_rate: float = 1e-3):
    """Build the Multi-Task Multi-Head MLP Regressor.

    Architecture (designed by hand, not auto-generated):

    Shared trunk
        Dense(256, relu) → BatchNorm → Dropout
        Dense(256, relu) → BatchNorm → Dropout
        Dense(128, relu) → BatchNorm

    Heads
        temperature_k : Dense(64, relu) → Dense(1, linear)
        pressure_pa   : Dense(64, relu) → Dense(1, linear)   # pressure trained as log-pressure
        wind_u        : Dense(128, relu) → Dense(64, relu) → Dense(1, linear)
        wind_v        : Dense(128, relu) → Dense(64, relu) → Dense(1, linear)

    Loss: Huber(delta=1.0) per head, on normalized targets. Huber is robust to
    the rare extreme wind values present in ERA5 while behaving like MSE near
    the median, which is what we want for a ballistic simulator.
    Optimizer: Adam, default lr=1e-3.
    """
    _, keras = _import_keras()

    inputs = keras.Input(shape=(n_features,), name="weather_features")

    # Shared trunk: learns a joint atmospheric representation from position + altitude + time.
    x = keras.layers.Dense(256, activation="relu", name="shared_dense_1")(inputs)
    x = keras.layers.BatchNormalization(name="shared_batchnorm_1")(x)
    x = keras.layers.Dropout(dropout_rate, name="shared_dropout_1")(x)

    x = keras.layers.Dense(256, activation="relu", name="shared_dense_2")(x)
    x = keras.layers.BatchNormalization(name="shared_batchnorm_2")(x)
    x = keras.layers.Dropout(dropout_rate, name="shared_dropout_2")(x)

    trunk = keras.layers.Dense(128, activation="relu", name="shared_dense_3")(x)
    trunk = keras.layers.BatchNormalization(name="shared_batchnorm_3")(trunk)

    # Smaller heads for relatively smooth targets (T, P).
    temp = keras.layers.Dense(64, activation="relu", name="temperature_head_dense_1")(trunk)
    temperature_k = keras.layers.Dense(1, activation="linear", name="temperature_k")(temp)

    pressure = keras.layers.Dense(64, activation="relu", name="pressure_head_dense_1")(trunk)
    pressure_pa = keras.layers.Dense(1, activation="linear", name="pressure_pa")(pressure)

    # Larger heads for noisier targets (U, V wind).
    wind_u = keras.layers.Dense(128, activation="relu", name="wind_u_head_dense_1")(trunk)
    wind_u = keras.layers.Dense(64, activation="relu", name="wind_u_head_dense_2")(wind_u)
    wind_u = keras.layers.Dense(1, activation="linear", name="wind_u")(wind_u)

    wind_v = keras.layers.Dense(128, activation="relu", name="wind_v_head_dense_1")(trunk)
    wind_v = keras.layers.Dense(64, activation="relu", name="wind_v_head_dense_2")(wind_v)
    wind_v = keras.layers.Dense(1, activation="linear", name="wind_v")(wind_v)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "temperature_k": temperature_k,
            "pressure_pa": pressure_pa,
            "wind_u": wind_u,
            "wind_v": wind_v,
        },
        name="multi_task_multi_head_mlp_regressor",
    )

    losses = {name: keras.losses.Huber(delta=1.0, name=f"{name}_huber") for name in TARGET_OUTPUTS}
    metrics = {
        name: [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]
        for name in TARGET_OUTPUTS
    }

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        metrics=metrics,
    )
    return model


class MultiHeadMLPWeatherModel:
    """Prediction wrapper around the saved Multi-Task Multi-Head MLP model."""

    def __init__(
        self,
        keras_model: Any,
        feature_builder: WeatherFeatureBuilder,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: Mapping[str, float],
        y_std: Mapping[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = keras_model
        self.feature_builder = feature_builder
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = _safe_std(np.asarray(x_std, dtype=np.float32)).reshape(1, -1)
        self.y_mean = {k: float(v) for k, v in y_mean.items()}
        self.y_std = {k: float(v) if abs(float(v)) >= 1e-8 else 1.0 for k, v in y_std.items()}
        self._metadata = metadata or {}

        # Validate that the loaded model and the feature builder agree on width.
        n_features_builder = len(feature_builder.feature_names)
        if self.x_mean.shape[1] != n_features_builder:
            raise ValueError(
                f"Normalization width ({self.x_mean.shape[1]}) does not match "
                f"feature builder width ({n_features_builder}). The feature config "
                f"saved in feature_metadata.json is out of sync with the codebase."
            )
        try:
            input_shape = self.model.input_shape
            n_features_model = int(input_shape[-1])
            if n_features_model != n_features_builder:
                raise ValueError(
                    f"Loaded Keras model expects {n_features_model} features but the feature "
                    f"builder produces {n_features_builder}. Retrain the model after any feature change."
                )
        except (AttributeError, TypeError, IndexError):
            # Older / non-standard Keras models may not expose input_shape; skip.
            pass

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def feature_names(self) -> List[str]:
        return list(self.feature_builder.feature_names)

    @staticmethod
    def compute_normalization(
        X_train_raw: np.ndarray,
        train_targets: Mapping[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
        """Compute per-feature and per-target normalization stats from the TRAIN split only.

        Reductions are done in float64 to avoid catastrophic precision loss
        when summing tens of millions of float32 values, then cast back to
        float32 for storage. This is important because float32 mean over 20M
        rows can drift by O(1) units, which silently breaks normalization.
        """
        X64 = X_train_raw.astype(np.float64, copy=False) if X_train_raw.dtype != np.float64 else X_train_raw
        x_mean = X64.mean(axis=0).astype(np.float32)
        x_std = _safe_std(X64.std(axis=0).astype(np.float32))

        y_mean: Dict[str, float] = {}
        y_std: Dict[str, float] = {}
        y_transformed: Dict[str, np.ndarray] = {}

        for era5_key, output_name in ERA5_TO_OUTPUT.items():
            if era5_key not in train_targets:
                raise KeyError(f"Missing training target: {era5_key}")
            y = _forward_target_transform(output_name, train_targets[era5_key])
            mean = float(np.mean(y.astype(np.float64)))
            std = float(np.std(y.astype(np.float64)))
            if abs(std) < 1e-8:
                std = 1.0
            y_mean[output_name] = mean
            y_std[output_name] = std
            y_transformed[output_name] = y

        return x_mean, x_std, y_mean, y_std, y_transformed

    @staticmethod
    def normalize_targets(
        raw_targets: Mapping[str, np.ndarray],
        y_mean: Mapping[str, float],
        y_std: Mapping[str, float],
    ) -> Dict[str, np.ndarray]:
        y_norm: Dict[str, np.ndarray] = {}
        for era5_key, output_name in ERA5_TO_OUTPUT.items():
            y = _forward_target_transform(output_name, raw_targets[era5_key])
            y_norm[output_name] = ((y - float(y_mean[output_name])) / float(y_std[output_name])).astype(np.float32)
        return y_norm

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "MultiHeadMLPWeatherModel":
        """Load a trained model from ``artifact_dir`` with explicit, readable errors."""
        _, keras = _import_keras()
        artifact_path = Path(artifact_dir).resolve()

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {artifact_path}. "
                f"Train the model first (python train_multi_head_mlp.py) "
                f"or set WEATHER_ARTIFACT_DIR to point at an existing trained directory."
            )

        model_path = artifact_path / MODEL_FILENAME
        norm_path = artifact_path / NORMALIZATION_FILENAME
        metadata_path = artifact_path / METADATA_FILENAME
        feature_path = artifact_path / FEATURE_METADATA_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(
                f"Multi-head MLP model file not found: {model_path}. "
                f"Run train_multi_head_mlp.py to produce {MODEL_FILENAME}."
            )
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {norm_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature metadata file not found: {feature_path}")

        try:
            keras_model = keras.models.load_model(model_path)
        except Exception as exc:  # pragma: no cover - depends on TF version
            raise RuntimeError(
                f"TensorFlow could not load {model_path}. This usually means a TF "
                f"version mismatch or a corrupted .keras file. Original error: {exc}"
            ) from exc

        with open(norm_path, "r", encoding="utf-8") as f:
            norm = json.load(f)
        with open(feature_path, "r", encoding="utf-8") as f:
            feature_meta = json.load(f)

        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        feature_cfg = WeatherFeatureConfig(**feature_meta.get("config", {}))
        feature_builder = WeatherFeatureBuilder(feature_cfg)

        return cls(
            keras_model=keras_model,
            feature_builder=feature_builder,
            x_mean=np.asarray(norm["x_mean"], dtype=np.float32),
            x_std=np.asarray(norm["x_std"], dtype=np.float32),
            y_mean={k: v["mean"] for k, v in norm["targets"].items()},
            y_std={k: v["std"] for k, v in norm["targets"].items()},
            metadata=metadata,
        )

    def _normalize_features(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        X_raw = self.feature_builder.transform(dict(features))
        if X_raw.shape[1] != self.x_mean.shape[1]:
            raise ValueError(
                f"Feature width mismatch at predict time: got {X_raw.shape[1]}, "
                f"expected {self.x_mean.shape[1]}. The serving feature builder is out of "
                f"sync with the trained model. Retrain or downgrade the codebase."
            )
        return ((X_raw - self.x_mean) / self.x_std).astype(np.float32)

    def _raw_model_predict(self, X_norm: np.ndarray) -> Dict[str, np.ndarray]:
        raw = self.model.predict(X_norm, verbose=0)
        if isinstance(raw, dict):
            return {name: np.asarray(raw[name]).reshape(-1) for name in TARGET_OUTPUTS}

        if isinstance(raw, (list, tuple)):
            output_names = list(getattr(self.model, "output_names", TARGET_OUTPUTS))
            return {
                name: np.asarray(values).reshape(-1)
                for name, values in zip(output_names, raw)
                if name in TARGET_OUTPUTS
            }

        raise TypeError(f"Unexpected Keras predict output type: {type(raw)!r}")

    def predict_features(self, features: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X_norm = self._normalize_features(features)
        pred_norm = self._raw_model_predict(X_norm)

        out: Dict[str, np.ndarray] = {}
        for output_name in TARGET_OUTPUTS:
            y_norm = pred_norm[output_name]
            transformed = y_norm * self.y_std[output_name] + self.y_mean[output_name]
            physical = _inverse_target_transform(output_name, transformed)
            if output_name == "pressure_pa":
                physical = np.clip(physical, 1.0, None)
            out[output_name] = physical.astype(np.float32)
        return out

    def predict_one_from_parts(
        self,
        lat: float,
        lon: float,
        altitude_m: float,
        day_of_year: float,
        utc_hour: float,
        year: int | None = None,
    ) -> Dict[str, float]:
        # Compute local solar hour using the SAME convention as training:
        # longitude is first folded into [-180, 180].
        lon_corrected = float(lon)
        if lon_corrected > 180.0:
            lon_corrected -= 360.0
        local_solar_hour = (float(utc_hour) + lon_corrected / 15.0) % 24.0

        features = {
            "lat": np.array([lat], dtype=np.float32),
            "lon": np.array([lon], dtype=np.float32),
            "altitude_m": np.array([altitude_m], dtype=np.float32),
            "day_of_year": np.array([day_of_year], dtype=np.float32),
            "utc_hour": np.array([utc_hour], dtype=np.float32),
            "local_solar_hour": np.array([local_solar_hour], dtype=np.float32),
        }
        pred = self.predict_features(features)
        return {name: float(pred[name][0]) for name in TARGET_OUTPUTS}

    def predict_one(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        sim_datetime: str | datetime | None,
    ) -> Dict[str, float]:
        day_of_year, utc_hour, year = datetime_to_day_hour(sim_datetime)
        return self.predict_one_from_parts(
            lat=lat,
            lon=lon,
            altitude_m=alt_m,
            day_of_year=day_of_year,
            utc_hour=utc_hour,
            year=year,
        )

    def predict_batch_features(self, features: Mapping[str, np.ndarray]) -> List[Dict[str, float]]:
        pred = self.predict_features(features)
        n = len(pred["temperature_k"])
        return [
            {name: float(pred[name][i]) for name in TARGET_OUTPUTS}
            for i in range(n)
        ]

    def predict_batch(self, requests: Iterable[Mapping[str, Any]]) -> List[Dict[str, float]]:
        rows = list(requests)
        if not rows:
            return []

        lat: List[float] = []
        lon: List[float] = []
        altitude_m: List[float] = []
        day_of_year: List[float] = []
        utc_hour: List[float] = []

        for row in rows:
            lat.append(float(row.get("lat", row.get("latitude"))))
            lon.append(float(row.get("lon", row.get("longitude"))))
            altitude_m.append(float(row.get("altitude_m", row.get("alt_m", row.get("alt")))))

            if "day_of_year" in row and "utc_hour" in row:
                day_of_year.append(float(row["day_of_year"]))
                utc_hour.append(float(row["utc_hour"]))
            else:
                doy, hour, _ = datetime_to_day_hour(row.get("sim_datetime"))
                day_of_year.append(float(doy))
                utc_hour.append(float(hour))

        features = {
            "lat": np.asarray(lat, dtype=np.float32),
            "lon": np.asarray(lon, dtype=np.float32),
            "altitude_m": np.asarray(altitude_m, dtype=np.float32),
            "day_of_year": np.asarray(day_of_year, dtype=np.float32),
            "utc_hour": np.asarray(utc_hour, dtype=np.float32),
        }
        return self.predict_batch_features(features)

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "Multi-Task Multi-Head MLP Regressor",
            "backend": "multi_head_mlp",
            "keras_model_name": getattr(self.model, "name", "multi_task_multi_head_mlp_regressor"),
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "targets": list(TARGET_OUTPUTS),
            "target_transforms": dict(TARGET_TRANSFORMS),
            "metadata": self.metadata,
        }

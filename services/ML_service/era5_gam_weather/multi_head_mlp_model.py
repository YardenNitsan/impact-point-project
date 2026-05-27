"""Multi-Task Multi-Head MLP weather model.

This module:
  * defines the Keras architecture (shared trunk + 4 task-specific heads),
  * computes train-only normalization statistics (no leakage),
  * provides a serving wrapper (``MultiHeadMLPWeatherModel``) that loads a
    saved Keras model + JSON statistics and produces predictions in physical
    units.

The architecture is designed BY HAND. TensorFlow only provides:
  * the layer math (Dense, BatchNorm, Dropout)
  * the activations (ReLU, linear)
  * the optimizer (Adam) and Huber loss
  * autodiff (backprop) and ``model.fit``

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

# Pressure is trained as log(P): atmospheric pressure decays exponentially with
# altitude, so log(P) is much closer to a smooth function of the inputs we feed
# the network. Predicting raw Pa would force the network to fit values spanning
# 5,000 → 102,000 with a single linear head, which is unstable.
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


DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "temperature_k": 0.5,
    "pressure_pa": 0.5,
    "wind_u": 1.5,
    "wind_v": 1.5,
}


def _residual_block(x, units: int, dropout_rate: float, regularizer, idx: int, keras):
    """Pre-activation residual block: BN → ReLU → Dense → BN → ReLU → (Dropout) → Dense → Add.

    The skip connection lets later layers reuse the heavily Fourier-encoded
    inputs without re-learning them, which matters because our feature builder
    emits sin/cos pairs at multiple harmonics.
    """
    residual = x
    y = keras.layers.BatchNormalization(name=f"resblock_{idx}_bn1")(x)
    y = keras.layers.Activation("relu", name=f"resblock_{idx}_relu1")(y)
    y = keras.layers.Dense(units, name=f"resblock_{idx}_dense1", kernel_regularizer=regularizer)(y)
    y = keras.layers.BatchNormalization(name=f"resblock_{idx}_bn2")(y)
    y = keras.layers.Activation("relu", name=f"resblock_{idx}_relu2")(y)
    if dropout_rate > 0.0:
        y = keras.layers.Dropout(dropout_rate, name=f"resblock_{idx}_dropout")(y)
    y = keras.layers.Dense(units, name=f"resblock_{idx}_dense2", kernel_regularizer=regularizer)(y)
    return keras.layers.Add(name=f"resblock_{idx}_add")([residual, y])


def build_multi_head_mlp(
    n_features: int,
    dropout_rate: float = 0.4,
    learning_rate: Any = 2e-3,
    l2_weight_decay: float = 0.0,
    loss_weights: Optional[Mapping[str, float]] = None,
    n_residual_blocks: int = 4,
    block_width: int = 512,
    head_hidden_width: int = 128,
    huber_delta: float = 0.5,
    weight_decay: float = 1e-5,
    use_adamw: bool = True,
):
    """Build the Multi-Task Multi-Head MLP Regressor with a residual trunk.

    Architecture
    ------------
    Stem
        Dense(block_width, linear)
    Trunk
        n_residual_blocks × pre-activation ResNet blocks of width block_width
        BN → ReLU (final pre-activation)
    Heads
        temperature_k : Dense(head_hidden_width, relu) → Dense(1, linear)
        pressure_pa   : Dense(head_hidden_width, relu) → Dense(1, linear)   # trained as log(P)
        wind_u        : Dense(2*head_hidden_width, relu) → Dense(head_hidden_width, relu) → Dense(1, linear)
        wind_v        : Dense(2*head_hidden_width, relu) → Dense(head_hidden_width, relu) → Dense(1, linear)

    Loss: Huber(delta=huber_delta) per head, on normalized targets. Per-target
    loss_weights bias the optimizer toward the harder targets (wind). Default
    weights are {T:0.5, P:0.5, U:1.5, V:1.5}.

    Optimizer: AdamW (default) with decoupled weight_decay, or plain Adam if
    use_adamw=False. ``learning_rate`` may be a float or a Keras LR schedule.

    Parameters
    ----------
    n_features : int
        Width of the engineered feature vector.
    dropout_rate : float, default 0.0
        Dropout inside each residual block. Default off — the network is now
        regularized via AdamW weight decay, BatchNorm, and (optionally) L2.
    learning_rate : float or LearningRateSchedule, default 2e-3
        Optimizer learning rate. Pass a Keras schedule for cosine/warmup.
    l2_weight_decay : float, default 0.0
        L2 kernel regularization on every Dense in the trunk and heads. Usually
        leave at 0 when use_adamw=True (decoupled WD already regularizes).
    loss_weights : Mapping[str, float], optional
        Per-target loss weights. If omitted, DEFAULT_LOSS_WEIGHTS is used.
    n_residual_blocks : int, default 4
        Depth of the residual trunk.
    block_width : int, default 512
        Width of every Dense in the residual trunk.
    head_hidden_width : int, default 128
        Width of the first hidden layer in each head.
    huber_delta : float, default 0.5
        Huber loss delta on normalized targets. Smaller values penalize the
        long tail less aggressively, which helps wind heads.
    weight_decay : float, default 1e-5
        AdamW decoupled weight decay. Ignored when use_adamw=False.
    use_adamw : bool, default True
        If True, use AdamW; falls back to Adam if AdamW is not present in the
        installed Keras version.
    """
    _, keras = _import_keras()

    if l2_weight_decay > 0.0:
        kernel_regularizer = keras.regularizers.l2(l2_weight_decay)
    else:
        kernel_regularizer = None

    def dense(units: int, name: str, activation: str = "relu"):
        return keras.layers.Dense(
            units, activation=activation, name=name,
            kernel_regularizer=kernel_regularizer,
        )

    inputs = keras.Input(shape=(n_features,), name="weather_features")

    # Stem projection: bring features into block_width before residual blocks.
    x = keras.layers.Dense(
        block_width, name="stem_dense", kernel_regularizer=kernel_regularizer,
    )(inputs)

    for i in range(n_residual_blocks):
        x = _residual_block(x, block_width, dropout_rate, kernel_regularizer, i + 1, keras)

    # Final pre-activation so heads see a clean ReLU output.
    trunk = keras.layers.BatchNormalization(name="trunk_final_bn")(x)
    trunk = keras.layers.Activation("relu", name="trunk_final_relu")(trunk)

    # Smaller heads for relatively smooth targets (T, P).
    temp = dense(head_hidden_width, "temperature_head_dense_1")(trunk)
    temperature_k = dense(1, "temperature_k", activation="linear")(temp)

    pressure = dense(head_hidden_width, "pressure_head_dense_1")(trunk)
    pressure_pa = dense(1, "pressure_pa", activation="linear")(pressure)

    # Larger heads for noisier targets (U, V wind).
    wind_u = dense(2 * head_hidden_width, "wind_u_head_dense_1")(trunk)
    wind_u = dense(head_hidden_width, "wind_u_head_dense_2")(wind_u)
    wind_u = dense(1, "wind_u", activation="linear")(wind_u)

    wind_v = dense(2 * head_hidden_width, "wind_v_head_dense_1")(trunk)
    wind_v = dense(head_hidden_width, "wind_v_head_dense_2")(wind_v)
    wind_v = dense(1, "wind_v", activation="linear")(wind_v)

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

    losses = {
        name: keras.losses.Huber(delta=float(huber_delta), name=f"{name}_huber")
        for name in TARGET_OUTPUTS
    }
    metrics = {
        name: [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ]
        for name in TARGET_OUTPUTS
    }

    if loss_weights is None:
        weight_dict = dict(DEFAULT_LOSS_WEIGHTS)
    else:
        weight_dict = {
            name: float(loss_weights.get(name, DEFAULT_LOSS_WEIGHTS[name]))
            for name in TARGET_OUTPUTS
        }

    optimizer = None
    if use_adamw:
        adamw_cls = getattr(keras.optimizers, "AdamW", None)
        if adamw_cls is not None:
            optimizer = adamw_cls(learning_rate=learning_rate, weight_decay=float(weight_decay))
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=weight_dict,
        metrics=metrics,
    )
    return model


# ---------------------------------------------------------------------------
# Welford online accumulator for streaming feature normalization.
#
# The previous implementation cast the entire X_train_raw to float64 to compute
# per-column mean/std without precision drift. For (N, F) = (3e6, 42) that is a
# ~1 GB transient copy. Welford's running update lets us achieve the same
# numerical stability while consuming the matrix in chunks of, say, 100k rows.
# ---------------------------------------------------------------------------
class WelfordAccumulator:
    """Per-column online mean/variance using Welford's algorithm.

    All running stats are kept in float64 so chunks of float32 samples can be
    summed without catastrophic cancellation, even for tens of millions of rows.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = int(n_features)
        self._count = 0
        self._mean = np.zeros(self.n_features, dtype=np.float64)
        self._m2 = np.zeros(self.n_features, dtype=np.float64)

    def update(self, X_chunk: np.ndarray) -> None:
        if X_chunk.size == 0:
            return
        if X_chunk.ndim != 2 or X_chunk.shape[1] != self.n_features:
            raise ValueError(
                f"WelfordAccumulator: expected (N, {self.n_features}) chunk, got {X_chunk.shape}"
            )
        # Convert chunk to float64 just for the reduction; this is a single-chunk
        # cost rather than full-matrix cost.
        chunk = X_chunk.astype(np.float64, copy=False)
        nb = chunk.shape[0]
        chunk_mean = chunk.mean(axis=0)
        chunk_m2 = ((chunk - chunk_mean) ** 2).sum(axis=0)

        if self._count == 0:
            self._count = nb
            self._mean = chunk_mean
            self._m2 = chunk_m2
            return

        na = self._count
        delta = chunk_mean - self._mean
        n_total = na + nb
        # Parallel-algorithm form of Welford's update for combining batches.
        self._mean = self._mean + delta * (nb / n_total)
        self._m2 = self._m2 + chunk_m2 + (delta ** 2) * (na * nb / n_total)
        self._count = n_total

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> np.ndarray:
        return self._mean.astype(np.float32)

    def std(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.n_features, dtype=np.float32)
        var = self._m2 / max(1, self._count - 1)
        return _safe_std(np.sqrt(var).astype(np.float32))


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
        chunk_rows: int = 200_000,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
        """Compute per-feature and per-target normalization stats from the TRAIN split only.

        Uses :class:`WelfordAccumulator` to stream over X in chunks of
        ``chunk_rows`` rows. This avoids the previous full-matrix float64 copy
        (which doubled peak memory during normalization) while keeping the same
        numerical stability for tens of millions of rows.
        """
        if X_train_raw.ndim != 2:
            raise ValueError(f"compute_normalization expects 2-D X, got shape {X_train_raw.shape}")

        n_rows, n_features = X_train_raw.shape
        acc = WelfordAccumulator(n_features=n_features)
        step = max(1, int(chunk_rows))
        for start in range(0, n_rows, step):
            acc.update(X_train_raw[start:start + step])
        x_mean = acc.mean()
        x_std = _safe_std(acc.std())

        y_mean: Dict[str, float] = {}
        y_std: Dict[str, float] = {}
        y_transformed: Dict[str, np.ndarray] = {}

        for era5_key, output_name in ERA5_TO_OUTPUT.items():
            if era5_key not in train_targets:
                raise KeyError(f"Missing training target: {era5_key}")
            y = _forward_target_transform(output_name, train_targets[era5_key])
            # Targets are 1-D; Welford on a single column would just be overhead.
            y64 = y.astype(np.float64, copy=False)
            mean = float(np.mean(y64))
            std = float(np.std(y64))
            if abs(std) < 1e-8:
                std = 1.0
            y_mean[output_name] = mean
            y_std[output_name] = std
            y_transformed[output_name] = y

        return x_mean, x_std, y_mean, y_std, y_transformed

    @staticmethod
    def compute_normalization_streaming(
        x_chunks: Iterable[np.ndarray],
        target_chunks: Mapping[str, Iterable[np.ndarray]],
        n_features: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
        """Welford-based normalization for fully streamed pipelines.

        Use this when you cannot materialize X_train_raw in RAM at all (e.g.
        very large multi-year training runs). Each input is an iterable of
        chunks; the function consumes them once and never holds the full
        dataset.
        """
        x_acc = WelfordAccumulator(n_features=n_features)
        for chunk in x_chunks:
            x_acc.update(np.asarray(chunk))

        y_mean: Dict[str, float] = {}
        y_std: Dict[str, float] = {}
        for era5_key, output_name in ERA5_TO_OUTPUT.items():
            chunks = target_chunks.get(era5_key)
            if chunks is None:
                raise KeyError(f"Missing streaming target: {era5_key}")
            y_acc = WelfordAccumulator(n_features=1)
            for raw in chunks:
                y = _forward_target_transform(output_name, np.asarray(raw)).reshape(-1, 1)
                y_acc.update(y)
            y_mean[output_name] = float(y_acc.mean()[0])
            y_std[output_name] = float(y_acc.std()[0])

        return x_acc.mean(), _safe_std(x_acc.std()), y_mean, y_std

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
            keras_model = keras.models.load_model(model_path, compile=False)
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

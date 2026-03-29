from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .tree_features import TreeFeatureBuilder, TreeFeatureConfig

TARGETS: Tuple[str, ...] = ("T", "P", "U", "V")
TARGET_TRANSFORMS = {
    "T": "identity",
    "P": "log",
    "U": "identity",
    "V": "identity",
}

DEFAULT_MODEL_CONFIGS: Dict[str, Dict] = {
    "T": {
        "loss": "squared_error",
        "learning_rate": 0.035,
        "max_iter": 900,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 48,
        "l2_regularization": 1e-3,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 35,
        "random_state": 42,
    },
    "P": {
        "loss": "squared_error",
        "learning_rate": 0.035,
        "max_iter": 900,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 64,
        "l2_regularization": 1e-3,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 35,
        "random_state": 42,
    },
    "U": {
        "loss": "squared_error",
        "learning_rate": 0.03,
        "max_iter": 1200,
        "max_leaf_nodes": 191,
        "min_samples_leaf": 32,
        "l2_regularization": 5e-4,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 45,
        "random_state": 42,
    },
    "V": {
        "loss": "squared_error",
        "learning_rate": 0.03,
        "max_iter": 1200,
        "max_leaf_nodes": 191,
        "min_samples_leaf": 32,
        "l2_regularization": 5e-4,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 45,
        "random_state": 42,
    },
}


@dataclass
class EvalMetrics:
    mae: Dict[str, float]
    rmse: Dict[str, float]
    max_abs: Dict[str, float]
    n_samples: int


def _forward_transform(name: str, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if TARGET_TRANSFORMS.get(name) == "log":
        return np.log(np.clip(y, 1e-9, None))
    return y


def _inverse_transform(name: str, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if TARGET_TRANSFORMS.get(name) == "log":
        y = np.exp(y)
    return y


def _validate_target_names(target_names: Iterable[str]) -> Tuple[str, ...]:
    names = tuple(target_names)
    if not names:
        raise ValueError("target_names cannot be empty")
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise ValueError(f"Unknown targets requested: {unknown}")
    return names


class WeatherTreeTrainer:
    def __init__(
        self,
        feature_builder: TreeFeatureBuilder | None = None,
        model_configs: Dict[str, Dict] | None = None,
        target_names: Iterable[str] | None = None,
    ):
        self.feature_builder = feature_builder or TreeFeatureBuilder(TreeFeatureConfig())
        self.model_configs = model_configs or DEFAULT_MODEL_CONFIGS
        self.target_names = _validate_target_names(target_names or TARGETS)

    def fit(
        self,
        features: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        sample_weight: np.ndarray | None = None,
    ) -> "WeatherTreeBundle":
        X = self.feature_builder.transform(features)
        models: Dict[str, HistGradientBoostingRegressor] = {}
        fitted_configs: Dict[str, Dict] = {}

        for name in self.target_names:
            if name not in targets:
                raise KeyError(f"Missing training target: {name}")
            cfg = dict(self.model_configs.get(name, DEFAULT_MODEL_CONFIGS[name]))
            model = HistGradientBoostingRegressor(**cfg)
            y = _forward_transform(name, np.asarray(targets[name], dtype=np.float64))
            model.fit(X, y, sample_weight=sample_weight)
            models[name] = model
            fitted_configs[name] = cfg

        return WeatherTreeBundle(
            feature_builder=self.feature_builder,
            models=models,
            model_configs=fitted_configs,
            target_names=self.target_names,
        )


class WeatherTreeBundle:
    def __init__(
        self,
        feature_builder: TreeFeatureBuilder,
        models: Dict[str, HistGradientBoostingRegressor],
        model_configs: Dict[str, Dict] | None = None,
        target_names: Iterable[str] | None = None,
    ):
        self.feature_builder = feature_builder
        self.models = models
        self.target_names = _validate_target_names(target_names or tuple(models.keys()) or TARGETS)
        self.model_configs = model_configs or {name: DEFAULT_MODEL_CONFIGS[name] for name in self.target_names}

    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = self.feature_builder.transform(features)
        out: Dict[str, np.ndarray] = {}
        for name in self.target_names:
            raw = self.models[name].predict(X)
            out[name] = _inverse_transform(name, raw)
        return out

    def predict_one(
        self,
        lat: float,
        lon: float,
        altitude_m: float,
        day_of_year: float,
        utc_hour: float,
    ) -> Dict[str, float]:
        local_solar_hour = (utc_hour + lon / 15.0) % 24.0
        features = {
            "lat": np.array([lat], dtype=np.float64),
            "lon": np.array([lon], dtype=np.float64),
            "altitude_m": np.array([altitude_m], dtype=np.float64),
            "day_of_year": np.array([day_of_year], dtype=np.float64),
            "utc_hour": np.array([utc_hour], dtype=np.float64),
            "local_solar_hour": np.array([local_solar_hour], dtype=np.float64),
        }
        pred = self.predict(features)
        out: Dict[str, float] = {}
        if "T" in pred:
            out["temperature_k"] = float(pred["T"][0])
        if "P" in pred:
            out["pressure_pa"] = float(max(pred["P"][0], 1.0))
        if "U" in pred:
            out["wind_u"] = float(pred["U"][0])
        if "V" in pred:
            out["wind_v"] = float(pred["V"][0])
        return out

    def evaluate(self, features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]) -> EvalMetrics:
        pred = self.predict(features)
        mae: Dict[str, float] = {}
        rmse: Dict[str, float] = {}
        max_abs: Dict[str, float] = {}
        n = len(next(iter(targets.values())))
        for name in self.target_names:
            if name not in targets:
                raise KeyError(f"Missing evaluation target: {name}")
            y_true = np.asarray(targets[name], dtype=np.float64)
            y_pred = np.asarray(pred[name], dtype=np.float64)
            err = y_pred - y_true
            mae[name] = float(np.mean(np.abs(err)))
            rmse[name] = float(np.sqrt(np.mean(err ** 2)))
            max_abs[name] = float(np.max(np.abs(err)))
        return EvalMetrics(mae=mae, rmse=rmse, max_abs=max_abs, n_samples=n)

    def save(self, path: str) -> None:
        payload = {
            "metadata": self.feature_builder.metadata(),
            "model_configs": self.model_configs,
            "target_transforms": TARGET_TRANSFORMS,
            "target_names": list(self.target_names),
            "models": self.models,
        }
        joblib.dump(payload, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "WeatherTreeBundle":
        payload = joblib.load(path)
        metadata = payload["metadata"]
        config = TreeFeatureConfig(**metadata["config"])
        fb = TreeFeatureBuilder(config)
        return cls(
            feature_builder=fb,
            models=payload["models"],
            model_configs=payload.get("model_configs", DEFAULT_MODEL_CONFIGS),
            target_names=payload.get("target_names") or tuple(payload["models"].keys()),
        )

    def describe(self) -> Dict:
        return {
            "feature_metadata": self.feature_builder.metadata(),
            "target_names": list(self.target_names),
            "target_transforms": json.loads(json.dumps(TARGET_TRANSFORMS)),
            "model_configs": json.loads(json.dumps(self.model_configs)),
        }
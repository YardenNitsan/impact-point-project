from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import BasisConfig
from .feature_builder import FeatureBuilder

TARGETS: Tuple[str, ...] = ("T", "P", "U", "V")
TARGET_INDEX = {name: i for i, name in enumerate(TARGETS)}
TARGET_TRANSFORMS = {
    "T": "identity",
    "P": "log",
    "U": "identity",
    "V": "identity",
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
        return np.exp(y)
    return y


class WeatherGAMTrainer:
    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        p = feature_builder.n_features
        self.xtx = np.zeros((p, p), dtype=np.float64)
        self.xty = np.zeros((p, len(TARGETS)), dtype=np.float64)
        self.n_rows = 0

    def update(self, features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]) -> None:
        X = self.feature_builder.transform(features)
        self.xtx += X.T @ X

        y_cols = []
        for name in TARGETS:
            y_cols.append(_forward_transform(name, np.asarray(targets[name], dtype=np.float64)))
        Y = np.column_stack(y_cols)
        self.xty += X.T @ Y
        self.n_rows += X.shape[0]

    def finalize(self) -> "WeatherGAMBundle":
        A = self.xtx.copy()
        diag_idx = np.diag_indices_from(A)
        A[diag_idx] += self.feature_builder.penalty_diag
        coef_matrix = np.linalg.solve(A, self.xty)
        return WeatherGAMBundle(self.feature_builder, coef_matrix)

    def save_state(self, path: str) -> None:
        np.savez_compressed(
            path,
            xtx=self.xtx,
            xty=self.xty,
            n_rows=np.array([self.n_rows], dtype=np.int64),
            metadata_json=np.array([self.feature_builder.metadata_json()], dtype=object),
            target_transforms_json=np.array([json.dumps(TARGET_TRANSFORMS)], dtype=object),
        )

    @classmethod
    def load_state(cls, path: str) -> "WeatherGAMTrainer":
        z = np.load(path, allow_pickle=True)
        metadata = json.loads(z["metadata_json"][0])
        config = BasisConfig(**metadata["config"])
        fb = FeatureBuilder(config)
        trainer = cls(fb)
        trainer.xtx = np.asarray(z["xtx"], dtype=np.float64)
        if "xty" in z:
            trainer.xty = np.asarray(z["xty"], dtype=np.float64)
        else:
            trainer.xty = np.column_stack(
                [
                    np.asarray(z["xty_T"], dtype=np.float64),
                    np.asarray(z["xty_P"], dtype=np.float64),
                    np.asarray(z["xty_U"], dtype=np.float64),
                    np.asarray(z["xty_V"], dtype=np.float64),
                ]
            )
        trainer.n_rows = int(z["n_rows"][0])
        return trainer


class WeatherGAMBundle:
    def __init__(self, feature_builder: FeatureBuilder, coefficients: np.ndarray):
        self.feature_builder = feature_builder
        coef = np.asarray(coefficients, dtype=np.float64)
        if coef.ndim == 1:
            coef = coef[:, None]
        self.coefficients = coef

    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X = self.feature_builder.transform(features)
        raw = X @ self.coefficients
        out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(TARGETS):
            out[name] = _inverse_transform(name, raw[:, i])
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
        return {
            "temperature_k": float(pred["T"][0]),
            "pressure_pa": float(pred["P"][0]),
            "wind_u": float(pred["U"][0]),
            "wind_v": float(pred["V"][0]),
        }

    def evaluate(self, features: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]) -> EvalMetrics:
        pred = self.predict(features)
        mae: Dict[str, float] = {}
        rmse: Dict[str, float] = {}
        max_abs: Dict[str, float] = {}
        n = len(next(iter(targets.values())))
        for name in TARGETS:
            y_true = np.asarray(targets[name], dtype=np.float64)
            y_pred = np.asarray(pred[name], dtype=np.float64)
            err = y_pred - y_true
            mae[name] = float(np.mean(np.abs(err)))
            rmse[name] = float(np.sqrt(np.mean(err ** 2)))
            max_abs[name] = float(np.max(np.abs(err)))
        return EvalMetrics(mae=mae, rmse=rmse, max_abs=max_abs, n_samples=n)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            coef_matrix=self.coefficients,
            metadata_json=np.array([self.feature_builder.metadata_json()], dtype=object),
            target_transforms_json=np.array([json.dumps(TARGET_TRANSFORMS)], dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "WeatherGAMBundle":
        z = np.load(path, allow_pickle=True)
        metadata = json.loads(z["metadata_json"][0])
        config = BasisConfig(**metadata["config"])
        fb = FeatureBuilder(config)

        if "coef_matrix" in z:
            coefficients = np.asarray(z["coef_matrix"], dtype=np.float64)
        else:
            coefficients = np.column_stack(
                [
                    np.asarray(z["coef_T"], dtype=np.float64),
                    np.asarray(z["coef_P"], dtype=np.float64),
                    np.asarray(z["coef_U"], dtype=np.float64),
                    np.asarray(z["coef_V"], dtype=np.float64),
                ]
            )
        return cls(fb, coefficients)

    def term_contributions(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        X = self.feature_builder.transform(features)
        out: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in TARGETS}
        for ts in self.feature_builder.term_slices:
            block = X[:, ts.start:ts.end]
            coef_block = self.coefficients[ts.start:ts.end, :]
            contrib = block @ coef_block
            for i, target in enumerate(TARGETS):
                if ts.name == "intercept":
                    out[target][ts.name] = _inverse_transform(target, contrib[:, i])
                else:
                    out[target][ts.name] = contrib[:, i]
        return out
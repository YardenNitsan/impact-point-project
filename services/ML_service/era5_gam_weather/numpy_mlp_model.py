"""
NumpyMLPWeatherModel – Weather prediction model using 4 separate NumPy MLPs.

Each target variable (temperature_k, pressure_pa, wind_u, wind_v) has its
own independently trained neural network.  All four networks share the same
engineered input features but have separate weights, target normalisation,
and training histories.

This module provides:
  - NumpyMLPWeatherModel.train_from_data()   → train from raw feature/target dicts
  - NumpyMLPWeatherModel.load(path)          → load a saved .npz artefact
  - model.predict_one(lat, lon, alt, dt)     → single prediction
  - model.predict_batch(requests)            → batch prediction
  - model.save(path)                         → save all weights & stats

No sklearn, PyTorch, TensorFlow, or any other ML library is used.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .numpy_nn import (
    Normaliser,
    NumpyMLP,
    TrainingHistory,
    manual_mae,
    manual_max_abs_error,
    manual_rmse,
    train_mlp,
)
from .tree_features import TreeFeatureBuilder, TreeFeatureConfig

# The four target variables – same names as the tree model
TARGET_MAP = {
    "T": "temperature_k",
    "P": "pressure_pa",
    "U": "wind_u",
    "V": "wind_v",
}
TARGET_KEYS = ("T", "P", "U", "V")

# Default architecture per target (can be overridden)
DEFAULT_LAYER_SIZES = (128, 128, 64, 1)

# Per-target training hyper-parameters
DEFAULT_HPARAMS: Dict[str, Dict[str, Any]] = {
    "T": {"lr": 5e-4, "epochs": 400, "batch_size": 512, "patience": 40, "weight_decay": 1e-5},
    "P": {"lr": 5e-4, "epochs": 400, "batch_size": 512, "patience": 40, "weight_decay": 1e-5},
    "U": {"lr": 3e-4, "epochs": 500, "batch_size": 512, "patience": 50, "weight_decay": 1e-5},
    "V": {"lr": 3e-4, "epochs": 500, "batch_size": 512, "patience": 50, "weight_decay": 1e-5},
}

# Pressure is trained in log-space for numerical stability
TARGET_TRANSFORMS = {
    "T": "identity",
    "P": "log",
    "U": "identity",
    "V": "identity",
}


def _forward_transform(name: str, y: np.ndarray) -> np.ndarray:
    if TARGET_TRANSFORMS.get(name) == "log":
        return np.log(np.clip(y, 1e-9, None))
    return y


def _inverse_transform(name: str, y: np.ndarray) -> np.ndarray:
    if TARGET_TRANSFORMS.get(name) == "log":
        return np.exp(y)
    return y


class NumpyMLPWeatherModel:
    """
    Weather prediction model with 4 independent NumPy MLP networks.
    """

    def __init__(self) -> None:
        self.feature_builder = TreeFeatureBuilder(TreeFeatureConfig())
        self.input_normaliser = Normaliser()

        # One MLP + target normaliser per target
        self.models: Dict[str, NumpyMLP] = {}
        self.target_normalisers: Dict[str, Normaliser] = {}
        self.histories: Dict[str, TrainingHistory] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def feature_names(self) -> List[str]:
        return list(self.feature_builder.feature_names)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train_from_data(
        cls,
        train_features: Dict[str, np.ndarray],
        train_targets: Dict[str, np.ndarray],
        val_features: Dict[str, np.ndarray],
        val_targets: Dict[str, np.ndarray],
        layer_sizes: Tuple[int, ...] = DEFAULT_LAYER_SIZES,
        hparams: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> "NumpyMLPWeatherModel":
        """
        Train 4 separate MLPs from raw feature dictionaries.

        Parameters
        ----------
        train_features : dict with keys lat, lon, altitude_m, day_of_year, utc_hour, local_solar_hour
        train_targets  : dict with keys T, P, U, V
        val_features   : same structure, for validation
        val_targets    : same structure, for validation
        """
        hparams = hparams or DEFAULT_HPARAMS
        model = cls()

        # 1. Build engineered feature matrices
        X_train_raw = model.feature_builder.transform(train_features)
        X_val_raw = model.feature_builder.transform(val_features)
        n_features = X_train_raw.shape[1]

        if verbose:
            print(f"Feature matrix: {X_train_raw.shape[1]} features, "
                  f"{X_train_raw.shape[0]} train rows, {X_val_raw.shape[0]} val rows")

        # 2. Fit input normaliser on training data only
        model.input_normaliser.fit(X_train_raw)
        X_train = model.input_normaliser.transform(X_train_raw)
        X_val = model.input_normaliser.transform(X_val_raw)

        # 3. Train one network per target
        for tgt in TARGET_KEYS:
            hp = hparams.get(tgt, DEFAULT_HPARAMS[tgt])
            if verbose:
                print(f"\n{'='*60}")
                print(f"Training network for target: {tgt}  ({TARGET_MAP[tgt]})")
                print(f"  architecture: {n_features} -> {' -> '.join(map(str, layer_sizes))}")
                print(f"  lr={hp['lr']}, epochs={hp['epochs']}, batch={hp['batch_size']}, "
                      f"patience={hp['patience']}, wd={hp['weight_decay']}")
                print(f"{'='*60}")

            # Apply forward transform (e.g. log for pressure)
            y_train_raw = _forward_transform(tgt, train_targets[tgt]).reshape(-1, 1)
            y_val_raw = _forward_transform(tgt, val_targets[tgt]).reshape(-1, 1)

            # Normalise targets
            tgt_norm = Normaliser()
            tgt_norm.fit(y_train_raw)
            y_train = tgt_norm.transform(y_train_raw)
            y_val = tgt_norm.transform(y_val_raw)
            model.target_normalisers[tgt] = tgt_norm

            # Create MLP
            mlp = NumpyMLP(
                layer_sizes=layer_sizes,
                n_input=n_features,
                seed=seed + hash(tgt) % 10000,
                lr=hp["lr"],
                weight_decay=hp["weight_decay"],
            )

            # Train with mini-batches, early stopping, and Adam
            mlp = train_mlp(
                model=mlp,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=hp["epochs"],
                batch_size=hp["batch_size"],
                patience=hp["patience"],
                verbose=verbose,
                target_name=tgt,
            )

            model.models[tgt] = mlp
            model.histories[tgt] = mlp.history

        # 4. Store metadata
        model._metadata = {
            "layer_sizes": list(layer_sizes),
            "n_input_features": n_features,
            "feature_names": model.feature_names,
            "feature_config": model.feature_builder.config.to_dict(),
            "hparams": {k: dict(v) for k, v in hparams.items()},
            "seed": seed,
            "target_transforms": dict(TARGET_TRANSFORMS),
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
        }

        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        features: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        split_name: str = "test",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute MAE, RMSE, max_abs_error for each target on a given split.
        Returns dict[target_name] -> {mae, rmse, max_abs}.
        """
        X_raw = self.feature_builder.transform(features)
        X = self.input_normaliser.transform(X_raw)
        results: Dict[str, Dict[str, float]] = {}

        for tgt in TARGET_KEYS:
            y_true = np.asarray(targets[tgt], dtype=np.float64)

            # Predict in normalised space, then denormalise + inverse transform
            y_pred_norm = self.models[tgt].predict(X)
            y_pred_transformed = self.target_normalisers[tgt].inverse_transform(y_pred_norm).ravel()
            y_pred = _inverse_transform(tgt, y_pred_transformed)

            # Clip pressure to positive
            if tgt == "P":
                y_pred = np.clip(y_pred, 1.0, None)

            results[tgt] = {
                "mae": manual_mae(y_pred, y_true),
                "rmse": manual_rmse(y_pred, y_true),
                "max_abs": manual_max_abs_error(y_pred, y_true),
            }

        self.metrics[split_name] = results
        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_array(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict for an array of features. Returns dict[T/P/U/V] -> array."""
        X_raw = self.feature_builder.transform(features)
        X = self.input_normaliser.transform(X_raw)
        out: Dict[str, np.ndarray] = {}

        for tgt in TARGET_KEYS:
            y_norm = self.models[tgt].predict(X)
            y_transformed = self.target_normalisers[tgt].inverse_transform(y_norm).ravel()
            y = _inverse_transform(tgt, y_transformed)
            if tgt == "P":
                y = np.clip(y, 1.0, None)
            out[tgt] = y

        return out

    def predict_one(
        self,
        lat: float,
        lon: float,
        altitude_m: float,
        day_of_year: float,
        utc_hour: float,
    ) -> Dict[str, float]:
        """Single-point prediction matching the tree model interface."""
        local_solar_hour = (utc_hour + lon / 15.0) % 24.0
        features = {
            "lat": np.array([lat], dtype=np.float64),
            "lon": np.array([lon], dtype=np.float64),
            "altitude_m": np.array([altitude_m], dtype=np.float64),
            "day_of_year": np.array([day_of_year], dtype=np.float64),
            "utc_hour": np.array([utc_hour], dtype=np.float64),
            "local_solar_hour": np.array([local_solar_hour], dtype=np.float64),
        }
        pred = self._predict_array(features)
        return {
            "temperature_k": float(pred["T"][0]),
            "pressure_pa": float(max(pred["P"][0], 1.0)),
            "wind_u": float(pred["U"][0]),
            "wind_v": float(pred["V"][0]),
        }

    def predict_batch_features(self, features: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """Batch prediction from raw feature arrays."""
        pred = self._predict_array(features)
        n = len(pred["T"])
        return [
            {
                "temperature_k": float(pred["T"][i]),
                "pressure_pa": float(pred["P"][i]),
                "wind_u": float(pred["U"][i]),
                "wind_v": float(pred["V"][i]),
            }
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Save / Load  (.npz format – no sklearn, no joblib)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save all model weights, normalisation stats, and metadata to a .npz file.
        """
        save_dict: Dict[str, Any] = {}

        # Input normaliser
        inp_state = self.input_normaliser.state_dict()
        save_dict["input_norm_mean"] = inp_state["mean"]
        save_dict["input_norm_std"] = inp_state["std"]

        # Per-target: weights + target normaliser
        for tgt in TARGET_KEYS:
            weights = self.models[tgt].get_weights()
            for k, v in weights.items():
                save_dict[f"{tgt}_{k}"] = v

            tgt_state = self.target_normalisers[tgt].state_dict()
            save_dict[f"{tgt}_target_norm_mean"] = tgt_state["mean"]
            save_dict[f"{tgt}_target_norm_std"] = tgt_state["std"]

            # Training history
            save_dict[f"{tgt}_train_loss"] = np.array(self.histories[tgt].train_loss)
            save_dict[f"{tgt}_val_loss"] = np.array(self.histories[tgt].val_loss)

        # Metadata as JSON string stored in a numpy array
        save_dict["metadata_json"] = np.array([json.dumps(self._metadata)])

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "NumpyMLPWeatherModel":
        """
        Load a saved model from a .npz file.
        """
        data = np.load(path, allow_pickle=False)
        model = cls()

        # Metadata
        meta_str = str(data["metadata_json"][0])
        model._metadata = json.loads(meta_str)

        layer_sizes = tuple(model._metadata["layer_sizes"])
        n_input = model._metadata["n_input_features"]
        seed = model._metadata.get("seed", 42)

        # Feature builder
        feat_cfg = model._metadata.get("feature_config", {})
        if feat_cfg:
            model.feature_builder = TreeFeatureBuilder(TreeFeatureConfig(**feat_cfg))

        # Input normaliser
        model.input_normaliser.load_state_dict({
            "mean": data["input_norm_mean"],
            "std": data["input_norm_std"],
        })

        # Per-target
        for tgt in TARGET_KEYS:
            mlp = NumpyMLP(
                layer_sizes=layer_sizes,
                n_input=n_input,
                seed=seed,
                lr=1e-3,  # not used at inference
            )
            weights = {
                k.replace(f"{tgt}_", ""): data[k]
                for k in data.files
                if k.startswith(f"{tgt}_layer_")
            }
            mlp.set_weights(weights)
            model.models[tgt] = mlp

            tgt_norm = Normaliser()
            tgt_norm.load_state_dict({
                "mean": data[f"{tgt}_target_norm_mean"],
                "std": data[f"{tgt}_target_norm_std"],
            })
            model.target_normalisers[tgt] = tgt_norm

            # History
            history = TrainingHistory()
            if f"{tgt}_train_loss" in data.files:
                history.train_loss = data[f"{tgt}_train_loss"].tolist()
                history.val_loss = data[f"{tgt}_val_loss"].tolist()
            model.histories[tgt] = history

        return model

    def describe(self) -> Dict[str, Any]:
        """Return a human-readable description of the model."""
        return {
            "type": "NumpyMLPWeatherModel",
            "backend": "numpy_mlp",
            "layer_sizes": self._metadata.get("layer_sizes", []),
            "n_input_features": self._metadata.get("n_input_features", 0),
            "feature_names": self.feature_names,
            "targets": list(TARGET_MAP.values()),
            "target_transforms": dict(TARGET_TRANSFORMS),
            "metrics": self.metrics,
        }

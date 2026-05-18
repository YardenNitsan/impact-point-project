"""Raw K-Nearest-Neighbours weather model.

Hand-written brute-force implementation — no scikit-learn, no kd-tree library.
The data set is small enough (tens of thousands of points) that a vectorised
NumPy distance computation finishes in a few ms per query.

Pipeline mirrors the description in the project book:

    1. Min-max normalise every feature to [0, 1] using the per-column min/max
       observed at training time.
    2. At inference, compute the Euclidean distance from the (normalised)
       query to every training row.
    3. Keep the K rows with the smallest distance.
    4. Aggregate their targets with inverse-distance weighting.
    5. Reject queries that fall outside the training envelope as
       out-of-distribution (the book calls this the OOD guard).

Two feature representations live side by side:

* ``raw_inputs`` are the 5 numbers the user actually queries
  (lat, lon, altitude_m, day_of_year, utc_hour). The OOD envelope check runs
  on these so a query is "in distribution" iff every raw value sits inside
  its observed range.
* ``cyclic_features`` add (sin, cos) of day-of-year and hour so the Euclidean
  metric treats Dec 31 and Jan 1 as neighbours. Distance work uses these.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


RAW_INPUT_NAMES: Tuple[str, ...] = (
    "lat",
    "lon",
    "altitude_m",
    "day_of_year",
    "utc_hour",
)

CYCLIC_FEATURE_NAMES: Tuple[str, ...] = (
    "lat",
    "lon",
    "altitude_m",
    "day_of_year_sin",
    "day_of_year_cos",
    "utc_hour_sin",
    "utc_hour_cos",
)

TARGET_NAMES: Tuple[str, ...] = (
    "temperature_K",
    "pressure_Pa",
    "wind_u_east_mps",
    "wind_v_north_mps",
)

# Distances smaller than this are clamped before inverting so a query that
# coincides exactly with a training point does not produce an infinite weight.
DISTANCE_FLOOR = 1e-6


def cyclic_encode(raw_inputs: np.ndarray) -> np.ndarray:
    """Expand the 5-column raw input into the 7-column cyclic feature matrix."""
    if raw_inputs.ndim != 2 or raw_inputs.shape[1] != len(RAW_INPUT_NAMES):
        raise ValueError(
            f"raw_inputs must have shape (n, {len(RAW_INPUT_NAMES)}), got {raw_inputs.shape}"
        )
    lat = raw_inputs[:, 0].astype(np.float32)
    lon = raw_inputs[:, 1].astype(np.float32)
    altitude_m = raw_inputs[:, 2].astype(np.float32)
    day_of_year = raw_inputs[:, 3].astype(np.float32)
    utc_hour = raw_inputs[:, 4].astype(np.float32)

    doy_angle = 2.0 * np.pi * (day_of_year / 365.25)
    hour_angle = 2.0 * np.pi * (utc_hour / 24.0)

    return np.stack(
        [
            lat,
            lon,
            altitude_m,
            np.sin(doy_angle, dtype=np.float32),
            np.cos(doy_angle, dtype=np.float32),
            np.sin(hour_angle, dtype=np.float32),
            np.cos(hour_angle, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class MinMaxStats:
    """Per-column min and max observed at training time."""

    feature_min: np.ndarray
    feature_max: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        span = self.feature_max - self.feature_min
        span = np.where(span > 0, span, np.float32(1.0))
        return ((x - self.feature_min) / span).astype(np.float32, copy=False)

    def envelope_excursion(self, x: np.ndarray) -> np.ndarray:
        """Per-sample max excursion outside the training envelope, in
        units of feature span. 0 means the sample is inside the envelope.
        """
        span = self.feature_max - self.feature_min
        span = np.where(span > 0, span, np.float32(1.0))
        below = np.maximum(0.0, self.feature_min - x) / span
        above = np.maximum(0.0, x - self.feature_max) / span
        return np.maximum(below, above).max(axis=1)


@dataclass(frozen=True)
class PredictionResult:
    """Single-query prediction with a few diagnostics for the API layer."""

    temperature_K: float
    pressure_Pa: float
    wind_u_east_mps: float
    wind_v_north_mps: float
    neighbor_distances: np.ndarray
    out_of_distribution: bool
    envelope_excursion: float


class KnnWeatherModel:
    """In-memory KNN regressor over the ERA5 weather feature set."""

    def __init__(
        self,
        raw_inputs: np.ndarray,
        targets: np.ndarray,
        k: int,
        ood_threshold: float = 0.05,
    ) -> None:
        if raw_inputs.ndim != 2 or raw_inputs.shape[1] != len(RAW_INPUT_NAMES):
            raise ValueError(
                f"raw_inputs must have shape (n, {len(RAW_INPUT_NAMES)}), got {raw_inputs.shape}"
            )
        if targets.ndim != 2 or targets.shape[1] != len(TARGET_NAMES):
            raise ValueError(
                f"targets must have shape (n, {len(TARGET_NAMES)}), got {targets.shape}"
            )
        if raw_inputs.shape[0] != targets.shape[0]:
            raise ValueError("raw_inputs and targets must have the same number of rows")
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > raw_inputs.shape[0]:
            raise ValueError(
                f"k={k} exceeds the training set size ({raw_inputs.shape[0]})"
            )

        self.k = int(k)
        self.ood_threshold = float(ood_threshold)
        self.raw_inputs = raw_inputs.astype(np.float32, copy=False)
        self.targets = targets.astype(np.float32, copy=False)

        # Envelope stats live in raw-input space so the OOD check reasons in
        # the same units the caller actually sent.
        self.envelope_stats = MinMaxStats(
            feature_min=self.raw_inputs.min(axis=0).astype(np.float32, copy=False),
            feature_max=self.raw_inputs.max(axis=0).astype(np.float32, copy=False),
        )

        cyclic = cyclic_encode(self.raw_inputs)
        self.distance_stats = MinMaxStats(
            feature_min=cyclic.min(axis=0).astype(np.float32, copy=False),
            feature_max=cyclic.max(axis=0).astype(np.float32, copy=False),
        )
        self.features_norm = self.distance_stats.normalize(cyclic)

    @property
    def n_train(self) -> int:
        return int(self.features_norm.shape[0])

    def predict(
        self,
        lat: float,
        lon: float,
        altitude_m: float,
        day_of_year: float,
        utc_hour: float,
    ) -> PredictionResult:
        raw = np.array(
            [[lat, lon, altitude_m, day_of_year, utc_hour]],
            dtype=np.float32,
        )
        excursion = float(self.envelope_stats.envelope_excursion(raw)[0])
        out_of_distribution = excursion > self.ood_threshold

        cyclic = cyclic_encode(raw)
        x_norm = self.distance_stats.normalize(cyclic)

        # Squared Euclidean distance, vectorised over the whole training
        # set. Sqrt is only needed for the K kept points, not all N.
        diffs = self.features_norm - x_norm
        sq_dist = np.einsum("ij,ij->i", diffs, diffs)

        # argpartition gives the K smallest indices in O(N), and we sort the
        # K kept rows afterwards so neighbours are reported nearest-first.
        if self.k < self.n_train:
            top_idx = np.argpartition(sq_dist, self.k)[: self.k]
        else:
            top_idx = np.arange(self.n_train)

        top_sq = sq_dist[top_idx]
        order = np.argsort(top_sq)
        top_idx = top_idx[order]
        top_dist = np.sqrt(top_sq[order]).astype(np.float32, copy=False)

        # Inverse-distance weighting (per the book), with a floor so weights
        # stay finite when the query lands exactly on a sample.
        clamped = np.maximum(top_dist, np.float32(DISTANCE_FLOOR))
        weights = 1.0 / clamped
        weights = (weights / weights.sum()).astype(np.float32, copy=False)

        weighted_mean = self.targets[top_idx].T @ weights

        return PredictionResult(
            temperature_K=float(weighted_mean[0]),
            pressure_Pa=float(weighted_mean[1]),
            wind_u_east_mps=float(weighted_mean[2]),
            wind_v_north_mps=float(weighted_mean[3]),
            neighbor_distances=top_dist,
            out_of_distribution=bool(out_of_distribution),
            envelope_excursion=excursion,
        )

    # --- persistence -----------------------------------------------------

    def save(self, artifact_dir: Path) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            artifact_dir / "dataset.npz",
            raw_inputs=self.raw_inputs,
            targets=self.targets,
        )

        meta = {
            "k": self.k,
            "ood_threshold": self.ood_threshold,
            "n_train": self.n_train,
            "raw_input_names": list(RAW_INPUT_NAMES),
            "cyclic_feature_names": list(CYCLIC_FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "envelope_min": self.envelope_stats.feature_min.tolist(),
            "envelope_max": self.envelope_stats.feature_max.tolist(),
        }
        with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, artifact_dir: Path) -> "KnnWeatherModel":
        artifact_dir = Path(artifact_dir)
        with (artifact_dir / "metadata.json").open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        bundle = np.load(artifact_dir / "dataset.npz")
        raw_inputs = bundle["raw_inputs"].astype(np.float32, copy=False)
        targets = bundle["targets"].astype(np.float32, copy=False)

        return cls(
            raw_inputs=raw_inputs,
            targets=targets,
            k=int(meta["k"]),
            ood_threshold=float(meta.get("ood_threshold", 0.05)),
        )

    def describe(self) -> dict:
        return {
            "k": self.k,
            "n_train": self.n_train,
            "ood_threshold": self.ood_threshold,
            "raw_input_names": list(RAW_INPUT_NAMES),
            "cyclic_feature_names": list(CYCLIC_FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "envelope_min": [float(v) for v in self.envelope_stats.feature_min],
            "envelope_max": [float(v) for v in self.envelope_stats.feature_max],
        }

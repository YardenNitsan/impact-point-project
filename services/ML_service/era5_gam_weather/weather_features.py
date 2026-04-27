"""Deterministic feature engineering for the Multi-Task Multi-Head MLP weather model.

This module is the single source of truth for the input vector that the network
consumes. It is intentionally pure NumPy so that training and serving produce
identical features. The implementation is memory-conscious: every intermediate
array is float32, which roughly halves peak RAM compared to the previous float64
pipeline. Trigonometric and polynomial precision in float32 is more than enough
for inputs in the lat/lon/altitude/time ranges we use.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class WeatherFeatureConfig:
    """Configuration for deterministic weather feature engineering.

    The model is still a neural network; these features only give the network a
    physically meaningful representation of position, altitude, and time. They
    are NOT a black-box feature transform: every column has a clear physical
    meaning (normalized lat/lon, altitude polynomials, sphere XYZ embedding,
    annual harmonics for seasonality, daily harmonics for diurnal cycle).
    """

    annual_harmonics: int = 2
    utc_hour_harmonics: int = 2
    local_hour_harmonics: int = 2
    lat_lon_fourier_harmonics: int = 2
    include_xyz: bool = True
    include_altitude_polynomials: bool = True
    include_altitude_directional_terms: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


class WeatherFeatureBuilder:
    """Builds the exact same engineered feature vector for training and serving."""

    def __init__(self, config: WeatherFeatureConfig | None = None):
        self.config = config or WeatherFeatureConfig()
        self.feature_names: List[str] = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = [
            "lat_norm",
            "lon_norm",
            "abs_lat_norm",
            "altitude_km",
            "log1p_altitude_km",
        ]

        if self.config.include_altitude_polynomials:
            names += ["altitude_km_sq", "altitude_km_cu"]

        names += [
            "sin_lat",
            "cos_lat",
            "sin_lon",
            "cos_lon",
        ]

        if self.config.include_xyz:
            names += ["sphere_x", "sphere_y", "sphere_z"]

        if self.config.include_altitude_directional_terms:
            names += [
                "altitude_x_sin_lat",
                "altitude_x_cos_lat",
                "altitude_x_sin_lon",
                "altitude_x_cos_lon",
                "altitude_x_abs_lat_norm",
            ]

        names += ["day_of_year_norm", "utc_hour_norm", "local_solar_hour_norm"]

        for k in range(1, self.config.annual_harmonics + 1):
            names += [f"doy_sin_{k}", f"doy_cos_{k}"]

        for k in range(1, self.config.utc_hour_harmonics + 1):
            names += [f"utc_sin_{k}", f"utc_cos_{k}"]

        for k in range(1, self.config.local_hour_harmonics + 1):
            names += [f"solar_sin_{k}", f"solar_cos_{k}"]

        for k in range(1, self.config.lat_lon_fourier_harmonics + 1):
            names += [
                f"lat_sin_{k}",
                f"lat_cos_{k}",
                f"lon_sin_{k}",
                f"lon_cos_{k}",
            ]

        return names

    @staticmethod
    def _array(features: Dict[str, np.ndarray], key: str) -> np.ndarray:
        if key not in features:
            raise KeyError(f"Missing feature key: {key}")
        # Force float32 from the start; this is the dominant memory saving.
        return np.asarray(features[key], dtype=np.float32).reshape(-1)

    def transform(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Return an (N, n_features) float32 matrix.

        Inputs may be int/float of any precision; we normalize to float32. We
        avoid creating any float64 intermediates so peak memory stays close to
        ``N * n_features * 4`` bytes plus a few small temporaries.
        """
        lat = self._array(features, "lat")
        lon = self._array(features, "lon")
        altitude_m = self._array(features, "altitude_m")
        day_of_year = self._array(features, "day_of_year")
        utc_hour = self._array(features, "utc_hour")

        if "local_solar_hour" in features:
            local_solar_hour = self._array(features, "local_solar_hour")
        else:
            # Compute on the corrected longitude (-180..180) so serving matches training.
            lon_for_solar = lon.copy()
            lon_for_solar[lon_for_solar > 180.0] -= 360.0
            local_solar_hour = np.mod(utc_hour + lon_for_solar / np.float32(15.0), np.float32(24.0)).astype(np.float32)
            del lon_for_solar

        n = lat.shape[0]
        for name, arr in (
            ("lon", lon),
            ("altitude_m", altitude_m),
            ("day_of_year", day_of_year),
            ("utc_hour", utc_hour),
            ("local_solar_hour", local_solar_hour),
        ):
            if arr.shape[0] != n:
                raise ValueError(f"Feature {name} has length {arr.shape[0]}, expected {n}")

        # Keep longitude in [-180, 180] because ERA5 longitudes may arrive as 0..360.
        lon = lon.copy()
        lon[lon > 180.0] -= 360.0

        deg2rad = np.float32(np.pi / 180.0)
        lat_rad = lat * deg2rad
        lon_rad = lon * deg2rad
        altitude_km = np.clip(altitude_m, 0.0, None) * np.float32(1.0 / 1000.0)
        abs_lat_norm = np.abs(lat) * np.float32(1.0 / 90.0)

        cols: List[np.ndarray] = [
            lat * np.float32(1.0 / 90.0),
            lon * np.float32(1.0 / 180.0),
            abs_lat_norm,
            altitude_km,
            np.log1p(altitude_km),
        ]

        if self.config.include_altitude_polynomials:
            cols += [altitude_km * altitude_km, altitude_km * altitude_km * altitude_km]

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        cols += [sin_lat, cos_lat, sin_lon, cos_lon]

        if self.config.include_xyz:
            cols += [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]

        if self.config.include_altitude_directional_terms:
            cols += [
                altitude_km * sin_lat,
                altitude_km * cos_lat,
                altitude_km * sin_lon,
                altitude_km * cos_lon,
                altitude_km * abs_lat_norm,
            ]

        cols += [
            (day_of_year - np.float32(1.0)) * np.float32(1.0 / 365.25),
            utc_hour * np.float32(1.0 / 24.0),
            local_solar_hour * np.float32(1.0 / 24.0),
        ]

        two_pi = np.float32(2.0 * np.pi)

        for k in range(1, self.config.annual_harmonics + 1):
            angle = (two_pi * k * np.float32(1.0 / 365.25)) * day_of_year
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.utc_hour_harmonics + 1):
            angle = (two_pi * k * np.float32(1.0 / 24.0)) * utc_hour
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.local_hour_harmonics + 1):
            angle = (two_pi * k * np.float32(1.0 / 24.0)) * local_solar_hour
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.lat_lon_fourier_harmonics + 1):
            cols += [
                np.sin(k * lat_rad),
                np.cos(k * lat_rad),
                np.sin(k * lon_rad),
                np.cos(k * lon_rad),
            ]

        # column_stack will allocate a single contiguous (N, F) float32 block.
        X = np.column_stack(cols).astype(np.float32, copy=False)
        if X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"Feature width mismatch: got {X.shape[1]}, expected {len(self.feature_names)} "
                f"(feature config out of sync with feature_names list)"
            )
        return X

    def metadata(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
        }

    def metadata_json(self) -> str:
        return json.dumps(self.metadata(), indent=2)

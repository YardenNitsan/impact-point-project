from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class WeatherFeatureConfig:
    """Configuration for deterministic weather feature engineering.

    The model is still a neural network; these features only give the network a
    physically meaningful representation of position, altitude, and time.
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
        return np.asarray(features[key], dtype=np.float64).reshape(-1)

    def transform(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        lat = self._array(features, "lat")
        lon = self._array(features, "lon")
        altitude_m = self._array(features, "altitude_m")
        day_of_year = self._array(features, "day_of_year")
        utc_hour = self._array(features, "utc_hour")

        if "local_solar_hour" in features:
            local_solar_hour = self._array(features, "local_solar_hour")
        else:
            local_solar_hour = (utc_hour + lon / 15.0) % 24.0

        n = lat.shape[0]
        for name, arr in {
            "lon": lon,
            "altitude_m": altitude_m,
            "day_of_year": day_of_year,
            "utc_hour": utc_hour,
            "local_solar_hour": local_solar_hour,
        }.items():
            if arr.shape[0] != n:
                raise ValueError(f"Feature {name} has length {arr.shape[0]}, expected {n}")

        # Keep longitude in [-180, 180] because ERA5 longitudes may arrive as 0..360.
        lon = lon.copy()
        lon[lon > 180.0] -= 360.0

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        altitude_km = np.clip(altitude_m, 0.0, None) / 1000.0
        abs_lat_norm = np.abs(lat) / 90.0

        cols: List[np.ndarray] = [
            lat / 90.0,
            lon / 180.0,
            abs_lat_norm,
            altitude_km,
            np.log1p(altitude_km),
        ]

        if self.config.include_altitude_polynomials:
            cols += [altitude_km ** 2, altitude_km ** 3]

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        cols += [sin_lat, cos_lat, sin_lon, cos_lon]

        if self.config.include_xyz:
            sphere_x = cos_lat * cos_lon
            sphere_y = cos_lat * sin_lon
            sphere_z = sin_lat
            cols += [sphere_x, sphere_y, sphere_z]

        if self.config.include_altitude_directional_terms:
            cols += [
                altitude_km * sin_lat,
                altitude_km * cos_lat,
                altitude_km * sin_lon,
                altitude_km * cos_lon,
                altitude_km * abs_lat_norm,
            ]

        cols += [
            (day_of_year - 1.0) / 365.25,
            utc_hour / 24.0,
            local_solar_hour / 24.0,
        ]

        for k in range(1, self.config.annual_harmonics + 1):
            angle = 2.0 * np.pi * k * day_of_year / 365.25
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.utc_hour_harmonics + 1):
            angle = 2.0 * np.pi * k * utc_hour / 24.0
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.local_hour_harmonics + 1):
            angle = 2.0 * np.pi * k * local_solar_hour / 24.0
            cols += [np.sin(angle), np.cos(angle)]

        for k in range(1, self.config.lat_lon_fourier_harmonics + 1):
            cols += [
                np.sin(k * lat_rad),
                np.cos(k * lat_rad),
                np.sin(k * lon_rad),
                np.cos(k * lon_rad),
            ]

        X = np.column_stack(cols).astype(np.float32)
        if X.shape[1] != len(self.feature_names):
            raise RuntimeError(
                f"Feature width mismatch: got {X.shape[1]}, expected {len(self.feature_names)}"
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

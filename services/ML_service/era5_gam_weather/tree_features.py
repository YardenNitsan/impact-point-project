from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class TreeFeatureConfig:
    annual_harmonics: int = 2
    utc_hour_harmonics: int = 2
    local_hour_harmonics: int = 2
    include_xyz: bool = True
    include_altitude_polynomials: bool = True
    include_altitude_directional_terms: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


class TreeFeatureBuilder:
    def __init__(self, config: TreeFeatureConfig | None = None):
        self.config = config or TreeFeatureConfig()
        self.feature_names: List[str] = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = [
            "lat_deg",
            "lon_deg",
            "abs_lat_deg",
            "alt_km",
            "log1p_alt_km",
        ]

        if self.config.include_altitude_polynomials:
            names += ["alt_km_sq", "alt_km_cu"]

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
                "alt_x_sin_lat",
                "alt_x_cos_lat",
                "alt_x_sin_lon",
                "alt_x_cos_lon",
                "alt_x_abs_lat",
            ]

        for k in range(1, self.config.annual_harmonics + 1):
            names += [f"doy_sin_{k}", f"doy_cos_{k}"]

        for k in range(1, self.config.utc_hour_harmonics + 1):
            names += [f"utc_sin_{k}", f"utc_cos_{k}"]

        for k in range(1, self.config.local_hour_harmonics + 1):
            names += [f"solar_sin_{k}", f"solar_cos_{k}"]

        return names

    def transform(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        lat = np.asarray(features["lat"], dtype=np.float64)
        lon = np.asarray(features["lon"], dtype=np.float64)
        altitude_m = np.asarray(features["altitude_m"], dtype=np.float64)
        day_of_year = np.asarray(features["day_of_year"], dtype=np.float64)
        utc_hour = np.asarray(features["utc_hour"], dtype=np.float64)
        local_solar_hour = np.asarray(features["local_solar_hour"], dtype=np.float64)

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        alt_km = np.clip(altitude_m, 0.0, None) / 1000.0
        abs_lat = np.abs(lat)

        cols: List[np.ndarray] = [
            lat,
            lon,
            abs_lat,
            alt_km,
            np.log1p(alt_km),
        ]

        if self.config.include_altitude_polynomials:
            cols += [alt_km ** 2, alt_km ** 3]

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
                alt_km * sin_lat,
                alt_km * cos_lat,
                alt_km * sin_lon,
                alt_km * cos_lon,
                alt_km * abs_lat / 90.0,
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

        X = np.column_stack(cols)
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
        return json.dumps(self.metadata())
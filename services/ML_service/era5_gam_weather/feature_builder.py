from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import BasisConfig
from .spline_basis import bspline_basis, harmonic_basis, open_uniform_knots, tensor_product_rows


@dataclass(frozen=True)
class TermSlice:
    name: str
    start: int
    end: int


class FeatureBuilder:
    def __init__(self, config: BasisConfig):
        self.config = config
        d = config.degree

        self.knots_lat_space = open_uniform_knots(*config.lat_range, config.lat_basis_space, d)
        self.knots_lon_space = open_uniform_knots(*config.lon_range, config.lon_basis_space, d)
        self.knots_alt = open_uniform_knots(*config.alt_range_m, config.alt_basis, d)
        self.knots_day = open_uniform_knots(*config.day_of_year_range, config.day_basis, d)

        self.knots_alt_interaction = open_uniform_knots(*config.alt_range_m, config.alt_basis_interaction, d)
        self.knots_lat_interaction = open_uniform_knots(*config.lat_range, config.lat_basis_interaction, d)
        self.knots_lon_interaction = open_uniform_knots(*config.lon_range, config.lon_basis_interaction, d)
        self.knots_day_interaction = open_uniform_knots(*config.day_of_year_range, config.day_basis_interaction, d)

        self.term_slices: List[TermSlice] = []
        cursor = 0

        def add_term(name: str, width: int) -> None:
            nonlocal cursor
            self.term_slices.append(TermSlice(name=name, start=cursor, end=cursor + width))
            cursor += width

        add_term("intercept", 1)
        add_term("space_lat_lon", config.lat_basis_space * config.lon_basis_space)
        add_term("altitude", config.alt_basis)
        add_term("day_of_year", config.day_basis)
        add_term("altitude_x_latitude", config.alt_basis_interaction * config.lat_basis_interaction)
        add_term("altitude_x_longitude", config.alt_basis_interaction * config.lon_basis_interaction)
        add_term("day_x_altitude", config.day_basis_interaction * config.alt_basis_interaction)
        add_term("day_x_latitude", config.day_basis_interaction * config.lat_basis_interaction)
        add_term("day_x_longitude", config.day_basis_interaction * config.lon_basis_interaction)
        add_term("local_hour_harmonics", 2 * config.local_hour_harmonics)
        add_term("utc_hour_harmonics", 2 * config.utc_hour_harmonics)

        self.n_features = cursor
        self.term_index = {ts.name: ts for ts in self.term_slices}
        self.penalty_diag = self._build_penalty_diag()

    def _build_penalty_diag(self) -> np.ndarray:
        penalty = np.zeros(self.n_features, dtype=np.float64)

        def apply_ridge(term_name: str, lam: float) -> None:
            ts = self.get_term_slice(term_name)
            if ts.name != "intercept":
                penalty[ts.start:ts.end] = lam

        apply_ridge("space_lat_lon", self.config.ridge_space)
        apply_ridge("altitude", self.config.ridge_alt)
        apply_ridge("day_of_year", self.config.ridge_day)
        apply_ridge("altitude_x_latitude", self.config.ridge_interaction)
        apply_ridge("altitude_x_longitude", self.config.ridge_interaction)
        apply_ridge("day_x_altitude", self.config.ridge_interaction)
        apply_ridge("day_x_latitude", self.config.ridge_interaction)
        apply_ridge("day_x_longitude", self.config.ridge_interaction)
        apply_ridge("local_hour_harmonics", self.config.ridge_time_harmonics)
        apply_ridge("utc_hour_harmonics", self.config.ridge_time_harmonics)
        return penalty

    def get_term_slice(self, name: str) -> TermSlice:
        try:
            return self.term_index[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    def transform(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        lat = np.clip(np.asarray(features["lat"], dtype=np.float64), *self.config.lat_range)
        lon = np.clip(np.asarray(features["lon"], dtype=np.float64), *self.config.lon_range)
        altitude_m = np.clip(np.asarray(features["altitude_m"], dtype=np.float64), *self.config.alt_range_m)
        day_of_year = np.clip(np.asarray(features["day_of_year"], dtype=np.float64), *self.config.day_of_year_range)
        utc_hour = np.asarray(features["utc_hour"], dtype=np.float64)
        local_solar_hour = np.asarray(features["local_solar_hour"], dtype=np.float64)

        B_lat_space = bspline_basis(lat, self.knots_lat_space, self.config.degree)
        B_lon_space = bspline_basis(lon, self.knots_lon_space, self.config.degree)
        B_alt = bspline_basis(altitude_m, self.knots_alt, self.config.degree)
        B_day = bspline_basis(day_of_year, self.knots_day, self.config.degree)

        B_alt_i = bspline_basis(altitude_m, self.knots_alt_interaction, self.config.degree)
        B_lat_i = bspline_basis(lat, self.knots_lat_interaction, self.config.degree)
        B_lon_i = bspline_basis(lon, self.knots_lon_interaction, self.config.degree)
        B_day_i = bspline_basis(day_of_year, self.knots_day_interaction, self.config.degree)

        H_local = harmonic_basis(local_solar_hour, period=24.0, n_harmonics=self.config.local_hour_harmonics)
        H_utc = harmonic_basis(utc_hour, period=24.0, n_harmonics=self.config.utc_hour_harmonics)

        n_rows = lat.shape[0]
        X = np.zeros((n_rows, self.n_features), dtype=np.float64)
        X[:, 0] = 1.0

        ts = self.get_term_slice("space_lat_lon")
        X[:, ts.start:ts.end] = tensor_product_rows(B_lat_space, B_lon_space)

        ts = self.get_term_slice("altitude")
        X[:, ts.start:ts.end] = B_alt

        ts = self.get_term_slice("day_of_year")
        X[:, ts.start:ts.end] = B_day

        ts = self.get_term_slice("altitude_x_latitude")
        X[:, ts.start:ts.end] = tensor_product_rows(B_alt_i, B_lat_i)

        ts = self.get_term_slice("altitude_x_longitude")
        X[:, ts.start:ts.end] = tensor_product_rows(B_alt_i, B_lon_i)

        ts = self.get_term_slice("day_x_altitude")
        X[:, ts.start:ts.end] = tensor_product_rows(B_day_i, B_alt_i)

        ts = self.get_term_slice("day_x_latitude")
        X[:, ts.start:ts.end] = tensor_product_rows(B_day_i, B_lat_i)

        ts = self.get_term_slice("day_x_longitude")
        X[:, ts.start:ts.end] = tensor_product_rows(B_day_i, B_lon_i)

        ts = self.get_term_slice("local_hour_harmonics")
        X[:, ts.start:ts.end] = H_local

        ts = self.get_term_slice("utc_hour_harmonics")
        X[:, ts.start:ts.end] = H_utc

        return X

    def metadata(self) -> Dict:
        return {
            "config": self.config.to_dict(),
            "term_slices": [ts.__dict__ for ts in self.term_slices],
            "n_features": self.n_features,
            "knots": {
                "knots_lat_space": self.knots_lat_space.tolist(),
                "knots_lon_space": self.knots_lon_space.tolist(),
                "knots_alt": self.knots_alt.tolist(),
                "knots_day": self.knots_day.tolist(),
                "knots_alt_interaction": self.knots_alt_interaction.tolist(),
                "knots_lat_interaction": self.knots_lat_interaction.tolist(),
                "knots_lon_interaction": self.knots_lon_interaction.tolist(),
                "knots_day_interaction": self.knots_day_interaction.tolist(),
            },
        }

    def metadata_json(self) -> str:
        return json.dumps(self.metadata())
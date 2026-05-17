"""Dataset construction for the KNN weather service.

Two paths, picked at runtime based on what is available:

* ``build_from_era5(...)`` — read raw ERA5 NetCDF files (the same files the
  MLP service trains on), draw a stratified random sample, and return raw
  feature/target arrays. Requires ``xarray`` + ``netcdf4``.

* ``build_synthetic(...)`` — generate a deterministic grid of points from
  the ISA atmosphere model with no wind. The numbers are physically
  reasonable but the spatial/temporal variation is much simpler than ERA5.
  Used as a fallback so the service is always serviceable for demos.

Both paths return numpy arrays that ``KnnWeatherModel`` can consume directly.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np


# Constants for the gravitational potential -> geometric altitude conversion
# (used when reading geopotential from ERA5).
G0_MPS2 = 9.80665

# ISA constants used by the synthetic fallback.
ISA_T0_K = 288.15
ISA_P0_PA = 101_325.0
ISA_LAPSE_RATE_K_PER_M = 0.0065
ISA_R = 287.05
ISA_GAS_EXP = 5.2561  # g / (R * lapse)


@dataclass(frozen=True)
class TrainingDataset:
    features: np.ndarray  # shape (n, 5) — lat, lon, altitude_m, day_of_year, utc_hour
    targets: np.ndarray   # shape (n, 4) — temperature_K, pressure_Pa, wind_u, wind_v
    source: str           # "era5" or "synthetic"


# --- ERA5 path ---------------------------------------------------------------


_DATE_RE = re.compile(r"era5_(\d{4})_(\d{2})_(\d{2})\.nc$")


def _discover_era5_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if _DATE_RE.search(p.name))


def _first_present(ds, names: Sequence[str]) -> str:
    for name in names:
        if name in ds or name in ds.coords:
            return name
    raise KeyError(f"None of the names exist in dataset: {names}")


def build_from_era5(
    era5_root: Path,
    samples_per_file: int,
    max_files: int,
    seed: int = 42,
) -> TrainingDataset:
    """Sample raw ERA5 NetCDF files into KNN training rows.

    Each file holds one calendar day. We pull ``samples_per_file`` random
    (time, level, lat, lon) cells from each file and convert the
    geopotential field to a geometric altitude.
    """
    import xarray as xr

    files = _discover_era5_files(era5_root)
    if not files:
        raise FileNotFoundError(f"No ERA5 NetCDF files found under {era5_root}")
    if max_files > 0:
        files = files[:max_files]

    rng = np.random.default_rng(seed)

    feature_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []

    for path in files:
        date_match = _DATE_RE.search(path.name)
        assert date_match is not None
        day_of_year = datetime(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        ).timetuple().tm_yday

        ds = xr.open_dataset(path, engine="netcdf4", cache=False)
        try:
            t_name = _first_present(ds, ["t", "temperature"])
            u_name = _first_present(ds, ["u", "u_component_of_wind"])
            v_name = _first_present(ds, ["v", "v_component_of_wind"])
            z_name = _first_present(ds, ["z", "geopotential"])
            time_name = _first_present(ds, ["valid_time", "time"])
            level_name = _first_present(ds, ["pressure_level", "level"])
            lat_name = _first_present(ds, ["latitude", "lat"])
            lon_name = _first_present(ds, ["longitude", "lon"])

            levels_hpa = np.asarray(ds[level_name].values, dtype=np.float64)
            lats = np.asarray(ds[lat_name].values, dtype=np.float64)
            lons = np.asarray(ds[lon_name].values, dtype=np.float64)
            times = np.asarray(ds[time_name].values)

            n_time = times.shape[0]
            n_level = levels_hpa.shape[0]
            n_lat = lats.shape[0]
            n_lon = lons.shape[0]

            # Uniform random indices — good enough for KNN, simpler than the
            # stratified scheme in the MLP service.
            count = int(samples_per_file)
            ti = rng.integers(0, n_time, size=count)
            li = rng.integers(0, n_level, size=count)
            la = rng.integers(0, n_lat, size=count)
            lo = rng.integers(0, n_lon, size=count)

            t_cube = ds[t_name].transpose(time_name, level_name, lat_name, lon_name).values
            u_cube = ds[u_name].transpose(time_name, level_name, lat_name, lon_name).values
            v_cube = ds[v_name].transpose(time_name, level_name, lat_name, lon_name).values
            z_cube = ds[z_name].transpose(time_name, level_name, lat_name, lon_name).values

            t_vals = t_cube[ti, li, la, lo].astype(np.float32)
            u_vals = u_cube[ti, li, la, lo].astype(np.float32)
            v_vals = v_cube[ti, li, la, lo].astype(np.float32)
            z_vals = z_cube[ti, li, la, lo].astype(np.float32)
            altitude_m = (z_vals / np.float32(G0_MPS2)).astype(np.float32)

            lat_vals = lats[la].astype(np.float32)
            lon_vals = lons[lo].astype(np.float32)
            lon_vals[lon_vals > 180.0] -= 360.0

            # The hour-of-day is taken from the time index of the cell.
            hour_per_time = np.array(
                [_hour_of_day(times[i]) for i in range(n_time)],
                dtype=np.float32,
            )
            utc_hour = hour_per_time[ti]

            pressure_pa = (levels_hpa[li] * 100.0).astype(np.float32)
            doy = np.full(count, float(day_of_year), dtype=np.float32)

            valid = (
                np.isfinite(t_vals)
                & np.isfinite(u_vals)
                & np.isfinite(v_vals)
                & np.isfinite(altitude_m)
            )

            feature_chunks.append(
                np.stack(
                    [lat_vals[valid], lon_vals[valid], altitude_m[valid], doy[valid], utc_hour[valid]],
                    axis=1,
                )
            )
            target_chunks.append(
                np.stack(
                    [t_vals[valid], pressure_pa[valid], u_vals[valid], v_vals[valid]],
                    axis=1,
                )
            )
        finally:
            ds.close()

    features = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    targets = np.concatenate(target_chunks, axis=0).astype(np.float32, copy=False)
    return TrainingDataset(features=features, targets=targets, source="era5")


def _hour_of_day(time_value) -> int:
    ts = np.datetime64(time_value)
    s = np.datetime_as_string(ts, unit="s")
    return datetime.fromisoformat(s).hour


# --- Synthetic fallback ------------------------------------------------------


def build_synthetic(
    n_lat: int = 12,
    n_lon: int = 12,
    n_alt: int = 8,
    n_doy: int = 6,
    n_hour: int = 4,
    seed: int = 42,
) -> TrainingDataset:
    """Build a dense ISA-based dataset on a regular grid.

    The output is purely deterministic and contains no wind variation, so it
    is a *demo* dataset, not a substitute for real ERA5 training. It exists
    so the KNN service can boot and answer queries when no ERA5 data is
    mounted into the container.
    """
    rng = np.random.default_rng(seed)

    lats = np.linspace(-80.0, 80.0, n_lat, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, n_lon, endpoint=False, dtype=np.float32)
    alts = np.linspace(0.0, 20_000.0, n_alt, dtype=np.float32)
    doys = np.linspace(15.0, 350.0, n_doy, dtype=np.float32)
    hours = np.linspace(0.0, 21.0, n_hour, dtype=np.float32)

    grid = np.stack(np.meshgrid(lats, lons, alts, doys, hours, indexing="ij"), axis=-1)
    features = grid.reshape(-1, 5).astype(np.float32, copy=False)

    altitude_m = features[:, 2]
    temperature_K = ISA_T0_K - ISA_LAPSE_RATE_K_PER_M * altitude_m
    temperature_K = np.maximum(temperature_K, np.float32(216.65))
    pressure_Pa = ISA_P0_PA * np.power(
        np.maximum(temperature_K / ISA_T0_K, np.float32(1e-3)),
        ISA_GAS_EXP,
    )

    # A small deterministic wind that varies with latitude so the KNN has
    # *something* non-trivial to fit. Real ERA5 training overrides this.
    wind_u = 5.0 * np.sin(np.deg2rad(features[:, 0])).astype(np.float32)
    wind_v = 2.0 * np.cos(np.deg2rad(features[:, 0])).astype(np.float32)

    # Light Gaussian jitter on the targets so neighbour distances are
    # informative (a perfectly noiseless grid is degenerate for KNN).
    jitter = rng.normal(0.0, 0.5, size=temperature_K.shape).astype(np.float32)
    temperature_K = temperature_K + jitter

    targets = np.stack(
        [temperature_K.astype(np.float32), pressure_Pa.astype(np.float32), wind_u, wind_v],
        axis=1,
    )
    return TrainingDataset(features=features, targets=targets, source="synthetic")


# --- High-level entry point used by both the training CLI and the service ---


def load_or_build_dataset(
    artifact_dir: Path,
    era5_root: Path | None,
    samples_per_file: int,
    max_files: int,
    rebuild: bool = False,
) -> TrainingDataset:
    """Return a training dataset, preferring ERA5 when available."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if era5_root is not None and era5_root.is_dir() and _discover_era5_files(era5_root):
        print(f"[knn-service] Building training dataset from ERA5 files at {era5_root}")
        return build_from_era5(
            era5_root=era5_root,
            samples_per_file=samples_per_file,
            max_files=max_files,
        )

    print(
        "[knn-service] ERA5 data not found — falling back to synthetic ISA "
        "dataset. Set ERA5_DATA_ROOT to use real training data."
    )
    del rebuild
    return build_synthetic()


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser()

"""ERA5 NetCDF sampling utilities.

The sampler reads one daily ERA5 file at a time, extracts a stratified random
subset of (time, level, lat, lon) cells, and returns float32 features and
targets. Returning float32 (instead of float64) roughly halves the RAM needed
to hold a full month of training samples — the dominant memory cost in this
project.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import xarray as xr

from .config import SamplingConfig

G0 = 9.80665
DATE_RE = re.compile(r"era5_(\d{4})_(\d{2})_(\d{2})\.nc$")


@dataclass
class SampleBatch:
    features: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]


def discover_era5_files(root: str, year: int, months: Sequence[int]) -> List[str]:
    """List ERA5 files matching ``era5_{year}_{month:02d}_DD.nc`` for each month.

    Only the requested year/month prefixes are scanned; we do not look at any
    other months even if they exist in the directory.
    """
    if not os.path.isdir(root):
        return []
    wanted_prefixes = tuple(f"era5_{year}_{int(month):02d}_" for month in months)
    out: List[str] = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(".nc"):
                continue
            if entry.name.startswith(wanted_prefixes):
                out.append(entry.path)
    return sorted(out)


def parse_date_from_path(path: str) -> datetime:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse date from {path}")
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def split_files_by_day(files: Sequence[str], train_end: int, val_end: int) -> Dict[str, List[str]]:
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []
    for path in sorted(files):
        day = parse_date_from_path(path).day
        if day <= train_end:
            train.append(path)
        elif day <= val_end:
            val.append(path)
        else:
            test.append(path)
    return {"train": train, "val": val, "test": test}


def _first_existing(ds: xr.Dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in ds or name in ds.coords:
            return name
    raise KeyError(f"None of the names exist: {names}")


def _extract_time_parts(time_value) -> Tuple[int, int]:
    ts = np.datetime64(time_value)
    s = np.datetime_as_string(ts, unit="s")
    dt = datetime.fromisoformat(s)
    return dt.timetuple().tm_yday, dt.hour


def _make_rng(path: str, seed: int) -> np.random.Generator:
    dt = parse_date_from_path(path)
    file_seed = seed + dt.year * 10000 + dt.month * 100 + dt.day
    return np.random.default_rng(file_seed)


def _build_indices(
    n_time: int,
    n_level: int,
    n_lat: int,
    n_lon: int,
    sample_count: int,
    rng: np.random.Generator,
    stratified: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four (sample_count,) int32 index arrays for (time, level, lat, lon).

    Using int32 instead of int64 halves the memory cost of these helper arrays.
    With sample_count ~= 80,000 the savings are small per file, but they add up
    across many files because we keep these arrays in flight while we sort.
    """
    if not stratified:
        return (
            rng.integers(0, n_time, size=sample_count, dtype=np.int32),
            rng.integers(0, n_level, size=sample_count, dtype=np.int32),
            rng.integers(0, n_lat, size=sample_count, dtype=np.int32),
            rng.integers(0, n_lon, size=sample_count, dtype=np.int32),
        )

    n_slices = n_time * n_level
    base = sample_count // n_slices
    rem = sample_count % n_slices

    flat_order = np.arange(n_slices, dtype=np.int64)
    rng.shuffle(flat_order)

    time_parts = []
    level_parts = []
    lat_parts = []
    lon_parts = []

    for rank, flat_id in enumerate(flat_order):
        count = base + (1 if rank < rem else 0)
        if count <= 0:
            continue
        ti = int(flat_id) // n_level
        li = int(flat_id) % n_level

        time_parts.append(np.full(count, ti, dtype=np.int32))
        level_parts.append(np.full(count, li, dtype=np.int32))
        lat_parts.append(rng.integers(0, n_lat, size=count, dtype=np.int32))
        lon_parts.append(rng.integers(0, n_lon, size=count, dtype=np.int32))

    time_idx = np.concatenate(time_parts)
    level_idx = np.concatenate(level_parts)
    lat_idx = np.concatenate(lat_parts)
    lon_idx = np.concatenate(lon_parts)

    perm = rng.permutation(time_idx.shape[0])
    return time_idx[perm], level_idx[perm], lat_idx[perm], lon_idx[perm]


def sample_from_file(path: str, config: SamplingConfig) -> SampleBatch:
    """Read random samples from a single daily ERA5 file.

    All output arrays are float32. We never load the full 4D cube; we only read
    one (time, level) slice at a time and gather the required (lat, lon) cells.
    """
    rng = _make_rng(path, config.seed)
    ds = xr.open_dataset(path, engine="netcdf4", cache=False)
    try:
        t_name = _first_existing(ds, ["t", "temperature"])
        u_name = _first_existing(ds, ["u", "u_component_of_wind"])
        v_name = _first_existing(ds, ["v", "v_component_of_wind"])
        z_name = _first_existing(ds, ["z", "geopotential"])
        time_name = _first_existing(ds, ["valid_time", "time"])
        level_name = _first_existing(ds, ["pressure_level", "level"])
        lat_name = _first_existing(ds, ["latitude", "lat"])
        lon_name = _first_existing(ds, ["longitude", "lon"])

        times = np.asarray(ds[time_name].values)
        # levels and lats/lons stay float64 here for index correctness; tiny.
        levels_hpa = np.asarray(ds[level_name].values, dtype=np.float64)
        lats = np.asarray(ds[lat_name].values, dtype=np.float64)
        lons = np.asarray(ds[lon_name].values, dtype=np.float64)

        n_time = times.shape[0]
        n_level = levels_hpa.shape[0]
        n_lat = lats.shape[0]
        n_lon = lons.shape[0]

        sample_count = int(config.samples_per_file)
        time_idx, level_idx, lat_idx, lon_idx = _build_indices(
            n_time=n_time,
            n_level=n_level,
            n_lat=n_lat,
            n_lon=n_lon,
            sample_count=sample_count,
            rng=rng,
            stratified=bool(getattr(config, "stratified_time_level", False)),
        )

        # Group by (time, level) so we read each slice exactly once.
        flat_slice = time_idx.astype(np.int64) * n_level + level_idx.astype(np.int64)
        order = np.argsort(flat_slice, kind="mergesort")
        flat_sorted = flat_slice[order]
        group_starts = np.flatnonzero(np.r_[True, flat_sorted[1:] != flat_sorted[:-1]])

        # Output arrays in float32: this is the dominant memory saving.
        lat_out = np.empty(sample_count, dtype=np.float32)
        lon_out = np.empty(sample_count, dtype=np.float32)
        alt_out = np.empty(sample_count, dtype=np.float32)
        doy_out = np.empty(sample_count, dtype=np.float32)
        utc_hour_out = np.empty(sample_count, dtype=np.float32)
        solar_hour_out = np.empty(sample_count, dtype=np.float32)

        T_out = np.empty(sample_count, dtype=np.float32)
        P_out = np.empty(sample_count, dtype=np.float32)
        U_out = np.empty(sample_count, dtype=np.float32)
        V_out = np.empty(sample_count, dtype=np.float32)

        valid_mask = np.ones(sample_count, dtype=bool)
        clip_lo, clip_hi = float(config.altitude_clip_m[0]), float(config.altitude_clip_m[1])

        for g, start in enumerate(group_starts):
            end = group_starts[g + 1] if g + 1 < group_starts.shape[0] else sample_count
            group_order = order[start:end]
            flat_id = int(flat_sorted[start])
            ti = flat_id // n_level
            li = flat_id % n_level

            lat_g = lat_idx[group_order]
            lon_g = lon_idx[group_order]

            # Read a single (time, level) horizontal slice as float32.
            t_slice = np.asarray(ds[t_name].isel({time_name: ti, level_name: li}).values, dtype=np.float32)
            u_slice = np.asarray(ds[u_name].isel({time_name: ti, level_name: li}).values, dtype=np.float32)
            v_slice = np.asarray(ds[v_name].isel({time_name: ti, level_name: li}).values, dtype=np.float32)
            z_slice = np.asarray(ds[z_name].isel({time_name: ti, level_name: li}).values, dtype=np.float32)

            T_vals = t_slice[lat_g, lon_g]
            U_vals = u_slice[lat_g, lon_g]
            V_vals = v_slice[lat_g, lon_g]
            Z_vals = z_slice[lat_g, lon_g]
            del t_slice, u_slice, v_slice, z_slice

            altitude_vals = np.clip(Z_vals * np.float32(1.0 / G0), clip_lo, clip_hi)
            day_of_year, utc_hour = _extract_time_parts(times[ti])
            lon_vals = lons[lon_g].astype(np.float32, copy=True)
            lon_vals[lon_vals > 180.0] -= 360.0
            solar_vals = np.mod(np.float32(utc_hour) + lon_vals * np.float32(1.0 / 15.0), np.float32(24.0))

            group_valid = (
                np.isfinite(T_vals)
                & np.isfinite(U_vals)
                & np.isfinite(V_vals)
                & np.isfinite(Z_vals)
                & np.isfinite(altitude_vals)
            )

            valid_mask[group_order] = group_valid
            lat_out[group_order] = lats[lat_g].astype(np.float32, copy=False)
            lon_out[group_order] = lon_vals
            alt_out[group_order] = altitude_vals
            doy_out[group_order] = np.float32(day_of_year)
            utc_hour_out[group_order] = np.float32(utc_hour)
            solar_hour_out[group_order] = solar_vals

            T_out[group_order] = T_vals
            P_out[group_order] = np.float32(levels_hpa[li] * 100.0)
            U_out[group_order] = U_vals
            V_out[group_order] = V_vals
    finally:
        ds.close()

    valid_idx = np.flatnonzero(valid_mask)
    return SampleBatch(
        features={
            "lat": lat_out[valid_idx],
            "lon": lon_out[valid_idx],
            "altitude_m": alt_out[valid_idx],
            "day_of_year": doy_out[valid_idx],
            "utc_hour": utc_hour_out[valid_idx],
            "local_solar_hour": solar_hour_out[valid_idx],
        },
        targets={
            "T": T_out[valid_idx],
            "P": P_out[valid_idx],
            "U": U_out[valid_idx],
            "V": V_out[valid_idx],
        },
    )

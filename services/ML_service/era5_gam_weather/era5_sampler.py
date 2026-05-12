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


def _lat_sampling_probs(lats_deg: np.ndarray) -> np.ndarray:
    """Return a length-n_lat probability vector ∝ cos(lat) (area-weighted).

    ERA5 grids are uniform in lat/lon degrees, so polar cells cover much
    smaller physical area than equatorial cells. Without this weighting the
    sampler over-represents tiny polar cells and biases wind statistics.
    """
    lats = np.asarray(lats_deg, dtype=np.float64)
    weights = np.clip(np.cos(np.deg2rad(lats)), 1e-6, None)
    return (weights / weights.sum()).astype(np.float64)


def _draw_lat_indices(rng: np.random.Generator, n_lat: int, count: int, lat_probs: np.ndarray | None) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int32)
    if lat_probs is None:
        return rng.integers(0, n_lat, size=count, dtype=np.int32)
    return rng.choice(n_lat, size=count, replace=True, p=lat_probs).astype(np.int32, copy=False)


def _level_budget(
    sample_count: int,
    n_time: int,
    n_level: int,
    level_weights: np.ndarray | None,
) -> np.ndarray:
    """Compute integer sample budget per (time, level) slice.

    Returns a length-(n_time*n_level) integer array summing to sample_count.
    When ``level_weights`` is None, slices are uniformly weighted (preserving
    the previous behavior). Otherwise the level dimension is weighted by
    ``level_weights`` and time is uniform.
    """
    n_slices = n_time * n_level
    if level_weights is None:
        base = sample_count // n_slices
        rem = sample_count % n_slices
        budget = np.full(n_slices, base, dtype=np.int64)
        # Distribute the remainder across slices uniformly (deterministic).
        budget[:rem] += 1
        return budget

    if level_weights.shape[0] != n_level:
        raise ValueError(
            f"level_weights length {level_weights.shape[0]} does not match n_level={n_level}"
        )
    weights = np.clip(level_weights, 0.0, None).astype(np.float64)
    if not np.any(weights > 0):
        raise ValueError("level_weights must contain at least one positive value")
    weights = weights / weights.sum()
    # Per-(time, level) target = sample_count * (1/n_time) * level_weight.
    per_slice = (sample_count / n_time) * np.tile(weights, n_time)
    floor = np.floor(per_slice).astype(np.int64)
    rem = int(sample_count - floor.sum())
    if rem > 0:
        # Hand out the remaining samples to slices with the largest fractional part.
        frac = per_slice - floor
        order = np.argsort(-frac, kind="stable")
        floor[order[:rem]] += 1
    return floor


def _build_indices(
    n_time: int,
    n_level: int,
    n_lat: int,
    n_lon: int,
    sample_count: int,
    rng: np.random.Generator,
    stratified: bool,
    lat_probs: np.ndarray | None = None,
    level_weights: np.ndarray | None = None,
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
            _draw_lat_indices(rng, n_lat, sample_count, lat_probs),
            rng.integers(0, n_lon, size=sample_count, dtype=np.int32),
        )

    n_slices = n_time * n_level
    flat_order = np.arange(n_slices, dtype=np.int64)
    rng.shuffle(flat_order)
    budget_by_slice = _level_budget(sample_count, n_time, n_level, level_weights)

    time_parts = []
    level_parts = []
    lat_parts = []
    lon_parts = []

    for flat_id in flat_order:
        count = int(budget_by_slice[flat_id])
        if count <= 0:
            continue
        ti = int(flat_id) // n_level
        li = int(flat_id) % n_level

        time_parts.append(np.full(count, ti, dtype=np.int32))
        level_parts.append(np.full(count, li, dtype=np.int32))
        lat_parts.append(_draw_lat_indices(rng, n_lat, count, lat_probs))
        lon_parts.append(rng.integers(0, n_lon, size=count, dtype=np.int32))

    time_idx = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=np.int32)
    level_idx = np.concatenate(level_parts) if level_parts else np.empty(0, dtype=np.int32)
    lat_idx = np.concatenate(lat_parts) if lat_parts else np.empty(0, dtype=np.int32)
    lon_idx = np.concatenate(lon_parts) if lon_parts else np.empty(0, dtype=np.int32)

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
        lat_probs = (
            _lat_sampling_probs(lats)
            if bool(getattr(config, "area_weighted_lat", False))
            else None
        )
        level_weights_cfg = getattr(config, "level_weights", None)
        level_weights_arr = (
            np.asarray(level_weights_cfg, dtype=np.float64)
            if level_weights_cfg is not None
            else None
        )
        time_idx, level_idx, lat_idx, lon_idx = _build_indices(
            n_time=n_time,
            n_level=n_level,
            n_lat=n_lat,
            n_lon=n_lon,
            sample_count=sample_count,
            rng=rng,
            stratified=bool(getattr(config, "stratified_time_level", False)),
            lat_probs=lat_probs,
            level_weights=level_weights_arr,
        )

        # Load all 4 variables fully into memory upfront: 4 sequential reads
        # instead of O(n_time * n_levels * n_vars) per-slice isel() disk reads.
        # Transposing to (time, level, lat, lon) is safe regardless of the
        # original dimension order in the file.
        T_full = np.asarray(ds[t_name].transpose(time_name, level_name, lat_name, lon_name).values, dtype=np.float32)
        U_full = np.asarray(ds[u_name].transpose(time_name, level_name, lat_name, lon_name).values, dtype=np.float32)
        V_full = np.asarray(ds[v_name].transpose(time_name, level_name, lat_name, lon_name).values, dtype=np.float32)
        Z_full = np.asarray(ds[z_name].transpose(time_name, level_name, lat_name, lon_name).values, dtype=np.float32)

        # Group by (time, level) so each combination is processed once.
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

            T_vals = T_full[ti, li, lat_g, lon_g]
            U_vals = U_full[ti, li, lat_g, lon_g]
            V_vals = V_full[ti, li, lat_g, lon_g]
            Z_vals = Z_full[ti, li, lat_g, lon_g]

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

        del T_full, U_full, V_full, Z_full
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

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Dict, Sequence, Tuple

import numpy as np
import xarray as xr

G0 = 9.80665
_CACHE_MAX_FILES = 3
_CACHE_LOCK = RLock()
_FILE_CACHE: "OrderedDict[str, _DailyERA5File]" = OrderedDict()


def _first_existing(ds: xr.Dataset, names: Sequence[str]) -> str:
    for name in names:
        if name in ds or name in ds.coords:
            return name
    raise KeyError(f"None of the names exist: {names}")


def _date_from_year_and_day(year: int, day_of_year: float) -> datetime:
    day_int = int(day_of_year)
    if day_int < 1 or day_int > 366:
        raise ValueError(f"Invalid day_of_year: {day_of_year}")
    return datetime(year, 1, 1) + timedelta(days=day_int - 1)


def _target_datetime64(year: int, day_of_year: float, utc_hour: float) -> np.datetime64:
    base = _date_from_year_and_day(year, day_of_year)
    dt = base + timedelta(hours=float(utc_hour))
    return np.datetime64(dt, "s")


def _daily_file_path(data_root: str, year: int, day_of_year: float) -> Path:
    dt = _date_from_year_and_day(year, day_of_year)
    return Path(data_root) / f"era5_{dt.year}_{dt.month:02d}_{dt.day:02d}.nc"


def _normalize_lon_for_grid(lon: float, lon_values: np.ndarray) -> float:
    lon = float(lon)
    if np.nanmax(lon_values) > 180.0 and lon < 0.0:
        return lon + 360.0
    return lon


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _bracket_indices(coords: np.ndarray, x: float) -> Tuple[int, int, float]:
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n == 1:
        return 0, 0, 0.0

    if coords[0] <= coords[-1]:
        x = _clip(x, coords[0], coords[-1])
        hi = int(np.searchsorted(coords, x, side="right"))
        if hi <= 0:
            return 0, 0, 0.0
        if hi >= n:
            return n - 1, n - 1, 0.0
        lo = hi - 1
        den = coords[hi] - coords[lo]
        frac = 0.0 if den == 0.0 else float((x - coords[lo]) / den)
        return lo, hi, frac

    rev = coords[::-1]
    x = _clip(x, rev[0], rev[-1])
    hi_r = int(np.searchsorted(rev, x, side="right"))
    if hi_r <= 0:
        idx = n - 1
        return idx, idx, 0.0
    if hi_r >= n:
        return 0, 0, 0.0
    lo_r = hi_r - 1
    lo = n - 1 - lo_r
    hi = n - 1 - hi_r
    coord_lo = coords[lo]
    coord_hi = coords[hi]
    den = coord_hi - coord_lo
    frac = 0.0 if den == 0.0 else float((x - coord_lo) / den)
    return lo, hi, frac


class _DailyERA5File:
    def __init__(self, path: Path):
        self.path = path
        self.ds = xr.open_dataset(path, engine="netcdf4", cache=False)

        self.t_name = _first_existing(self.ds, ["t", "temperature"])
        self.u_name = _first_existing(self.ds, ["u", "u_component_of_wind"])
        self.v_name = _first_existing(self.ds, ["v", "v_component_of_wind"])
        self.z_name = _first_existing(self.ds, ["z", "geopotential"])
        self.time_name = _first_existing(self.ds, ["valid_time", "time"])
        self.level_name = _first_existing(self.ds, ["pressure_level", "level"])
        self.lat_name = _first_existing(self.ds, ["latitude", "lat"])
        self.lon_name = _first_existing(self.ds, ["longitude", "lon"])

        self.times = np.asarray(self.ds[self.time_name].values).astype("datetime64[s]")
        self.time_seconds = self.times.astype("int64")
        self.levels_hpa = np.asarray(self.ds[self.level_name].values, dtype=np.float64)
        self.levels_pa = self.levels_hpa * 100.0
        self.lats = np.asarray(self.ds[self.lat_name].values, dtype=np.float64)
        self.lons = np.asarray(self.ds[self.lon_name].values, dtype=np.float64)

    def close(self) -> None:
        self.ds.close()

    def _subcube(self, var_name: str, time_idx: int, lat_i0: int, lat_i1: int, lon_i0: int, lon_i1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lat_lo = min(lat_i0, lat_i1)
        lat_hi = max(lat_i0, lat_i1)
        lon_lo = min(lon_i0, lon_i1)
        lon_hi = max(lon_i0, lon_i1)

        arr = np.asarray(
            self.ds[var_name]
            .isel(
                {
                    self.time_name: int(time_idx),
                    self.lat_name: slice(lat_lo, lat_hi + 1),
                    self.lon_name: slice(lon_lo, lon_hi + 1),
                }
            )
            .values,
            dtype=np.float64,
        )
        lat_vals = self.lats[lat_lo : lat_hi + 1].copy()
        lon_vals = self.lons[lon_lo : lon_hi + 1].copy()

        if lat_vals.shape[0] == 2 and lat_vals[0] > lat_vals[1]:
            lat_vals = lat_vals[::-1]
            arr = arr[:, ::-1, :]
        if lon_vals.shape[0] == 2 and lon_vals[0] > lon_vals[1]:
            lon_vals = lon_vals[::-1]
            arr = arr[:, :, ::-1]
        return arr, lat_vals, lon_vals

    def _bilinear_profile(self, var_name: str, time_idx: int, lat: float, lon: float) -> np.ndarray:
        lat_i0, lat_i1, _ = _bracket_indices(self.lats, lat)
        lon_i0, lon_i1, _ = _bracket_indices(self.lons, lon)

        arr, lat_vals, lon_vals = self._subcube(var_name, time_idx, lat_i0, lat_i1, lon_i0, lon_i1)
        n_levels = arr.shape[0]

        if lat_vals.shape[0] == 1 and lon_vals.shape[0] == 1:
            return arr[:, 0, 0]
        if lat_vals.shape[0] == 1:
            x = _clip(lon, lon_vals[0], lon_vals[-1])
            fx = 0.0 if lon_vals.shape[0] == 1 or lon_vals[1] == lon_vals[0] else (x - lon_vals[0]) / (lon_vals[1] - lon_vals[0])
            return (1.0 - fx) * arr[:, 0, 0] + fx * arr[:, 0, 1]
        if lon_vals.shape[0] == 1:
            y = _clip(lat, lat_vals[0], lat_vals[-1])
            fy = 0.0 if lat_vals.shape[0] == 1 or lat_vals[1] == lat_vals[0] else (y - lat_vals[0]) / (lat_vals[1] - lat_vals[0])
            return (1.0 - fy) * arr[:, 0, 0] + fy * arr[:, 1, 0]

        x = _clip(lon, lon_vals[0], lon_vals[-1])
        y = _clip(lat, lat_vals[0], lat_vals[-1])
        fx = 0.0 if lon_vals[1] == lon_vals[0] else (x - lon_vals[0]) / (lon_vals[1] - lon_vals[0])
        fy = 0.0 if lat_vals[1] == lat_vals[0] else (y - lat_vals[0]) / (lat_vals[1] - lat_vals[0])

        v00 = arr[:, 0, 0]
        v01 = arr[:, 0, -1]
        v10 = arr[:, -1, 0]
        v11 = arr[:, -1, -1]

        return (
            (1.0 - fy) * (1.0 - fx) * v00
            + (1.0 - fy) * fx * v01
            + fy * (1.0 - fx) * v10
            + fy * fx * v11
        ).reshape(n_levels)

    def _profile_at_time(self, time_idx: int, lat: float, lon: float) -> Dict[str, np.ndarray]:
        return {
            "T": self._bilinear_profile(self.t_name, time_idx, lat, lon),
            "U": self._bilinear_profile(self.u_name, time_idx, lat, lon),
            "V": self._bilinear_profile(self.v_name, time_idx, lat, lon),
            "Z": self._bilinear_profile(self.z_name, time_idx, lat, lon),
        }

    def _time_interp_profiles(self, target_dt: np.datetime64, lat: float, lon: float) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        target_sec = target_dt.astype("datetime64[s]").astype("int64")
        t0, t1, tfrac = _bracket_indices(self.time_seconds, float(target_sec))
        p0 = self._profile_at_time(t0, lat, lon)

        if t1 == t0:
            meta = {
                "time_interp_fraction": 0.0,
                "time_index_0": int(t0),
                "time_index_1": int(t1),
                "time_0_iso": np.datetime_as_string(self.times[t0], unit="s"),
                "time_1_iso": np.datetime_as_string(self.times[t1], unit="s"),
            }
            return p0, meta

        p1 = self._profile_at_time(t1, lat, lon)
        blended = {name: (1.0 - tfrac) * p0[name] + tfrac * p1[name] for name in p0}
        meta = {
            "time_interp_fraction": float(tfrac),
            "time_index_0": int(t0),
            "time_index_1": int(t1),
            "time_0_iso": np.datetime_as_string(self.times[t0], unit="s"),
            "time_1_iso": np.datetime_as_string(self.times[t1], unit="s"),
        }
        return blended, meta

    def interpolate(self, target_dt: np.datetime64, lat: float, lon: float, altitude_m: float) -> Dict:
        lon_query = _normalize_lon_for_grid(lon, self.lons)
        profile, time_meta = self._time_interp_profiles(target_dt=target_dt, lat=float(lat), lon=lon_query)

        altitude_profile_m = profile["Z"] / G0
        valid = (
            np.isfinite(altitude_profile_m)
            & np.isfinite(profile["T"])
            & np.isfinite(profile["U"])
            & np.isfinite(profile["V"])
            & np.isfinite(self.levels_pa)
        )
        if not np.any(valid):
            raise ValueError("No valid ERA5 values found for the requested point")

        alt = altitude_profile_m[valid]
        temp = profile["T"][valid]
        wind_u = profile["U"][valid]
        wind_v = profile["V"][valid]
        pressure_pa = self.levels_pa[valid]

        order = np.argsort(alt)
        alt = alt[order]
        temp = temp[order]
        wind_u = wind_u[order]
        wind_v = wind_v[order]
        pressure_pa = pressure_pa[order]

        unique_alt, unique_idx = np.unique(alt, return_index=True)
        alt = unique_alt
        temp = temp[unique_idx]
        wind_u = wind_u[unique_idx]
        wind_v = wind_v[unique_idx]
        pressure_pa = pressure_pa[unique_idx]

        altitude_query = _clip(float(altitude_m), float(alt[0]), float(alt[-1]))
        temperature_k = float(np.interp(altitude_query, alt, temp))
        wind_u_val = float(np.interp(altitude_query, alt, wind_u))
        wind_v_val = float(np.interp(altitude_query, alt, wind_v))
        pressure_val = float(np.exp(np.interp(altitude_query, alt, np.log(np.clip(pressure_pa, 1.0, None)))))

        nearest_idx = int(np.argmin(np.abs(alt - altitude_query)))
        matched_lon = float(lon_query)
        if matched_lon > 180.0:
            matched_lon -= 360.0

        return {
            "real": {
                "temperature_k": temperature_k,
                "pressure_pa": pressure_val,
                "wind_u": wind_u_val,
                "wind_v": wind_v_val,
            },
            "meta": {
                "era5_file": str(self.path),
                "lookup_method": "bilinear_xy + linear_time + linear_vertical + log_pressure_vertical",
                "requested_time_iso": np.datetime_as_string(target_dt.astype("datetime64[s]"), unit="s"),
                "requested_lat": float(lat),
                "requested_lon": float(lon),
                "requested_altitude_m": float(altitude_m),
                "clipped_altitude_m": float(altitude_query),
                "matched_lat": float(lat),
                "matched_lon": matched_lon,
                "nearest_profile_altitude_m": float(alt[nearest_idx]),
                "nearest_profile_pressure_pa": float(pressure_pa[nearest_idx]),
                "nearest_profile_altitude_abs_error_m": float(abs(alt[nearest_idx] - altitude_query)),
                **time_meta,
            },
        }


def _get_daily_file(path: Path) -> _DailyERA5File:
    key = str(path)
    with _CACHE_LOCK:
        cached = _FILE_CACHE.get(key)
        if cached is not None:
            _FILE_CACHE.move_to_end(key)
            return cached

        daily = _DailyERA5File(path)
        _FILE_CACHE[key] = daily
        _FILE_CACHE.move_to_end(key)

        while len(_FILE_CACHE) > _CACHE_MAX_FILES:
            _, evicted = _FILE_CACHE.popitem(last=False)
            evicted.close()
        return daily


def lookup_real_era5_point(
    data_root: str,
    year: int,
    day_of_year: float,
    utc_hour: float,
    lat: float,
    lon: float,
    altitude_m: float,
) -> Dict:
    path = _daily_file_path(data_root, year, day_of_year)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {path}")

    daily = _get_daily_file(path)
    target_dt = _target_datetime64(year, day_of_year, utc_hour)
    return daily.interpolate(target_dt=target_dt, lat=lat, lon=lon, altitude_m=altitude_m)
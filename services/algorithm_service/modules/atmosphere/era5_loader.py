import os
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr

G0 = 9.80665

ERA5_DIR = os.environ.get(
    "ERA5_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data/era5")),
)

MAX_CACHED_HOURS = int(os.environ.get("ERA5_MAX_CACHED_HOURS", "2"))

_CACHE_LOCK = threading.RLock()
_GLOBAL_NETCDF_LOAD_LOCK = threading.Lock()
_HOUR_CACHE: "OrderedDict[Tuple[str, np.datetime64], Dict[str, np.ndarray]]" = OrderedDict()


def _normalize_time(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _normalize_longitude(lon: float) -> float:
    return lon if lon >= 0.0 else lon + 360.0


def _get_file_path(date: datetime) -> str:
    return os.path.join(
        ERA5_DIR,
        f"era5_{date.year}_{date.month:02d}_{date.day:02d}.nc"
    )


def _pick_name(ds, *candidates: str) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    raise KeyError(
        f"None of {candidates} found. "
        f"coords={list(ds.coords)}, dims={list(ds.dims)}, vars={list(ds.variables)}"
    )


def _nearest_index(arr: np.ndarray, value) -> int:
    return int(np.abs(arr - value).argmin())


def _evict_if_needed() -> None:
    while len(_HOUR_CACHE) > MAX_CACHED_HOURS:
        old_key, _ = _HOUR_CACHE.popitem(last=False)
        print(f"Evicted ERA5 hour from cache: {old_key}")


def _build_hour_slice(path: str, hour_dt: datetime) -> Dict[str, np.ndarray]:
    target_hour = np.datetime64(_normalize_time(hour_dt))
    print(f"Loading ERA5 hour into RAM: {path} @ {target_hour}")

    with xr.open_dataset(path, engine="netcdf4") as ds:
        time_name = _pick_name(ds, "valid_time", "time")
        level_name = _pick_name(ds, "pressure_level", "level")
        lat_name = _pick_name(ds, "latitude", "lat")
        lon_name = _pick_name(ds, "longitude", "lon")

        z_name = _pick_name(ds, "z", "geopotential")
        t_name = _pick_name(ds, "t", "temperature")
        u_name = _pick_name(ds, "u", "u_component_of_wind")
        v_name = _pick_name(ds, "v", "v_component_of_wind")

        # בוחרים רק שעה אחת, לא יום שלם
        hour_slice = ds.sel(
            {time_name: target_hour},
            method="nearest",
        )

        levels_hpa = np.asarray(hour_slice[level_name].values, dtype=np.float32)
        latitudes = np.asarray(hour_slice[lat_name].values, dtype=np.float32)
        longitudes = np.asarray(hour_slice[lon_name].values, dtype=np.float32)

        # shape: [level, lat, lon]
        z_m = np.asarray(
            hour_slice[z_name].transpose(level_name, lat_name, lon_name).values,
            dtype=np.float32,
        ) / G0

        t_k = np.asarray(
            hour_slice[t_name].transpose(level_name, lat_name, lon_name).values,
            dtype=np.float32,
        )

        u_east = np.asarray(
            hour_slice[u_name].transpose(level_name, lat_name, lon_name).values,
            dtype=np.float32,
        )

        v_north = np.asarray(
            hour_slice[v_name].transpose(level_name, lat_name, lon_name).values,
            dtype=np.float32,
        )

    return {
        "hour": target_hour,
        "levels_hpa": levels_hpa,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "z_m": z_m,
        "temperature_K": t_k,
        "u_east_mps": u_east,
        "v_north_mps": v_north,
    }


def _get_hour_key(time: datetime) -> Tuple[str, np.datetime64]:
    path = _get_file_path(time)
    hour = np.datetime64(_normalize_time(time))
    return path, hour


def preload_era5_hour(time: datetime) -> None:
    _ = _get_hour_data(time)


def _get_hour_data(time: datetime) -> Dict[str, np.ndarray]:
    path, hour = _get_hour_key(time)

    if not os.path.exists(path):
        raise FileNotFoundError(f"ERA5 file not found: {path}")

    key = (path, hour)

    with _CACHE_LOCK:
        cached = _HOUR_CACHE.get(key)
        if cached is not None:
            _HOUR_CACHE.move_to_end(key)
            return cached

    # רק טעינת netCDF עצמה נעשית בצורה סדרתית
    with _GLOBAL_NETCDF_LOAD_LOCK:
        with _CACHE_LOCK:
            cached = _HOUR_CACHE.get(key)
            if cached is not None:
                _HOUR_CACHE.move_to_end(key)
                return cached

        hour_data = _build_hour_slice(path, time)

        with _CACHE_LOCK:
            _HOUR_CACHE[key] = hour_data
            _HOUR_CACHE.move_to_end(key)
            _evict_if_needed()

        return hour_data


def load_era5_column(lat: float, lon: float, time: datetime) -> dict:
    hour_data = _get_hour_data(time)

    target_lat = float(lat)
    target_lon = _normalize_longitude(float(lon))

    lat_idx = _nearest_index(hour_data["latitudes"], target_lat)
    lon_idx = _nearest_index(hour_data["longitudes"], target_lon)

    z_m = hour_data["z_m"][:, lat_idx, lon_idx].astype(np.float64)
    t_k = hour_data["temperature_K"][:, lat_idx, lon_idx].astype(np.float64)
    u_east = hour_data["u_east_mps"][:, lat_idx, lon_idx].astype(np.float64)
    v_north = hour_data["v_north_mps"][:, lat_idx, lon_idx].astype(np.float64)
    p_pa = hour_data["levels_hpa"].astype(np.float64) * 100.0

    order = np.argsort(z_m)

    return {
        "z_m": z_m[order],
        "temperature_K": t_k[order],
        "pressure_Pa": p_pa[order],
        "u_east_mps": u_east[order],
        "v_north_mps": v_north[order],
    }


def sample_column_at_altitude(column: dict, altitude_m: float) -> dict:
    z = column["z_m"]
    h = float(np.clip(float(altitude_m), float(z[0]), float(z[-1])))

    return {
        "temperature_K": float(np.interp(h, z, column["temperature_K"])),
        "pressure_Pa": float(np.interp(h, z, column["pressure_Pa"])),
        "u_east_mps": float(np.interp(h, z, column["u_east_mps"])),
        "v_north_mps": float(np.interp(h, z, column["v_north_mps"])),
    }
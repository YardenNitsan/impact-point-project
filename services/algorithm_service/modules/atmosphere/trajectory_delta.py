from datetime import datetime
from typing import Dict, Any

import numpy as np

from modules.atmosphere.era5_loader import load_era5_column

ALT_GRID_M = np.arange(0.0, 20000.0 + 500.0, 500.0, dtype=np.float64)


def _get_first(d: Dict[str, Any], *keys: str):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of keys {keys} found in dict")


def _build_time_features(sim_datetime: str) -> Dict[str, float]:
    dt = datetime.fromisoformat(sim_datetime)

    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour

    day_angle = 2.0 * np.pi * day_of_year / 365.0
    hour_angle = 2.0 * np.pi * hour / 24.0

    return {
        "day_of_year": int(day_of_year),
        "hour_of_day": int(hour),
        "day_sin": float(np.sin(day_angle)),
        "day_cos": float(np.cos(day_angle)),
        "hour_sin": float(np.sin(hour_angle)),
        "hour_cos": float(np.cos(hour_angle)),
    }


def _build_environment_profile(sample: Dict[str, Any]) -> Dict[str, Any]:
    latitude = float(_get_first(sample, "latitude", "lat"))
    longitude = float(_get_first(sample, "longitude", "lon"))
    altitude = float(_get_first(sample, "altitude", "alt"))
    sim_datetime = str(sample["sim_datetime"])

    column = load_era5_column(
        latitude,
        longitude,
        datetime.fromisoformat(sim_datetime),
    )

    temperature_profile = np.interp(
        ALT_GRID_M, column["z_m"], column["temperature_K"]
    )
    pressure_profile = np.interp(
        ALT_GRID_M, column["z_m"], column["pressure_Pa"]
    )
    u_profile = np.interp(
        ALT_GRID_M, column["z_m"], column["u_east_mps"]
    )
    v_profile = np.interp(
        ALT_GRID_M, column["z_m"], column["v_north_mps"]
    )

    launch_temperature = float(np.interp(
        altitude, column["z_m"], column["temperature_K"]
    ))
    launch_pressure = float(np.interp(
        altitude, column["z_m"], column["pressure_Pa"]
    ))
    launch_u = float(np.interp(
        altitude, column["z_m"], column["u_east_mps"]
    ))
    launch_v = float(np.interp(
        altitude, column["z_m"], column["v_north_mps"]
    ))

    return {
        "launch": {
            "temperature_K": launch_temperature,
            "pressure_Pa": launch_pressure,
            "u_east_mps": launch_u,
            "v_north_mps": launch_v,
        },
        "profile": {
            "altitudes_m": ALT_GRID_M.tolist(),
            "temperature_K": temperature_profile.tolist(),
            "pressure_Pa": pressure_profile.tolist(),
            "u_east_mps": u_profile.tolist(),
            "v_north_mps": v_profile.tolist(),
        },
    }


def build_dataset_row(sample: Dict[str, Any], simulation_result: Dict[str, Any]) -> Dict[str, Any]:
    latitude = float(_get_first(sample, "latitude", "lat"))
    longitude = float(_get_first(sample, "longitude", "lon"))
    altitude = float(_get_first(sample, "altitude", "alt"))
    speed = float(_get_first(sample, "speed", "initialSpeed"))
    mass = float(sample["mass"])
    sim_datetime = str(sample["sim_datetime"])

    time_features = _build_time_features(sim_datetime)
    environment = _build_environment_profile(sample)

    impact = simulation_result["impact"]

    row = {
        "initial_conditions": {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "speed": speed,
            "mass": mass,
            "sim_datetime": sim_datetime,
            **time_features,
        },
        "environment": environment,
        "target": {
            "impact_lat": float(impact["lat"]),
            "impact_lon": float(impact["lon"]),
            "impact_alt": float(impact["alt"]),
            "impact_vx": float(impact["vx"]),
            "impact_vz": float(impact["vz"]),
            "impact_theta": float(impact["theta"]),
            "flight_time_s": float(simulation_result["physical_time"]),
            "raw_points_count": int(simulation_result["raw_points_count"]),
        },
    }

    if "sin_az" in sample:
        row["initial_conditions"]["sin_az"] = float(sample["sin_az"])
    if "cos_az" in sample:
        row["initial_conditions"]["cos_az"] = float(sample["cos_az"])
    if "sin_el" in sample:
        row["initial_conditions"]["sin_el"] = float(sample["sin_el"])
    if "cos_el" in sample:
        row["initial_conditions"]["cos_el"] = float(sample["cos_el"])

    if "azimuth" in sample:
        row["initial_conditions"]["azimuth_deg"] = float(sample["azimuth"])
    if "elevation" in sample:
        row["initial_conditions"]["elevation_deg"] = float(sample["elevation"])

    return row
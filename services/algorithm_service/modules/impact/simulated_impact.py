from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Dict, List, NotRequired, TypedDict

from modules.aerodynamics.aero_tables import default_demo_table
from modules.aerodynamics.aerodynamics import AeroRef
from modules.atmosphere.weather_client import (
    HTTPWeatherProviderClient,
    StaticWeatherProviderClient,
)
from modules.atmosphere.weather_runtime import (
    TrajectoryWeatherRuntime,
    WeatherSample,
    enu_displacement_to_latlon,
)
from modules.solver.run_simulation import (
    run_simulation_impact_only,
    run_simulation_sampled,
)
from modules.state.state import State3DOF


class SimulationInput(TypedDict):
    lat: float
    lon: float
    alt: float
    azimuth: float
    elevation: float
    mass: float
    initialSpeed: float
    sim_datetime: NotRequired[str | datetime]

    weather_source: NotRequired[str]

    T0_K: NotRequired[float]
    P0_Pa: NotRequired[float]
    wind_x: NotRequired[float]
    wind_z: NotRequired[float]


class TrajectoryPoint(TypedDict):
    lat: float
    lon: float
    alt: float
    vx: float
    vz: float
    theta: float
    temperature_K: float
    pressure_Pa: float
    wind_u_east_mps: float
    wind_v_north_mps: float
    wind_along_track_mps: float
    wind_vertical_mps: float


class SimulationOutput(TypedDict):
    impact: TrajectoryPoint
    physical_time: float
    raw_points_count: int
    environment: Dict[str, float | str | int]
    trajectory: NotRequired[List[TrajectoryPoint]]
    points_count: NotRequired[int]


DEFAULT_TIME_STEP_S: float = 0.001
DEFAULT_MAX_SIM_TIME_S: float = 800.0
DEFAULT_REFERENCE_AREA_M2: float = 0.002
DEFAULT_REFERENCE_LENGTH_M: float = 0.30
DEFAULT_MOMENT_OF_INERTIA_KGM2: float = 2.0
DEFAULT_GRAVITY_MPS2: float = 9.81
DEFAULT_CENTER_OF_GRAVITY_OFFSET: float = 0.02
GROUND_ALTITUDE_M: float = 0.0

AERO_TABLE = default_demo_table()
AERO_REF = AeroRef(
    reference_area=DEFAULT_REFERENCE_AREA_M2,
    reference_length=DEFAULT_REFERENCE_LENGTH_M,
)

DEFAULT_WEATHER_SERVICE_URL = os.environ.get("WEATHER_SERVICE_URL", "")
DEFAULT_WEATHER_TIMEOUT_S = float(os.environ.get("WEATHER_PROVIDER_TIMEOUT_S", "3.0"))
WEATHER_SERVICE_PATH = os.environ.get("WEATHER_SERVICE_PATH", "/weather")


def join_base_url_and_path(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    if not base:
        raise ValueError(f"Missing base URL for weather provider path {suffix}")
    return f"{base}{suffix}"


def normalize_launch_angles(
    azimuth_rad: float,
    elevation_rad: float,
) -> tuple[float, float]:
    two_pi = 2.0 * math.pi
    half_pi = 0.5 * math.pi

    normalized_elev = (elevation_rad + math.pi) % two_pi - math.pi
    corrected_azimuth = azimuth_rad

    if normalized_elev > half_pi:
        normalized_elev = math.pi - normalized_elev
        corrected_azimuth += math.pi
    elif normalized_elev < -half_pi:
        normalized_elev = -math.pi - normalized_elev
        corrected_azimuth += math.pi

    corrected_azimuth = corrected_azimuth % two_pi
    return corrected_azimuth, normalized_elev


def build_point_environment(sample: WeatherSample) -> Dict[str, float]:
    return {
        "temperature_K": float(sample.temperature_K),
        "pressure_Pa": float(sample.pressure_Pa),
        "wind_u_east_mps": float(sample.wind_east_mps),
        "wind_v_north_mps": float(sample.wind_north_mps),
        "wind_along_track_mps": float(sample.wind_x_mps),
        "wind_vertical_mps": float(sample.wind_z_mps),
    }


def _has_manual_weather_override(initial_conditions: SimulationInput) -> bool:
    return all(
        initial_conditions.get(key) is not None
        for key in ("T0_K", "P0_Pa", "wind_x", "wind_z")
    )


def _select_weather_provider(initial_conditions: SimulationInput):
    if _has_manual_weather_override(initial_conditions):
        return StaticWeatherProviderClient(
            temperature_K=float(initial_conditions["T0_K"]),
            pressure_Pa=float(initial_conditions["P0_Pa"]),
            wind_x_mps=float(initial_conditions["wind_x"]),
            wind_z_mps=float(initial_conditions["wind_z"]),
        ), "manual"

    requested_source = str(initial_conditions.get("weather_source", "machine")).lower()

    if requested_source not in {"api", "machine"}:
        raise ValueError("weather_source must be 'api' or 'machine'")

    return HTTPWeatherProviderClient(
        name=requested_source,
        url=join_base_url_and_path(DEFAULT_WEATHER_SERVICE_URL, WEATHER_SERVICE_PATH),
        timeout_s=DEFAULT_WEATHER_TIMEOUT_S,
        requested_source=requested_source,
    ), requested_source


def simulate_impact(
    initial_conditions: SimulationInput,
    environment_override=None,
    return_trajectory: bool = False,
    dx_sample_m: float = 2.0,
) -> SimulationOutput:
    del environment_override

    initial_altitude_m = float(initial_conditions["alt"])

    azimuth_rad = math.radians(float(initial_conditions["azimuth"]))
    elevation_rad = math.radians(float(initial_conditions["elevation"]))
    azimuth_rad, elevation_rad = normalize_launch_angles(azimuth_rad, elevation_rad)

    launch_lat_deg = float(initial_conditions["lat"])
    launch_lon_deg = float(initial_conditions["lon"])

    mass_kg = float(initial_conditions["mass"])
    initial_speed_mps = float(initial_conditions["initialSpeed"])

    vx0 = initial_speed_mps * math.cos(elevation_rad)
    vz0 = initial_speed_mps * math.sin(elevation_rad)

    state0 = State3DOF(
        x=0.0,
        z=initial_altitude_m,
        vx=vx0,
        vz=vz0,
        theta=elevation_rad,
        q=0.0,
    )

    sim_time_raw = initial_conditions.get("sim_datetime")
    sim_time = (
        datetime.fromisoformat(sim_time_raw)
        if isinstance(sim_time_raw, str)
        else sim_time_raw
    )

    provider_client, requested_source = _select_weather_provider(initial_conditions)
    weather_runtime = TrajectoryWeatherRuntime(
        provider_client=provider_client,
        launch_lat_deg=launch_lat_deg,
        launch_lon_deg=launch_lon_deg,
        azimuth_rad=azimuth_rad,
        launch_datetime=sim_time,
    )

    sim_kwargs = dict(
        state0=state0,
        dt=DEFAULT_TIME_STEP_S,
        max_time=DEFAULT_MAX_SIM_TIME_S,
        weather_runtime=weather_runtime,
        mass_kg=mass_kg,
        pitch_inertia_kg_m2=DEFAULT_MOMENT_OF_INERTIA_KGM2,
        gravity_mps2=DEFAULT_GRAVITY_MPS2,
        aero_reference=AERO_REF,
        aero_table=AERO_TABLE,
        center_of_gravity_offset_m=DEFAULT_CENTER_OF_GRAVITY_OFFSET,
    )

    sin_az = math.sin(azimuth_rad)
    cos_az = math.cos(azimuth_rad)

    if not return_trajectory:
        impact_state, raw_points = run_simulation_impact_only(**sim_kwargs)

        impact_east_m = impact_state.x * sin_az
        impact_north_m = impact_state.x * cos_az
        impact_lat, impact_lon = enu_displacement_to_latlon(
            impact_east_m,
            impact_north_m,
            launch_lat_deg,
            launch_lon_deg,
        )

        physical_time_s = DEFAULT_TIME_STEP_S * (raw_points - 1)
        impact_sample = weather_runtime.lookup_sample_for_x(impact_state.x)
        runtime_summary = weather_runtime.summary()

        return {
            "impact": {
                "lat": impact_lat,
                "lon": impact_lon,
                "alt": GROUND_ALTITUDE_M,
                "vx": impact_state.vx,
                "vz": impact_state.vz,
                "theta": impact_state.theta,
                **build_point_environment(impact_sample),
            },
            "physical_time": physical_time_s,
            "raw_points_count": raw_points,
            "environment": {
                "requested_source": requested_source,
                "active_source": impact_sample.source,
                "provider": impact_sample.provider,
                "note": impact_sample.note,
                "T0_K": impact_sample.temperature_K,
                "P0_Pa": impact_sample.pressure_Pa,
                "wind_x_mps": impact_sample.wind_x_mps,
                "wind_z_mps": impact_sample.wind_z_mps,
                "refresh_count": int(runtime_summary.get("refresh_count", 0)),
                "fetch_count": int(runtime_summary.get("fetch_count", 0)),
                "state_key": str(runtime_summary.get("state_key", "")),
            },
        }

    impact_state, sampled_states, raw_points = run_simulation_sampled(
        **sim_kwargs,
        dx_sample_m=dx_sample_m,
    )

    trajectory_path: List[TrajectoryPoint] = []

    for s in sampled_states:
        east_m = s.x * sin_az
        north_m = s.x * cos_az
        lat, lon = enu_displacement_to_latlon(
            east_m,
            north_m,
            launch_lat_deg,
            launch_lon_deg,
        )
        point_sample = weather_runtime.lookup_sample_for_x(s.x)
        trajectory_path.append(
            {
                "lat": lat,
                "lon": lon,
                "alt": s.z,
                "vx": s.vx,
                "vz": s.vz,
                "theta": s.theta,
                **build_point_environment(point_sample),
            }
        )

    impact_east_m = impact_state.x * sin_az
    impact_north_m = impact_state.x * cos_az
    impact_lat, impact_lon = enu_displacement_to_latlon(
        impact_east_m,
        impact_north_m,
        launch_lat_deg,
        launch_lon_deg,
    )

    impact_sample = weather_runtime.lookup_sample_for_x(impact_state.x)
    impact_point: TrajectoryPoint = {
        "lat": impact_lat,
        "lon": impact_lon,
        "alt": GROUND_ALTITUDE_M,
        "vx": impact_state.vx,
        "vz": impact_state.vz,
        "theta": impact_state.theta,
        **build_point_environment(impact_sample),
    }

    if trajectory_path:
        trajectory_path[-1] = impact_point
    else:
        trajectory_path.append(impact_point)

    physical_time_s = DEFAULT_TIME_STEP_S * (raw_points - 1)
    runtime_summary = weather_runtime.summary()

    return {
        "impact": impact_point,
        "trajectory": trajectory_path,
        "physical_time": physical_time_s,
        "points_count": len(trajectory_path),
        "raw_points_count": raw_points,
        "environment": {
            "requested_source": requested_source,
            "active_source": impact_sample.source,
            "provider": impact_sample.provider,
            "note": impact_sample.note,
            "T0_K": impact_sample.temperature_K,
            "P0_Pa": impact_sample.pressure_Pa,
            "wind_x_mps": impact_sample.wind_x_mps,
            "wind_z_mps": impact_sample.wind_z_mps,
            "refresh_count": int(runtime_summary.get("refresh_count", 0)),
            "fetch_count": int(runtime_summary.get("fetch_count", 0)),
            "state_key": str(runtime_summary.get("state_key", "")),        },
    }
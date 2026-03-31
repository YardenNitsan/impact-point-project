from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Dict, List, NotRequired, TypedDict

from modules.aerodynamics.aero_tables import default_demo_table
from modules.aerodynamics.aerodynamics import AeroRef
from modules.atmosphere.calculated_weather_runtime import CalculatedWeatherRuntime
from modules.atmosphere.environment import EnvironmentalConditions
from modules.atmosphere.weather_client import (
    HTTPWeatherProviderClient,
    ProviderWeatherSample,
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

MIN_OUTPUT_TRAJECTORY_POINTS: int = 350
MAX_OUTPUT_TRAJECTORY_POINTS: int = 700
OUTPUT_TRAJECTORY_SPACING_M: float = 25.0


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _interpolate_state(a: State3DOF, b: State3DOF, alpha: float) -> State3DOF:
    return State3DOF(
        x=_lerp(a.x, b.x, alpha),
        z=_lerp(a.z, b.z, alpha),
        vx=_lerp(a.vx, b.vx, alpha),
        vz=_lerp(a.vz, b.vz, alpha),
        theta=_lerp(a.theta, b.theta, alpha),
        q=_lerp(a.q, b.q, alpha),
    )


def _build_cumulative_path_lengths(states: List[State3DOF]) -> List[float]:
    if not states:
        return []

    lengths: List[float] = [0.0]

    for i in range(1, len(states)):
        dx = float(states[i].x - states[i - 1].x)
        dz = float(states[i].z - states[i - 1].z)
        lengths.append(lengths[-1] + math.hypot(dx, dz))

    return lengths


def _resample_states_to_target_count(
    states: List[State3DOF],
    target_count: int,
) -> List[State3DOF]:
    if len(states) <= target_count:
        return states[:]

    if target_count <= 2:
        return [states[0], states[-1]]

    cumulative_lengths = _build_cumulative_path_lengths(states)
    total_length = cumulative_lengths[-1]

    if not math.isfinite(total_length) or total_length <= 0.0:
        step = max(1, math.ceil((len(states) - 1) / max(1, target_count - 1)))
        reduced = [states[i] for i in range(0, len(states), step)]
        if reduced[-1] != states[-1]:
            reduced.append(states[-1])
        return reduced[:target_count]

    output: List[State3DOF] = [states[0]]
    segment_index = 0

    for i in range(1, target_count - 1):
        target_length = (total_length * i) / (target_count - 1)

        while (
            segment_index < len(states) - 2
            and cumulative_lengths[segment_index + 1] < target_length
        ):
            segment_index += 1

        start_state = states[segment_index]
        end_state = states[segment_index + 1]

        start_length = cumulative_lengths[segment_index]
        end_length = cumulative_lengths[segment_index + 1]
        segment_length = end_length - start_length

        if segment_length <= 0.0:
            output.append(start_state)
            continue

        alpha = (target_length - start_length) / segment_length
        alpha = max(0.0, min(1.0, alpha))
        output.append(_interpolate_state(start_state, end_state, alpha))

    output.append(states[-1])
    return output


def _compress_trajectory_states(states: List[State3DOF]) -> List[State3DOF]:
    if len(states) <= MAX_OUTPUT_TRAJECTORY_POINTS:
        return states

    cumulative_lengths = _build_cumulative_path_lengths(states)
    total_length = cumulative_lengths[-1] if cumulative_lengths else 0.0

    target_count = int(total_length / OUTPUT_TRAJECTORY_SPACING_M) + 1
    target_count = max(
        MIN_OUTPUT_TRAJECTORY_POINTS,
        min(MAX_OUTPUT_TRAJECTORY_POINTS, target_count),
    )

    if len(states) <= target_count:
        return states

    return _resample_states_to_target_count(states, target_count)

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


def _build_manual_environment_override(
    initial_conditions: SimulationInput,
    azimuth_rad: float,
) -> EnvironmentalConditions:
    wind_x = float(initial_conditions["wind_x"])

    wind_east = wind_x * math.sin(azimuth_rad)
    wind_north = wind_x * math.cos(azimuth_rad)

    return EnvironmentalConditions(
        sea_level_temperature_K=float(initial_conditions["T0_K"]),
        sea_level_pressure_Pa=float(initial_conditions["P0_Pa"]),
        wind_east_10m_mps=wind_east,
        wind_north_10m_mps=wind_north,
        wind_east_100m_mps=wind_east,
        wind_north_100m_mps=wind_north,
        data_source="manual-calculations",
        diagnostic_note="manual environment override with internal calculations",
    )


def _normalize_sim_datetime(sim_time: datetime | None) -> datetime:
    if sim_time is None:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if sim_time.tzinfo is None:
        return sim_time.replace(tzinfo=timezone.utc)
    return sim_time.astimezone(timezone.utc)


def _seed_environment_from_provider_sample(
    *,
    provider_sample: ProviderWeatherSample,
    azimuth_rad: float,
) -> EnvironmentalConditions:
    note_parts: list[str] = [
        f"seeded once from weather service source={provider_sample.source or 'api'}",
    ]

    if provider_sample.wind_east_10m_mps is not None and provider_sample.wind_north_10m_mps is not None:
        east_10 = float(provider_sample.wind_east_10m_mps)
        north_10 = float(provider_sample.wind_north_10m_mps)
    elif provider_sample.wind_east_mps is not None and provider_sample.wind_north_mps is not None:
        east_10 = float(provider_sample.wind_east_mps)
        north_10 = float(provider_sample.wind_north_mps)
        note_parts.append("10m wind seeded from general ENU wind fields")
    else:
        along = float(provider_sample.wind_x_mps or 0.0)
        east_10 = along * math.sin(azimuth_rad)
        north_10 = along * math.cos(azimuth_rad)
        note_parts.append("10m wind seeded from along-track wind fallback")

    if provider_sample.wind_east_100m_mps is not None and provider_sample.wind_north_100m_mps is not None:
        east_100 = float(provider_sample.wind_east_100m_mps)
        north_100 = float(provider_sample.wind_north_100m_mps)
    else:
        east_100 = east_10
        north_100 = north_10
        note_parts.append("100m wind missing from weather service; reused 10m seed")

    if provider_sample.note:
        note_parts.append(str(provider_sample.note))

    return EnvironmentalConditions(
        sea_level_temperature_K=float(provider_sample.temperature_K),
        sea_level_pressure_Pa=float(provider_sample.pressure_Pa),
        wind_east_10m_mps=float(east_10),
        wind_north_10m_mps=float(north_10),
        wind_east_100m_mps=float(east_100),
        wind_north_100m_mps=float(north_100),
        data_source=str(provider_sample.source or "api-seed"),
        diagnostic_note=" | ".join(note_parts),
    )


def _fetch_calculations_seed_from_weather_service(
    *,
    launch_lat_deg: float,
    launch_lon_deg: float,
    initial_altitude_m: float,
    azimuth_rad: float,
    sim_time: datetime | None,
) -> EnvironmentalConditions:
    provider_client = HTTPWeatherProviderClient(
        name="calculations-seed",
        url=join_base_url_and_path(DEFAULT_WEATHER_SERVICE_URL, WEATHER_SERVICE_PATH),
        timeout_s=DEFAULT_WEATHER_TIMEOUT_S,
        requested_source="api",
    )

    seed_sample = provider_client.fetch(
        lat=float(launch_lat_deg),
        lon=float(launch_lon_deg),
        alt=float(initial_altitude_m),
        when=_normalize_sim_datetime(sim_time),
    )

    return _seed_environment_from_provider_sample(
        provider_sample=seed_sample,
        azimuth_rad=azimuth_rad,
    )


def _build_calculations_runtime(
    *,
    initial_conditions: SimulationInput,
    launch_lat_deg: float,
    launch_lon_deg: float,
    initial_altitude_m: float,
    azimuth_rad: float,
    sim_time: datetime | None,
) -> tuple[CalculatedWeatherRuntime, EnvironmentalConditions, str]:
    if _has_manual_weather_override(initial_conditions):
        env = _build_manual_environment_override(initial_conditions, azimuth_rad)
        seed_fetch_count = 0
    else:
        env = _fetch_calculations_seed_from_weather_service(
            launch_lat_deg=launch_lat_deg,
            launch_lon_deg=launch_lon_deg,
            initial_altitude_m=initial_altitude_m,
            azimuth_rad=azimuth_rad,
            sim_time=sim_time,
        )
        seed_fetch_count = 1

    runtime = CalculatedWeatherRuntime(
        environment=env,
        azimuth_rad=azimuth_rad,
        seed_fetch_count=seed_fetch_count,
    )
    return runtime, env, "calculations"


def _build_service_runtime(
    *,
    initial_conditions: SimulationInput,
    launch_lat_deg: float,
    launch_lon_deg: float,
    azimuth_rad: float,
    sim_time: datetime | None,
) -> tuple[TrajectoryWeatherRuntime, str]:
    if _has_manual_weather_override(initial_conditions):
        provider_client = StaticWeatherProviderClient(
            temperature_K=float(initial_conditions["T0_K"]),
            pressure_Pa=float(initial_conditions["P0_Pa"]),
            wind_x_mps=float(initial_conditions["wind_x"]),
            wind_z_mps=float(initial_conditions["wind_z"]),
        )
        requested_source = "manual"
    else:
        requested_source = str(initial_conditions.get("weather_source", "machine")).lower()
        if requested_source not in {"api", "machine"}:
            raise ValueError("weather_source must be 'calculations', 'api', or 'machine'")

        provider_client = HTTPWeatherProviderClient(
            name=requested_source,
            url=join_base_url_and_path(DEFAULT_WEATHER_SERVICE_URL, WEATHER_SERVICE_PATH),
            timeout_s=DEFAULT_WEATHER_TIMEOUT_S,
            requested_source=requested_source,
        )

    runtime = TrajectoryWeatherRuntime(
        provider_client=provider_client,
        launch_lat_deg=launch_lat_deg,
        launch_lon_deg=launch_lon_deg,
        azimuth_rad=azimuth_rad,
        launch_datetime=sim_time,
    )
    return runtime, requested_source


def simulate_impact(
    initial_conditions: SimulationInput,
    environment_override=None,
    return_trajectory: bool = False,
    dx_sample_m: float = 5.0,
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

    requested_mode = str(initial_conditions.get("weather_source", "machine")).lower()

    if requested_mode == "calculations":
        weather_runtime, seed_environment, requested_source = _build_calculations_runtime(
            initial_conditions=initial_conditions,
            launch_lat_deg=launch_lat_deg,
            launch_lon_deg=launch_lon_deg,
            initial_altitude_m=initial_altitude_m,
            azimuth_rad=azimuth_rad,
            sim_time=sim_time,
        )
        using_calculations = True
    else:
        weather_runtime, requested_source = _build_service_runtime(
            initial_conditions=initial_conditions,
            launch_lat_deg=launch_lat_deg,
            launch_lon_deg=launch_lon_deg,
            azimuth_rad=azimuth_rad,
            sim_time=sim_time,
        )
        seed_environment = None
        using_calculations = False

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

    def sample_for_state(state: State3DOF) -> WeatherSample:
        if using_calculations:
            return weather_runtime.sample_for_state(state)
        return weather_runtime.lookup_sample_for_x(state.x)

    def build_environment_block(impact_sample: WeatherSample, runtime_summary: dict) -> Dict[str, float | str | int]:
        if using_calculations:
            assert seed_environment is not None
            return {
                "requested_source": requested_source,
                "active_source": impact_sample.source,
                "provider": impact_sample.provider,
                "note": impact_sample.note,
                "T0_K": float(seed_environment.sea_level_temperature_K),
                "P0_Pa": float(seed_environment.sea_level_pressure_Pa),
                "wind_x_mps": float(impact_sample.wind_x_mps),
                "wind_z_mps": float(impact_sample.wind_z_mps),
                "refresh_count": int(runtime_summary.get("refresh_count", 0)),
                "fetch_count": int(runtime_summary.get("fetch_count", 0)),
                "evaluate_count": int(runtime_summary.get("evaluate_count", 0)),
            }

        return {
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
        }

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
        impact_sample = sample_for_state(impact_state)
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
            "environment": build_environment_block(impact_sample, runtime_summary),
        }

    impact_state, sampled_states, raw_points = run_simulation_sampled(
        **sim_kwargs,
        dx_sample_m=dx_sample_m,
    )
    sampled_states = _compress_trajectory_states(sampled_states)

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
        point_sample = sample_for_state(s)
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

    impact_sample = sample_for_state(impact_state)
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
        "environment": build_environment_block(impact_sample, runtime_summary),
    }
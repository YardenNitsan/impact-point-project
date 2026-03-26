"""
Simulation orchestration and Cesium-ready output formatting.

This module acts as the high-level entry point for a single 3DOF
simulation run. It performs:

1) Parsing of launch parameters (lat/lon/alt, azimuth/elevation, mass, speed)
2) Fetching atmospheric conditions (temperature, pressure, wind)
3) Building a deterministic wind profile model along the launch azimuth
4) Running the 3DOF numerical solver
5) Detecting the ground-impact event via interpolation
6) Converting local downrange displacement into geographic coordinates
7) Downsampling the trajectory for efficient rendering (e.g., Cesium)

Coordinate conventions
----------------------
• The solver state uses an inertial planar frame:
    x : downrange distance along the launch direction [m]
    z : altitude above ground (positive upward) [m]

• Azimuth is measured in the local ENU frame:
    azimuth = 0   → North
    azimuth = 90° → East

Geographic conversion
---------------------
Local displacement is converted into latitude/longitude using a spherical
Earth approximation (sufficient for short ranges; not a full geodesic).
"""

from __future__ import annotations

import math
from typing import Dict, List, TypedDict, NotRequired

from modules.state.state import State3DOF
from functools import lru_cache

from modules.solver.run_simulation import (
    run_simulation_impact_only,
    run_simulation_sampled,
)

from modules.atmosphere.environment import fetch_environmental_conditions
from modules.atmosphere.wind import AlongTrackWindShearModel

from modules.aerodynamics.aero_tables import default_demo_table
from modules.aerodynamics.aerodynamics import AeroRef


from datetime import datetime
from modules.atmosphere.era5_loader import (
    load_era5_column,
    sample_column_at_altitude,
)

# ============================================================
# Typed I/O schemas (professional clarity)
# ============================================================


class SimulationInput(TypedDict):
    lat: float
    lon: float
    alt: float
    azimuth: float
    elevation: float
    mass: float
    initialSpeed: float
    sim_datetime: str | datetime

    T0_K: NotRequired[float]
    P0_Pa: NotRequired[float]
    wind_x: NotRequired[float]
    wind_z: NotRequired[float]


class TrajectoryPoint(TypedDict):
    """
    A single trajectory sample formatted for visualization.
    """
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


class SimulationOutput(TypedDict):
    impact: TrajectoryPoint
    physical_time: float
    raw_points_count: int
    environment: Dict[str, float | str]

    # מוחזרים רק אם ביקשו trajectory
    trajectory: NotRequired[List[TrajectoryPoint]]
    points_count: NotRequired[int]


# ============================================================
# Physical / geographic constants
# ============================================================

EARTH_RADIUS_M: float = 6_371_000.0
"""Mean Earth radius used for spherical approximation."""

DEG_PER_RAD: float = 180.0 / math.pi
"""Conversion from radians to degrees."""

METERS_PER_DEGREE_APPROX: float = 111_000.0
"""
Approximate meters per degree of latitude/longitude.
Used only for distance estimation in downsampling (flat approximation).
"""


# ============================================================
# Simulation configuration constants
# ============================================================

DEFAULT_TIME_STEP_S: float = 0.001
"""Integrator time step [s]."""

DEFAULT_MAX_SIM_TIME_S: float = 800.0
"""Maximum simulated time horizon [s]."""

DEFAULT_TARGET_TRAJECTORY_POINTS: int = 500
"""Target number of points after downsampling (visualization constraint)."""

DEFAULT_MIN_DOWNSAMPLE_DISTANCE_M: float = 5.0
"""Minimum spacing between output points (reduces Cesium load)."""

DEFAULT_REFERENCE_AREA_M2: float = 0.002
"""Aerodynamic reference area Sref [m²]."""

DEFAULT_REFERENCE_LENGTH_M: float = 0.30
"""Aerodynamic reference length lref [m]."""

DEFAULT_MOMENT_OF_INERTIA_KGM2: float = 2.0
"""Pitch moment of inertia Iyy [kg·m²] (placeholder / project parameter)."""

DEFAULT_GRAVITY_MPS2: float = 9.81
"""Gravitational acceleration [m/s²]."""

DEFAULT_CENTER_OF_GRAVITY_OFFSET: float = 0.02
"""
Non-dimensional CG offset term for moment correction (project parameter).
Interpretation depends on your coefficient model convention.
"""

GROUND_ALTITUDE_M: float = 0.0
"""Ground altitude used for output impact altitude."""

# ============================================================
# Reusable singletons / caches
# ============================================================

# לא ליצור מחדש את טבלת האווירודינמיקה בכל בקשה.
AERO_TABLE = default_demo_table()

# גם את AeroRef אין צורך לבנות בכל סימולציה.
AERO_REF = AeroRef(
    reference_area=DEFAULT_REFERENCE_AREA_M2,
    reference_length=DEFAULT_REFERENCE_LENGTH_M,
)

@lru_cache(maxsize=2048)
def _fetch_environment_cached(lat_rounded: float, lon_rounded: float):
    """
    Cache קטן לתנאי סביבה כדי למנוע fetch חוזר על אותם אזורים.
    העיגול מתבצע לפני הקריאה לפונקציה.
    """
    return fetch_environmental_conditions(lat_rounded, lon_rounded)


# ============================================================
# Coordinate conversion
# ============================================================

def enu_displacement_to_latlon(
    east_m: float,
    north_m: float,
    reference_lat_deg: float,
    reference_lon_deg: float,
) -> tuple[float, float]:
    """
    Convert a local ENU displacement (east/north in meters) into
    latitude/longitude degrees using a spherical Earth approximation.
    """

    lat_offset_deg = (north_m / EARTH_RADIUS_M) * DEG_PER_RAD

    lon_offset_deg = (
        east_m / (EARTH_RADIUS_M * math.cos(math.radians(reference_lat_deg)))
    ) * DEG_PER_RAD

    return reference_lat_deg + lat_offset_deg, reference_lon_deg + lon_offset_deg


# ============================================================
# Angle normalization
# ============================================================

def normalize_launch_angles(
    azimuth_rad: float,
    elevation_rad: float,
) -> tuple[float, float]:
    """
    Normalize elevation into [-π/2, +π/2] and adjust azimuth accordingly.

    This prevents ambiguous representations such as elevation > 90°,
    which is physically equivalent to flipping direction by 180°.
    """

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


# ============================================================
# Trajectory downsampling
# ============================================================
def downsample_states_by_dx(
    states: List[State3DOF],
    dx_min_m: float,
) -> List[State3DOF]:
    """
    Downsample solver states by keeping points at least dx_min_m apart in downrange (x).
    This avoids lat/lon approximations and produces smooth spacing for Cesium.
    """
    if len(states) < 2:
        return states

    out: List[State3DOF] = [states[0]]
    last_x = float(states[0].x)

    for s in states[1:]:
        x = float(s.x)
        if (x - last_x) >= dx_min_m:
            out.append(s)
            last_x = x

    # always keep last point
    if out[-1] is not states[-1]:
        out.append(states[-1])

    return out

def downsample_by_distance(
    trajectory: List[TrajectoryPoint],
    min_distance_m: float,
) -> List[TrajectoryPoint]:
    """
    Keep only points that are at least min_distance_m apart (horizontal distance),
    ensuring the last point is always preserved.
    """

    if len(trajectory) < 2:
        return trajectory

    filtered: List[TrajectoryPoint] = [trajectory[0]]

    for point in trajectory[1:]:
        prev = filtered[-1]

        dlon_m = (point["lon"] - prev["lon"]) * METERS_PER_DEGREE_APPROX
        dlat_m = (point["lat"] - prev["lat"]) * METERS_PER_DEGREE_APPROX

        horizontal_dist_m = math.hypot(dlon_m, dlat_m)

        if horizontal_dist_m >= min_distance_m:
            filtered.append(point)

    if filtered[-1] != trajectory[-1]:
        filtered.append(trajectory[-1])

    return filtered


def adaptive_downsample(
    trajectory: List[TrajectoryPoint],
    target_points: int,
) -> List[TrajectoryPoint]:
    """
    Compute a reasonable spacing threshold based on total range and
    downsample accordingly.
    """

    if len(trajectory) <= target_points:
        return trajectory

    start = trajectory[0]
    end = trajectory[-1]

    dlon_m = (end["lon"] - start["lon"]) * METERS_PER_DEGREE_APPROX
    dlat_m = (end["lat"] - start["lat"]) * METERS_PER_DEGREE_APPROX

    total_range_m = math.hypot(dlon_m, dlat_m)

    spacing_m = max(
        DEFAULT_MIN_DOWNSAMPLE_DISTANCE_M,
        total_range_m / float(target_points),
    )

    return downsample_by_distance(trajectory, spacing_m)


# ============================================================
# Main simulation entry point
# ============================================================

class ERA5WindModel:
    """
    TEMP DATASET MODE:
    רוח נלקחת ישירות מעמודת ERA5 לפי גובה,
    בלי power-law shear ובלי חישוב 10m/100m.
    """
    def __init__(self, *, azimuth_rad: float, era5_column: dict):
        self.azimuth_rad = float(azimuth_rad)
        self.era5_column = era5_column

    def wind_at_height(self, z_m: float):
        sample = sample_column_at_altitude(self.era5_column, z_m)

        wind_x = (
            sample["u_east_mps"] * math.sin(self.azimuth_rad)
            + sample["v_north_mps"] * math.cos(self.azimuth_rad)
        )

        return float(wind_x), 0.0

def build_point_environment(
    *,
    era5_column: dict,
    altitude_m: float,
    azimuth_rad: float,
) -> Dict[str, float]:
    atm = sample_column_at_altitude(era5_column, max(0.0, float(altitude_m)))

    wind_along_track = (
        atm["u_east_mps"] * math.sin(azimuth_rad)
        + atm["v_north_mps"] * math.cos(azimuth_rad)
    )

    return {
        "temperature_K": float(atm["temperature_K"]),
        "pressure_Pa": float(atm["pressure_Pa"]),
        "wind_u_east_mps": float(atm["u_east_mps"]),
        "wind_v_north_mps": float(atm["v_north_mps"]),
        "wind_along_track_mps": float(wind_along_track),
    }


def simulate_impact(
    initial_conditions: SimulationInput,
    environment_override=None,
    return_trajectory: bool = False,
    dx_sample_m: float = 2.0,
) -> SimulationOutput:
    """
    Run one full 3DOF simulation and return impact + Cesium-ready trajectory.
    """

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

    # Solver inertial frame: x is downrange along azimuth, z is altitude.
    state0 = State3DOF(
        x=0.0,
        z=initial_altitude_m,
        vx=vx0,
        vz=vz0,
        theta=elevation_rad,
        q=0.0,
    )

    # Environment (external override or API)

    # if (
    #     initial_conditions.get("T0_K") is not None and
    #     initial_conditions.get("P0_Pa") is not None and
    #     initial_conditions.get("wind_x") is not None and
    #     initial_conditions.get("wind_z") is not None
    # ):

    #     class EnvOverride:
    #         def __init__(self):
    #             self.sea_level_temperature_K = initial_conditions["T0_K"]
    #             self.sea_level_pressure_Pa = initial_conditions["P0_Pa"]

    #             self.wind_east_10m_mps = initial_conditions["wind_x"]
    #             self.wind_north_10m_mps = initial_conditions["wind_z"]

    #             self.wind_east_100m_mps = initial_conditions["wind_x"]
    #             self.wind_north_100m_mps = initial_conditions["wind_z"]

    #             self.data_source = "dataset"
    #             self.diagnostic_note = "environment override"

    #     env = EnvOverride()     

    # else:
    #     env = _fetch_environment_cached(round(launch_lat_deg, 3), round(launch_lon_deg, 3))

    # ========================================================
    # TEMP DATASET MODE:
    # לא API, לא ISA fallback, לא shear model מחושב.
    # לוקחים עמודת ERA5 מקומית פעם אחת לסימולציה.
    # ========================================================

    sim_time_raw = initial_conditions["sim_datetime"]
    sim_time = (
        datetime.fromisoformat(sim_time_raw)
        if isinstance(sim_time_raw, str)
        else sim_time_raw
    )

    era5_column = load_era5_column(
        launch_lat_deg,
        launch_lon_deg,
        sim_time,
    )

    launch_atm = sample_column_at_altitude(era5_column, initial_altitude_m)

    class EnvFromERA5:
        def __init__(self):
            self.sea_level_temperature_K = launch_atm["temperature_K"]
            self.sea_level_pressure_Pa = launch_atm["pressure_Pa"]

            # שומרים את השדות הישנים כדי לא לשבור את שאר הקוד
            self.wind_east_10m_mps = launch_atm["u_east_mps"]
            self.wind_north_10m_mps = launch_atm["v_north_mps"]
            self.wind_east_100m_mps = launch_atm["u_east_mps"]
            self.wind_north_100m_mps = launch_atm["v_north_mps"]

            self.data_source = "era5-local-file"
            self.diagnostic_note = "TEMP dataset mode"

    env = EnvFromERA5()

    # Wind model projected onto the trajectory axis (along-track).
    # wind_model = AlongTrackWindShearModel(
    #     azimuth_rad=azimuth_rad,
    #     wind_east_10m_mps=env.wind_east_10m_mps,
    #     wind_north_10m_mps=env.wind_north_10m_mps,
    #     wind_east_100m_mps=env.wind_east_100m_mps,
    #     wind_north_100m_mps=env.wind_north_100m_mps,
    # )
    # TEMP DATASET MODE:
    # משתמשים ברוח ישירות מ-ERA5 לפי גובה
    wind_model = ERA5WindModel(
        azimuth_rad=azimuth_rad,
        era5_column=era5_column,
    )

    sim_kwargs = dict(
        state0=state0,
        dt=DEFAULT_TIME_STEP_S,
        max_time=DEFAULT_MAX_SIM_TIME_S,
        wind_model=wind_model,
        mass_kg=mass_kg,
        pitch_inertia_kg_m2=DEFAULT_MOMENT_OF_INERTIA_KGM2,
        gravity_mps2=DEFAULT_GRAVITY_MPS2,
        sea_level_temperature_K=env.sea_level_temperature_K,
        sea_level_pressure_Pa=env.sea_level_pressure_Pa,
        aero_reference=AERO_REF,
        aero_table=AERO_TABLE,
        center_of_gravity_offset_m=DEFAULT_CENTER_OF_GRAVITY_OFFSET,
        era5_column=era5_column,
    )

    sin_az = math.sin(azimuth_rad)
    cos_az = math.cos(azimuth_rad)

    # ========================================================
    # Fast path: impact בלבד (ל-dataset / עומס / throughput גבוה)
    # ========================================================
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

        impact_env = build_point_environment(
            era5_column=era5_column,
            altitude_m=GROUND_ALTITUDE_M,
            azimuth_rad=azimuth_rad,
        )

        return {
            "impact": {
                "lat": impact_lat,
                "lon": impact_lon,
                "alt": GROUND_ALTITUDE_M,
                "vx": impact_state.vx,
                "vz": impact_state.vz,
                "theta": impact_state.theta,
                **impact_env,
            },
            "physical_time": physical_time_s,
            "raw_points_count": raw_points,
            "environment": {
                "source": env.data_source,
                "note": env.diagnostic_note,
                "T0_K": env.sea_level_temperature_K,
                "P0_Pa": env.sea_level_pressure_Pa,
                "wind_u_east_mps": env.wind_east_10m_mps,
                "wind_v_north_mps": env.wind_north_10m_mps,
            },
        }
    # ========================================================
    # Visual path: trajectory מדוגם בלבד
    # ========================================================
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

        point_env = build_point_environment(
            era5_column=era5_column,
            altitude_m=s.z,
            azimuth_rad=azimuth_rad,
        )

        trajectory_path.append(
            {
                "lat": lat,
                "lon": lon,
                "alt": s.z,
                "vx": s.vx,
                "vz": s.vz,
                "theta": s.theta,
                **point_env,
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

  
    impact_env = build_point_environment(
        era5_column=era5_column,
        altitude_m=GROUND_ALTITUDE_M,
        azimuth_rad=azimuth_rad,
    )

    impact_point: TrajectoryPoint = {
        "lat": impact_lat,
        "lon": impact_lon,
        "alt": GROUND_ALTITUDE_M,
        "vx": impact_state.vx,
        "vz": impact_state.vz,
        "theta": impact_state.theta,
        **impact_env,
    }

    if trajectory_path:
        trajectory_path[-1] = impact_point
    else:
        trajectory_path.append(impact_point)

    physical_time_s = DEFAULT_TIME_STEP_S * (raw_points - 1)

    return {
        "impact": impact_point,
        "trajectory": trajectory_path,
        "physical_time": physical_time_s,
        "points_count": len(trajectory_path),
        "raw_points_count": raw_points,
        "environment": {
            "source": env.data_source,
            "note": env.diagnostic_note,
            "T0_K": env.sea_level_temperature_K,
            "P0_Pa": env.sea_level_pressure_Pa,
            "wind_u_east_mps": env.wind_east_10m_mps,
            "wind_v_north_mps": env.wind_north_10m_mps,
        },
    }
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
from typing import Dict, List, TypedDict

from modules.state.state import State3DOF
from modules.solver.run_simulation import run_simulation
from modules.impact.impact import compute_impact_from_trajectory

from modules.atmosphere.environment import fetch_environmental_conditions
from modules.atmosphere.wind import AlongTrackWindShearModel

from modules.aerodynamics.aero_tables import default_demo_table
from modules.aerodynamics.aerodynamics import AeroRef


# ============================================================
# Typed I/O schemas (professional clarity)
# ============================================================

class SimulationInput(TypedDict):
    """
    Input parameters for a single simulation case.

    All angles are provided in DEGREES (UI-friendly).
    """
    lat: float
    lon: float
    alt: float
    azimuth: float          # degrees (0=N, 90=E)
    elevation: float        # degrees (positive up)
    mass: float             # kg
    initialSpeed: float     # m/s


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


class SimulationOutput(TypedDict):
    impact: TrajectoryPoint
    trajectory: List[TrajectoryPoint]
    physical_time: float
    points_count: int
    raw_points_count: int
    environment: Dict[str, float | str]


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

DEFAULT_TIME_STEP_S: float = 0.005
"""Integrator time step [s]."""

DEFAULT_MAX_SIM_TIME_S: float = 300.0
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

def simulate_impact(initial_conditions: SimulationInput) -> SimulationOutput:
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

    # Environment (real-time API with ISA fallback).
    env = fetch_environmental_conditions(launch_lat_deg, launch_lon_deg)

    # Wind model projected onto the trajectory axis (along-track).
    wind_model = AlongTrackWindShearModel(
        azimuth_rad=azimuth_rad,
        wind_east_10m_mps=env.wind_east_10m_mps,
        wind_north_10m_mps=env.wind_north_10m_mps,
        wind_east_100m_mps=env.wind_east_100m_mps,
        wind_north_100m_mps=env.wind_north_100m_mps,
    )

    aero_ref = AeroRef(
        reference_area=DEFAULT_REFERENCE_AREA_M2,
        reference_length=DEFAULT_REFERENCE_LENGTH_M,
    )

    trajectory = run_simulation(
        state0=state0,
        dt=DEFAULT_TIME_STEP_S,
        max_time=DEFAULT_MAX_SIM_TIME_S,
        wind_model=wind_model,
        mass_kg=mass_kg,
        pitch_inertia_kg_m2=DEFAULT_MOMENT_OF_INERTIA_KGM2,
        gravity_mps2=DEFAULT_GRAVITY_MPS2,
        sea_level_temperature_K=env.sea_level_temperature_K,
        sea_level_pressure_Pa=env.sea_level_pressure_Pa,
        aero_reference=aero_ref,
        aero_table=default_demo_table(),
        center_of_gravity_offset_m=DEFAULT_CENTER_OF_GRAVITY_OFFSET,
    )

    impact = compute_impact_from_trajectory(trajectory)
    if impact is None:
        raise RuntimeError("No ground impact detected (trajectory did not cross z=0).")

    impact_state = impact.state_at_impact

    # Convert solver x (downrange) into ENU east/north displacement and then lat/lon.
    trajectory_path: List[TrajectoryPoint] = []

    sin_az = math.sin(azimuth_rad)
    cos_az = math.cos(azimuth_rad)

    for s in trajectory:
        east_m = s.x * sin_az
        north_m = s.x * cos_az

        lat, lon = enu_displacement_to_latlon(east_m, north_m, launch_lat_deg, launch_lon_deg)

        trajectory_path.append(
            {
                "lat": lat,
                "lon": lon,
                "alt": s.z,
                "vx": s.vx,
                "vz": s.vz,
                "theta": s.theta,
            }
        )

    # Compute impact geographic location (using interpolated impact x).
    impact_east_m = impact_state.x * sin_az
    impact_north_m = impact_state.x * cos_az

    impact_lat, impact_lon = enu_displacement_to_latlon(
        impact_east_m,
        impact_north_m,
        launch_lat_deg,
        launch_lon_deg,
    )

    # Replace last sample with exact interpolated impact.
    if trajectory_path:
        trajectory_path.pop()

    trajectory_path.append(
        {
            "lat": impact_lat,
            "lon": impact_lon,
            "alt": GROUND_ALTITUDE_M,
            "vx": impact_state.vx,
            "vz": impact_state.vz,
            "theta": impact_state.theta,
        }
    )

    trajectory_path = adaptive_downsample(
        trajectory_path,
        DEFAULT_TARGET_TRAJECTORY_POINTS,
    )

    physical_time_s = DEFAULT_TIME_STEP_S * (len(trajectory) - 1)

    return {
        "impact": {
            "lat": impact_lat,
            "lon": impact_lon,
            "alt": GROUND_ALTITUDE_M,
            "vx": impact_state.vx,
            "vz": impact_state.vz,
            "theta": impact_state.theta,
        },
        "trajectory": trajectory_path,
        "physical_time": physical_time_s,
        "points_count": len(trajectory_path),
        "raw_points_count": len(trajectory),
        "environment": {
            "source": env.data_source,
            "note": env.diagnostic_note,
            "T0_K": env.sea_level_temperature_K,
            "P0_Pa": env.sea_level_pressure_Pa,
        },
    }

from __future__ import annotations

"""
Planar 3DOF rigid-body flight dynamics model.

This module implements the continuous-time equations of motion for a
projectile / vehicle in a vertical plane with aerodynamic forces and pitch dynamics.

Mathematical model
------------------

Translational dynamics:

    m * dv/dt = F_aero + F_gravity

where:

    v = (vx, vz)   inertial velocity vector
    F_aero         aerodynamic force vector
    F_gravity      = (0, -m g)

Rotational dynamics (pitch):

    Iyy * dq/dt = My

    dθ/dt = q

Coordinate frame
----------------

x-axis : downrange inertial axis [m]
z-axis : altitude above ground (positive upward) [m]

State vector:

    x     : downrange position [m]
    z     : altitude [m]
    vx    : inertial x velocity [m/s]
    vz    : inertial z velocity [m/s]
    theta : pitch angle [rad]
    q     : pitch rate [rad/s]

Aerodynamic model
-----------------

Aerodynamic forces are computed from lookup tables as a function of:

    alpha : angle of attack [rad]
    M     : Mach number [-]

using atmospheric properties from ISA.

Assumptions
-----------

• Flat Earth
• No Coriolis
• Point-mass translation + pitch DOF
• ISA atmosphere
• Aerodynamic coefficients from tabulated data
"""

import math

from modules.state.state import State3DOF, StateDerivatives3DOF
from modules.atmosphere.isa import compute_isa_troposphere_state, compute_speed_of_sound
from modules.aerodynamics.aerodynamics import compute_aerodynamic_loads_from_lookup_table, AeroRef
from modules.aerodynamics.aero_tables import AeroTable2D, wrap_to_pi


# ============================================================
# Numerical safety constants
# ============================================================

MIN_SPEED_OF_SOUND_MPS: float = 1e-6
"""Lower bound to avoid division by zero in Mach computation."""


# ============================================================
# Public API
# ============================================================

def derivatives(
    state: State3DOF,
    *,
    mass_kg: float,
    pitch_inertia_kg_m2: float,
    gravity_mps2: float,
    sea_level_temperature_K: float,
    sea_level_pressure_Pa: float,
    aero_reference: AeroRef,
    aero_table: AeroTable2D,
    wind_x_mps: float,
    wind_z_mps: float,
    center_of_gravity_offset_m: float = 0.0,
) -> StateDerivatives3DOF:
    """
    Compute time derivatives of the 3DOF state.

    Parameters
    ----------
    state:
        Current 3DOF state.
    mass_kg:
        Vehicle mass [kg].
    pitch_inertia_kg_m2:
        Pitch moment of inertia Iyy [kg·m²].
    gravity_mps2:
        Gravitational acceleration [m/s²].
    sea_level_temperature_K:
        Reference ISA sea-level temperature [K].
    sea_level_pressure_Pa:
        Reference ISA sea-level pressure [Pa].
    aero_reference:
        Aerodynamic reference geometry.
    aero_table:
        2D aerodynamic lookup table (alpha, Mach).
    wind_x_mps, wind_z_mps:
        Wind components in inertial frame [m/s].
    center_of_gravity_offset_m:
        CG offset parameter used by the aero model.

    Returns
    -------
    StateDerivatives3DOF
        Time derivatives of the state.
    """

    # ========================================================
    # State unpacking
    # ========================================================

    altitude_m = state.z
    vx_inertial = state.vx
    vz_inertial = state.vz
    theta_rad = state.theta
    pitch_rate_radps = state.q

    # ========================================================
    # Atmosphere model (ISA)
    # ========================================================

    temperature_K, pressure_Pa, _density = compute_isa_troposphere_state(
        altitude_m,
        sea_level_temperature_K,
        sea_level_pressure_Pa,
    )

    a_sound = max(compute_speed_of_sound(temperature_K), MIN_SPEED_OF_SOUND_MPS)

    # ========================================================
    # Relative airflow
    # ========================================================

    vx_rel = vx_inertial - wind_x_mps
    vz_rel = vz_inertial - wind_z_mps

    airspeed_mps = math.hypot(vx_rel, vz_rel)
    flight_path_angle_rad = math.atan2(vz_rel, vx_rel)

    alpha_rad = wrap_to_pi(theta_rad - flight_path_angle_rad)
    mach_number = airspeed_mps / a_sound

    # clamp mach to table limits if available
    if hasattr(aero_table, "mach_min") and hasattr(aero_table, "mach_max"):
        mach_number = max(aero_table.mach_min, min(aero_table.mach_max, mach_number))

    # ========================================================
    # Aerodynamic forces & moments
    # ========================================================

    coeffs = aero_table.lookup(alpha_rad, mach_number)

    force_x_N, force_z_N, moment_y_Nm, _, _ = compute_aerodynamic_loads_from_lookup_table(
        static_pressure=pressure_Pa,
        static_temperature=temperature_K,
        velocity_x=vx_inertial,
        velocity_z=vz_inertial,
        wind_velocity_x=wind_x_mps,
        wind_velocity_z=wind_z_mps,
        alpha=alpha_rad,
        mach=mach_number,
        aerodynamic_coefficients=coeffs,
        aero_reference=aero_reference,
        pitch_rate=pitch_rate_radps,
        lcg=center_of_gravity_offset_m,
    )

    # ========================================================
    # Equations of motion
    # ========================================================

    ax = force_x_N / mass_kg
    az = (force_z_N / mass_kg) - gravity_mps2
    q_dot = moment_y_Nm / pitch_inertia_kg_m2

    return StateDerivatives3DOF(
        x=vx_inertial,
        z=vz_inertial,
        vx=ax,
        vz=az,
        theta=pitch_rate_radps,
        q=q_dot,
    )

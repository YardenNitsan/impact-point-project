from __future__ import annotations

"""
Planar 3DOF rigid-body flight dynamics model.
"""

import math

from modules.state.state import State3DOF, StateDerivatives3DOF
from modules.atmosphere.isa import compute_speed_of_sound
from modules.aerodynamics.aerodynamics import compute_aerodynamic_loads_from_lookup_table, AeroRef
from modules.aerodynamics.aero_tables import AeroTable2D, wrap_to_pi

MIN_SPEED_OF_SOUND_MPS: float = 1e-6
MIN_VALID_TEMPERATURE_K: float = 1e-6
MIN_VALID_PRESSURE_PA: float = 1e-6


def derivatives(
    state: State3DOF,
    *,
    mass_kg: float,
    pitch_inertia_kg_m2: float,
    gravity_mps2: float,
    temperature_K: float,
    pressure_Pa: float,
    aero_reference: AeroRef,
    aero_table: AeroTable2D,
    wind_x_mps: float,
    wind_z_mps: float,
    center_of_gravity_offset_m: float = 0.0,
) -> StateDerivatives3DOF:
    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    if pitch_inertia_kg_m2 <= 0.0:
        raise ValueError("pitch_inertia_kg_m2 must be positive")
    if temperature_K <= MIN_VALID_TEMPERATURE_K:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa <= MIN_VALID_PRESSURE_PA:
        raise ValueError("pressure_Pa must be positive")

    vx_inertial = float(state.vx)
    vz_inertial = float(state.vz)
    theta_rad = float(state.theta)
    pitch_rate_radps = float(state.q)

    a_sound = max(compute_speed_of_sound(float(temperature_K)), MIN_SPEED_OF_SOUND_MPS)

    vx_rel = vx_inertial - float(wind_x_mps)
    vz_rel = vz_inertial - float(wind_z_mps)

    airspeed_mps = math.hypot(vx_rel, vz_rel)
    flight_path_angle_rad = math.atan2(vz_rel, vx_rel)

    alpha_rad = wrap_to_pi(theta_rad - flight_path_angle_rad)
    mach_number = airspeed_mps / a_sound

    coeffs = aero_table.lookup(alpha_rad, mach_number)

    force_x_N, force_z_N, moment_y_Nm, _, _ = compute_aerodynamic_loads_from_lookup_table(
        static_pressure=float(pressure_Pa),
        static_temperature=float(temperature_K),
        velocity_x=vx_inertial,
        velocity_z=vz_inertial,
        wind_velocity_x=float(wind_x_mps),
        wind_velocity_z=float(wind_z_mps),
        alpha=alpha_rad,
        mach=mach_number,
        aerodynamic_coefficients=coeffs,
        aero_reference=aero_reference,
        pitch_rate=pitch_rate_radps,
        lcg=float(center_of_gravity_offset_m),
    )

    ax = force_x_N / float(mass_kg)
    az = (force_z_N / float(mass_kg)) - float(gravity_mps2)
    q_dot = moment_y_Nm / float(pitch_inertia_kg_m2)

    return StateDerivatives3DOF(
        x=vx_inertial,
        z=vz_inertial,
        vx=ax,
        vz=az,
        theta=pitch_rate_radps,
        q=q_dot,
    )
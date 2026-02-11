"""
3DOF rigid-body inertial dynamics model.

This module defines the state vector representation and
equations of motion for planar 3DOF rigid-body dynamics
in an inertial coordinate frame.

State vector definition
-----------------------

The rigid-body state is represented as:

    S = [x, z, vx, vz, θ, q]^T

where:

    x  : horizontal position [m]
    z  : altitude (positive upward) [m]
    vx : inertial x velocity [m/s]
    vz : inertial z velocity [m/s]
    θ  : pitch angle [rad]
    q  : pitch rate [rad/s]

System equations
----------------

The dynamics follow:

    dS/dt = f(S, t)

with:

    x_ddot     = X_force / mass
    z_ddot     = Z_force / mass − gravity
    θ_ddot     = pitching_moment / inertia
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ============================================================
# State containers
# ============================================================

@dataclass(frozen=True)
class RigidBodyState3DOF:
    """
    3DOF rigid-body state vector.

    Represents the inertial state:

        S = [x, z, vx, vz, θ, q]
    """

    x_position_m: float
    z_altitude_m: float

    x_velocity_mps: float
    z_velocity_mps: float

    pitch_angle_rad: float
    pitch_rate_radps: float


@dataclass(frozen=True)
class RigidBodyStateDerivatives3DOF:
    """
    Time derivative of the 3DOF state vector.

    Represents:

        dS/dt = [vx, vz, ax, az, q, θ_ddot]
    """

    x_position_rate_mps: float
    z_altitude_rate_mps: float

    x_acceleration_mps2: float
    z_acceleration_mps2: float

    pitch_angle_rate_radps: float
    pitch_acceleration_radps2: float


# ============================================================
# Equations of motion
# ============================================================

def compute_rigid_body_accelerations(
    x_force_N: float,
    z_force_N: float,
    pitching_moment_Nm: float,
    mass_kg: float,
    pitch_inertia_kgm2: float,
    gravity_mps2: float,
) -> Tuple[float, float, float]:
    """
    Compute inertial accelerations from applied forces
    and pitching moment.

    Implements the rigid-body equations of motion.
    """

    x_inertial_acceleration = x_force_N / mass_kg

    z_inertial_acceleration = (
        z_force_N / mass_kg
        - gravity_mps2
    )

    pitch_angular_acceleration = (
        pitching_moment_Nm
        / pitch_inertia_kgm2
    )

    return x_inertial_acceleration, z_inertial_acceleration, pitch_angular_acceleration


# ============================================================
# State derivative assembly
# ============================================================

def compute_rigid_body_state_derivatives(
    rigid_body_state: RigidBodyState3DOF,
    x_force_N: float,
    z_force_N: float,
    pitching_moment_Nm: float,
    mass_kg: float,
    pitch_inertia_kgm2: float,
    gravity_mps2: float,
) -> RigidBodyStateDerivatives3DOF:
    """
    Assemble the time derivative of the rigid-body state.

    This function defines the system dynamics:

        dS/dt = f(S, t)
    """

    (
        x_acceleration,
        z_acceleration,
        pitch_acceleration,
    ) = compute_rigid_body_accelerations(
        x_force_N,
        z_force_N,
        pitching_moment_Nm,
        mass_kg,
        pitch_inertia_kgm2,
        gravity_mps2,
    )

    return RigidBodyStateDerivatives3DOF(
        x_position_rate_mps=rigid_body_state.x_velocity_mps,
        z_altitude_rate_mps=rigid_body_state.z_velocity_mps,
        x_acceleration_mps2=x_acceleration,
        z_acceleration_mps2=z_acceleration,
        pitch_angle_rate_radps=rigid_body_state.pitch_rate_radps,
        pitch_acceleration_radps2=pitch_acceleration,
    )

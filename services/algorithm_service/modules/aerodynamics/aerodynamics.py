"""
Aerodynamic force and pitching moment model
for planar 3DOF flight simulation.

This module converts aerodynamic coefficients (CD, CL, CM)
into inertial-frame forces (X, Z) and a pitching moment (My)
using standard aerodynamic relations:

    q_bar = 0.5 * rho * V^2
    D     = q_bar * Sref * CD
    L     = q_bar * Sref * CL
    My    = q_bar * Sref * lref * CM_eff

The flow-relative (wind-relative) velocity is used, i.e. the
vehicle velocity relative to the air mass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Physical constants
# ============================================================

R_AIR: float = 287.05
"""Specific gas constant for dry air [J/(kg·K)]."""

DEFAULT_GAMMA: float = 1.4
"""Ratio of specific heats for air (used elsewhere if needed)."""


# ============================================================
# Numerical stability thresholds
# ============================================================

MIN_AIRSPEED_FOR_FORCES: float = 1e-6
"""Below this airspeed [m/s], aerodynamic forces are treated as zero."""

MIN_AIRSPEED_FOR_MOMENT: float = 1e-3
"""Minimum airspeed [m/s] used to avoid division-by-zero in pitch damping."""

PITCH_DAMPING_SCALE: float = 2.0
"""
Standard aerodynamic normalization factor for pitch damping.

In many flight dynamics formulations, pitch damping is normalized using (2V).
This constant makes the nondimensional term consistent with that convention.
"""


# ============================================================
# Aerodynamic reference geometry
# ============================================================

@dataclass(frozen=True)
class AeroRef:
    """
    Aerodynamic reference geometry.

    Attributes
    ----------
    reference_area : float
        Reference area [m²]
    reference_length : float
        Reference length [m]
    """

    reference_area: float
    reference_length: float


# ============================================================
# Aerodynamic force & moment model
# ============================================================

def compute_aerodynamic_loads_from_lookup_table(
    *,
    static_pressure: float,
    static_temperature: float,
    velocity_x: float,
    velocity_z: float,
    wind_velocity_x: float,
    wind_velocity_z: float,
    alpha: float,
    mach: float,
    aerodynamic_coefficients,
    aero_reference: AeroRef,
    pitch_rate: float,
    lcg: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """
    Convert aerodynamic coefficients into inertial-frame forces and pitching moment.

    Parameters
    ----------
    static_pressure : float
        Static pressure [Pa]
    static_temperature : float
        Static temperature [K]
    velocity_x, velocity_z : float
        Vehicle inertial velocity components [m/s]
    wind_velocity_x, wind_velocity_z : float
        Wind velocity components expressed in solver axes [m/s]
        (wind is subtracted to obtain air-relative velocity)
    alpha : float
        Angle of attack [rad]
    mach : float
        Mach number (passed through for diagnostics)
    aerodynamic_coefficients : object
        Aerodynamic coefficients container with:
        CD, CL, CM0, Cm_alpha, Cmq
    aero_reference : AeroRef
        Aerodynamic reference geometry
    pitch_rate : float
        Pitch rate [rad/s]
    lcg : float, optional
        Non-dimensional center-of-gravity offset contribution.
        Included as +CL * lcg in the effective pitching moment coefficient.

    Returns
    -------
    Xv : float
        Aerodynamic force along inertial x-axis [N]
    Zv : float
        Aerodynamic force along inertial z-axis [N]
    My : float
        Pitching moment about CG / reference point [N·m]
    mach : float
        Mach number (passed through)
    CM_eff : float
        Effective pitching moment coefficient used in My calculation

    Notation
    --------
    vx_rel, vz_rel : air-relative velocity components
    airspeed              : airspeed magnitude
    rho            : air density (ideal gas law)
    q_bar          : dynamic pressure = 0.5 * rho * airspeed^2
    flow_relative_flight_path_angle      : flow-relative flight-path angle = atan2(vz_rel, vx_rel)
    """

    # --------------------------------------------------------
    # Extract aerodynamic coefficients
    # --------------------------------------------------------
    CD = float(aerodynamic_coefficients.CD)        # drag coefficient
    CL = float(aerodynamic_coefficients.CL)        # lift coefficient
    CM0 = float(aerodynamic_coefficients.CM0)      # baseline moment coefficient
    Cm_alpha = float(aerodynamic_coefficients.Cm_alpha)  # moment slope vs. alpha
    Cmq = float(aerodynamic_coefficients.Cmq)      # pitch-rate damping coefficient

    # --------------------------------------------------------
    # Air-relative velocity (vehicle velocity relative to wind)
    # --------------------------------------------------------
    vx_rel = velocity_x - wind_velocity_x
    vz_rel = velocity_z - wind_velocity_z
    airspeed = math.hypot(vx_rel, vz_rel)

    if airspeed < MIN_AIRSPEED_FOR_FORCES:
        # If airspeed is essentially zero, aerodynamic loads are negligible.
        return 0.0, 0.0, 0.0, mach, CM0

    # --------------------------------------------------------
    # Flow-relative flight-path angle
    # --------------------------------------------------------
    flow_relative_flight_path_angle = math.atan2(vz_rel, vx_rel)

    # --------------------------------------------------------
    # Density and dynamic pressure
    # --------------------------------------------------------
    rho = static_pressure / (R_AIR * static_temperature)
    q_bar = 0.5 * rho * airspeed * airspeed

    # --------------------------------------------------------
    # Aerodynamic forces in wind axes (Drag opposite flow, Lift perpendicular)
    # --------------------------------------------------------
    drag_force = q_bar * aero_reference.reference_area * CD
    lift_force = q_bar * aero_reference.reference_area * CL

    # --------------------------------------------------------
    # Transform to inertial axes using flow_relative_flight_path_angle
    # --------------------------------------------------------
    cos_flow_angle = math.cos(flow_relative_flight_path_angle)
    sin_flow_angle = math.sin(flow_relative_flight_path_angle)

    Xv = (-drag_force * cos_flow_angle) - (lift_force * sin_flow_angle)
    Zv = (-drag_force * sin_flow_angle) + (lift_force * cos_flow_angle)

    # --------------------------------------------------------
    # Pitching moment coefficient model
    # --------------------------------------------------------
    safe_airspeed_for_damping = max(airspeed, MIN_AIRSPEED_FOR_MOMENT)

    # Cmq term uses standard normalization ~ (q * lref) / (2V)
    pitch_damping_contribution = Cmq * (pitch_rate * aero_reference.reference_length / (PITCH_DAMPING_SCALE * safe_airspeed_for_damping))

    CM_eff = (
        CM0
        + Cm_alpha * alpha
        + pitch_damping_contribution
        + CL * lcg
    )

    My = q_bar * aero_reference.reference_area * aero_reference.reference_length * CM_eff

    return Xv, Zv, My, mach, CM_eff

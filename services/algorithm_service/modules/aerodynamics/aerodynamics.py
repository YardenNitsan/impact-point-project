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
    Sref : float
        Reference area [m²]
    lref : float
        Reference length [m]
    """

    Sref: float
    lref: float


# ============================================================
# Aerodynamic force & moment model
# ============================================================

def compute_Xv_Zv_My_from_table(
    *,
    P: float,
    T: float,
    vx: float,
    vz: float,
    wind_x: float,
    wind_z: float,
    alpha: float,
    mach: float,
    coeffs,
    ref: AeroRef,
    q: float,
    lcg: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """
    Convert aerodynamic coefficients into inertial-frame forces and pitching moment.

    Parameters
    ----------
    P : float
        Static pressure [Pa]
    T : float
        Static temperature [K]
    vx, vz : float
        Vehicle inertial velocity components [m/s]
    wind_x, wind_z : float
        Wind velocity components expressed in solver axes [m/s]
        (wind is subtracted to obtain air-relative velocity)
    alpha : float
        Angle of attack [rad]
    mach : float
        Mach number (passed through for diagnostics)
    coeffs : object
        Aerodynamic coefficients container with:
        CD, CL, CM0, Cm_alpha, Cmq
    ref : AeroRef
        Aerodynamic reference geometry
    q : float
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
    V              : airspeed magnitude
    rho            : air density (ideal gas law)
    q_bar          : dynamic pressure = 0.5 * rho * V^2
    gamma_rel      : flow-relative flight-path angle = atan2(vz_rel, vx_rel)
    """

    # --------------------------------------------------------
    # Extract aerodynamic coefficients
    # --------------------------------------------------------
    CD = float(coeffs.CD)        # drag coefficient
    CL = float(coeffs.CL)        # lift coefficient
    CM0 = float(coeffs.CM0)      # baseline moment coefficient
    Cm_alpha = float(coeffs.Cm_alpha)  # moment slope vs. alpha
    Cmq = float(coeffs.Cmq)      # pitch-rate damping coefficient

    # --------------------------------------------------------
    # Air-relative velocity (vehicle velocity relative to wind)
    # --------------------------------------------------------
    vx_rel = vx - wind_x
    vz_rel = vz - wind_z
    V = math.hypot(vx_rel, vz_rel)

    if V < MIN_AIRSPEED_FOR_FORCES:
        # If airspeed is essentially zero, aerodynamic loads are negligible.
        return 0.0, 0.0, 0.0, mach, CM0

    # --------------------------------------------------------
    # Flow-relative flight-path angle
    # --------------------------------------------------------
    gamma_rel = math.atan2(vz_rel, vx_rel)

    # --------------------------------------------------------
    # Density and dynamic pressure
    # --------------------------------------------------------
    rho = P / (R_AIR * T)
    q_bar = 0.5 * rho * V * V

    # --------------------------------------------------------
    # Aerodynamic forces in wind axes (Drag opposite flow, Lift perpendicular)
    # --------------------------------------------------------
    D = q_bar * ref.Sref * CD
    L = q_bar * ref.Sref * CL

    # --------------------------------------------------------
    # Transform to inertial axes using gamma_rel
    # --------------------------------------------------------
    cos_g = math.cos(gamma_rel)
    sin_g = math.sin(gamma_rel)

    Xv = (-D * cos_g) - (L * sin_g)
    Zv = (-D * sin_g) + (L * cos_g)

    # --------------------------------------------------------
    # Pitching moment coefficient model
    # --------------------------------------------------------
    V_safe = max(V, MIN_AIRSPEED_FOR_MOMENT)

    # Cmq term uses standard normalization ~ (q * lref) / (2V)
    cmq_term = Cmq * (q * ref.lref / (PITCH_DAMPING_SCALE * V_safe))

    CM_eff = (
        CM0
        + Cm_alpha * alpha
        + cmq_term
        + CL * lcg
    )

    My = q_bar * ref.Sref * ref.lref * CM_eff

    return Xv, Zv, My, mach, CM_eff

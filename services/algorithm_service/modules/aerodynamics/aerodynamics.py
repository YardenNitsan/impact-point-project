from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

DEFAULT_GAMMA = 1.4
R_AIR = 287.05


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class AeroRef:
    Sref: float
    lref: float


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
    """Convert aerodynamic coefficients to forces and moment in inertial frame.

    This follows your project proposal requirement:
    - include Pitch Damping derivative Cmq (moment depends on pitch rate q)
    - include CG correction via +CL * lcg
    (see proposal Pitch Damping + CG correction section).
    """

    # Table object expected: AeroCoeffsTable (CD, CL, CM0, Cm_alpha, Cmq)
    CD = float(coeffs.CD)
    CL = float(coeffs.CL)
    CM0 = float(coeffs.CM0)
    Cm_alpha = float(coeffs.Cm_alpha)
    Cmq = float(coeffs.Cmq)

    # Relative velocity
    vx_rel = vx - wind_x
    vz_rel = vz - wind_z
    V = math.hypot(vx_rel, vz_rel)

    if V < 1e-6:
        return 0.0, 0.0, 0.0, mach, CM0

    # Flight-path angle of RELATIVE airflow
    gamma_rel = math.atan2(vz_rel, vx_rel)

    # Density and dynamic pressure
    rho = P / (R_AIR * T)
    qbar = 0.5 * rho * V * V

    # Aerodynamic forces in wind axes
    D = qbar * ref.Sref * CD
    L = qbar * ref.Sref * CL

    # Convert to inertial frame (x forward, z up)
    Xv = -D * math.cos(gamma_rel) - L * math.sin(gamma_rel)
    Zv = -D * math.sin(gamma_rel) + L * math.cos(gamma_rel)

    # Pitching moment coefficient with required corrections:
    V_safe = max(V, 1e-3)
    CM_eff = (
        CM0
        + Cm_alpha * alpha
        + Cmq * (q * ref.lref / (2.0 * V_safe))
        + CL * lcg
    )

    My = qbar * ref.Sref * ref.lref * CM_eff

    return Xv, Zv, My, mach, CM_eff


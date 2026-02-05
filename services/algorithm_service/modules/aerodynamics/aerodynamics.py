from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math

from modules.atmosphere.isa import speed_of_sound

DEFAULT_GAMMA = 1.4


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class AeroRef:
    Sref: float   # [m^2]
    lref: float   # [m]


def compressible_dynamic_pressure(P: float, M: float, gamma: float = DEFAULT_GAMMA) -> float:
    return 0.5 * gamma * P * (M * M)


def aerodynamic_DLM(P: float, M: float, CD: float, CL: float, CM: float, ref: AeroRef,
                    gamma: float = DEFAULT_GAMMA) -> Tuple[float, float, float]:
    qbar = compressible_dynamic_pressure(P, M, gamma)
    D = qbar * ref.Sref * CD
    L = qbar * ref.Sref * CL
    Mp = qbar * ref.Sref * ref.lref * CM
    return D, L, Mp


def effective_CM(
    *,
    CM0: float,
    CL: float,
    alpha: float,
    Cm_alpha: float,
    Cmq: float,
    q: float,
    V: float,
    ref: AeroRef,
    lcg: float = 0.0,
) -> float:
    alpha_lim = math.radians(20.0)
    alpha_eff = _clamp(alpha, -alpha_lim, alpha_lim)

    q_eff = _clamp(q, -50.0, 50.0)

    CM_static = CM0 + Cm_alpha * alpha_eff
    CM_cg = CM_static + (lcg / ref.lref) * CL

    V_eff = max(V, 10.0)
    damp = Cmq * (q_eff * ref.lref / (2.0 * V_eff))

    CM = CM_cg + damp
    CM = _clamp(CM, -1.5, 1.5)
    return CM


def compute_Xv_Zv_My_from_table(
    *,
    P: float,
    T: float,
    vx: float,
    vz: float,
    q: float,
    alpha: float,
    CD: float,
    CL: float,
    CM0: float,
    Cm_alpha: float,
    Cmq: float,
    ref: AeroRef,
    lcg: float = 0.0,
    gamma: float = DEFAULT_GAMMA
) -> Tuple[float, float, float, float, float]:
    """
    Returns aerodynamic forces in the SAME convention as simulation:
      +x forward, +z UP.

    Drag acts opposite the velocity vector.
    Lift acts perpendicular to velocity in the x-z plane (positive lift -> +z).
    """

    V = math.hypot(vx, vz)
    a = speed_of_sound(T, gamma=gamma)
    Mach = (V / a) if (a > 1e-9) else 0.0

    CM_eff = effective_CM(
        CM0=CM0,
        CL=CL,
        alpha=alpha,
        Cm_alpha=Cm_alpha,
        Cmq=Cmq,
        q=q,
        V=V,
        ref=ref,
        lcg=lcg,
    )

    D, L, My = aerodynamic_DLM(P, Mach, CD, CL, CM_eff, ref, gamma=gamma)

    if V < 1e-6:
        # basically no aerodynamic direction
        return 0.0, 0.0, My, Mach, CM_eff

    ux = vx / V
    uz = vz / V

    # Drag opposite velocity
    Fx_drag = -D * ux
    Fz_drag = -D * uz

    # Lift: perpendicular to velocity, choose direction that gives +z for positive CL
    # Perp unit in x-z plane: (-uz, +ux)
    Fx_lift = -L * uz
    Fz_lift =  L * ux

    Xv = Fx_drag + Fx_lift
    Zv = Fz_drag + Fz_lift

    return Xv, Zv, My, Mach, CM_eff

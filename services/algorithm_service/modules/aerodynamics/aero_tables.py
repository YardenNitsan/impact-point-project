# modules/aerodynamics/aero_tables.py

from __future__ import annotations
from dataclasses import dataclass
import bisect
import math
from typing import Sequence, Tuple

# ============================================================
# Numerical tolerances
# ============================================================

INTERPOLATION_EPSILON = 1e-12

# ============================================================
# Hoerner-style (high AoA / 360°) model parameters
# ============================================================
# Hoerner discusses high-angle behavior with lift component ~ k' * sin(a)*cos(a)
# and typical k'≈2 for very low aspect ratio / A≈0 case. :contentReference[oaicite:2]{index=2}
HOERNER_K_LIFT = 2.0

# Drag behavior at high AoA is dominated by normal-flow pressure drag,
# often modeled ~ sin^2(a) (classic plate/bluff-body behavior).
# We keep this as a parameter (project-dependent).
DEFAULT_CD_MIN = 0.10     # baseline drag at alpha ~ 0 (you can tune)
DEFAULT_CD_SIN2_GAIN = 1.20  # magnitude toward 90° (tunable, Hoerner discusses plate-like magnitudes) :contentReference[oaicite:3]{index=3}

# Moment model: without real data, the most defensible “simple” choice
# is neutral CM0 and user-chosen Cm_alpha, Cmq (already in your pipeline).
DEFAULT_CM0 = 0.0
DEFAULT_CM_ALPHA = -0.2
DEFAULT_CMQ = -8.0

# Angle normalization
PI = math.pi
TWO_PI = 2.0 * math.pi


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    a = (a + PI) % (TWO_PI) - PI
    return a


@dataclass(frozen=True)
class AeroCoeffsTable:
    CD: float
    CL: float
    CM0: float
    Cm_alpha: float
    Cmq: float


@dataclass(frozen=True)
class AeroTable2D:
    alpha_grid: Sequence[float]
    mach_grid: Sequence[float]
    data: Sequence[Sequence[AeroCoeffsTable]]

    def _idx_pair(self, grid: Sequence[float], x: float) -> Tuple[int, int, float]:
        if len(grid) < 2:
            raise ValueError("Grid must have at least 2 points.")

        x = _clamp(x, grid[0], grid[-1])

        i1 = bisect.bisect_right(grid, x)
        if i1 <= 0:
            return 0, 1, 0.0
        if i1 >= len(grid):
            return len(grid) - 2, len(grid) - 1, 1.0

        i0 = i1 - 1
        g0 = grid[i0]
        g1 = grid[i1]

        if abs(g1 - g0) < INTERPOLATION_EPSILON:
            return i0, i1, 0.0

        t = (x - g0) / (g1 - g0)
        return i0, i1, t

    def lookup(self, alpha: float, mach: float) -> AeroCoeffsTable:
        a0, a1, ta = self._idx_pair(self.alpha_grid, alpha)
        m0, m1, tm = self._idx_pair(self.mach_grid, mach)

        c00 = self.data[a0][m0]
        c01 = self.data[a0][m1]
        c10 = self.data[a1][m0]
        c11 = self.data[a1][m1]

        def lerp(u: float, v: float, t: float) -> float:
            return u + (v - u) * t

        def bilinear(f00: float, f01: float, f10: float, f11: float) -> float:
            f0 = lerp(f00, f01, tm)
            f1 = lerp(f10, f11, tm)
            return lerp(f0, f1, ta)

        return AeroCoeffsTable(
            CD=bilinear(c00.CD, c01.CD, c10.CD, c11.CD),
            CL=bilinear(c00.CL, c01.CL, c10.CL, c11.CL),
            CM0=bilinear(c00.CM0, c01.CM0, c10.CM0, c11.CM0),
            Cm_alpha=bilinear(c00.Cm_alpha, c01.Cm_alpha, c10.Cm_alpha, c11.Cm_alpha),
            Cmq=bilinear(c00.Cmq, c01.Cmq, c10.Cmq, c11.Cmq),
        )


def hoerner_style_coeffs_360(
    alpha: float,
    *,
    cd_min: float = DEFAULT_CD_MIN,
    cd_sin2_gain: float = DEFAULT_CD_SIN2_GAIN,
    k_lift: float = HOERNER_K_LIFT,
    cm0: float = DEFAULT_CM0,
    cm_alpha: float = DEFAULT_CM_ALPHA,
    cmq: float = DEFAULT_CMQ,
) -> AeroCoeffsTable:
    """
    Simple 360° coefficient model suitable for 3DOF projects:

      CL(alpha) ≈ k * sin(alpha) * cos(alpha)         (Hoerner-style high-AoA lift component) :contentReference[oaicite:4]{index=4}
      CD(alpha) ≈ CDmin + Kd * sin(alpha)^2           (classic bluff/plate-like pressure drag trend)

    This is not CFD; it's the most defensible “simple” model when no real tables exist.
    """

    a = wrap_to_pi(alpha)

    sa = math.sin(a)
    ca = math.cos(a)

    CL = k_lift * sa * ca
    CD = cd_min + cd_sin2_gain * (sa * sa)

    return AeroCoeffsTable(
        CD=CD,
        CL=CL,
        CM0=cm0,
        Cm_alpha=cm_alpha,
        Cmq=cmq,
    )


def build_hoerner_style_table_360(
    *,
    alpha_grid: Sequence[float],
    mach_grid: Sequence[float],
    cd_min: float = DEFAULT_CD_MIN,
    cd_sin2_gain: float = DEFAULT_CD_SIN2_GAIN,
    k_lift: float = HOERNER_K_LIFT,
    cm0: float = DEFAULT_CM0,
    cm_alpha: float = DEFAULT_CM_ALPHA,
    cmq: float = DEFAULT_CMQ,
) -> AeroTable2D:
    """
    Build a full AeroTable2D for (alpha, mach).
    In this simplified level, coefficients are Mach-independent (reasonable for low subsonic),
    so the same alpha curve is replicated across mach_grid.
    """
    data = []
    for a in alpha_grid:
        row = []
        coeff = hoerner_style_coeffs_360(
            a,
            cd_min=cd_min,
            cd_sin2_gain=cd_sin2_gain,
            k_lift=k_lift,
            cm0=cm0,
            cm_alpha=cm_alpha,
            cmq=cmq,
        )
        for _m in mach_grid:
            row.append(coeff)
        data.append(row)

    return AeroTable2D(alpha_grid=list(alpha_grid), mach_grid=list(mach_grid), data=data)


# ============================================================
# Default demo table (used by simulated_impact)
# ============================================================

# ============================================================
# Default grids (used by default_demo_table)
# ============================================================

DEFAULT_ALPHA_GRID = [math.radians(a) for a in range(-180, 181, 5)]  # 5° resolution, full 360°
DEFAULT_MACH_GRID = [0.0, 0.3, 0.6, 0.9]

def default_demo_table():
    """Build a default Hoerner-style 360° aerodynamic table.

    Used as a robust fallback/demo aerodynamic model when you don't have CFD/DatCOM tables.
    """
    return build_hoerner_style_table_360(
        alpha_grid=DEFAULT_ALPHA_GRID,
        mach_grid=DEFAULT_MACH_GRID,
    )

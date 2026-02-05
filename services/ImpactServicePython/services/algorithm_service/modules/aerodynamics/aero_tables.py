from __future__ import annotations
from dataclasses import dataclass
import bisect
import math
from typing import List, Sequence, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class AeroCoeffsTable:
    """
    Aerodynamic coefficients at a specific (alpha, mach) cell.
    CM0 is the base pitching moment coefficient (before CG correction and damping).
    """
    CD: float
    CL: float
    CM0: float
    Cm_alpha: float
    Cmq: float


@dataclass(frozen=True)
class AeroTable2D:
    """
    2D grid table for coefficients versus (alpha, mach).
    alpha_grid: radians, ascending (e.g. [-0.2, 0.0, 0.2])
    mach_grid : ascending (e.g. [0.5, 1.0, 2.0])
    data[ia][im] = AeroCoeffsTable
    """
    alpha_grid: Sequence[float]
    mach_grid: Sequence[float]
    data: Sequence[Sequence[AeroCoeffsTable]]

    def _idx_pair(self, grid: Sequence[float], x: float) -> Tuple[int, int, float]:
        """
        Returns (i0, i1, t) where x lies between grid[i0] and grid[i1],
        and t in [0,1] is the interpolation weight.
        """
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
        if abs(g1 - g0) < 1e-12:
            return i0, i1, 0.0
        t = (x - g0) / (g1 - g0)
        return i0, i1, t

    def lookup(self, alpha: float, mach: float) -> AeroCoeffsTable:
        """
        Bilinear interpolation.
        """
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


def default_demo_table() -> AeroTable2D:
    alphas = [-0.2, -0.1, 0.0, 0.1, 0.2]
    machs = [0.3, 0.8, 1.2, 2.0]

    CL_alpha = 6.0
    CD0 = 0.12
    k = 0.6
    Cm_alpha = -0.4
    Cmq = -8.0

    data = []

    for a in alphas:
        row = []
        CL = CL_alpha * a
        CD = CD0 + k * CL * CL

        for M in machs:
            row.append(
                AeroCoeffsTable(
                    CD=CD,
                    CL=CL,
                    CM0=0.0,
                    Cm_alpha=Cm_alpha,
                    Cmq=Cmq
                )
            )
        data.append(row)

    return AeroTable2D(alpha_grid=alphas, mach_grid=machs, data=data)


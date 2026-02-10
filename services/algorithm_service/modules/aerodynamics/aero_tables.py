"""
Aerodynamic lookup and simplified coefficient model
for planar 3DOF flight simulation.

This module provides:

1. A 2D aerodynamic coefficient lookup table with bilinear interpolation.
2. A simplified Hoerner-style fallback model for cases where
   measured aerodynamic data is unavailable.

The fallback model approximates lift and drag behavior over a full
360° angle-of-attack range using trigonometric functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


# ============================================================
# Numerical constants
# ============================================================

INTERPOLATION_EPSILON: float = 1e-12
"""Tolerance used to avoid division-by-zero in interpolation."""


# ============================================================
# Simplified aerodynamic model parameters
# ============================================================

HOERNER_K_LIFT: float = 2.0
"""Lift gain factor in the simplified Hoerner-style model."""

DEFAULT_CD_MIN: float = 0.10
"""Baseline drag coefficient near zero angle of attack."""

DEFAULT_CD_SIN2_GAIN: float = 1.20
"""Drag growth factor at high angle of attack."""

DEFAULT_CM0: float = 0.0
"""Baseline pitching moment coefficient."""

DEFAULT_CM_ALPHA: float = -0.2
"""Pitching moment sensitivity to angle of attack."""

DEFAULT_CMQ: float = -8.0
"""Pitch-rate damping coefficient."""


# ============================================================
# Mathematical constants
# ============================================================

PI: float = math.pi
TWO_PI: float = 2.0 * math.pi


# ============================================================
# Utility functions
# ============================================================

def clamp(value: float, lower: float, upper: float) -> float:
    """
    Clamp a scalar value to the inclusive range [lower, upper].
    """
    return max(lower, min(upper, value))


def wrap_to_pi(angle: float) -> float:
    """
    Normalize an angle to the interval (-π, π].

    This ensures consistent angular representation.
    """
    return (angle + PI) % TWO_PI - PI


def binary_search_right(grid: Sequence[float], x: float) -> int:
    """
    Binary search equivalent to bisect.bisect_right.

    Parameters
    ----------
    grid : Sequence[float]
        Sorted ascending grid.
    x : float
        Query value.

    Returns
    -------
    int
        Index i such that grid[i-1] <= x < grid[i].
    """
    lo = 0
    hi = len(grid)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < grid[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo


# ============================================================
# Aerodynamic data structures
# ============================================================

@dataclass(frozen=True)
class AeroCoeffsTable:
    """
    Aerodynamic coefficient container.

    Attributes
    ----------
    CD : float
        Drag coefficient.
    CL : float
        Lift coefficient.
    CM0 : float
        Baseline pitching moment.
    Cm_alpha : float
        Pitching moment slope vs. angle of attack.
    Cmq : float
        Pitch-rate damping coefficient.
    """

    CD: float
    CL: float
    CM0: float
    Cm_alpha: float
    Cmq: float


@dataclass(frozen=True)
class AeroTable2D:
    """
    Two-dimensional aerodynamic lookup table.

    Parameters
    ----------
    alpha_grid : Sequence[float]
        Angle-of-attack grid.
    mach_grid : Sequence[float]
        Mach number grid.
    data : Sequence[Sequence[AeroCoeffsTable]]
        Coefficient table indexed by (alpha, mach).
    """

    alpha_grid: Sequence[float]
    mach_grid: Sequence[float]
    data: Sequence[Sequence[AeroCoeffsTable]]

    # --------------------------------------------------------

    def idx_pair(
        self,
        grid: Sequence[float],
        x: float
    ) -> Tuple[int, int, float]:
        """
        Compute interpolation interval and weight.

        Parameters
        ----------
        grid : Sequence[float]
            Sorted interpolation grid.
        x : float
            Query value.

        Returns
        -------
        i0 : int
            Lower index.
        i1 : int
            Upper index.
        t : float
            Linear interpolation fraction.

        Notation
        --------
        g0, g1 : bounding grid values
        t      : (x - g0) / (g1 - g0)
        """

        if len(grid) < 2:
            raise ValueError("Grid must contain at least two points.")

        x = clamp(x, grid[0], grid[-1])

        i1 = binary_search_right(grid, x)

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

    # --------------------------------------------------------

    def lookup(self, alpha: float, mach: float) -> AeroCoeffsTable:
        """
        Perform bilinear interpolation in the aerodynamic table.

        Notation
        --------
        a0, a1 : alpha indices
        m0, m1 : mach indices
        ta, tm : interpolation fractions
        """

        a0, a1, ta = self.idx_pair(self.alpha_grid, alpha)
        m0, m1, tm = self.idx_pair(self.mach_grid, mach)

        c00 = self.data[a0][m0]
        c01 = self.data[a0][m1]
        c10 = self.data[a1][m0]
        c11 = self.data[a1][m1]

        def lerp(u: float, v: float, t: float) -> float:
            """Linear interpolation."""
            return u + (v - u) * t

        def bilinear(f00: float, f01: float, f10: float, f11: float) -> float:
            """Bilinear interpolation."""
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


# ============================================================
# Simplified Hoerner-style fallback model
# ============================================================

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
    Generate simplified aerodynamic coefficients.

    Notation
    --------
    a  : wrapped angle of attack
    sa : sin(a)
    ca : cos(a)

    Model
    -----
    CL = k * sin(a) * cos(a)
    CD = CD_min + K * sin²(a)
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
    Construct a full aerodynamic lookup table from the fallback model.
    """

    data = []

    for a in alpha_grid:
        coeff = hoerner_style_coeffs_360(
            a,
            cd_min=cd_min,
            cd_sin2_gain=cd_sin2_gain,
            k_lift=k_lift,
            cm0=cm0,
            cm_alpha=cm_alpha,
            cmq=cmq,
        )

        row = [coeff for _ in mach_grid]
        data.append(row)

    return AeroTable2D(
        alpha_grid=list(alpha_grid),
        mach_grid=list(mach_grid),
        data=data,
    )


# ============================================================
# Default demo configuration
# ============================================================

DEFAULT_ALPHA_GRID = [math.radians(a) for a in range(-180, 181, 5)]
DEFAULT_MACH_GRID = [0.0, 0.3, 0.6, 0.9]


def default_demo_table() -> AeroTable2D:
    """
    Create a default aerodynamic lookup table.
    """
    return build_hoerner_style_table_360(
        alpha_grid=DEFAULT_ALPHA_GRID,
        mach_grid=DEFAULT_MACH_GRID,
    )

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

PI: float = math.pi
TWO_PI: float = 2.0 * math.pi
"""Both are for math calculations"""


def clamp(value: float, lower: float, upper: float) -> float:
    """
    Clamp a scalar value to the inclusive range [lower, upper].
    """
    return max(lower, min(upper, value))


def wrap_to_pi(angle_radians: float) -> float:
    """
    Normalize an angle to the interval (-π, π].
    This ensures consistent angular representation.
    """
    return (angle_radians + PI) % TWO_PI - PI


def binary_search_right(sorted_grid: Sequence[float], query_value: float) -> int:
    """
    Binary search equivalent to bisect.bisect_right.

    Parameters
    ----------
    sorted_grid : Sequence[float]
        Sorted ascending grid.
    query_value : float
        Query value.

    Returns
    -------
    int
        Index i such that grid[i-1] <= x < grid[i].
    """
    left_index = 0
    right_index = len(sorted_grid)

    while left_index < right_index:
        middle_index = (left_index + right_index) // 2
        if query_value < sorted_grid[middle_index]:
            right_index = middle_index
        else:
            left_index = middle_index + 1

    return left_index


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


    def idx_pair(
        self,
        interpolation_grid: Sequence[float],
        query_value: float
    ) -> Tuple[int, int, float]:
        """
        Compute interpolation interval and weight.

        Parameters
        ----------
        interpolation_grid : Sequence[float]
            Sorted interpolation grid.
        query_value : float
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

        if len(interpolation_grid) < 2:
            raise ValueError("Grid must contain at least two points.")

        query_value = clamp(query_value, interpolation_grid[0], interpolation_grid[-1])

        upper_index = binary_search_right(interpolation_grid, query_value)

        if upper_index <= 0:
            return 0, 1, 0.0

        if upper_index >= len(interpolation_grid):
            return len(interpolation_grid) - 2, len(interpolation_grid) - 1, 1.0

        lower_index = upper_index - 1
        lower_grid_value = interpolation_grid[lower_index]
        upper_grid_value = interpolation_grid[upper_index]

        if abs(upper_grid_value - lower_grid_value) < INTERPOLATION_EPSILON:
            return lower_index, upper_index, 0.0

        interpolation_weight = (query_value - lower_grid_value) / (upper_grid_value - lower_grid_value)
        return lower_index, upper_index, interpolation_weight


    def lookup(self, alpha: float, mach: float) -> AeroCoeffsTable:
        """
        Perform bilinear interpolation in the aerodynamic table.

        Notation
        --------
        alpha_lower_index, alpha_upper_index : alpha indices
        mach_lower_index, mach_upper_index : mach indices
        alpha_interpolation_weight, mach_interpolation_weight : interpolation fractions
        """

        alpha_lower_index, alpha_upper_index, alpha_interpolation_weight = self.idx_pair(self.alpha_grid, alpha)
        mach_lower_index, mach_upper_index, mach_interpolation_weight = self.idx_pair(self.mach_grid, mach)

        coeff_alpha0_mach0 = self.data[alpha_lower_index][mach_lower_index]
        coeff_alpha0_mach1 = self.data[alpha_lower_index][mach_upper_index]
        coeff_alpha1_mach0 = self.data[alpha_upper_index][mach_lower_index]
        coeff_alpha1_mach1 = self.data[alpha_upper_index][mach_upper_index]

        def lerp(start_value: float, end_value: float, interpolation_weight: float) -> float:
            """Linear interpolation."""
            return start_value + (end_value - start_value) * interpolation_weight

        def bilinear(value_alpha0_mach0: float, value_alpha0_mach1: float, value_alpha1_mach0: float, value_alpha1_mach1: float) -> float:
            """Bilinear interpolation."""
            interpolated_alpha0_value = lerp(value_alpha0_mach0, value_alpha0_mach1, mach_interpolation_weight)
            interpolated_alpha1_value = lerp(value_alpha1_mach0, value_alpha1_mach1, mach_interpolation_weight)
            return lerp(interpolated_alpha0_value, interpolated_alpha1_value, alpha_interpolation_weight)

        return AeroCoeffsTable(
            CD=bilinear(coeff_alpha0_mach0.CD, coeff_alpha0_mach1.CD, coeff_alpha1_mach0.CD, coeff_alpha1_mach1.CD),
            CL=bilinear(coeff_alpha0_mach0.CL, coeff_alpha0_mach1.CL, coeff_alpha1_mach0.CL, coeff_alpha1_mach1.CL),
            CM0=bilinear(coeff_alpha0_mach0.CM0, coeff_alpha0_mach1.CM0, coeff_alpha1_mach0.CM0, coeff_alpha1_mach1.CM0),
            Cm_alpha=bilinear(coeff_alpha0_mach0.Cm_alpha, coeff_alpha0_mach1.Cm_alpha, coeff_alpha1_mach0.Cm_alpha, coeff_alpha1_mach1.Cm_alpha),
            Cmq=bilinear(coeff_alpha0_mach0.Cmq, coeff_alpha0_mach1.Cmq, coeff_alpha1_mach0.Cmq, coeff_alpha1_mach1.Cmq),
        )


# ============================================================
# Hoerner-style fallback model
# ============================================================

def compute_hoerner_aerodynamic_coefficients_full_alpha(
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
    wrapped_alpha  : wrapped angle of attack
    sin_alpha : sin(wrapped_alpha)
    cos_alpha : cos(wrapped_alpha)

    Model
    -----
    CL = k * sin(wrapped_alpha) * cos(wrapped_alpha)
    CD = CD_min + K * sin²(wrapped_alpha)
    """

    wrapped_alpha = wrap_to_pi(alpha)

    sin_alpha = math.sin(wrapped_alpha)
    cos_alpha = math.cos(wrapped_alpha)

    CL = k_lift * sin_alpha * cos_alpha
    CD = cd_min + cd_sin2_gain * (sin_alpha * sin_alpha)

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
    Construct alpha_value full aerodynamic lookup table from the fallback model.
    """

    aerodynamic_table_data = []

    for alpha_value in alpha_grid:
        aerodynamic_coefficients = compute_hoerner_aerodynamic_coefficients_full_alpha(
            alpha_value,
            cd_min=cd_min,
            cd_sin2_gain=cd_sin2_gain,
            k_lift=k_lift,
            cm0=cm0,
            cm_alpha=cm_alpha,
            cmq=cmq,
        )

        mach_row_coefficients = [aerodynamic_coefficients for mach_value in mach_grid]
        aerodynamic_table_data.append(mach_row_coefficients)

    return AeroTable2D(
        alpha_grid=list(alpha_grid),
        mach_grid=list(mach_grid),
        data=aerodynamic_table_data,
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

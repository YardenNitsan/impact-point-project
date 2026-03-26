"""
Deterministic along-track wind shear model.

This module implements a power-law wind shear profile derived
from two reference wind measurements (10 m and 100 m). ENU wind
vectors are projected onto a trajectory-aligned axis and interpolated
as a function of altitude.

Wind shear model
----------------

The altitude-dependent wind speed follows a power-law relation:

    V(h) = V_ref (h / h_ref)^α

where:

    h     : altitude
    V_ref : reference wind speed
    α     : shear exponent

The implementation is deterministic, numerically stable,
and designed for 3DOF flight simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Wind profile model constants
# ============================================================

REFERENCE_HEIGHT_LOW_M: float = 10.0
"""Lower reference measurement height [m]."""

REFERENCE_HEIGHT_HIGH_M: float = 100.0
"""Upper reference measurement height [m]."""

MIN_VALID_HEIGHT_M: float = 1.0
"""Minimum altitude used by the wind model [m]."""

MIN_NONZERO_WIND_SPEED_MPS: float = 0.1
"""Minimum wind magnitude to avoid singularities [m/s]."""

MIN_SHEAR_EXPONENT: float = -1.0
MAX_SHEAR_EXPONENT: float = 1.0
"""Physical bounds on the wind shear exponent."""


# ============================================================
# Metadata container
# ============================================================

@dataclass(frozen=True)
class WindProfileMeta:
    """
    Diagnostic metadata describing the wind profile.

    Attributes
    ----------
    wind_along_track_10m_mps : float
        Along-track wind at 10 m
    wind_along_track_100m_mps : float
        Along-track wind at 100 m
    shear_exponent : float
        Estimated power-law shear exponent
    """

    wind_along_track_10m_mps: float
    wind_along_track_100m_mps: float
    shear_exponent: float


# ============================================================
# Along-track wind shear model
# ============================================================

class AlongTrackWindShearModel:
    """
    Deterministic altitude-dependent wind model.

    ENU wind vectors are projected onto the trajectory axis
    defined by azimuth angle:

        V_track = V_east sin(ψ) + V_north cos(ψ)

    where ψ is the trajectory azimuth.

    A power-law shear model is then applied to compute
    altitude-dependent wind.
    """

    def __init__(
        self,
        *,
        azimuth_rad: float,
        wind_east_10m_mps: float,
        wind_north_10m_mps: float,
        wind_east_100m_mps: float,
        wind_north_100m_mps: float,
    ) -> None:

        self.azimuth_rad = float(azimuth_rad)

        self._wind_along_10m_mps = self._project_to_track(
            wind_east_10m_mps,
            wind_north_10m_mps,
        )

        self._wind_along_100m_mps = self._project_to_track(
            wind_east_100m_mps,
            wind_north_100m_mps,
        )

        self._shear_exponent = self._compute_shear_exponent(
            self._wind_along_10m_mps,
            self._wind_along_100m_mps,
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    @staticmethod
    def _compute_shear_exponent(
        wind_10m_mps: float,
        wind_100m_mps: float,
    ) -> float:
        """
        Estimate the power-law wind shear exponent α.
        """

        v10 = max(abs(wind_10m_mps), MIN_NONZERO_WIND_SPEED_MPS)
        v100 = max(abs(wind_100m_mps), MIN_NONZERO_WIND_SPEED_MPS)

        height_ratio = (
            REFERENCE_HEIGHT_HIGH_M
            / REFERENCE_HEIGHT_LOW_M
        )

        alpha = (
            math.log(v100 / v10)
            / math.log(height_ratio)
        )

        return max(
            MIN_SHEAR_EXPONENT,
            min(MAX_SHEAR_EXPONENT, alpha),
        )

    # --------------------------------------------------------

    def _project_to_track(
        self,
        wind_east_mps: float,
        wind_north_mps: float,
    ) -> float:
        """
        Project ENU wind vector onto trajectory axis.
        """

        unit_e = math.sin(self.azimuth_rad)
        unit_n = math.cos(self.azimuth_rad)

        return (
            float(wind_east_mps) * unit_e
            + float(wind_north_mps) * unit_n
        )

    # --------------------------------------------------------
    # Public interface
    # --------------------------------------------------------

    def compute_wind_at_altitude(
        self,
        altitude_m: float,
    ) -> Tuple[float, float]:
        """
        Compute wind at a given altitude.

        Returns
        -------
        wind_x : float
            Along-track wind [m/s]
        wind_z : float
            Vertical wind (zero in this model)
        """

        effective_altitude_m = max(float(altitude_m), MIN_VALID_HEIGHT_M)

        reference_wind_direction_sign = 1.0 if self._wind_along_10m_mps >= 0.0 else -1.0

        reference_wind_magnitude = max(
            abs(self._wind_along_10m_mps),
            MIN_NONZERO_WIND_SPEED_MPS,
        )

        along_track_wind_speed = (
            reference_wind_direction_sign
            * reference_wind_magnitude
            * (effective_altitude_m / REFERENCE_HEIGHT_LOW_M) ** self._shear_exponent
        )

        return along_track_wind_speed, 0.0

    # --------------------------------------------------------

    def get_profile_metadata(self) -> WindProfileMeta:
        """
        Return diagnostic wind profile metadata.
        """

        return WindProfileMeta(
            wind_along_track_10m_mps=self._wind_along_10m_mps,
            wind_along_track_100m_mps=self._wind_along_100m_mps,
            shear_exponent=self._shear_exponent,
        )

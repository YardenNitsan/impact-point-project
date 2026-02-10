"""
International Standard Atmosphere (ISA) troposphere model
and thermodynamic utilities.

This module implements a deterministic ISA atmospheric model
valid in the troposphere (0–11 km) and provides the speed-of-sound
relation used in 3DOF flight simulation.

ISA troposphere model
---------------------

Temperature:
    T(h) = T0 − L h

Pressure:
    P(h) = P0 (T/T0)^n

Density:
    rho = P / (R T)

where:

    h  : altitude
    L  : temperature lapse rate
    R  : specific gas constant
    n  : pressure exponent
"""

from __future__ import annotations

import math
from typing import Tuple


# ============================================================
# Physical constants (ISA troposphere)
# ============================================================

STANDARD_GRAVITY_ACCELERATION_MPS2: float = 9.81
"""Standard gravitational acceleration [m/s²]."""

AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K: float = 287.05
"""Specific gas constant for dry air [J/(kg·K)]."""

TROPOSPHERE_TEMPERATURE_LAPSE_RATE_K_PER_M: float = 0.0065
"""ISA tropospheric temperature lapse rate [K/m]."""

TROPOSPHERE_MAX_VALID_ALTITUDE_M: float = 11_000.0
"""Maximum altitude where the troposphere model is valid [m]."""


# ============================================================
# Derived ISA constants
# ============================================================

ISA_PRESSURE_EXPONENT: float = (
    STANDARD_GRAVITY_ACCELERATION_MPS2
    / (
        AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K
        * TROPOSPHERE_TEMPERATURE_LAPSE_RATE_K_PER_M
    )
)
"""Exponent used in the ISA pressure–altitude relation."""


# ============================================================
# Numerical bounds
# ============================================================

MIN_VALID_ALTITUDE_M: float = 0.0
"""Minimum supported altitude [m]."""


# ============================================================
# ISA troposphere model
# ============================================================

def isa_atmosphere(
    altitude_m: float,
    sea_level_temperature_K: float,
    sea_level_pressure_Pa: float,
) -> Tuple[float, float, float]:
    """
    Compute ISA atmospheric state in the troposphere.

    Parameters
    ----------
    altitude_m : float
        Geometric altitude [m]
    sea_level_temperature_K : float
        Sea-level temperature reference [K]
    sea_level_pressure_Pa : float
        Sea-level pressure reference [Pa]

    Returns
    -------
    temperature_K : float
        Atmospheric temperature [K]
    pressure_Pa : float
        Atmospheric pressure [Pa]
    density_kg_per_m3 : float
        Air density [kg/m³]

    Notation
    --------
    h   : altitude
    T0  : sea-level temperature
    P0  : sea-level pressure
    L   : lapse rate
    rho : air density
    """

    h = max(
        MIN_VALID_ALTITUDE_M,
        min(float(altitude_m), TROPOSPHERE_MAX_VALID_ALTITUDE_M),
    )

    T = (
        sea_level_temperature_K
        - TROPOSPHERE_TEMPERATURE_LAPSE_RATE_K_PER_M * h
    )

    P = sea_level_pressure_Pa * (
        T / sea_level_temperature_K
    ) ** ISA_PRESSURE_EXPONENT

    rho = P / (
        AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K * T
    )

    return T, P, rho


# ============================================================
# Speed of sound model
# ============================================================

DEFAULT_AIR_HEAT_CAPACITY_RATIO: float = 1.4
"""Ratio of specific heats for air (gamma)."""


def speed_of_sound(
    temperature_K: float,
    heat_capacity_ratio: float = DEFAULT_AIR_HEAT_CAPACITY_RATIO,
    gas_constant: float = AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K,
) -> float:
    """
    Compute speed of sound in air.

    Formula
    -------
        a = sqrt(gamma * R * T)

    Parameters
    ----------
    temperature_K : float
        Air temperature [K]
    heat_capacity_ratio : float, optional
        Ratio of specific heats (gamma)
    gas_constant : float, optional
        Specific gas constant

    Returns
    -------
    float
        Speed of sound [m/s]
    """

    return math.sqrt(
        heat_capacity_ratio
        * gas_constant
        * temperature_K
    )

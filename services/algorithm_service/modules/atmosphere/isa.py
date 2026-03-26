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

STRATOSPHERE_BASE_ALTITUDE_M: float = 11_000.0
"""Altitude where the troposphere transitions to the stratosphere [m]."""

MAX_MODEL_ALTITUDE_M: float = 20_000.0
"""Maximum altitude supported by the ISA model [m]."""

STRATOSPHERE_TEMPERATURE_K: float = 216.65
"""ISA temperature in the lower stratosphere (11–20 km) [K]."""


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

def compute_isa_atmosphere_state(
    altitude_m: float,
    sea_level_temperature_K: float,
    sea_level_pressure_Pa: float,
) -> Tuple[float, float, float]:

    h = max(0.0, min(float(altitude_m), MAX_MODEL_ALTITUDE_M))

    # -------------------------------------------------
    # Troposphere (0–11 km)
    # -------------------------------------------------
    if h <= STRATOSPHERE_BASE_ALTITUDE_M:

        temperature_K = (
            sea_level_temperature_K
            - TROPOSPHERE_TEMPERATURE_LAPSE_RATE_K_PER_M * h
        )

        pressure_Pa = sea_level_pressure_Pa * (
            temperature_K / sea_level_temperature_K
        ) ** ISA_PRESSURE_EXPONENT

    # -------------------------------------------------
    # Lower Stratosphere (11–20 km)
    # -------------------------------------------------
    else:

        # Conditions at 11 km
        T11 = (
            sea_level_temperature_K
            - TROPOSPHERE_TEMPERATURE_LAPSE_RATE_K_PER_M
            * STRATOSPHERE_BASE_ALTITUDE_M
        )

        P11 = sea_level_pressure_Pa * (
            T11 / sea_level_temperature_K
        ) ** ISA_PRESSURE_EXPONENT

        temperature_K = STRATOSPHERE_TEMPERATURE_K

        pressure_Pa = P11 * math.exp(
            -STANDARD_GRAVITY_ACCELERATION_MPS2
            * (h - STRATOSPHERE_BASE_ALTITUDE_M)
            / (
                AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K
                * temperature_K
            )
        )

    rho = pressure_Pa / (
        AIR_SPECIFIC_GAS_CONSTANT_J_PER_KG_K
        * temperature_K
    )

    return temperature_K, pressure_Pa, rho


# ============================================================
# Speed of sound model
# ============================================================

DEFAULT_AIR_HEAT_CAPACITY_RATIO: float = 1.4
"""Ratio of specific heats for air (gamma)."""


def compute_speed_of_sound(
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

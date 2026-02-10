"""
Environmental data interface for planar 3DOF flight simulation.

This module retrieves atmospheric conditions from the Open-Meteo API
and converts them into SI units suitable for simulation. If the API
fails or returns invalid data, a deterministic ISA sea-level fallback
model is used.

The interface guarantees:

- Physically bounded outputs
- Deterministic fallback behavior
- Consistent SI unit representation

Wind vectors are converted from meteorological convention
(direction FROM) to ENU solver coordinates (direction TO).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import requests


# ============================================================
# Unit conversion constants
# ============================================================

CELSIUS_TO_KELVIN_OFFSET_K: float = 273.15
"""Offset used to convert Celsius to Kelvin."""

HECTOPASCAL_TO_PASCAL_FACTOR: float = 100.0
"""Conversion factor from hectopascal (hPa) to pascal (Pa)."""


# ============================================================
# ISA fallback environment (sea level)
# ============================================================

ISA_FALLBACK_TEMPERATURE_K: float = 288.15
"""Standard ISA sea-level temperature [K]."""

ISA_FALLBACK_PRESSURE_PA: float = 101_325.0
"""Standard ISA sea-level pressure [Pa]."""

CALM_WIND_SPEED_MPS: float = 0.0
"""Wind speed used for fallback conditions [m/s]."""


# ============================================================
# API configuration
# ============================================================

OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
"""Open-Meteo forecast endpoint."""

API_TIMEOUT_SECONDS: float = 5.0
"""Maximum allowed API request duration [s]."""

REQUESTED_WIND_SPEED_UNIT: str = "m/s"
REQUESTED_PRESSURE_UNIT: str = "hPa"
REQUESTED_TEMPERATURE_UNIT: str = "celsius"


# ============================================================
# Physical sanity bounds
# ============================================================

MIN_REASONABLE_TEMPERATURE_K: float = 150.0
MAX_REASONABLE_TEMPERATURE_K: float = 350.0

MIN_REASONABLE_PRESSURE_PA: float = 50_000.0
MAX_REASONABLE_PRESSURE_PA: float = 120_000.0


# ============================================================
# Required API fields
# ============================================================

REQUIRED_CURRENT_FIELDS: Tuple[str, ...] = (
    "temperature_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
)


# ============================================================
# Data container
# ============================================================

@dataclass(frozen=True)
class CurrentConditions:
    """
    Atmospheric conditions used by the simulation.

    Attributes
    ----------
    sea_level_temperature_K : float
        Sea-level temperature [K]
    sea_level_pressure_Pa : float
        Sea-level pressure [Pa]
    wind_east_10m_mps : float
        Eastward wind component at 10 m [m/s]
    wind_north_10m_mps : float
        Northward wind component at 10 m [m/s]
    wind_east_100m_mps : float
        Eastward wind component at 100 m [m/s]
    wind_north_100m_mps : float
        Northward wind component at 100 m [m/s]
    data_source : str
        Source label ("open-meteo" or "isa-fallback")
    diagnostic_note : str
        Optional diagnostic information
    """

    sea_level_temperature_K: float
    sea_level_pressure_Pa: float

    wind_east_10m_mps: float
    wind_north_10m_mps: float
    wind_east_100m_mps: float
    wind_north_100m_mps: float

    data_source: str
    diagnostic_note: str = ""


# ============================================================
# Wind conversion helper
# ============================================================

FULL_CIRCLE_DEGREES: float = 360.0
WIND_DIRECTION_OFFSET_DEGREES: float = 180.0


def _convert_meteorological_wind_to_enu(
    wind_speed_mps: float,
    direction_from_degrees: float,
) -> Tuple[float, float]:
    """
    Convert meteorological wind to ENU vector components.

    Parameters
    ----------
    wind_speed_mps : float
        Wind speed magnitude [m/s]
    direction_from_degrees : float
        Meteorological wind direction (degrees FROM)

    Returns
    -------
    east : float
        Eastward wind component [m/s]
    north : float
        Northward wind component [m/s]

    Notes
    -----
    Meteorological wind direction specifies the direction
    FROM which wind originates. Adding 180° converts it
    to the direction TO which the wind flows.
    """

    direction_to_degrees = (
        float(direction_from_degrees)
        + WIND_DIRECTION_OFFSET_DEGREES
    ) % FULL_CIRCLE_DEGREES

    direction_to_radians = math.radians(direction_to_degrees)

    east = float(wind_speed_mps) * math.sin(direction_to_radians)
    north = float(wind_speed_mps) * math.cos(direction_to_radians)

    return east, north


# ============================================================
# Fallback builder
# ============================================================

FALLBACK_SOURCE_LABEL: str = "isa-fallback"
API_SOURCE_LABEL: str = "open-meteo"


def _build_fallback_conditions(note: str) -> CurrentConditions:
    """
    Construct deterministic ISA fallback conditions.
    """

    return CurrentConditions(
        sea_level_temperature_K=ISA_FALLBACK_TEMPERATURE_K,
        sea_level_pressure_Pa=ISA_FALLBACK_PRESSURE_PA,
        wind_east_10m_mps=CALM_WIND_SPEED_MPS,
        wind_north_10m_mps=CALM_WIND_SPEED_MPS,
        wind_east_100m_mps=CALM_WIND_SPEED_MPS,
        wind_north_100m_mps=CALM_WIND_SPEED_MPS,
        data_source=FALLBACK_SOURCE_LABEL,
        diagnostic_note=note,
    )


# ============================================================
# Main API fetch function
# ============================================================

def fetch_current_conditions(
    latitude: float,
    longitude: float,
) -> CurrentConditions:
    """
    Retrieve atmospheric conditions from Open-Meteo.

    Any API failure or physically invalid value triggers
    deterministic ISA fallback behavior.
    """

    request_parameters: Dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "current": list(REQUIRED_CURRENT_FIELDS),
        "windspeed_unit": REQUESTED_WIND_SPEED_UNIT,
        "pressure_unit": REQUESTED_PRESSURE_UNIT,
        "temperature_unit": REQUESTED_TEMPERATURE_UNIT,
    }

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=request_parameters,
            timeout=API_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        payload = response.json()
        current = payload.get("current", {})

        for field in REQUIRED_CURRENT_FIELDS:
            if field not in current:
                raise ValueError(f"Missing API field: {field}")

        temp_C = float(current["temperature_2m"])
        pressure_hPa = float(current["surface_pressure"])

        temp_K = temp_C + CELSIUS_TO_KELVIN_OFFSET_K
        pressure_Pa = pressure_hPa * HECTOPASCAL_TO_PASCAL_FACTOR

        if not (MIN_REASONABLE_TEMPERATURE_K <= temp_K <= MAX_REASONABLE_TEMPERATURE_K):
            return _build_fallback_conditions("Temperature out of bounds")

        if not (MIN_REASONABLE_PRESSURE_PA <= pressure_Pa <= MAX_REASONABLE_PRESSURE_PA):
            return _build_fallback_conditions("Pressure out of bounds")

        east_10, north_10 = _convert_meteorological_wind_to_enu(
            float(current["wind_speed_10m"]),
            float(current["wind_direction_10m"]),
        )

        east_100, north_100 = _convert_meteorological_wind_to_enu(
            float(current["wind_speed_100m"]),
            float(current["wind_direction_100m"]),
        )

        return CurrentConditions(
            sea_level_temperature_K=temp_K,
            sea_level_pressure_Pa=pressure_Pa,
            wind_east_10m_mps=east_10,
            wind_north_10m_mps=north_10,
            wind_east_100m_mps=east_100,
            wind_north_100m_mps=north_100,
            data_source=API_SOURCE_LABEL,
        )

    except Exception as error:
        # Broad exception handling is intentional:
        # any API/network/format failure triggers deterministic fallback
        return _build_fallback_conditions(f"API failure: {error}")


# ============================================================
# Backward-compatible helper
# ============================================================

def get_sea_level_environment(
    latitude: float,
    longitude: float,
) -> Tuple[float, float]:
    """
    Return sea-level temperature and pressure in SI units.
    """

    conditions = fetch_current_conditions(latitude, longitude)

    return (
        conditions.sea_level_temperature_K,
        conditions.sea_level_pressure_Pa,
    )

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class WeatherRequest(BaseModel):
    """
    Canonical request that this weather service receives from physics.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)

    alt_m: float = Field(
        ...,
        validation_alias=AliasChoices("alt_m", "alt", "altitude"),
    )

    sim_datetime: datetime | None = Field(
        default=None,
        validation_alias=AliasChoices("sim_datetime", "timestamp", "datetime"),
    )

    source: Literal["api", "machine"] = Field(
        ...,
        validation_alias=AliasChoices("source", "weather_source"),
    )

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class WeatherResponse(BaseModel):
    """
    Canonical response returned to physics.
    Physics should depend only on this schema.
    """

    temperature_K: float
    pressure_Pa: float
    wind_east_mps: float
    wind_north_mps: float
    provider_used: Literal["api", "machine"]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
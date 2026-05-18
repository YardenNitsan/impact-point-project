from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Same envelope the MLP service exposes — keeps the upstream contract aligned.
ALTITUDE_TRAIN_MIN_M = 0.0
ALTITUDE_TRAIN_MAX_M = 32_000.0


class WeatherValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_K: float
    pressure_Pa: float
    wind_u_east_mps: float
    wind_v_north_mps: float


class PhysicsWeatherRequest(BaseModel):
    """The contract used by ``weather_service`` when it routes ``source=knn``.

    Matches the shape the MLP service accepts on the same path, so the
    upstream provider does not need a knn-specific code path.
    """

    model_config = ConfigDict(extra="ignore")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    alt: float = Field(..., ge=ALTITUDE_TRAIN_MIN_M, le=ALTITUDE_TRAIN_MAX_M)
    sim_datetime: Optional[str] = None


class PhysicsWeatherResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_K: float
    pressure_Pa: float
    wind_u_east_mps: float
    wind_v_north_mps: float
    source: str
    provider: str
    note: str


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    artifact_dir: str
    model_loaded: bool
    k: int
    n_train: int
    feature_names: List[str]
    target_names: List[str]

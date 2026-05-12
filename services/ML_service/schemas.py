from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class WeatherValues(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_k: float
    pressure_pa: float
    wind_u: float
    wind_v: float


# Inclusive altitude range the multi-head MLP was trained on. Requests
# outside this band would force the network to extrapolate well past where
# any ERA5 sample exists, so we reject at the edge of the API rather than
# silently produce garbage predictions. Keep this in sync with
# SamplingConfig.altitude_clip_m in era5_gam_weather/config.py.
ALTITUDE_TRAIN_MIN_M = 0.0
ALTITUDE_TRAIN_MAX_M = 32000.0


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude_m: float = Field(..., ge=ALTITUDE_TRAIN_MIN_M, le=ALTITUDE_TRAIN_MAX_M)
    day_of_year: float = Field(..., ge=1.0, le=366.0)
    utc_hour: float = Field(..., ge=0.0, le=24.0)
    year: int = Field(2025, ge=1900, le=2100)
    prediction_mode: Literal["model", "exact", "hybrid"] = "model"
    include_real_era5: bool = False


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: List[PredictRequest] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted: WeatherValues
    model_used: Optional[str] = None
    prediction_source: Literal["model_tree", "model_numpy_mlp", "model_multi_head_mlp", "era5_exact"]
    real_era5: Optional[WeatherValues] = None
    prediction_minus_real: Optional[WeatherValues] = None
    comparison_meta: Optional[Dict[str, Any]] = None


class ServiceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    data_root: str
    artifact_dir: str
    default_model_loaded: bool
    model_cache_size: int
    candidate_models: List[str]
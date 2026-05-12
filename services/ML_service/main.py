from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from era5_gam_weather.prediction_service import WeatherPredictionService
from schemas import (
    ALTITUDE_TRAIN_MAX_M,
    ALTITUDE_TRAIN_MIN_M,
    BatchPredictRequest,
    PredictRequest,
    PredictResponse,
    ServiceInfo,
)

app = FastAPI(title="Weather ML Service", version="8.0.0")
service = WeatherPredictionService()


@app.on_event("startup")
def startup_event() -> None:
    service.warm_start()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "Machine service is running",
        "backend": service.backend,
        "artifact_dir": str(service.artifact_dir),
    }


@app.get("/model-info")
def model_info() -> dict:
    """Return information about the currently active model backend."""
    info: dict = {
        "backend": service.backend,
        "artifact_dir": str(service.artifact_dir),
        "loaded_at_startup": service.default_model_loaded,
    }

    try:
        if service.backend == "multi_head_mlp":
            info["model"] = service._load_multi_head_mlp().describe()
        elif service.backend == "numpy_mlp":
            info["model"] = service._load_numpy_mlp().describe()
        else:
            info["legacy_note"] = "tree backend selected"
    except FileNotFoundError as exc:
        info["model_error"] = str(exc)

    return info


@app.get("/health", response_model=ServiceInfo)
def health() -> ServiceInfo:
    return service.health()


@app.post("/predict-weather", response_model=PredictResponse)
def predict_weather(req: PredictRequest) -> PredictResponse:
    try:
        return service.predict(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict-weather-batch", response_model=list[PredictResponse])
def predict_weather_batch(req: BatchPredictRequest) -> list[PredictResponse]:
    try:
        return service.batch_predict(req.points)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


class PhysicsWeatherRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    alt: float = Field(..., ge=ALTITUDE_TRAIN_MIN_M, le=ALTITUDE_TRAIN_MAX_M)
    sim_datetime: Optional[str] = None


@app.post("/predict-weather-physics")
def predict_weather_physics(req: PhysicsWeatherRequest):
    try:
        if req.sim_datetime:
            dt = datetime.fromisoformat(req.sim_datetime.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        day_of_year = float(dt.timetuple().tm_yday)
        utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        model_req = PredictRequest(
            lat=req.lat,
            lon=req.lon,
            altitude_m=req.alt,
            day_of_year=day_of_year,
            utc_hour=utc_hour,
            year=dt.year,
            prediction_mode="model",
            include_real_era5=False,
        )

        result = service.predict(model_req)

        return {
            "temperature_K": result.predicted.temperature_k,
            "pressure_Pa": result.predicted.pressure_pa,
            "wind_u_east_mps": result.predicted.wind_u,
            "wind_v_north_mps": result.predicted.wind_v,
            "source": "machine",
            "provider": "weather-ml",
            "note": f"prediction_source={result.prediction_source}",
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

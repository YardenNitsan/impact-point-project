from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from era5_gam_weather.era5_lookup import lookup_real_era5_point
from era5_gam_weather.model import WeatherGAMBundle

DATA_ROOT = r"../../data/era5"
DEFAULT_YEAR = 2025

MODEL_CANDIDATES = [
    Path("artifacts/weather_model_bundle_2025_05.npz"),
    Path("artifacts/weather_model_bundle_mar_apr_may.npz"),
    Path("artifacts/weather_model_bundle.npz"),
]

app = FastAPI(title="Weather ML Service", version="3.0.0")
MODEL_CACHE: Dict[str, WeatherGAMBundle] = {}
DEFAULT_MODEL: Optional[WeatherGAMBundle] = None


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    altitude_m: float = Field(..., ge=0.0, le=32000.0)
    day_of_year: float = Field(..., ge=1.0, le=366.0)
    utc_hour: float = Field(..., ge=0.0, le=23.9999)
    year: int = Field(DEFAULT_YEAR, ge=1900, le=2100)
    include_real_era5: bool = True
    prediction_mode: Literal["model", "hybrid", "exact"] = "model"


class WeatherValues(BaseModel):
    temperature_k: float
    pressure_pa: float
    wind_u: float
    wind_v: float


class PredictResponse(BaseModel):
    predicted: WeatherValues
    real_era5: Optional[WeatherValues] = None
    prediction_minus_real: Optional[WeatherValues] = None
    comparison_meta: Optional[Dict] = None
    model_used: Optional[str] = None
    prediction_source: str


def _date_from_year_and_day(year: int, day_of_year: float) -> datetime:
    day_int = int(day_of_year)
    if day_int < 1 or day_int > 366:
        raise ValueError(f"Invalid day_of_year: {day_of_year}")
    return datetime(year, 1, 1) + timedelta(days=day_int - 1)


def load_model(path: Path) -> WeatherGAMBundle:
    key = str(path)
    cached = MODEL_CACHE.get(key)
    if cached is None:
        cached = WeatherGAMBundle.load(str(path))
        MODEL_CACHE[key] = cached
    return cached


def choose_model(year: int, day_of_year: float) -> Tuple[WeatherGAMBundle, str]:
    dt = _date_from_year_and_day(year, day_of_year)
    monthly = Path(f"artifacts/weather_model_bundle_{dt.year}_{dt.month:02d}.npz")
    if monthly.exists():
        return load_model(monthly), str(monthly)

    for path in MODEL_CANDIDATES:
        if path.exists():
            return load_model(path), str(path)

    raise FileNotFoundError(
        "No trained model found. Expected one of: " + ", ".join(str(p) for p in MODEL_CANDIDATES)
    )


def _predict_with_model(req: PredictRequest) -> Tuple[Dict[str, float], str]:
    model, model_path = choose_model(req.year, req.day_of_year)
    predicted = model.predict_one(
        lat=req.lat,
        lon=req.lon,
        altitude_m=req.altitude_m,
        day_of_year=req.day_of_year,
        utc_hour=req.utc_hour,
    )
    return predicted, model_path


def _predict_with_exact(req: PredictRequest) -> Tuple[Dict[str, float], Dict]:
    payload = lookup_real_era5_point(
        data_root=DATA_ROOT,
        year=req.year,
        day_of_year=req.day_of_year,
        utc_hour=req.utc_hour,
        lat=req.lat,
        lon=req.lon,
        altitude_m=req.altitude_m,
    )
    return payload["real"], payload["meta"]


@app.on_event("startup")
def startup_event() -> None:
    global DEFAULT_MODEL
    try:
        DEFAULT_MODEL, _ = choose_model(DEFAULT_YEAR, 135.0)
    except FileNotFoundError:
        DEFAULT_MODEL = None


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "Weather ML service is running"}


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "ok": True,
        "model_cache_size": len(MODEL_CACHE),
        "default_model_loaded": DEFAULT_MODEL is not None,
    }


@app.post("/predict-weather", response_model=PredictResponse)
def predict_weather(req: PredictRequest) -> PredictResponse:
    exact_payload: Optional[Dict] = None
    model_path: Optional[str] = None

    if req.prediction_mode == "exact":
        try:
            predicted, exact_meta = _predict_with_exact(req)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Exact ERA5 lookup failed: {exc}")
        exact_payload = {"real": predicted, "meta": exact_meta}
        prediction_source = "era5_exact"
    elif req.prediction_mode == "hybrid":
        try:
            predicted, exact_meta = _predict_with_exact(req)
            exact_payload = {"real": predicted, "meta": exact_meta}
            prediction_source = "era5_exact"
        except FileNotFoundError:
            try:
                predicted, model_path = _predict_with_model(req)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            prediction_source = "model"
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Hybrid exact lookup failed: {exc}")
    else:
        try:
            predicted, model_path = _predict_with_model(req)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        prediction_source = "model"

    predicted_model = WeatherValues(**predicted)

    if not req.include_real_era5:
        return PredictResponse(
            predicted=predicted_model,
            model_used=model_path,
            prediction_source=prediction_source,
        )

    if exact_payload is None:
        try:
            exact_real, exact_meta = _predict_with_exact(req)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ERA5 comparison failed: {exc}")
        exact_payload = {"real": exact_real, "meta": exact_meta}

    real = exact_payload["real"]
    delta = {
        "temperature_k": predicted["temperature_k"] - real["temperature_k"],
        "pressure_pa": predicted["pressure_pa"] - real["pressure_pa"],
        "wind_u": predicted["wind_u"] - real["wind_u"],
        "wind_v": predicted["wind_v"] - real["wind_v"],
    }

    return PredictResponse(
        predicted=predicted_model,
        real_era5=WeatherValues(**real),
        prediction_minus_real=WeatherValues(**delta),
        comparison_meta=exact_payload["meta"],
        model_used=model_path,
        prediction_source=prediction_source,
    )
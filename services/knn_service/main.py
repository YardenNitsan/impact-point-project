"""FastAPI app for the KNN weather service.

Loads the trained KNN model on startup (auto-training from a synthetic
dataset if no artifact exists yet) and exposes the same
``/predict-weather-physics`` contract the MLP service offers. The
``weather_service`` upstream picks between the two based on the
``weather_source`` field set in the original simulation request.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from knn_model import KnnWeatherModel, CYCLIC_FEATURE_NAMES, TARGET_NAMES
from schemas import (
    PhysicsWeatherRequest,
    PhysicsWeatherResponse,
    ServiceInfo,
)
from training_data import env_path, load_or_build_dataset


load_dotenv()

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "knn_weather"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _resolve_artifact_dir() -> Path:
    return env_path("KNN_ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR


def _load_or_bootstrap_model() -> KnnWeatherModel:
    """Load the persisted KNN model, or train a fresh one if none exists.

    Auto-training keeps the service runnable in a clean environment: a
    developer doing ``docker compose up`` for the first time gets a working
    service backed by the synthetic ISA fallback. Real ERA5 training is a
    separate explicit step (``python train_knn.py``).
    """
    artifact_dir = _resolve_artifact_dir()
    dataset_path = artifact_dir / "dataset.npz"
    metadata_path = artifact_dir / "metadata.json"

    if dataset_path.exists() and metadata_path.exists():
        print(f"[knn-service] Loading KNN model from {artifact_dir}")
        return KnnWeatherModel.load(artifact_dir)

    print(
        f"[knn-service] No KNN artifacts at {artifact_dir} — building a "
        "starter model in-process. Run train_knn.py for a real training run."
    )
    dataset = load_or_build_dataset(
        artifact_dir=artifact_dir,
        era5_root=env_path("ERA5_DATA_ROOT"),
        samples_per_file=_env_int("KNN_SAMPLES_PER_FILE", 4000),
        max_files=_env_int("KNN_MAX_FILES", 30),
    )
    model = KnnWeatherModel(
        raw_inputs=dataset.features,
        targets=dataset.targets,
        k=_env_int("KNN_K", 8),
        ood_threshold=_env_float("KNN_OOD_THRESHOLD", 0.05),
    )
    model.save(artifact_dir)
    return model


app = FastAPI(title="KNN Weather Service", version="1.0.0")
model: Optional[KnnWeatherModel] = None


@app.on_event("startup")
def _startup() -> None:
    global model
    model = _load_or_bootstrap_model()
    print(
        f"[knn-service] Ready — k={model.k}, n_train={model.n_train}, "
        f"ood_threshold={model.ood_threshold}"
    )


def _resolve_doy_and_hour(sim_datetime: Optional[str]) -> tuple[float, float]:
    if sim_datetime:
        dt = datetime.fromisoformat(sim_datetime.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    day_of_year = float(dt.timetuple().tm_yday)
    utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return day_of_year, utc_hour


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "KNN weather service is running",
        "artifact_dir": str(_resolve_artifact_dir()),
    }


@app.get("/health", response_model=ServiceInfo)
def health() -> ServiceInfo:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ServiceInfo(
        ok=True,
        artifact_dir=str(_resolve_artifact_dir()),
        model_loaded=True,
        k=model.k,
        n_train=model.n_train,
        feature_names=list(CYCLIC_FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
    )


@app.get("/model-info")
def model_info() -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "artifact_dir": str(_resolve_artifact_dir()),
        "model": model.describe(),
    }


@app.post("/predict-weather-physics", response_model=PhysicsWeatherResponse)
def predict_weather_physics(req: PhysicsWeatherRequest) -> PhysicsWeatherResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        day_of_year, utc_hour = _resolve_doy_and_hour(req.sim_datetime)
        result = model.predict(
            lat=req.lat,
            lon=req.lon,
            altitude_m=req.alt,
            day_of_year=day_of_year,
            utc_hour=utc_hour,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"KNN prediction failed: {exc}") from exc

    note_parts = [
        f"k={model.k}",
        f"mean_neighbor_dist={float(result.neighbor_distances.mean()):.4f}",
    ]
    if result.out_of_distribution:
        note_parts.append(
            f"out_of_distribution (envelope_excursion={result.envelope_excursion:.3f})"
        )

    return PhysicsWeatherResponse(
        temperature_K=result.temperature_K,
        pressure_Pa=result.pressure_Pa,
        wind_u_east_mps=result.wind_u_east_mps,
        wind_v_north_mps=result.wind_v_north_mps,
        source="knn",
        provider="weather-knn",
        note=" | ".join(note_parts),
    )

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.clients.machine_client import MachineClient
from app.clients.openmeteo_client import OpenMeteoClient
from app.providers.machine_provider import MachineProvider
from app.providers.openmeteo_provider import OpenMeteoProvider
from app.router import WeatherRouter
from app.schemas import HealthResponse, WeatherRequest, WeatherResponse

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "weather-service")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20"))

MACHINE_BASE_URL = os.getenv(
    "MACHINE_BASE_URL",
    "http://host.docker.internal:8001",
)
MACHINE_PREDICT_PATH = os.getenv(
    "MACHINE_PREDICT_PATH",
    "/predict",
)

OPENMETEO_ARCHIVE_URL = os.getenv(
    "OPENMETEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)

machine_client = MachineClient(
    base_url=MACHINE_BASE_URL,
    predict_path=MACHINE_PREDICT_PATH,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
)

openmeteo_client = OpenMeteoClient(
    archive_url=OPENMETEO_ARCHIVE_URL,
    timeout_seconds=REQUEST_TIMEOUT_SECONDS,
)

machine_provider = MachineProvider(client=machine_client)
openmeteo_provider = OpenMeteoProvider(client=openmeteo_client)

weather_router = WeatherRouter(
    machine_provider=machine_provider,
    openmeteo_provider=openmeteo_provider,
)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
    )


@app.post("/weather", response_model=WeatherResponse)
async def get_weather(req: WeatherRequest) -> WeatherResponse:
    return await weather_router.get_weather(req)


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )
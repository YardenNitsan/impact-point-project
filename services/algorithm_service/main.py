from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import os
import time
import traceback

from modules.impact.simulated_impact import simulate_impact

app = FastAPI(
    title="Impact Simulation API",
    version="1.2.0",
)

request_counter = 0


class SimulationInput(BaseModel):
    alt: float = Field(..., description="Launch altitude above ground [m]")
    azimuth: float = Field(..., description="Launch azimuth [deg], 0=N, 90=E")
    elevation: float = Field(..., description="Launch elevation [deg]")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Launch latitude [deg]")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Launch longitude [deg]")
    mass: float = Field(..., gt=0.0, description="Projectile mass [kg]")
    initialSpeed: float = Field(..., ge=0.0, description="Initial speed magnitude [m/s]")
    sim_datetime: str | None = Field(
        default=None,
        description="Optional simulation datetime in ISO-8601 format",
    )

    weather_source: Literal["api", "machine", "calculations"] = Field(
        default="machine",
        description="Weather mode: machine/api use weather service; calculations seeds once from API then uses internal calculations",
    )

    T0_K: float | None = Field(default=None, gt=0.0, description="Manual temperature override [K]")
    P0_Pa: float | None = Field(default=None, gt=0.0, description="Manual pressure override [Pa]")
    wind_x: float | None = Field(default=None, description="Manual along-track wind override [m/s]")
    wind_z: float | None = Field(default=None, description="Manual vertical wind override [m/s]")

    return_trajectory: bool = Field(default=False)
    sample_dx_m: float = Field(default=2.0, gt=0.0, description="Trajectory output spacing [m]")


@app.get("/")
def health_check():
    return {"status": "Impact Simulation API is running"}


@app.post("/simulate-impact")
def simulate(input: SimulationInput):
    global request_counter
    request_counter += 1

    pid = os.getpid()
    request_id = request_counter
    start_time = time.time()

    print("\n==============================")
    print(f"START PID {pid} time {start_time}")
    print(f"Worker PID: {pid}")
    print(f"Request number: {request_id}")
    print("Simulation request received")
    print("Received simulation request:")
    print(input.model_dump())
    print("==============================\n")

    try:
        payload = input.model_dump(
            exclude={"return_trajectory", "sample_dx_m"},
            exclude_none=True,
        )

        result = simulate_impact(
            payload,
            return_trajectory=input.return_trajectory,
            dx_sample_m=input.sample_dx_m,
        )

        end_time = time.time()

        print("\n==============================")
        print(f"Simulation completed by worker PID: {pid}")
        print(f"Request number: {request_id}")
        print(f"START time: {start_time}")
        print(f"END time: {end_time}")
        print(f"Runtime: {end_time - start_time:.3f} seconds")
        print(f"Trajectory points: {len(result.get('trajectory', []))}")
        print("==============================\n")

        return result

    except Exception as exc:
        end_time = time.time()

        print("\n==============================")
        print(f"Simulation failed by worker PID: {pid}")
        print(f"Request number: {request_id}")
        print(f"START time: {start_time}")
        print(f"END time: {end_time}")
        print(f"Runtime before failure: {end_time - start_time:.3f} seconds")
        print(f"Error: {repr(exc)}")
        print("Traceback:")
        traceback.print_exc()
        print("==============================\n")

        raise HTTPException(status_code=500, detail=str(exc))
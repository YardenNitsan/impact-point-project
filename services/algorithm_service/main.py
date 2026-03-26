from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import traceback

from modules.impact.simulated_impact import simulate_impact

app = FastAPI(
    title="Impact Simulation API",
    version="1.0.0",
)

request_counter = 0


class SimulationInput(BaseModel):
    alt: float
    azimuth: float
    elevation: float
    lat: float
    lon: float
    mass: float
    initialSpeed: float
    sim_datetime: str
    T0_K: float | None = None
    P0_Pa: float | None = None
    wind_x: float | None = None
    wind_z: float | None = None
    return_trajectory: bool = False
    sample_dx_m: float = 2.0


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
        payload = input.model_dump(exclude={"return_trajectory", "sample_dx_m"})

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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os

# Import your algorithm
from modules.impact.simulated_impact import simulate_impact

# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="Impact Simulation API",
    version="1.0.0"
)

request_counter = 0

# ===============================
# Request Model
# ===============================
class SimulationInput(BaseModel):
    alt: float
    azimuth: float
    elevation: float
    lat: float
    lon: float
    mass: float
    initialSpeed: float
    T0_K: float | None = None
    P0_Pa: float | None = None
    wind_x: float | None = None
    wind_z: float | None = None

# ===============================
# Health Check
# ===============================
@app.get("/")
def health_check():
    return {"status": "Impact Simulation API is running"}

# ===============================
# Simulation Endpoint
# ===============================
@app.post("/simulate-impact")
def simulate(input: SimulationInput):

    print("================================")
    print(f"START PID {os.getpid()} time {time.time()}")
    print("================================")

    global request_counter
    request_counter += 1

    worker_pid = os.getpid()

    print("\n==============================")
    print(f"Worker PID: {worker_pid}")
    print(f"Request number: {request_counter}")
    print("Simulation request received")

    start_time = time.time()

    try:
        data = input.model_dump()

        print("\nReceived simulation request:")
        print(data)

        result = simulate_impact(data)

        duration = time.time() - start_time

        print(f"\nSimulation completed by worker PID: {worker_pid}")
        print(f"Runtime: {duration:.3f} seconds")

        if isinstance(result, dict) and "trajectory" in result:
            print(f"Trajectory points: {len(result['trajectory'])}")
        
        print("================================")
        print(f"END PID {os.getpid()} time {time.time()}")
        print("================================")

        return result

    except Exception as e:
        print("\nSimulation error:", e)
        raise HTTPException(status_code=500, detail=str(e))

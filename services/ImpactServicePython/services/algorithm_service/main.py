from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# Import your algorithm
from services.algorithm_service.modules.impact.simulated_impact import simulate_impact

# ===============================
# FastAPI App
# ===============================
app = FastAPI(
    title="Impact Simulation API",
    version="1.0.0"
)

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
    start_time = time.time()

    try:
        data = input.model_dump()

        print("\n📥 Received simulation request:")
        print(data)

        result = simulate_impact(data)

        duration = time.time() - start_time

        print("\n✅ Simulation completed")
        print(f"⏱ Runtime: {duration:.3f} seconds")

        if isinstance(result, dict) and "trajectory" in result:
            print(f"📈 Trajectory points: {len(result['trajectory'])}")

        return result

    except Exception as e:
        print("\n❌ Simulation error:", e)
        raise HTTPException(status_code=500, detail=str(e))

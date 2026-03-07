from fastapi import FastAPI
import requests
from concurrent.futures import ThreadPoolExecutor
from dataset_modules.lhs_sample import generate_samples

app = FastAPI()

PHYSICS_URL = "http://localhost:8000/simulate-impact"


@app.get("/")
def read_root():
    return {"status": "ML service is running"}


def run_simulation(sample):

    payload = {
        "lat": sample["latitude"],
        "lon": sample["longitude"],
        "alt": sample["altitude"],
        "azimuth": sample["azimuth"],
        "elevation": sample["elevation"],
        "mass": sample["mass"],
        "initialSpeed": sample["speed"],
        "T0_K": sample["T0"],
        "P0_Pa": sample["P0"],
        "wind_x": sample["wind_x"],
        "wind_z": sample["wind_z"]
    }

    response = requests.post(PHYSICS_URL, json=payload, timeout=60)

    return response.json()


@app.post("/generate-dataset")
def generate_dataset(n_samples: int = 10):

    samples = generate_samples(n_samples)

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(run_simulation, samples))

    return {"simulations": len(results)}
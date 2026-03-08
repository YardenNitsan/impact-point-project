import time

from fastapi import FastAPI
import requests
from concurrent.futures import ThreadPoolExecutor
from dataset_modules.lhs_sample import generate_samples
import threading

from requests.adapters import HTTPAdapter


import os

app = FastAPI()

PHYSICS_URL = "http://localhost:8000/simulate-impact"

session = requests.Session()
session.mount("http://", HTTPAdapter(pool_connections=100, pool_maxsize=100))


@app.get("/")
def read_root():
    return {"status": "ML service is running"}


def run_simulation(sample):

    thread_id = threading.get_ident()

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

    print(f"\nThread {thread_id} sending simulation request")

    start = time.time()

    try:
        response = session.post(PHYSICS_URL, json=payload, timeout=60)

        response.raise_for_status()

        duration = time.time() - start

        print(f"Thread {thread_id} simulation completed in {duration:.3f}s")

        return response.json()
    except Exception as e:
        print("Simulation failed:", e)
        return None




@app.post("/generate-dataset")
def generate_dataset(n_samples: int = 10):

    print("\n=================================")
    print(f"Generating dataset with {n_samples} samples")
    print("=================================\n")

    samples = generate_samples(n_samples)

    start = time.time()

    max_workers = min(32, os.cpu_count() * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_simulation, samples))

    results = [r for r in results if r is not None]

    total_time = time.time() - start

    print("\n=================================")
    print(f"Dataset generation finished")
    print(f"Total simulations: {len(results)}")
    print(f"Total time: {total_time:.3f}s")
    print("=================================\n")

    return {"simulations": len(results)}
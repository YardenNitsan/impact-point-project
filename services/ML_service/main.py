import time
import json
import os
import threading

from fastapi import FastAPI
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dataset_modules.lhs_sample import generate_samples
from dataset_modules.trajectory_delta import build_dataset_row

from requests.adapters import HTTPAdapter
from pymongo import MongoClient, InsertOne

# ============================================================
# Config
# ============================================================

PHYSICS_URL = "http://localhost:8001/simulate-impact"

MONGO_URI = "mongodb://localhost:27018"

DATASET_FILE = "dataset.jsonl"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

MONGO_BATCH_SIZE = 100

# ============================================================
# FastAPI
# ============================================================

app = FastAPI()

# ============================================================
# Mongo
# ============================================================

mongo_client = MongoClient(
    MONGO_URI,
    maxPoolSize=200,
)

db = mongo_client["impact_dataset"]
collection = db["simulations"]

# create useful index (for future ML queries)
collection.create_index("flight_time")

# ============================================================
# HTTP session (connection pool)
# ============================================================

session = requests.Session()
session.mount(
    "http://",
    HTTPAdapter(
        pool_connections=200,
        pool_maxsize=200
    )
)

# ============================================================
# Worker
# ============================================================

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

    for attempt in range(MAX_RETRIES):

        try:

            response = session.post(
                PHYSICS_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            simulation_result = response.json()

            dataset_row = build_dataset_row(sample, simulation_result)

            return dataset_row

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                print("Simulation failed:", e)
                return None

            time.sleep(1)
    return None


# ============================================================
# Dataset generator
# ============================================================

@app.post("/generate-dataset")
def generate_dataset(n_samples: int = 1000):

    print("\n=================================")
    print(f"Generating dataset with {n_samples} samples")
    print("=================================\n")

    samples = generate_samples(n_samples)

    start_time = time.time()

    max_workers = min(16, os.cpu_count())

    print(f"Using {max_workers} worker threads")

    written = 0

    mongo_buffer = []
    mongo_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(run_simulation, s) for s in samples]

        with open(DATASET_FILE, "a", buffering=1024 * 1024) as f:

            for future in tqdm(
                as_completed(futures),
                total=n_samples,
                desc="Simulations",
                unit="sim"
            ):

                result = future.result()

                if result is None:
                    continue

                written += 1

                # -------------------
                # write JSONL
                # -------------------

                f.write(json.dumps(result) + "\n")

                # -------------------
                # buffer Mongo insert
                # -------------------

                with mongo_lock:

                    mongo_buffer.append(InsertOne(result))

                    if len(mongo_buffer) >= MONGO_BATCH_SIZE:

                        collection.bulk_write(mongo_buffer)

                        mongo_buffer.clear()

    # flush remaining Mongo docs

    if mongo_buffer:
        collection.bulk_write(mongo_buffer)

    total_time = time.time() - start_time

    print("\n=================================")
    print("Dataset generation finished")
    print(f"Total simulations requested: {n_samples}")
    print(f"Successful simulations: {written}")
    print(f"Total time: {total_time:.2f}s")
    print("=================================\n")

    return {
        "requested": n_samples,
        "successful": written,
        "dataset_file": DATASET_FILE
    }


# ============================================================
# Health check
# ============================================================

@app.get("/")
def read_root():
    return {"status": "ML service is running"}
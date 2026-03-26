import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterator, Dict, Any, List

import numpy as np

from modules.atmosphere.era5_loader import preload_era5_hour
from modules.atmosphere.trajectory_delta import build_dataset_row
from modules.impact.simulated_impact import simulate_impact


GLOBAL_RANGES = {
    "latitude": (-89.5, 89.5),
    "longitude": (-180.0, 180.0),
    "azimuth": (0.0, 360.0),
    "day": (1.0, 32.0),
    "hour": (0.0, 24.0),
}

REGIMES: List[Dict[str, Any]] = [
    {
        "name": "ground_low_speed",
        "weight": 0.30,
        "ranges": {
            "altitude": (0.0, 1500.0),
            "speed": (30.0, 250.0),
            "elevation": (-20.0, 45.0),
            "mass": (5.0, 800.0),
        },
    },
    {
        "name": "ground_medium_speed",
        "weight": 0.30,
        "ranges": {
            "altitude": (0.0, 2500.0),
            "speed": (150.0, 500.0),
            "elevation": (-25.0, 50.0),
            "mass": (20.0, 2000.0),
        },
    },
    {
        "name": "air_release",
        "weight": 0.25,
        "ranges": {
            "altitude": (1000.0, 12000.0),
            "speed": (120.0, 450.0),
            "elevation": (-35.0, 20.0),
            "mass": (20.0, 2500.0),
        },
    },
    {
        "name": "high_energy",
        "weight": 0.15,
        "ranges": {
            "altitude": (3000.0, 18000.0),
            "speed": (400.0, 1100.0),
            "elevation": (-15.0, 35.0),
            "mass": (100.0, 5000.0),
        },
    },
]

DATASET_YEAR = int(os.environ.get("DATASET_YEAR", "2025"))
DATASET_MONTH = int(os.environ.get("DATASET_MONTH", "5"))

SKIP_DAYS = {
    int(x.strip())
    for x in os.environ.get("SKIP_ERA5_DAYS", "19").split(",")
    if x.strip()
}

ALLOWED_DAYS_ENV = os.environ.get("ALLOWED_DAYS", "").strip()
if ALLOWED_DAYS_ENV:
    ALLOWED_DAYS = sorted({
        int(x.strip())
        for x in ALLOWED_DAYS_ENV.split(",")
        if x.strip()
    })
else:
    ALLOWED_DAYS = list(range(1, 32))

VALID_DAYS = [d for d in ALLOWED_DAYS if d not in SKIP_DAYS]

if not VALID_DAYS:
    raise ValueError("No valid days left after ALLOWED_DAYS / SKIP_ERA5_DAYS")


def latin_hypercube_sampling(
    n_samples: int,
    dimensions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty((n_samples, dimensions), dtype=np.float64)
    for i in range(dimensions):
        perm = rng.permutation(n_samples)
        result[:, i] = (perm + rng.random(n_samples)) / n_samples
    return result


def _scale(unit_value: float, low: float, high: float) -> float:
    return low + unit_value * (high - low)


def _build_regime_plan(
    n_samples: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    weights = np.array([r["weight"] for r in REGIMES], dtype=float)
    weights = weights / weights.sum()

    raw_counts = weights * n_samples
    counts = np.floor(raw_counts).astype(int)

    remainder = n_samples - counts.sum()
    if remainder > 0:
        frac_order = np.argsort(raw_counts - counts)[::-1]
        for idx in frac_order[:remainder]:
            counts[idx] += 1

    plan: List[Dict[str, Any]] = []
    for regime, count in zip(REGIMES, counts):
        plan.extend([regime] * count)

    rng.shuffle(plan)
    return plan


def _sample_from_ranges(
    unit_row: np.ndarray,
    keys: List[str],
    ranges: Dict[str, tuple],
) -> Dict[str, float]:
    sample: Dict[str, float] = {}
    for i, key in enumerate(keys):
        low, high = ranges[key]
        sample[key] = float(_scale(unit_row[i], low, high))
    return sample


def _sample_valid_day(unit_value: float) -> int:
    idx = min(int(unit_value * len(VALID_DAYS)), len(VALID_DAYS) - 1)
    return VALID_DAYS[idx]


def generate_samples(
    n_samples: int,
    seed: int | None = None,
) -> Iterator[Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    global_keys = list(GLOBAL_RANGES.keys())
    global_lhs = latin_hypercube_sampling(n_samples, len(global_keys), rng)

    local_keys = ["altitude", "speed", "elevation", "mass"]
    local_lhs = latin_hypercube_sampling(n_samples, len(local_keys), rng)

    regime_plan = _build_regime_plan(n_samples, rng)

    day_idx = global_keys.index("day")

    for i in range(n_samples):
        sample: Dict[str, Any] = {}

        sample.update(_sample_from_ranges(global_lhs[i], global_keys, GLOBAL_RANGES))

        regime = regime_plan[i]
        sample.update(_sample_from_ranges(local_lhs[i], local_keys, regime["ranges"]))

        az_rad = math.radians(sample["azimuth"])
        el_rad = math.radians(sample["elevation"])

        sample["sin_az"] = math.sin(az_rad)
        sample["cos_az"] = math.cos(az_rad)
        sample["sin_el"] = math.sin(el_rad)
        sample["cos_el"] = math.cos(el_rad)

        day = _sample_valid_day(float(global_lhs[i][day_idx]))
        hour = max(0, min(23, int(sample["hour"])))

        sample["day"] = day
        sample["hour"] = hour
        sample["sim_datetime"] = (
            f"{DATASET_YEAR:04d}-{DATASET_MONTH:02d}-{day:02d}T{hour:02d}:00:00"
        )

        yield sample


def _build_simulation_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lat": float(sample["latitude"]),
        "lon": float(sample["longitude"]),
        "alt": float(sample["altitude"]),
        "mass": float(sample["mass"]),
        "initialSpeed": float(sample["speed"]),
        "sim_datetime": str(sample["sim_datetime"]),
        "azimuth": float(sample["azimuth"]),
        "elevation": float(sample["elevation"]),
    }


def run_simulation_direct(sample: Dict[str, Any]) -> Dict[str, Any]:
    payload = _build_simulation_payload(sample)
    simulation_result = simulate_impact(payload, return_trajectory=False)
    return build_dataset_row(sample, simulation_result)


def normalize_payload_hour(sample: Dict[str, Any]) -> str:
    dt = datetime.fromisoformat(str(sample["sim_datetime"]))
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat()


def group_payloads_by_hour(
    payloads: Iterator[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        groups[normalize_payload_hour(payload)].append(payload)
    return groups


def process_hour_batch(
    hour_key: str,
    batch: List[Dict[str, Any]],
    out_file,
    threads: int = 1,
    max_in_flight: int = 1,
) -> tuple[int, int]:
    hour_dt = datetime.fromisoformat(hour_key)

    try:
        preload_era5_hour(hour_dt)
    except Exception as exc:
        print(f"[{hour_key}] preload failed: {repr(exc)}")
        return 0, len(batch)

    total = len(batch)
    done = 0
    failed = 0

    if threads < 1:
        threads = 1
    if max_in_flight < 1:
        max_in_flight = 1

    for i in range(0, total, max_in_flight):
        window = batch[i:i + max_in_flight]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(run_simulation_direct, sample) for sample in window]

            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done += 1
                except Exception as exc:
                    failed += 1
                    print(f"[{hour_key}] simulation failed: {repr(exc)}")

        print(f"[{hour_key}] done {done}/{total}, failed={failed}")

    return done, failed


def generate_dataset(
    n_samples: int,
    output_path: str,
    threads: int = 1,
    max_in_flight: int = 1,
    seed: int | None = None,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples} samples with seed={seed}")
    print(
        f"Dataset month: {DATASET_YEAR:04d}-{DATASET_MONTH:02d}, "
        f"allowed days={VALID_DAYS}, skipped days={sorted(SKIP_DAYS) if SKIP_DAYS else []}"
    )

    grouped = group_payloads_by_hour(generate_samples(n_samples, seed=seed))
    ordered_hours = sorted(grouped.keys())

    total_success = 0
    total_failed = 0

    print(f"Distinct hours: {len(ordered_hours)}")

    with open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        for hour_key in ordered_hours:
            batch = grouped[hour_key]
            print(f"Processing hour {hour_key} with {len(batch)} payloads")
            success, failed = process_hour_batch(
                hour_key=hour_key,
                batch=batch,
                out_file=f,
                threads=threads,
                max_in_flight=max_in_flight,
            )
            total_success += success
            total_failed += failed

    print("Dataset generation complete.")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Output: {output_path}")


def main() -> None:
    n_samples = int(os.environ.get("N_SAMPLES", "1000"))
    output_dataset = os.environ.get("OUTPUT_DATASET", "../../data/dataset_1000.jsonl")
    threads = int(os.environ.get("GEN_THREADS", "1"))
    max_in_flight = int(os.environ.get("GEN_MAX_IN_FLIGHT", "1"))
    seed_env = os.environ.get("GEN_SEED")
    seed = int(seed_env) if seed_env is not None else None

    generate_dataset(
        n_samples=n_samples,
        output_path=output_dataset,
        threads=threads,
        max_in_flight=max_in_flight,
        seed=seed,
    )


if __name__ == "__main__":
    main()
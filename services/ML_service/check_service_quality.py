from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from era5_gam_weather.prediction_service import WeatherPredictionService
from schemas import PredictRequest

OUT_PATH = Path("artifacts/service_quality_check.json")


def compute_metrics(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(arr))),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "max_abs": float(np.max(np.abs(arr))),
    }


def main() -> None:
    service = WeatherPredictionService()
    rng = np.random.default_rng(42)

    errors_t = []
    errors_p = []
    errors_u = []
    errors_v = []

    kept = 0
    skipped = 0

    for _ in range(300):
        lat = float(rng.uniform(-90.0, 90.0))
        lon = float(rng.uniform(-180.0, 180.0))

        r = rng.random()
        if r < 0.5:
            altitude_m = float(rng.uniform(0.0, 3000.0))
        elif r < 0.8:
            altitude_m = float(rng.uniform(3000.0, 8000.0))
        elif r < 0.95:
            altitude_m = float(rng.uniform(8000.0, 16000.0))
        else:
            altitude_m = float(rng.uniform(16000.0, 30000.0))

        day_of_year = float(rng.integers(121, 152))  # May
        utc_hour = float(rng.integers(0, 24))

        req = PredictRequest(
            lat=lat,
            lon=lon,
            altitude_m=altitude_m,
            day_of_year=day_of_year,
            utc_hour=utc_hour,
            year=2025,
            prediction_mode="model",
            include_real_era5=True,
        )

        res = service.predict(req)

        if res.prediction_minus_real is None:
            skipped += 1
            continue

        errors_t.append(res.prediction_minus_real.temperature_k)
        errors_p.append(res.prediction_minus_real.pressure_pa)
        errors_u.append(res.prediction_minus_real.wind_u)
        errors_v.append(res.prediction_minus_real.wind_v)
        kept += 1

    if kept == 0:
        raise RuntimeError("No valid comparisons were produced. ERA5 exact lookup likely failed.")

    report = {
        "n_valid_points": kept,
        "n_skipped_points": skipped,
        "temperature_k": compute_metrics(errors_t),
        "pressure_pa": compute_metrics(errors_p),
        "wind_u": compute_metrics(errors_u),
        "wind_v": compute_metrics(errors_v),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
"""Evaluate the trained Multi-Head MLP against actual ERA5 on a held-out day.

The training script reports aggregated train/val/test metrics, but those metrics
hide *where* the model is bad. In particular, surface wind and high-altitude
jet errors get averaged out across the rest of the atmosphere. This harness:

  1. Loads the trained model from ``WEATHER_ARTIFACT_DIR``.
  2. Picks a single held-out ERA5 day (default: the last day under
     ``ERA5_DATA_ROOT/era5_YYYY_*.nc``, configurable via ``EVAL_DATE``).
  3. Samples N random points from that file using the same sampler as training.
  4. Runs the model on those points and compares to the actual ERA5 values at
     the same coordinates.
  5. Reports MAE/RMSE/MaxAbs per target and per altitude band, plus an overall
     score, both to stdout and to ``held_out_eval.json`` next to the artifact.

Configuration via environment variables:
  WEATHER_ARTIFACT_DIR  artifact directory containing model.keras
  ERA5_DATA_ROOT        directory of era5_YYYY_MM_DD.nc files
  EVAL_DATE             YYYY-MM-DD; if unset uses last available file
  EVAL_SAMPLES          how many samples to draw from the held-out file (default 50000)
  EVAL_SEED             RNG seed (default 1234, distinct from training seed)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from era5_gam_weather.config import SamplingConfig
from era5_gam_weather.era5_sampler import (
    discover_era5_files,
    parse_date_from_path,
    sample_from_file,
)
from era5_gam_weather.multi_head_mlp_model import (
    ERA5_TO_OUTPUT,
    MultiHeadMLPWeatherModel,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


# Altitude bands chosen to match meteorologically distinct regimes.
ALTITUDE_BANDS_M: List[Tuple[str, float, float]] = [
    ("surface_0_2km",        0.0,     2_000.0),
    ("low_trop_2_5km",       2_000.0, 5_000.0),
    ("mid_trop_5_8km",       5_000.0, 8_000.0),
    ("upper_trop_8_12km",    8_000.0, 12_000.0),
    ("lower_strato_12_18km", 12_000.0, 18_000.0),
    ("strato_18_32km",       18_000.0, 32_000.0),
]


def _metrics(residual: np.ndarray) -> Dict[str, float]:
    if residual.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    err = residual.astype(np.float64)
    return {
        "n": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def _pick_held_out_file(data_root: Path) -> str:
    eval_date = os.getenv("EVAL_DATE")
    if eval_date:
        from datetime import datetime as _dt

        try:
            target = _dt.strptime(eval_date, "%Y-%m-%d")
        except ValueError as exc:
            raise SystemExit(f"Invalid EVAL_DATE={eval_date!r}; expected YYYY-MM-DD") from exc

        files = discover_era5_files(str(data_root), target.year, [target.month])
        for path in files:
            if parse_date_from_path(path).date() == target.date():
                return path
        raise SystemExit(f"No ERA5 file found for EVAL_DATE={eval_date} under {data_root}")

    # No EVAL_DATE: pick the most recent file across any month/year.
    candidates = []
    for entry in os.scandir(data_root):
        if entry.is_file() and entry.name.startswith("era5_") and entry.name.endswith(".nc"):
            candidates.append(entry.path)
    if not candidates:
        raise SystemExit(f"No era5_*.nc files in {data_root}")
    candidates.sort(key=lambda p: parse_date_from_path(p))
    return candidates[-1]


def main() -> int:
    artifact_dir = Path(
        os.getenv("WEATHER_ARTIFACT_DIR")
        or str(THIS_DIR / "artifacts" / "multi_head_mlp_weather")
    ).expanduser().resolve()
    data_root = Path(
        os.getenv("ERA5_DATA_ROOT")
        or str(THIS_DIR.parent / "data" / "era5")
    ).expanduser().resolve()

    if not artifact_dir.exists():
        print(f"ERROR: artifact directory not found: {artifact_dir}", file=sys.stderr)
        return 1
    if not data_root.exists():
        print(f"ERROR: ERA5 data root not found: {data_root}", file=sys.stderr)
        return 1

    samples = _env_int("EVAL_SAMPLES", 50_000)
    seed = _env_int("EVAL_SEED", 1234)

    held_out_path = _pick_held_out_file(data_root)
    held_out_date = parse_date_from_path(held_out_path).date().isoformat()

    print(f"Loading model from {artifact_dir}", flush=True)
    model = MultiHeadMLPWeatherModel.load(artifact_dir)

    print(f"Held-out day: {held_out_date}  ({held_out_path})", flush=True)
    print(f"Drawing {samples} samples (seed={seed})", flush=True)

    sampling = SamplingConfig(
        samples_per_file=samples,
        seed=seed,
        stratified_time_level=True,
        # Match training distribution by default; flip via env if desired.
        area_weighted_lat=True,
    )
    batch = sample_from_file(held_out_path, sampling)

    n = batch.features["lat"].shape[0]
    if n == 0:
        print("ERROR: held-out day produced zero valid samples.", file=sys.stderr)
        return 1
    print(f"Got {n} valid samples", flush=True)

    pred = model.predict_features(batch.features)

    altitude = batch.features["altitude_m"].astype(np.float64)
    overall: Dict[str, Dict[str, float]] = {}
    by_band: Dict[str, Dict[str, Dict[str, float]]] = {}

    for era5_key, output_name in ERA5_TO_OUTPUT.items():
        truth = batch.targets[era5_key].astype(np.float64)
        guess = np.asarray(pred[output_name], dtype=np.float64)
        residual = guess - truth
        overall[output_name] = _metrics(residual)

        per_band: Dict[str, Dict[str, float]] = {}
        for band_name, lo, hi in ALTITUDE_BANDS_M:
            mask = (altitude >= lo) & (altitude < hi)
            per_band[band_name] = _metrics(residual[mask])
        by_band[output_name] = per_band

    # ------------------------------- Pretty print ------------------------------
    bar = "=" * 84
    print(bar)
    print(f"OVERALL HELD-OUT-DAY METRICS ({held_out_date})")
    print(bar)
    print(f"{'target':<18}{'n':>10}{'MAE':>14}{'RMSE':>14}{'MaxAbs':>14}")
    print("-" * 70)
    for target, m in overall.items():
        print(f"{target:<18}{m['n']:>10}{m['mae']:>14.4f}{m['rmse']:>14.4f}{m['max_abs']:>14.4f}")
    print()

    print(bar)
    print("PER-ALTITUDE-BAND MAE")
    print(bar)
    headers = ["target"] + [b[0] for b in ALTITUDE_BANDS_M]
    print(("{:<18}" + "{:>20}" * len(ALTITUDE_BANDS_M)).format(*headers))
    print("-" * (18 + 20 * len(ALTITUDE_BANDS_M)))
    for target in overall.keys():
        row = [target]
        for band_name, _lo, _hi in ALTITUDE_BANDS_M:
            m = by_band[target][band_name]
            if m["n"] == 0:
                row.append("(none)")
            else:
                row.append(f"{m['mae']:.3f} (n={m['n']})")
        print(("{:<18}" + "{:>20}" * len(ALTITUDE_BANDS_M)).format(*row))

    payload = {
        "held_out_date": held_out_date,
        "held_out_path": held_out_path,
        "n_samples": int(n),
        "seed": seed,
        "altitude_bands_m": [{"name": n_, "lo": lo, "hi": hi} for n_, lo, hi in ALTITUDE_BANDS_M],
        "overall": overall,
        "by_band": by_band,
        "sampling_config": asdict(sampling),
    }
    out_path = artifact_dir / "held_out_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

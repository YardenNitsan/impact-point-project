from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from era5_gam_weather.config import SamplingConfig, SplitConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day
from era5_gam_weather.model import WeatherGAMTrainer

DATA_ROOT = "../../data/era5"
YEAR = 2025
MONTHS = [3, 4]
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

INPUT_STATE = ARTIFACT_DIR / "may_training_state.npz"
OUTPUT_STATE = ARTIFACT_DIR / "march_april_may_training_state.npz"
OUTPUT_MODEL = ARTIFACT_DIR / "weather_model_bundle_mar_apr_may.npz"
OUTPUT_METRICS = ARTIFACT_DIR / "eval_metrics_mar_apr_may.json"


def run_continue() -> None:
    if not INPUT_STATE.exists():
        raise FileNotFoundError(f"Missing state file: {INPUT_STATE}")

    trainer = WeatherGAMTrainer.load_state(str(INPUT_STATE))
    sampling_config = SamplingConfig(samples_per_file=12000, seed=42, stratified_time_level=True)
    eval_sampling_config = SamplingConfig(samples_per_file=4000, seed=142, stratified_time_level=True)
    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)

    files = discover_era5_files(DATA_ROOT, YEAR, MONTHS)
    if not files:
        raise FileNotFoundError(f"No March/April ERA5 files found under {DATA_ROOT}")

    splits = split_files_by_day(
        files,
        train_end=split_config.train_end_day_inclusive,
        val_end=split_config.val_end_day_inclusive,
    )

    for path in splits["train"]:
        print(f"[CONTINUE TRAIN] {path}")
        batch = sample_from_file(path, sampling_config)
        trainer.update(batch.features, batch.targets)

    trainer.save_state(str(OUTPUT_STATE))
    model = trainer.finalize()
    model.save(str(OUTPUT_MODEL))

    all_eval_features = {k: [] for k in ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]}
    all_eval_targets = {k: [] for k in ["T", "P", "U", "V"]}

    for split_name in ["val", "test"]:
        for path in splits[split_name]:
            print(f"[{split_name.upper()}] {path}")
            batch = sample_from_file(path, eval_sampling_config)
            for k, v in batch.features.items():
                all_eval_features[k].append(v)
            for k, v in batch.targets.items():
                all_eval_targets[k].append(v)

    eval_features = {k: np.concatenate(v) for k, v in all_eval_features.items()}
    eval_targets = {k: np.concatenate(v) for k, v in all_eval_targets.items()}
    metrics = model.evaluate(eval_features, eval_targets)

    payload = {
        "months_added": MONTHS,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "max_abs": metrics.max_abs,
        "n_samples": metrics.n_samples,
        "sampling_config": sampling_config.to_dict(),
        "eval_sampling_config": eval_sampling_config.to_dict(),
        "split_config": split_config.to_dict(),
    }
    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved continued model.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run_continue()
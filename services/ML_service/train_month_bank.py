from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from era5_gam_weather.config import BasisConfig, SamplingConfig, SplitConfig
from era5_gam_weather.era5_sampler import discover_era5_files, sample_from_file, split_files_by_day
from era5_gam_weather.feature_builder import FeatureBuilder
from era5_gam_weather.model import WeatherGAMTrainer

DATA_ROOT = "../../data/era5"
YEAR = 2025
MONTHS = [3, 4, 5]

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def train_one_month(month: int) -> None:
    basis_config = BasisConfig()
    sampling_config = SamplingConfig(samples_per_file=12000, seed=42, stratified_time_level=True)
    eval_sampling_config = SamplingConfig(samples_per_file=4000, seed=142, stratified_time_level=True)
    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)

    files = discover_era5_files(DATA_ROOT, YEAR, [month])
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {YEAR}-{month:02d} under {DATA_ROOT}")

    splits = split_files_by_day(
        files,
        train_end=split_config.train_end_day_inclusive,
        val_end=split_config.val_end_day_inclusive,
    )

    trainer = WeatherGAMTrainer(FeatureBuilder(basis_config))

    for path in splits["train"]:
        print(f"[{month:02d} TRAIN] {path}")
        batch = sample_from_file(path, sampling_config)
        trainer.update(batch.features, batch.targets)

    model = trainer.finalize()
    model_path = ARTIFACT_DIR / f"weather_model_bundle_{YEAR}_{month:02d}.npz"
    model.save(str(model_path))

    all_eval_features = {k: [] for k in ["lat", "lon", "altitude_m", "day_of_year", "utc_hour", "local_solar_hour"]}
    all_eval_targets = {k: [] for k in ["T", "P", "U", "V"]}

    for split_name in ["val", "test"]:
        for path in splits[split_name]:
            print(f"[{month:02d} {split_name.upper()}] {path}")
            batch = sample_from_file(path, eval_sampling_config)
            for k, v in batch.features.items():
                all_eval_features[k].append(v)
            for k, v in batch.targets.items():
                all_eval_targets[k].append(v)

    eval_features = {k: np.concatenate(v) for k, v in all_eval_features.items()}
    eval_targets = {k: np.concatenate(v) for k, v in all_eval_targets.items()}
    metrics = model.evaluate(eval_features, eval_targets)

    metrics_path = ARTIFACT_DIR / f"eval_metrics_{YEAR}_{month:02d}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "month": month,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "max_abs": metrics.max_abs,
                "n_samples": metrics.n_samples,
                "basis_config": basis_config.to_dict(),
                "sampling_config": sampling_config.to_dict(),
                "eval_sampling_config": eval_sampling_config.to_dict(),
                "split_config": split_config.to_dict(),
            },
            f,
            indent=2,
        )

    print(f"Saved {model_path}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    for month in MONTHS:
        train_one_month(month)
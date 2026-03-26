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
MONTHS = [5]

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

STATE_PATH = ARTIFACT_DIR / "may_training_state.npz"
PROGRESS_PATH = ARTIFACT_DIR / "may_train_progress.json"
MODEL_PATH = ARTIFACT_DIR / "weather_model_bundle_2025_05.npz"
METRICS_PATH = ARTIFACT_DIR / "eval_metrics_2025_05.json"


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_train_files": []}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def run_training() -> None:
    basis_config = BasisConfig()
    sampling_config = SamplingConfig(samples_per_file=12000, seed=42, stratified_time_level=True)
    eval_sampling_config = SamplingConfig(samples_per_file=4000, seed=142, stratified_time_level=True)
    split_config = SplitConfig(train_end_day_inclusive=23, val_end_day_inclusive=27)

    files = discover_era5_files(DATA_ROOT, YEAR, MONTHS)
    if not files:
        raise FileNotFoundError(f"No ERA5 files found under {DATA_ROOT}")

    splits = split_files_by_day(
        files,
        train_end=split_config.train_end_day_inclusive,
        val_end=split_config.val_end_day_inclusive,
    )

    if STATE_PATH.exists():
        print(f"Resuming from saved state: {STATE_PATH}")
        trainer = WeatherGAMTrainer.load_state(str(STATE_PATH))
    else:
        print("Starting new training state")
        trainer = WeatherGAMTrainer(FeatureBuilder(basis_config))

    progress = load_progress()
    processed_train_files = set(progress.get("processed_train_files", []))

    total_train = len(splits["train"])
    print("Training on May train files...")
    for idx, path in enumerate(splits["train"], start=1):
        if path in processed_train_files:
            print(f"[SKIP {idx}/{total_train}] already processed: {path}")
            continue

        print(f"[TRAIN {idx}/{total_train}] {path}")
        batch = sample_from_file(path, sampling_config)
        trainer.update(batch.features, batch.targets)
        trainer.save_state(str(STATE_PATH))

        processed_train_files.add(path)
        save_progress({"processed_train_files": sorted(processed_train_files)})
        print(
            f"  saved checkpoint | rows so far = {trainer.n_rows} | "
            f"processed train files = {len(processed_train_files)}/{total_train}"
        )

    print("Finalizing model...")
    model = trainer.finalize()
    model.save(str(MODEL_PATH))

    print("Evaluating on validation + test...")
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
        "month": 5,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "max_abs": metrics.max_abs,
        "n_samples": metrics.n_samples,
        "basis_config": basis_config.to_dict(),
        "sampling_config": sampling_config.to_dict(),
        "eval_sampling_config": eval_sampling_config.to_dict(),
        "split_config": split_config.to_dict(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Training completed.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run_training()
"""One-shot training script for the KNN weather service.

Reads ERA5 NetCDF files (or falls back to a synthetic ISA dataset) and writes
``dataset.npz`` + ``metadata.json`` under the artifact directory. The FastAPI
app on startup loads whatever this script produced.

Run with:

    python train_knn.py

Environment variables (all optional):

    ERA5_DATA_ROOT          directory holding era5_YYYY_MM_DD.nc files
    KNN_ARTIFACT_DIR        output directory (default: artifacts/knn_weather)
    KNN_SAMPLES_PER_FILE    rows drawn per ERA5 file (default: 4000)
    KNN_MAX_FILES           cap on files used (0 = no cap; default: 30)
    KNN_K                   neighbours used at inference (default: 8)
    KNN_OOD_THRESHOLD       envelope-excursion fraction allowed (default: 0.05)
"""
from __future__ import annotations

import os
from pathlib import Path

from knn_model import KnnWeatherModel
from training_data import env_path, load_or_build_dataset


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "knn_weather"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def main() -> None:
    artifact_dir = env_path("KNN_ARTIFACT_DIR") or DEFAULT_ARTIFACT_DIR
    era5_root = env_path("ERA5_DATA_ROOT")
    samples_per_file = _env_int("KNN_SAMPLES_PER_FILE", 4000)
    max_files = _env_int("KNN_MAX_FILES", 30)
    k = _env_int("KNN_K", 8)
    ood_threshold = _env_float("KNN_OOD_THRESHOLD", 0.05)

    print(f"[knn-service] Artifact directory: {artifact_dir}")
    print(f"[knn-service] ERA5 root: {era5_root}")
    print(f"[knn-service] samples_per_file={samples_per_file}, max_files={max_files}")
    print(f"[knn-service] k={k}, ood_threshold={ood_threshold}")

    dataset = load_or_build_dataset(
        artifact_dir=artifact_dir,
        era5_root=era5_root,
        samples_per_file=samples_per_file,
        max_files=max_files,
    )
    print(
        f"[knn-service] Dataset built from {dataset.source}: "
        f"{dataset.features.shape[0]} rows, {dataset.features.shape[1]} features"
    )

    model = KnnWeatherModel(
        raw_inputs=dataset.features,
        targets=dataset.targets,
        k=k,
        ood_threshold=ood_threshold,
    )
    model.save(artifact_dir)
    print(f"[knn-service] Saved KNN model to {artifact_dir}")


if __name__ == "__main__":
    main()

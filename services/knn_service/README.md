# KNN Weather Service

A drop-in alternative to `ML_service` that uses a hand-written K-Nearest-Neighbours
regressor instead of a neural network. Same upstream contract
(`POST /predict-weather-physics`), so `weather-service` can route between
the two by inspecting the `weather_source` field on the incoming request.

The implementation is intentionally simple — no scikit-learn, no kd-tree
library, no clever indexing. The training set is small enough
(tens of thousands of rows) that a vectorised NumPy distance computation
finishes well under a millisecond per query.

## How it works

1. **Training.** Read ERA5 NetCDF files, draw a stratified sample of
   `(lat, lon, altitude, time)` cells, store the raw inputs and the four
   weather targets (`temperature_K`, `pressure_Pa`, `wind_u_east_mps`,
   `wind_v_north_mps`) to `artifacts/knn_weather/dataset.npz`. Per-feature
   min/max are recorded as the training envelope.
2. **Inference.** Encode the query as a 7-d feature vector — `lat`, `lon`,
   `altitude`, plus `(sin, cos)` of day-of-year and hour so cyclic gaps
   close — min-max normalise, compute Euclidean distance to every training
   row, take the K smallest (`numpy.argpartition`), and aggregate the
   targets with inverse-distance weighting.
3. **Out-of-distribution guard.** The OOD check runs on the *raw* 5-input
   query (not the cyclic features), so a query is "in distribution" iff
   every raw input sits inside the observed min/max plus a small slack
   controlled by `KNN_OOD_THRESHOLD`.

## Quick start

```bash
# 1. Build the dataset (uses ERA5 if available, synthetic ISA otherwise).
ERA5_DATA_ROOT=~/impact-data/era5 python train_knn.py

# 2. Start the service.
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Smoke test.
curl -s -X POST http://localhost:8000/predict-weather-physics \
  -H "Content-Type: application/json" \
  -d '{"lat": 32.0, "lon": 35.0, "alt": 1500, "sim_datetime": "2025-05-16T12:00:00Z"}' \
  | python -m json.tool
```

`main.py` also self-bootstraps on first start: if no `dataset.npz` exists,
it builds one in-process (synthetic by default, ERA5 if `ERA5_DATA_ROOT`
is set). This is convenient for `docker compose up` in a clean
environment but a real evaluation should always go through
`train_knn.py` so the artifacts on disk reflect the data that was used.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ERA5_DATA_ROOT` | unset | directory holding `era5_YYYY_MM_DD.nc` files |
| `KNN_ARTIFACT_DIR` | `<service>/artifacts/knn_weather` | output directory |
| `KNN_SAMPLES_PER_FILE` | `4000` | rows drawn per ERA5 file |
| `KNN_MAX_FILES` | `30` | cap on ERA5 files used (0 = no cap) |
| `KNN_K` | `8` | number of neighbours used at inference |
| `KNN_OOD_THRESHOLD` | `0.05` | envelope-excursion fraction allowed |

## Files

```
main.py            # FastAPI app + startup bootstrap
schemas.py         # Pydantic request/response (mirrors ML_service)
knn_model.py       # Raw KNN: cyclic encode → normalise → distance → IDW
training_data.py   # ERA5 sampler + synthetic ISA fallback
train_knn.py       # CLI wrapper around training_data + knn_model
artifacts/
  knn_weather/
    dataset.npz    # produced by training
    metadata.json  # produced by training
```

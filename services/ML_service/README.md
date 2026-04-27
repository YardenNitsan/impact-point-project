# ERA5 Weather ML Service — Multi-Task Multi-Head MLP

This project predicts atmospheric weather variables (`temperature_k`,
`pressure_pa`, `wind_u`, `wind_v`) for a physics / ballistic simulator,
given `(lat, lon, altitude_m, datetime)`.

The main model is a **Multi-Task Multi-Head MLP Regressor** built with
TensorFlow / Keras: a shared trunk of fully-connected layers with
BatchNormalization + Dropout, branching into four task-specific regression
heads. The architecture, feature set, normalization scheme, training
configuration, and loss/optimizer choices were all designed by hand —
TensorFlow only provides the layers, the optimizer (Adam), backpropagation,
and standard callbacks.

The repository also contains two **legacy** backends (a sklearn
HistGradientBoosting tree bundle and a hand-written NumPy MLP) that are kept
for historical reproducibility. They are loaded only if you explicitly select
them via `WEATHER_MODEL_BACKEND`. The new path never imports sklearn / joblib /
tree-model code.

## Layout

```
era5_gam_weather/                  # main Python package
    multi_head_mlp_model.py        # main model: build + serving wrapper
    weather_features.py            # deterministic feature engineering
    era5_sampler.py                # streaming sampler over daily NetCDF files
    era5_lookup.py                 # exact-mode bilinear+vertical interpolation
    prediction_service.py          # backend dispatch (used by FastAPI)
    config.py                      # SamplingConfig / SplitConfig dataclasses
    tree_model.py / tree_features.py / numpy_mlp_model.py / numpy_nn.py
                                   # legacy backends (lazy-loaded)

train_multi_head_mlp.py            # MAIN training script
train_may_tree.py                  # legacy tree-model training
train_numpy_mlp.py                 # legacy numpy-MLP training
main.py                            # FastAPI entry point
schemas.py                         # Pydantic request/response models
check_service_quality.py           # quick MAE/RMSE check vs ERA5 ground truth

artifacts/multi_head_mlp_weather/  # produced by training (see below)
```

## Training the main model

### 1. Quick smoke test (May, small)

```bash
ERA5_DATA_ROOT=~/impact-data/era5 \
TRAIN_MONTHS=5 \
TRAIN_SAMPLES_PER_FILE=4000 \
EVAL_SAMPLES_PER_FILE=1000 \
TRAIN_MAX_EPOCHS=5 \
TRAIN_BATCH_SIZE=1024 \
python train_multi_head_mlp.py
```

This finishes in a few minutes on CPU and verifies that the data path,
feature builder, model, callbacks, and saving all work end-to-end.

### 2. Dry run (no training, just print plan + estimated RAM)

```bash
ERA5_DATA_ROOT=~/impact-data/era5 \
TRAIN_MONTHS=5 \
TRAIN_DRY_RUN=1 \
python train_multi_head_mlp.py
```

The script prints the resolved configuration, lists the train/val/test files
it would use, estimates peak RAM, and exits.

### 3. Serious May-only run (32 GB RAM, RTX 5070)

```bash
ERA5_DATA_ROOT=~/impact-data/era5 \
TRAIN_MONTHS=5 \
TRAIN_SAMPLES_PER_FILE=50000 \
EVAL_SAMPLES_PER_FILE=10000 \
TRAIN_MAX_EPOCHS=300 \
TRAIN_BATCH_SIZE=2048 \
TRAIN_LEARNING_RATE=0.001 \
TRAIN_DROPOUT_RATE=0.04 \
TRAIN_EARLY_STOPPING_PATIENCE=30 \
TRAIN_REDUCE_LR_PATIENCE=12 \
TRAIN_SEED=42 \
python train_multi_head_mlp.py
```

Memory headroom: with 31 May files × 50 000 samples = ~1.55 M training rows
× 42 features × float32 ≈ 0.26 GB for the engineered feature matrix. Even
with raw buffers, validation/test arrays, TF runtime, and Python overhead the
process should stay well under 16 GB resident.

If you hit memory pressure, lower `TRAIN_SAMPLES_PER_FILE` first
(`50000 → 30000` halves the in-memory footprint).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `ERA5_DATA_ROOT` | `<project>/data/era5` | directory containing `era5_YYYY_MM_DD.nc` files |
| `WEATHER_ARTIFACT_DIR` | `<project>/artifacts/multi_head_mlp_weather` | output directory |
| `TRAIN_YEAR` | `2025` | year to read |
| `TRAIN_MONTHS` | `5` | comma-separated months, e.g. `4,5` |
| `TRAIN_SAMPLES_PER_FILE` | `50000` | random samples drawn per training file |
| `EVAL_SAMPLES_PER_FILE` | `10000` | random samples drawn per val/test file |
| `TRAIN_MAX_TOTAL_SAMPLES` | `0` | hard cap on total training rows (0 = unlimited) |
| `TRAIN_BATCH_SIZE` | `1024` | mini-batch size for `model.fit` |
| `TRAIN_MAX_EPOCHS` | `250` | epoch ceiling (EarlyStopping usually stops earlier) |
| `TRAIN_LEARNING_RATE` | `0.001` | Adam learning rate |
| `TRAIN_DROPOUT_RATE` | `0.04` | dropout probability in the shared trunk |
| `TRAIN_EARLY_STOPPING_PATIENCE` | `25` | epochs of no `val_loss` improvement before stop |
| `TRAIN_REDUCE_LR_PATIENCE` | `10` | epochs before LR is halved |
| `TRAIN_SPLIT_TRAIN_END_DAY` | `23` | last day-of-month for train split |
| `TRAIN_SPLIT_VAL_END_DAY` | `27` | last day-of-month for val split (rest = test) |
| `TRAIN_SAVE_PLOTS` | `1` | set to `0` to skip plotting |
| `TRAIN_DRY_RUN` | `0` | set to `1` to exit after printing the plan |
| `TRAIN_SEED` | `42` | seeds NumPy + TensorFlow for determinism |

### WSL note

In WSL, place ERA5 NetCDF files under a Linux path (e.g.
`~/impact-data/era5`) — reading from `/mnt/c/...` is dramatically slower
because of the Windows filesystem driver. Use:

```bash
export ERA5_DATA_ROOT=~/impact-data/era5
```

The script will print the resolved `ERA5_DATA_ROOT` at startup so you can
verify which path is being read.

## Training artifacts

After a successful run, `artifacts/multi_head_mlp_weather/` contains:

```
model.keras                # the trained Keras model (best-val checkpoint)
normalization_stats.json   # x_mean, x_std, y_mean, y_std, target transforms
feature_metadata.json      # feature builder config + ordered feature names
metadata.json              # full training config + architecture description
training_history.json      # full Keras history (loss, val_loss, per-head…)
training_metrics.json      # MAE / RMSE / max-abs in physical units, all splits
training_log.csv           # per-epoch log (written incrementally during training)
training_plots/            # PNG plots of loss curves and test metrics
```

## Serving with FastAPI

Once `model.keras` exists in the artifact directory:

```bash
ERA5_DATA_ROOT=~/impact-data/era5 \
WEATHER_MODEL_BACKEND=multi_head_mlp \
WEATHER_ARTIFACT_DIR=$PWD/artifacts/multi_head_mlp_weather \
uvicorn main:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/health | python -m json.tool
curl -s http://localhost:8000/model-info | python -m json.tool

curl -s -X POST http://localhost:8000/predict-weather \
  -H "Content-Type: application/json" \
  -d '{
        "lat": 32.0, "lon": 35.0, "altitude_m": 1500,
        "day_of_year": 135, "utc_hour": 12.0,
        "prediction_mode": "model", "include_real_era5": false
      }'
```

## Service-quality check

`check_service_quality.py` samples 300 random points and compares the model's
prediction against the exact ERA5 lookup, reporting MAE / RMSE / max-abs per
target. It uses `prediction_mode="model"` and `include_real_era5=true` so the
service does both calls.

```bash
python check_service_quality.py
```

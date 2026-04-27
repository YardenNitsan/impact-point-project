from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .era5_lookup import lookup_real_era5_point
from schemas import PredictRequest, PredictResponse, ServiceInfo, WeatherValues

DEFAULT_YEAR = 2025

# Main neural-network backend. Legacy backends remain available for old artifacts.
WEATHER_MODEL_BACKEND = os.getenv("WEATHER_MODEL_BACKEND", "multi_head_mlp").strip().lower()
WEATHER_ARTIFACT_PATH = os.getenv("WEATHER_ARTIFACT_PATH", "")


class WeatherPredictionService:
    def __init__(
        self,
        data_root: Optional[str] = None,
        artifact_dir: Optional[str] = None,
        default_year: int = DEFAULT_YEAR,
        backend: Optional[str] = None,
    ):
        self.default_year = default_year
        self.this_dir = Path(__file__).resolve().parent.parent
        self.project_root = self._find_project_root(self.this_dir)

        self.backend = (backend or WEATHER_MODEL_BACKEND).strip().lower()

        self.data_root = Path(
            data_root or os.getenv("ERA5_DATA_ROOT", str(self.project_root / "data" / "era5"))
        ).resolve()

        if self.backend == "multi_head_mlp":
            default_artifact_dir = self.this_dir / "artifacts" / "multi_head_mlp_weather"
        else:
            default_artifact_dir = self.this_dir / "artifacts"

        self.artifact_dir = Path(
            artifact_dir or os.getenv("WEATHER_ARTIFACT_DIR", str(default_artifact_dir))
        ).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Legacy tree model cache. Import is lazy so old backends are not loaded on the new neural path.
        self.model_cache: Dict[str, Any] = {}
        self.model_candidates = [
            self.artifact_dir / "weather_tree_bundle_2025_04_05.joblib",
        ]

        # Legacy NumPy MLP backend. Import is lazy.
        self.numpy_mlp_model: Optional[Any] = None
        self.numpy_mlp_path: Optional[str] = None
        if WEATHER_ARTIFACT_PATH:
            self._numpy_artifact_path = Path(WEATHER_ARTIFACT_PATH)
        else:
            self._numpy_artifact_path = self.artifact_dir / "numpy_mlp_weather_model.npz"

        # New Keras multi-head backend.
        self.multi_head_mlp_model: Optional[Any] = None
        self.multi_head_mlp_path: Optional[str] = None

        self.default_model_loaded = False

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        start = start.resolve()
        for base in [start] + list(start.parents):
            if (base / "data" / "era5").exists():
                return base
        return start.parent

    @staticmethod
    def _date_from_year_and_day(year: int, day_of_year: float) -> datetime:
        day_int = int(day_of_year)
        if day_int < 1 or day_int > 366:
            raise ValueError(f"Invalid day_of_year: {day_of_year}")
        return datetime(year, 1, 1) + timedelta(days=day_int - 1)

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------
    def warm_start(self) -> None:
        try:
            if self.backend == "multi_head_mlp":
                self._load_multi_head_mlp()
            elif self.backend == "numpy_mlp":
                self._load_numpy_mlp()
            else:
                self.choose_tree_model(self.default_year, 135.0)
            self.default_model_loaded = True
        except FileNotFoundError:
            self.default_model_loaded = False

    # ------------------------------------------------------------------
    # Tree model legacy backend
    # ------------------------------------------------------------------
    def _load_tree_model(self, path: Path):
        from .tree_model import WeatherTreeBundle

        key = str(path.resolve())
        cached = self.model_cache.get(key)
        if cached is not None:
            return cached

        if path.suffix != ".joblib":
            raise ValueError(f"Unsupported model artifact: {path}")

        model = WeatherTreeBundle.load(str(path))
        self.model_cache[key] = model
        return model

    def choose_tree_model(self, year: int, day_of_year: float) -> Tuple[Any, str]:
        dt = self._date_from_year_and_day(year, day_of_year)
        candidates = [
            self.artifact_dir / f"weather_tree_bundle_{dt.year}_{dt.month:02d}.joblib",
            *self.model_candidates,
        ]

        seen = set()
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return self._load_tree_model(path), str(path)

        raise FileNotFoundError(
            "No trained tree model found. Expected one of: " + ", ".join(str(p) for p in candidates)
        )

    # ------------------------------------------------------------------
    # Legacy NumPy MLP backend
    # ------------------------------------------------------------------
    def _load_numpy_mlp(self):
        if self.numpy_mlp_model is not None:
            return self.numpy_mlp_model

        from .numpy_mlp_model import NumpyMLPWeatherModel

        path = self._numpy_artifact_path
        if not path.exists():
            raise FileNotFoundError(f"NumPy MLP model not found: {path}")

        self.numpy_mlp_model = NumpyMLPWeatherModel.load(str(path))
        self.numpy_mlp_path = str(path)
        return self.numpy_mlp_model

    # ------------------------------------------------------------------
    # New Multi-Head MLP backend
    # ------------------------------------------------------------------
    def _load_multi_head_mlp(self):
        if self.multi_head_mlp_model is not None:
            return self.multi_head_mlp_model

        from .multi_head_mlp_model import MultiHeadMLPWeatherModel

        self.multi_head_mlp_model = MultiHeadMLPWeatherModel.load(self.artifact_dir)
        self.multi_head_mlp_path = str(self.artifact_dir)
        return self.multi_head_mlp_model

    # ------------------------------------------------------------------
    # Unified prediction dispatch
    # ------------------------------------------------------------------
    def predict_model(self, req: PredictRequest) -> Tuple[Dict[str, float], str, str]:
        if self.backend == "multi_head_mlp":
            model = self._load_multi_head_mlp()
            predicted = model.predict_one_from_parts(
                lat=req.lat,
                lon=req.lon,
                altitude_m=req.altitude_m,
                day_of_year=req.day_of_year,
                utc_hour=req.utc_hour,
                year=req.year,
            )
            return predicted, self.multi_head_mlp_path or str(self.artifact_dir), "model_multi_head_mlp"

        if self.backend == "numpy_mlp":
            model = self._load_numpy_mlp()
            predicted = model.predict_one(
                lat=req.lat,
                lon=req.lon,
                altitude_m=req.altitude_m,
                day_of_year=req.day_of_year,
                utc_hour=req.utc_hour,
            )
            return predicted, self.numpy_mlp_path or "", "model_numpy_mlp"

        model, model_path = self.choose_tree_model(req.year, req.day_of_year)
        predicted = model.predict_one(
            lat=req.lat,
            lon=req.lon,
            altitude_m=req.altitude_m,
            day_of_year=req.day_of_year,
            utc_hour=req.utc_hour,
        )
        return predicted, model_path, "model_tree"

    def predict_exact(self, req: PredictRequest) -> Tuple[Dict[str, float], Dict]:
        payload = lookup_real_era5_point(
            data_root=str(self.data_root),
            year=req.year,
            day_of_year=req.day_of_year,
            utc_hour=req.utc_hour,
            lat=req.lat,
            lon=req.lon,
            altitude_m=req.altitude_m,
        )
        return payload["real"], payload["meta"]

    def _request_features(self, points: List[PredictRequest]) -> Dict[str, np.ndarray]:
        return {
            "lat": np.array([p.lat for p in points], dtype=np.float32),
            "lon": np.array([p.lon for p in points], dtype=np.float32),
            "altitude_m": np.array([p.altitude_m for p in points], dtype=np.float32),
            "day_of_year": np.array([p.day_of_year for p in points], dtype=np.float32),
            "utc_hour": np.array([p.utc_hour for p in points], dtype=np.float32),
            "local_solar_hour": np.array(
                [(p.utc_hour + p.lon / 15.0) % 24.0 for p in points], dtype=np.float32
            ),
        }

    def _predict_model_batch(self, points: List[PredictRequest]) -> Tuple[List[Dict[str, float]], str, str]:
        if not points:
            return [], "", ""

        if self.backend == "multi_head_mlp":
            model = self._load_multi_head_mlp()
            preds = model.predict_batch_features(self._request_features(points))
            return preds, self.multi_head_mlp_path or str(self.artifact_dir), "model_multi_head_mlp"

        if self.backend == "numpy_mlp":
            model = self._load_numpy_mlp()
            preds = model.predict_batch_features(self._request_features(points))
            return preds, self.numpy_mlp_path or "", "model_numpy_mlp"

        first = points[0]
        model, model_path = self.choose_tree_model(first.year, first.day_of_year)

        same_month = all(
            p.year == first.year
            and self._date_from_year_and_day(p.year, p.day_of_year).month
            == self._date_from_year_and_day(first.year, first.day_of_year).month
            for p in points
        )
        if not same_month:
            predictions = []
            last_model_path = ""
            for p in points:
                pred, last_model_path, _ = self.predict_model(p)
                predictions.append(pred)
            return predictions, last_model_path, "model_tree"

        pred = model.predict(self._request_features(points))
        out = []
        for i in range(len(points)):
            out.append({
                "temperature_k": float(pred["T"][i]),
                "pressure_pa": float(pred["P"][i]),
                "wind_u": float(pred["U"][i]),
                "wind_v": float(pred["V"][i]),
            })
        return out, model_path, "model_tree"

    def predict(self, req: PredictRequest) -> PredictResponse:
        exact_payload: Optional[Dict] = None
        model_path: Optional[str] = None

        if req.prediction_mode == "exact":
            predicted, exact_meta = self.predict_exact(req)
            exact_payload = {"real": predicted, "meta": exact_meta}
            prediction_source = "era5_exact"

        elif req.prediction_mode == "hybrid":
            try:
                predicted, exact_meta = self.predict_exact(req)
                exact_payload = {"real": predicted, "meta": exact_meta}
                prediction_source = "era5_exact"
            except FileNotFoundError:
                predicted, model_path, prediction_source = self.predict_model(req)
            except Exception:
                predicted, model_path, prediction_source = self.predict_model(req)

        else:
            predicted, model_path, prediction_source = self.predict_model(req)

        response = PredictResponse(
            predicted=WeatherValues(**predicted),
            model_used=model_path,
            prediction_source=prediction_source,
        )

        if not req.include_real_era5:
            return response

        if exact_payload is None:
            try:
                exact_real, exact_meta = self.predict_exact(req)
                exact_payload = {"real": exact_real, "meta": exact_meta}
            except Exception:
                return response

        real = exact_payload["real"]
        delta = {
            "temperature_k": predicted["temperature_k"] - real["temperature_k"],
            "pressure_pa": predicted["pressure_pa"] - real["pressure_pa"],
            "wind_u": predicted["wind_u"] - real["wind_u"],
            "wind_v": predicted["wind_v"] - real["wind_v"],
        }

        response.real_era5 = WeatherValues(**real)
        response.prediction_minus_real = WeatherValues(**delta)
        response.comparison_meta = exact_payload["meta"]
        return response

    def batch_predict(self, points: List[PredictRequest]) -> List[PredictResponse]:
        if not points:
            return []

        if all(p.prediction_mode == "model" and not p.include_real_era5 for p in points):
            preds, model_path, source = self._predict_model_batch(points)
            return [
                PredictResponse(
                    predicted=WeatherValues(**pred),
                    model_used=model_path,
                    prediction_source=source,
                )
                for pred in preds
            ]

        return [self.predict(point) for point in points]

    def health(self) -> ServiceInfo:
        if self.backend == "multi_head_mlp":
            candidates = [str(self.artifact_dir / "model.keras")]
        elif self.backend == "numpy_mlp":
            candidates = [str(self._numpy_artifact_path)]
        else:
            candidates = [str(p) for p in self.model_candidates]

        return ServiceInfo(
            ok=True,
            data_root=str(self.data_root),
            artifact_dir=str(self.artifact_dir),
            default_model_loaded=self.default_model_loaded,
            model_cache_size=len(self.model_cache),
            candidate_models=candidates,
        )

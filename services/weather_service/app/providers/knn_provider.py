from __future__ import annotations

from datetime import timezone
from typing import Any

from fastapi import HTTPException

from app.clients.knn_client import KnnClient
from app.schemas import WeatherRequest, WeatherResponse


class KnnProvider:
    """Provider adapter for the KNN weather service.

    The KNN service speaks the same request/response shape as the ML
    service, so this adapter is structurally identical to MachineProvider —
    it lives in its own file so the routing layer reads cleanly.
    """

    def __init__(self, client: KnnClient) -> None:
        self.client = client

    async def fetch(self, req: WeatherRequest) -> WeatherResponse:
        payload = self._build_payload(req)
        raw = await self.client.post_predict(payload)
        return self._normalize_response(raw)

    def _build_payload(self, req: WeatherRequest) -> dict[str, Any]:
        payload = {
            "lat": req.lat,
            "lon": req.lon,
            "alt": req.alt_m,
        }

        if req.sim_datetime is not None:
            dt_utc = (
                req.sim_datetime.replace(tzinfo=timezone.utc)
                if req.sim_datetime.tzinfo is None
                else req.sim_datetime.astimezone(timezone.utc)
            )
            payload["sim_datetime"] = dt_utc.isoformat()

        return payload

    def _normalize_response(self, data: dict[str, Any]) -> WeatherResponse:
        temperature_K = self._pick_float(
            data,
            ["temperature_K", "T0_K", "temperature", "T"],
        )
        pressure_Pa = self._pick_float(
            data,
            ["pressure_Pa", "P0_Pa", "pressure", "P"],
        )
        wind_east_mps = self._pick_float(
            data,
            ["wind_east_mps", "wind_u_east_mps", "wind_u", "u", "U"],
        )
        wind_north_mps = self._pick_float(
            data,
            ["wind_north_mps", "wind_v_north_mps", "wind_v", "v", "V"],
        )

        return WeatherResponse(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            wind_east_mps=wind_east_mps,
            wind_north_mps=wind_north_mps,
            provider_used="knn",
        )

    @staticmethod
    def _pick_float(data: dict[str, Any], keys: list[str]) -> float:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return float(value)
        raise HTTPException(
            status_code=502,
            detail=(
                f"KNN response missing required keys. "
                f"Tried: {keys}. Got: {list(data.keys())}"
            ),
        )

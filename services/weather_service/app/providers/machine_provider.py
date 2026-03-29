from __future__ import annotations

from datetime import timezone
from typing import Any

from fastapi import HTTPException

from app.clients.machine_client import MachineClient
from app.schemas import WeatherRequest, WeatherResponse


class MachineProvider:
    """
    Provider adapter for your ML weather service.

    This file is the ONLY place that should know the machine's exact
    request/response shape.
    """

    def __init__(self, client: MachineClient) -> None:
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

            payload.update({
                "sim_datetime": dt_utc.isoformat(),
                "timestamp": dt_utc.isoformat(),
                "month": dt_utc.month,
                "day": dt_utc.day,
                "hour": dt_utc.hour,
                "minute": dt_utc.minute,
                "day_of_year": dt_utc.timetuple().tm_yday,
                "hour_utc": dt_utc.hour + (dt_utc.minute / 60.0),
            })

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
            provider_used="machine",
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
                f"Machine response missing required keys. "
                f"Tried: {keys}. Got: {list(data.keys())}"
            ),
        )
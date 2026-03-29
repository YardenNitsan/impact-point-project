from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import HTTPException

from app.clients.openmeteo_client import OpenMeteoClient
from app.schemas import WeatherRequest, WeatherResponse


class OpenMeteoProvider:
    """
    Provider adapter for Open-Meteo.
    """

    def __init__(self, client: OpenMeteoClient) -> None:
        self.client = client

    async def fetch(self, req: WeatherRequest) -> WeatherResponse:
        target_dt = self._to_utc(req.sim_datetime)
        target_date = target_dt.strftime("%Y-%m-%d")
        target_hour = target_dt.replace(minute=0, second=0, microsecond=0)
        target_hour_key = target_hour.strftime("%Y-%m-%dT%H:%M")

        params = {
            "latitude": req.lat,
            "longitude": req.lon,
            "elevation": req.alt_m,
            "start_date": target_date,
            "end_date": target_date,
            "hourly": (
                "temperature_2m,"
                "surface_pressure,"
                "wind_speed_10m,"
                "wind_direction_10m"
            ),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }

        raw = await self.client.get_archive(params=params)

        hourly = raw.get("hourly")
        if not isinstance(hourly, dict):
            raise HTTPException(
                status_code=502,
                detail="Open-Meteo response missing 'hourly'",
            )

        times = hourly.get("time", [])
        if target_hour_key not in times:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Requested hour {target_hour_key} not found in "
                    f"Open-Meteo response"
                ),
            )

        idx = times.index(target_hour_key)

        try:
            temperature_c = float(hourly["temperature_2m"][idx])
            surface_pressure_hpa = float(hourly["surface_pressure"][idx])
            wind_speed_mps = float(hourly["wind_speed_10m"][idx])
            wind_dir_deg = float(hourly["wind_direction_10m"][idx])
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Open-Meteo hourly payload missing required values: {exc}",
            ) from exc

        wind_east_mps, wind_north_mps = self._direction_to_uv(
            wind_speed_mps,
            wind_dir_deg,
        )

        return WeatherResponse(
            temperature_K=temperature_c + 273.15,
            pressure_Pa=surface_pressure_hpa * 100.0,
            wind_east_mps=wind_east_mps,
            wind_north_mps=wind_north_mps,
            provider_used="api",
        )

    @staticmethod
    def _to_utc(dt: datetime | None) -> datetime:
        if dt is None:
            return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _direction_to_uv(
        speed_mps: float,
        direction_deg: float,
    ) -> tuple[float, float]:
        """
        Meteorological convention:
        direction_deg is where the wind COMES FROM.
        Convert to eastward (u) and northward (v).
        """
        rad = math.radians(direction_deg)
        east = -speed_mps * math.sin(rad)
        north = -speed_mps * math.cos(rad)
        return east, north
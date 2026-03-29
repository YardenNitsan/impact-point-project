from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class ProviderWeatherSample:
    temperature_K: float
    pressure_Pa: float
    wind_x_mps: float | None
    wind_z_mps: float | None
    wind_east_mps: float | None
    wind_north_mps: float | None
    source: str
    provider: str
    note: str = ""


class WeatherProvider(Protocol):
    is_static: bool

    def fetch(self, *, lat: float, lon: float, alt: float, when: datetime) -> ProviderWeatherSample:
        ...


class StaticWeatherProviderClient:
    is_static = True

    def __init__(
        self,
        *,
        temperature_K: float,
        pressure_Pa: float,
        wind_x_mps: float,
        wind_z_mps: float,
        source: str = "manual",
        provider: str = "manual-override",
        note: str = "manual override",
    ) -> None:
        self._sample = ProviderWeatherSample(
            temperature_K=float(temperature_K),
            pressure_Pa=float(pressure_Pa),
            wind_x_mps=float(wind_x_mps),
            wind_z_mps=float(wind_z_mps),
            wind_east_mps=None,
            wind_north_mps=None,
            source=source,
            provider=provider,
            note=note,
        )

    def fetch(self, *, lat: float, lon: float, alt: float, when: datetime) -> ProviderWeatherSample:
        del lat, lon, alt, when
        return self._sample


class HTTPWeatherProviderClient:
    is_static = False

    def __init__(self, *, name: str, url: str, timeout_s: float = 3.0) -> None:
        normalized_url = (url or "").strip()
        if not normalized_url:
            raise ValueError(f"Missing URL for weather provider '{name}'")

        self.name = str(name)
        self.url = normalized_url
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def fetch(self, *, lat: float, lon: float, alt: float, when: datetime) -> ProviderWeatherSample:
        payload = {
            "lat": float(lat),
            "lon": float(lon),
            "alt": float(alt),
            "sim_datetime": when.isoformat(),
        }

        response = self.session.post(
            self.url,
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Weather provider response must be a JSON object")

        return ProviderWeatherSample(
            temperature_K=_pick_required_float(body, "temperature_K", "T0_K", "T_K", "temp_K"),
            pressure_Pa=_pick_required_float(body, "pressure_Pa", "P0_Pa", "P_Pa", "pressure"),
            wind_x_mps=_pick_optional_float(body, "wind_x_mps", "wind_x"),
            wind_z_mps=_pick_optional_float(body, "wind_z_mps", "wind_z"),
            wind_east_mps=_pick_optional_float(body, "wind_u_east_mps", "u_east_mps", "u", "wind_east_mps"),
            wind_north_mps=_pick_optional_float(body, "wind_v_north_mps", "v_north_mps", "v", "wind_north_mps"),
            source=str(body.get("source", self.name)),
            provider=str(body.get("provider", self.name)),
            note=str(body.get("note", "")),
        )


def _pick_required_float(mapping: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    raise ValueError(f"Missing required weather field. Expected one of: {keys}")


def _pick_optional_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return None
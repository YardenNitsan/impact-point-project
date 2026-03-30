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

    # Optional richer seed fields for calculations mode.
    wind_east_10m_mps: float | None = None
    wind_north_10m_mps: float | None = None
    wind_east_100m_mps: float | None = None
    wind_north_100m_mps: float | None = None

    source: str = ""
    provider: str = ""
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
            wind_east_10m_mps=None,
            wind_north_10m_mps=None,
            wind_east_100m_mps=None,
            wind_north_100m_mps=None,
            source=source,
            provider=provider,
            note=note,
        )

    def fetch(self, *, lat: float, lon: float, alt: float, when: datetime) -> ProviderWeatherSample:
        del lat, lon, alt, when
        return self._sample


class HTTPWeatherProviderClient:
    is_static = False

    def __init__(
        self,
        *,
        name: str,
        url: str,
        timeout_s: float = 3.0,
        requested_source: str | None = None,
    ) -> None:
        normalized_url = (url or "").strip()
        if not normalized_url:
            raise ValueError(f"Missing URL for weather provider '{name}'")

        self.name = str(name)
        self.url = normalized_url
        self.timeout_s = float(timeout_s)
        self.requested_source = str(requested_source).lower() if requested_source else None
        self.session = requests.Session()

    def fetch(self, *, lat: float, lon: float, alt: float, when: datetime) -> ProviderWeatherSample:
        payload = {
            "lat": float(lat),
            "lon": float(lon),
            "alt": float(alt),
            "sim_datetime": when.isoformat(),
        }

        if self.requested_source is not None:
            payload["source"] = self.requested_source

        should_log_api = self.requested_source == "api" or self.name == "api" or self.name == "calculations-seed"

        if should_log_api:
            print("\n========== WEATHER API REQUEST ==========")
            print(f"POST {self.url}")
            print("Payload:")
            print(payload)
            print("========================================\n")

        response = self.session.post(
            self.url,
            json=payload,
            timeout=self.timeout_s,
        )

        if should_log_api:
            print("\n========== WEATHER API RESPONSE ==========")
            print(f"Status code: {response.status_code}")
            try:
                print("JSON body:")
                print(response.json())
            except Exception:
                print("Raw body:")
                print(response.text)
            print("=========================================\n")

        response.raise_for_status()

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Weather provider response must be a JSON object")

        source_used = str(
            body.get(
                "source",
                body.get("provider_used", self.requested_source or self.name),
            )
        )
        provider_used = str(
            body.get(
                "provider",
                body.get("provider_used", self.name),
            )
        )

        return ProviderWeatherSample(
            temperature_K=_pick_required_float(body, "temperature_K", "T0_K", "T_K", "temp_K"),
            pressure_Pa=_pick_required_float(body, "pressure_Pa", "P0_Pa", "P_Pa", "pressure"),
            wind_x_mps=_pick_optional_float(body, "wind_x_mps", "wind_x"),
            wind_z_mps=_pick_optional_float(body, "wind_z_mps", "wind_z"),
            wind_east_mps=_pick_optional_float(
                body,
                "wind_u_east_mps",
                "u_east_mps",
                "u",
                "wind_east_mps",
            ),
            wind_north_mps=_pick_optional_float(
                body,
                "wind_v_north_mps",
                "v_north_mps",
                "v",
                "wind_north_mps",
            ),
            wind_east_10m_mps=_pick_optional_float(
                body,
                "wind_u_east_10m_mps",
                "u_east_10m_mps",
                "wind_east_10m_mps",
            ),
            wind_north_10m_mps=_pick_optional_float(
                body,
                "wind_v_north_10m_mps",
                "v_north_10m_mps",
                "wind_north_10m_mps",
            ),
            wind_east_100m_mps=_pick_optional_float(
                body,
                "wind_u_east_100m_mps",
                "u_east_100m_mps",
                "wind_east_100m_mps",
            ),
            wind_north_100m_mps=_pick_optional_float(
                body,
                "wind_v_north_100m_mps",
                "v_north_100m_mps",
                "wind_north_100m_mps",
            ),
            source=source_used,
            provider=provider_used,
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
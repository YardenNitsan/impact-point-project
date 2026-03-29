from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import List

from modules.atmosphere.weather_client import ProviderWeatherSample, WeatherProvider

EARTH_RADIUS_M = 6_371_000.0
DEG_PER_RAD = 180.0 / math.pi


@dataclass(frozen=True)
class WeatherSample:
    temperature_K: float
    pressure_Pa: float
    wind_x_mps: float
    wind_z_mps: float
    wind_east_mps: float
    wind_north_mps: float
    source: str
    provider: str
    note: str


@dataclass(frozen=True)
class WeatherFetchRecord:
    x_m: float
    altitude_m: float
    elapsed_time_s: float
    lat_deg: float
    lon_deg: float
    state_key: str
    sample: WeatherSample


class TrajectoryWeatherRuntime:
    def __init__(
        self,
        *,
        provider_client: WeatherProvider,
        launch_lat_deg: float,
        launch_lon_deg: float,
        azimuth_rad: float,
        launch_datetime: datetime,
    ) -> None:
        self.provider_client = provider_client
        self.launch_lat_deg = float(launch_lat_deg)
        self.launch_lon_deg = float(launch_lon_deg)
        self.azimuth_rad = float(azimuth_rad)
        self.launch_datetime = launch_datetime
        self.history: List[WeatherFetchRecord] = []
        self.refresh_count = 0

    def get_sample(self, *, x_m: float, altitude_m: float, elapsed_time_s: float) -> WeatherSample:
        lat_deg, lon_deg, when = self._resolve_position_and_time(
            x_m=x_m,
            elapsed_time_s=elapsed_time_s,
        )

        state_key = build_weather_state_key(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            altitude_m=altitude_m,
            when=when,
            provider_name=getattr(self.provider_client, "name", self.provider_client.__class__.__name__),
        )

        if not self.history:
            return self._fetch_and_store(
                x_m=x_m,
                altitude_m=altitude_m,
                elapsed_time_s=elapsed_time_s,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                state_key=state_key,
                when=when,
            )

        last = self.history[-1]

        if state_key == last.state_key:
            return last.sample

        self.refresh_count += 1
        return self._fetch_and_store(
            x_m=x_m,
            altitude_m=altitude_m,
            elapsed_time_s=elapsed_time_s,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            state_key=state_key,
            when=when,
        )

    def lookup_sample_for_x(self, x_m: float) -> WeatherSample:
        if not self.history:
            raise RuntimeError("Weather runtime has no samples yet")

        selected = self.history[0].sample
        target_x = float(x_m)

        for record in self.history:
            if record.x_m <= target_x:
                selected = record.sample
            else:
                break

        return selected

    def summary(self) -> dict:
        if not self.history:
            return {"refresh_count": 0, "fetch_count": 0}

        first = self.history[0]
        last = self.history[-1]

        return {
            "refresh_count": int(self.refresh_count),
            "fetch_count": len(self.history),
            "first_fetch_lat": first.lat_deg,
            "first_fetch_lon": first.lon_deg,
            "last_fetch_lat": last.lat_deg,
            "last_fetch_lon": last.lon_deg,
            "last_fetch_alt_m": last.altitude_m,
            "last_fetch_time_s": last.elapsed_time_s,
            "state_key": last.state_key,
            "provider": last.sample.provider,
        }

    def _resolve_position_and_time(
        self,
        *,
        x_m: float,
        elapsed_time_s: float,
    ) -> tuple[float, float, datetime]:
        east_m = float(x_m) * math.sin(self.azimuth_rad)
        north_m = float(x_m) * math.cos(self.azimuth_rad)

        lat_deg, lon_deg = enu_displacement_to_latlon(
            east_m,
            north_m,
            self.launch_lat_deg,
            self.launch_lon_deg,
        )
        when = self.launch_datetime + timedelta(seconds=float(elapsed_time_s))
        return lat_deg, lon_deg, when

    def _fetch_and_store(
        self,
        *,
        x_m: float,
        altitude_m: float,
        elapsed_time_s: float,
        lat_deg: float,
        lon_deg: float,
        state_key: str,
        when: datetime,
    ) -> WeatherSample:
        provider_sample = self.provider_client.fetch(
            lat=lat_deg,
            lon=lon_deg,
            alt=float(altitude_m),
            when=when,
        )

        wind_x_mps, wind_z_mps, wind_east_mps, wind_north_mps = _normalize_provider_wind(
            provider_sample=provider_sample,
            azimuth_rad=self.azimuth_rad,
        )

        sample = WeatherSample(
            temperature_K=float(provider_sample.temperature_K),
            pressure_Pa=float(provider_sample.pressure_Pa),
            wind_x_mps=float(wind_x_mps),
            wind_z_mps=float(wind_z_mps),
            wind_east_mps=float(wind_east_mps),
            wind_north_mps=float(wind_north_mps),
            source=str(provider_sample.source),
            provider=str(provider_sample.provider),
            note=str(provider_sample.note),
        )

        self.history.append(
            WeatherFetchRecord(
                x_m=float(x_m),
                altitude_m=float(altitude_m),
                elapsed_time_s=float(elapsed_time_s),
                lat_deg=float(lat_deg),
                lon_deg=float(lon_deg),
                state_key=state_key,
                sample=sample,
            )
        )
        return sample


def build_weather_state_key(
    *,
    lat_deg: float,
    lon_deg: float,
    altitude_m: float,
    when: datetime,
    provider_name: str,
    lat_bin_deg: float = 0.25,
    lon_bin_deg: float = 0.25,
    alt_bin_m: float = 250.0,
    time_bin_minutes: int = 60,
) -> str:
    lat_bucket = round(float(lat_deg) / lat_bin_deg)
    lon_bucket = round(float(lon_deg) / lon_bin_deg)
    alt_bucket = round(float(altitude_m) / alt_bin_m)

    rounded_time = when.replace(minute=0, second=0, microsecond=0)

    return (
        f"provider={provider_name}|"
        f"time={rounded_time.isoformat()}|"
        f"lat_bucket={lat_bucket}|"
        f"lon_bucket={lon_bucket}|"
        f"alt_bucket={alt_bucket}"
    )


def _normalize_provider_wind(
    *,
    provider_sample: ProviderWeatherSample,
    azimuth_rad: float,
) -> tuple[float, float, float, float]:
    if provider_sample.wind_x_mps is not None:
        wind_x_mps = float(provider_sample.wind_x_mps)
        wind_z_mps = float(provider_sample.wind_z_mps or 0.0)
        wind_east_mps = wind_x_mps * math.sin(azimuth_rad)
        wind_north_mps = wind_x_mps * math.cos(azimuth_rad)
        return wind_x_mps, wind_z_mps, wind_east_mps, wind_north_mps

    east = float(provider_sample.wind_east_mps or 0.0)
    north = float(provider_sample.wind_north_mps or 0.0)
    wind_x_mps = east * math.sin(azimuth_rad) + north * math.cos(azimuth_rad)
    wind_z_mps = float(provider_sample.wind_z_mps or 0.0)
    return wind_x_mps, wind_z_mps, east, north


def enu_displacement_to_latlon(
    east_m: float,
    north_m: float,
    reference_lat_deg: float,
    reference_lon_deg: float,
) -> tuple[float, float]:
    lat_offset_deg = (float(north_m) / EARTH_RADIUS_M) * DEG_PER_RAD

    cos_lat = math.cos(math.radians(reference_lat_deg))
    if abs(cos_lat) < 1e-12:
        lon_offset_deg = 0.0
    else:
        lon_offset_deg = (
            float(east_m) / (EARTH_RADIUS_M * cos_lat)
        ) * DEG_PER_RAD

    return reference_lat_deg + lat_offset_deg, reference_lon_deg + lon_offset_deg
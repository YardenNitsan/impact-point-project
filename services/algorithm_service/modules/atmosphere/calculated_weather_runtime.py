from __future__ import annotations

import math

from modules.atmosphere.environment import EnvironmentalConditions
from modules.atmosphere.isa import compute_isa_atmosphere_state
from modules.atmosphere.weather_runtime import WeatherSample
from modules.state.state import State3DOF


class CalculatedWeatherRuntime:
    def __init__(
        self,
        *,
        environment: EnvironmentalConditions,
        azimuth_rad: float,
        seed_fetch_count: int = 1,
    ) -> None:
        self.environment = environment
        self.azimuth_rad = float(azimuth_rad)
        self.seed_fetch_count = int(seed_fetch_count)
        self.evaluate_count = 0

    def get_sample(self, *, x_m: float, altitude_m: float, elapsed_time_s: float) -> WeatherSample:
        del x_m, elapsed_time_s
        self.evaluate_count += 1
        return self.sample_for_altitude(altitude_m)

    def sample_for_state(self, state: State3DOF) -> WeatherSample:
        return self.sample_for_altitude(float(state.z))

    def sample_for_altitude(self, altitude_m: float) -> WeatherSample:
        temperature_K, pressure_Pa, _ = compute_isa_atmosphere_state(
            altitude_m=float(altitude_m),
            sea_level_temperature_K=float(self.environment.sea_level_temperature_K),
            sea_level_pressure_Pa=float(self.environment.sea_level_pressure_Pa),
        )

        wind_x_mps, wind_z_mps = self._wind_at_altitude(float(altitude_m))
        wind_east_mps = wind_x_mps * math.sin(self.azimuth_rad)
        wind_north_mps = wind_x_mps * math.cos(self.azimuth_rad)

        return WeatherSample(
            temperature_K=float(temperature_K),
            pressure_Pa=float(pressure_Pa),
            wind_x_mps=float(wind_x_mps),
            wind_z_mps=float(wind_z_mps),
            wind_east_mps=float(wind_east_mps),
            wind_north_mps=float(wind_north_mps),
            source=str(self.environment.data_source),
            provider="internal-calculations",
            note=str(self.environment.diagnostic_note),
        )

    def summary(self) -> dict:
        return {
            "refresh_count": 0,
            "fetch_count": int(self.seed_fetch_count),
            "evaluate_count": int(self.evaluate_count),
        }

    def _wind_at_altitude(self, altitude_m: float) -> tuple[float, float]:
        low_h = 10.0
        high_h = 100.0
        h = max(float(altitude_m), 1.0)

        v10 = self._project_env_wind(
            self.environment.wind_east_10m_mps,
            self.environment.wind_north_10m_mps,
        )
        v100 = self._project_env_wind(
            self.environment.wind_east_100m_mps,
            self.environment.wind_north_100m_mps,
        )

        abs_v10 = max(abs(v10), 0.1)
        abs_v100 = max(abs(v100), 0.1)
        alpha = math.log(abs_v100 / abs_v10) / math.log(high_h / low_h)
        alpha = max(-1.0, min(1.0, alpha))

        sign = 1.0 if v10 >= 0.0 else -1.0
        along_track = sign * abs_v10 * (h / low_h) ** alpha
        return float(along_track), 0.0

    def _project_env_wind(self, east_mps: float, north_mps: float) -> float:
        return (
            float(east_mps) * math.sin(self.azimuth_rad)
            + float(north_mps) * math.cos(self.azimuth_rad)
        )
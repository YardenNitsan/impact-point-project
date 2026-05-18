from __future__ import annotations

from app.providers.knn_provider import KnnProvider
from app.providers.machine_provider import MachineProvider
from app.providers.openmeteo_provider import OpenMeteoProvider
from app.schemas import WeatherRequest, WeatherResponse


class WeatherRouter:
    def __init__(
        self,
        machine_provider: MachineProvider,
        openmeteo_provider: OpenMeteoProvider,
        knn_provider: KnnProvider,
    ) -> None:
        self.machine_provider = machine_provider
        self.openmeteo_provider = openmeteo_provider
        self.knn_provider = knn_provider

    async def get_weather(self, req: WeatherRequest) -> WeatherResponse:
        if req.source == "machine":
            return await self.machine_provider.fetch(req)

        if req.source == "api":
            return await self.openmeteo_provider.fetch(req)

        if req.source == "knn":
            return await self.knn_provider.fetch(req)

        raise ValueError(f"Unsupported weather source: {req.source}")

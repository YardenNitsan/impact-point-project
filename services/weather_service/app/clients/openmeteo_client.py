from __future__ import annotations

import httpx
from fastapi import HTTPException


class OpenMeteoClient:
    """
    Thin HTTP client.
    Only transport logic lives here.
    """

    def __init__(
        self,
        archive_url: str,
        timeout_seconds: float,
    ) -> None:
        self.url = archive_url
        self.timeout_seconds = timeout_seconds

    async def get_archive(self, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(self.url, params=params)

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Open-Meteo request failed: "
                    f"{response.status_code} {response.text}"
                ),
            )

        try:
            return response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Open-Meteo returned invalid JSON: {exc}",
            ) from exc
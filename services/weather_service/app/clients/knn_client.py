from __future__ import annotations

import httpx
from fastapi import HTTPException


class KnnClient:
    """Thin HTTP client for the KNN weather service. Only transport logic."""

    def __init__(
        self,
        base_url: str,
        predict_path: str,
        timeout_seconds: float,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/{predict_path.lstrip('/')}"
        self.timeout_seconds = timeout_seconds

    async def post_predict(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.url, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"KNN provider request failed: "
                    f"{response.status_code} {response.text}"
                ),
            )

        try:
            return response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"KNN provider returned invalid JSON: {exc}",
            ) from exc

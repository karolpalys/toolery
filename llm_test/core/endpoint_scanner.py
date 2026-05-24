from __future__ import annotations

from pydantic import BaseModel


class EndpointInfo(BaseModel):
    port: int
    base_url: str
    model_id: str
    models: list[str]
    server_hint: str  # "vLLM" | "llama.cpp" | "unknown"


async def scan(ports: list[int], timeout: float = 1.0) -> list[EndpointInfo]:
    raise NotImplementedError

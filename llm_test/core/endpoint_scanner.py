from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EndpointInfo(BaseModel):
    port: int
    base_url: str
    model_id: str            # Friendly display name (basename of root if vLLM provides it; else served alias).
    served_model_id: str     # Alias the API expects in the `model` field of requests (id from /v1/models).
    models: list[str]
    server_hint: str  # "vLLM" | "llama.cpp" | "unknown"


def _classify_server(header_value: str | None) -> str:
    if not header_value:
        return "unknown"
    lower = header_value.lower()
    if "vllm" in lower:
        return "vLLM"
    if "llamacpp" in lower or "llama.cpp" in lower or "llama-server" in lower:
        return "llama.cpp"
    return "unknown"


async def _probe_one(client: httpx.AsyncClient, port: int) -> EndpointInfo | None:
    base_url = f"http://localhost:{port}"
    try:
        resp = await client.get(f"{base_url}/v1/models")
    except (httpx.ConnectError, httpx.TimeoutException):
        return None
    except httpx.HTTPError as exc:
        logger.debug("probe %s failed: %s", base_url, exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    models = [item["id"] for item in data if isinstance(item, dict) and "id" in item]
    primary_alias = models[0] if models else "(none loaded)"
    # Prefer basename of `root` — vLLM exposes the on-disk model path there,
    # which is the actual model name. /v1/models `id` is often a short alias
    # like "qwen" that doesn't tell the user which checkpoint is loaded.
    primary_root = ""
    if data and isinstance(data[0], dict):
        primary_root = (data[0].get("root") or "").strip()
    if primary_root:
        # Strip the directory and any "/models/" prefix → "Qwen3.6-35B-A3B-int4-mixed-AutoRound"
        display_name = primary_root.rstrip("/").rsplit("/", 1)[-1] or primary_alias
    else:
        display_name = primary_alias
    return EndpointInfo(
        port=port,
        base_url=base_url,
        model_id=display_name,
        served_model_id=primary_alias,
        models=models,
        server_hint=_classify_server(resp.headers.get("Server")),
    )


async def scan(ports: list[int], timeout: float = 1.0) -> list[EndpointInfo]:
    sem = asyncio.Semaphore(100)

    async def guarded(client: httpx.AsyncClient, port: int) -> EndpointInfo | None:
        async with sem:
            return await _probe_one(client, port)

    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(*[guarded(client, p) for p in ports])
    return [r for r in results if r is not None]

"""Cloud-API adapter: same wire protocol as OpenAIRawAdapter but targets a
remote OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, Anthropic-via-
proxy, etc.) rather than a local vLLM/Ollama/llama.cpp server.

Difference from `raw` is operational, not protocol-level:
  - requires an API key (raw can run without one against open local servers)
  - emits `name = "cloud"` so the DB / rankings can distinguish cloud-hosted
    runs from local-server runs even when the same model name appears in both
"""
from __future__ import annotations

from llm_test.adapters.openai_raw import OpenAIRawAdapter


class CloudAdapter(OpenAIRawAdapter):
    name = "cloud"
    version = "0.1"

    def __init__(self, base_url: str, api_key: str, concurrency: int = 4) -> None:
        if not api_key:
            raise ValueError(
                "CloudAdapter requires an API key. Set OPENAI_API_KEY (or the "
                "equivalent env var for your provider) before launching a cloud run.")
        super().__init__(base_url=base_url, api_key=api_key, concurrency=concurrency)

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime

import httpx

from toolery.core.models import Message, Scenario, ToolCall, TraceResult, TurnUsage
from toolery.core.text_utils import strip_reasoning_tags
from toolery.tools.mock_runtime import MockToolRuntime
from toolery.tools.registry import ToolRegistry

_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 30.0


def _retry_delay(attempt: int, resp: httpx.Response) -> float:
    """Seconds to wait before the next attempt. Honors a numeric ``Retry-After``
    header (RFC 7231 delta-seconds form) when present, otherwise falls back to
    capped exponential backoff (0.5, 1, 2, 4, ... ≤ 30s)."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), _BACKOFF_MAX_SECONDS)
        except ValueError:
            pass  # HTTP-date form unsupported; fall back to backoff
    return min(_BACKOFF_BASE_SECONDS * (2 ** attempt), _BACKOFF_MAX_SECONDS)


def _normalise_base_url(base_url: str) -> str:
    """OpenAI-compatible endpoints serve under `/vN/chat/completions`. Local
    vLLM/llama.cpp callers pass the bare host (`http://localhost:8888`) and
    expect the adapter to add `/v1`. We auto-detect to avoid the
    `/v1/v1/chat/completions` 404 if the caller already included `/v1`."""
    url = base_url.rstrip("/")
    if any(seg.startswith("v") and seg[1:].isdigit() for seg in url.split("/")):
        return url
    return url + "/v1"


class OpenAIRawAdapter:
    name = "raw"
    version = "0.2"

    def __init__(self, base_url: str, api_key: str = "", concurrency: int = 4,
                 max_retries: int = 4) -> None:
        self.base_url = _normalise_base_url(base_url)
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self):
        await self._client.aclose()

    async def _post_with_retry(self, payload: dict, headers: dict) -> tuple[httpx.Response, int]:
        """POST with retry on transient rate-limit (429) / server (5xx) errors.

        Cloud endpoints (MiniMax, OpenRouter, ...) throttle with 429 and may
        return transient 5xx from an overloaded gateway; without retry a single
        throttle gets recorded as a model failure and silently depresses the
        quality score. Other 4xx (bad request, auth) are real client errors and
        are NOT retried.

        Returns (response, latency_ms) where latency_ms times ONLY the
        successful attempt — retry backoff is excluded so the value is a clean
        denominator for effective-throughput math."""
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            resp = await self._client.post(url, json=payload, headers=headers)
            latency_ms = int((time.monotonic() - t0) * 1000)
            retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
            if retryable and attempt < self.max_retries:
                await asyncio.sleep(_retry_delay(attempt, resp))
                continue
            resp.raise_for_status()
            return resp, latency_ms
        raise RuntimeError("unreachable")  # pragma: no cover

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        runtime = MockToolRuntime(scenario)
        reg = ToolRegistry.default()
        tools_schema = reg.openai_schemas(scenario.tools)

        messages: list[dict] = []
        if scenario.system_prompt:
            messages.append({"role": "system", "content": scenario.system_prompt})
        messages.append({"role": "user", "content": scenario.prompt})

        tool_calls_recorded: list[ToolCall] = []
        usage_records: list[TurnUsage] = []
        final_response: str | None = None
        error: str | None = None
        turn_idx = 0
        try:
            for _ in range(scenario.budget.max_turns + 1):
                payload = {
                    "model": model, "messages": messages,
                    "temperature": 0.0,
                }
                # vLLM (and some other servers) reject `tools: []` with HTTP 400.
                # Only include the key when the scenario actually exposes tools.
                if tools_schema:
                    payload["tools"] = tools_schema
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                resp, req_latency_ms = await self._post_with_retry(payload, headers)
                data = resp.json()
                usage = data.get("usage") or {}
                usage_records.append(TurnUsage(
                    turn_index=turn_idx,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=req_latency_ms,
                ))
                msg = data["choices"][0]["message"]
                # Reasoning-model fallback: some servers (vLLM with DeepSeek/QwQ etc.)
                # put the final assistant text in `reasoning` or `reasoning_content` when
                # `content` is null. Normalize so downstream sees one field.
                if not msg.get("content"):
                    fallback = msg.get("reasoning") or msg.get("reasoning_content")
                    if fallback:
                        msg["content"] = fallback
                # Strip <think>/<thinking>/<reasoning> blocks from content. Some
                # models (MiniMax-M2, DeepSeek-R1, QwQ, etc.) emit chain-of-thought
                # inline in content; without stripping, structured-output checks
                # (JSON/YAML schema, regex, response_contains) fail on otherwise
                # correct answers and the model carries its own scratchpad into
                # subsequent turn context.
                msg["content"] = strip_reasoning_tags(msg.get("content"))
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    final_response = msg.get("content")
                    break
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    result, kind = runtime.respond(name, args)
                    tool_calls_recorded.append(ToolCall(
                        index=turn_idx, name=name, args=args,
                        result=result, result_kind=kind,
                    ))
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                    if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                        break
                turn_idx += 1
                if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                    break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[Message.model_validate(m) for m in messages],
            tool_calls=tool_calls_recorded,
            final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"base_url": self.base_url, "model": model},
            usage=usage_records,
        )

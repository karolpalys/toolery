from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import httpx

from llm_test.core.models import Message, Scenario, ToolCall, TraceResult
from llm_test.tools.mock_runtime import MockToolRuntime
from llm_test.tools.registry import ToolRegistry


class HermesAdapter:
    name = "hermes"
    version = "0.1"

    def __init__(self, api_url: str, gateway_url: str, token: str, workspace_id: str = "default"):
        self.api_url = api_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.workspace_id = workspace_id
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self):
        await self._client.aclose()

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
        final_response: str | None = None
        error: str | None = None
        turn_idx = 0
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Workspace-Id": self.workspace_id,
            "Content-Type": "application/json",
        }
        try:
            for _ in range(scenario.budget.max_turns + 1):
                payload = {"model": model, "messages": messages, "tools": tools_schema, "temperature": 0.0}
                resp = await self._client.post(
                    f"{self.api_url}/v1/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                messages.append(msg)
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    final_response = msg.get("content")
                    break
                for tc in tcs:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    result, kind = runtime.respond(name, args)
                    tool_calls_recorded.append(ToolCall(
                        index=turn_idx, name=name, args=args, result=result, result_kind=kind,
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
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[Message.model_validate(m) for m in messages],
            tool_calls=tool_calls_recorded, final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms, error=error,
            adapter_metadata={"api_url": self.api_url, "workspace": self.workspace_id, "model": model},
        )

import json

import httpx
import pytest
import respx

import llm_test.tools.generic  # noqa: F401  register tools
from llm_test.adapters.hermes import HermesAdapter
from llm_test.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule


def _scenario():
    return Scenario(
        id="t-h-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What's the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"}, returns={"temp_c": 7}),
            ],
        },
        scoring=Scoring(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_hermes_runs_loop_and_sends_token():
    first = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "tc_1", "type": "function",
         "function": {"name": "get_weather", "arguments": json.dumps({"location": "Warsaw"})}}
    ]}}]}
    second = {"choices": [{"message": {"role": "assistant", "content": "It's 7°C."}}]}
    route = respx.post("http://localhost:8644/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )
    adapter = HermesAdapter(
        api_url="http://localhost:8644", gateway_url="http://localhost:8642",
        token="hermes-token-xyz", workspace_id="default",
    )
    trace = await adapter.run_scenario(_scenario(), model="any", timeout=10)
    assert trace.error is None
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.final_response == "It's 7°C."
    last_call = route.calls.last.request
    assert last_call.headers.get("authorization") == "Bearer hermes-token-xyz"
    assert last_call.headers.get("x-workspace-id") == "default"

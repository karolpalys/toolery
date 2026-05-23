import json

import httpx
import pytest
import respx

import llm_test.tools.generic  # noqa: F401 — registers tools into ToolRegistry
from llm_test.adapters.openai_raw import OpenAIRawAdapter
from llm_test.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule


def _scenario():
    return Scenario(
        id="t-01-x", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What's the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"},
                                 returns={"temp_c": 7, "condition": "cloudy"}),
            ],
        },
        scoring=Scoring(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_openai_raw_handles_tool_call_then_final():
    first = {
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": json.dumps({"location": "Warsaw"})}}
            ]}
        }]
    }
    second = {
        "choices": [{"message": {"role": "assistant", "content": "It's 7°C and cloudy."}}]
    }
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C and cloudy."

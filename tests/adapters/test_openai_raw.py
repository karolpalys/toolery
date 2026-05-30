import json

import httpx
import pytest
import respx

import toolery.tools.generic  # noqa: F401 — registers tools into ToolRegistry
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule


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
async def test_openai_raw_strips_think_tags_from_content():
    """Regression: MiniMax-M2 (and similar) embed <think>...</think> in the
    same content field as the final answer. Adapter must strip so structured
    rubrics see the clean payload."""
    response = {
        "choices": [{"message": {
            "role": "assistant",
            "content": "<think>The user wants temp in Warsaw. I should answer in JSON.</think>\n\n"
                       '{"temp_c": 7, "condition": "cloudy"}',
        }}]
    }
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response)
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert trace.final_response == '{"temp_c": 7, "condition": "cloudy"}'
    # Also verify the message in conversation history is cleaned (so future
    # turns don't carry the model's scratchpad back to it as context).
    assistant_msgs = [m for m in trace.messages if m.role == "assistant"]
    assert all("<think>" not in (m.content or "") for m in assistant_msgs)


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

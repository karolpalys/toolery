import json

import httpx
import pytest
import respx

import toolery.adapters.openai_raw as openai_raw_mod
import toolery.tools.generic  # noqa: F401 — registers tools into ToolRegistry
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule


@pytest.fixture
def captured_sleeps(monkeypatch):
    """Replace asyncio.sleep in the adapter with a no-op that records the
    requested delays, so retry/backoff tests assert timing without waiting."""
    delays: list[float] = []

    async def _fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(openai_raw_mod.asyncio, "sleep", _fake_sleep)
    return delays


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


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_429_then_succeeds(captured_sleeps):
    """Cloud endpoints (MiniMax etc.) rate-limit with HTTP 429. The adapter must
    retry rather than record the scenario as a model failure."""
    ok = {"choices": [{"message": {"role": "assistant", "content": "It's 7°C."}}]}
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(429),
            httpx.Response(200, json=ok),
        ]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert trace.final_response == "It's 7°C."
    assert route.call_count == 3
    assert len(captured_sleeps) == 2  # backed off before each retry


@pytest.mark.asyncio
@respx.mock
async def test_429_honors_retry_after_header(captured_sleeps):
    """When the server tells us how long to wait via Retry-After, obey it
    instead of the default exponential backoff."""
    ok = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json=ok),
        ]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert captured_sleeps == [7.0]


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_5xx_then_succeeds(captured_sleeps):
    """Transient server errors (502/503 from an overloaded cloud gateway) are
    retryable, same as 429."""
    ok = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=ok)]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_does_not_retry_on_4xx_client_error(captured_sleeps):
    """A 400 (malformed request) or 401 (bad key) is a real client error — retry
    would just waste tokens/quota. Fail fast, record the error, no backoff."""
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(400), httpx.Response(200, json={})]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is not None and "400" in trace.error
    assert route.call_count == 1
    assert captured_sleeps == []


@pytest.mark.asyncio
@respx.mock
async def test_captures_usage_per_turn():
    # Single turn, no tool calls — model answers directly and reports usage.
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "It's cloudy."}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        })
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="m", timeout=30)
    await adapter.aclose()

    assert len(trace.usage) == 1
    u = trace.usage[0]
    assert u.turn_index == 0
    assert u.prompt_tokens == 123
    assert u.completion_tokens == 45
    assert u.latency_ms >= 0
    assert trace.token_totals()[:2] == (123, 45)


@pytest.mark.asyncio
@respx.mock
async def test_captures_usage_across_multiple_turns():
    # Two-request flow: first response is a tool call, second is the final
    # answer. Each turn reports DIFFERENT usage, so we can assert per-turn
    # capture and sequential turn_index — the scenario this feature exists for.
    first = {
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": json.dumps({"location": "Warsaw"})}}
            ]}
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    second = {
        "choices": [{"message": {"role": "assistant", "content": "It's 7°C and cloudy."}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 35, "total_tokens": 185},
    }
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="m", timeout=30)
    await adapter.aclose()

    assert trace.error is None
    assert len(trace.usage) == 2
    assert trace.usage[0].turn_index == 0
    assert trace.usage[1].turn_index == 1
    assert trace.usage[0].prompt_tokens == 100
    assert trace.usage[0].completion_tokens == 20
    assert trace.usage[1].prompt_tokens == 150
    assert trace.usage[1].completion_tokens == 35
    assert trace.token_totals()[:2] == (250, 55)


@pytest.mark.asyncio
@respx.mock
async def test_missing_usage_field_records_zeros():
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        })
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="m", timeout=30)
    await adapter.aclose()
    assert len(trace.usage) == 1
    assert trace.token_totals() == (0, 0, trace.usage[0].latency_ms)


@pytest.mark.asyncio
@respx.mock
async def test_gives_up_after_max_retries(captured_sleeps):
    """Persistent 429 must terminate after max_retries+1 attempts and record an
    error rather than loop forever."""
    route = respx.post("http://localhost:8000/v1/chat/completions").mock(
        return_value=httpx.Response(429)
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x", max_retries=2)
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is not None and "429" in trace.error
    assert route.call_count == 3  # initial + 2 retries
    assert len(captured_sleeps) == 2

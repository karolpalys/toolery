import json
from unittest.mock import AsyncMock, patch

import pytest

from llm_test.adapters.codex import CodexAdapter
from llm_test.core.models import Budget, Category, Scenario, Scoring, Tier


def _scenario():
    return Scenario(
        id="t-cx-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="x", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(),
    )


_FAKE_OUTPUT = "\n".join([
    json.dumps({"event": "tool_call", "name": "get_weather", "args": {"location": "Warsaw"}}),
    json.dumps({"event": "tool_result", "name": "get_weather", "result": {"temp_c": 7}}),
    json.dumps({"event": "final", "text": "It's 7°C."}),
])


@pytest.mark.asyncio
async def test_codex_parses_output():
    adapter = CodexAdapter(cli_path="codex", backend_url="http://localhost:8000")
    with patch("asyncio.create_subprocess_exec") as mp:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_OUTPUT.encode(), b""))
        proc.returncode = 0
        mp.return_value = proc
        trace = await adapter.run_scenario(_scenario(), model="x", timeout=10)
    assert trace.error is None
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C."


_FAKE_OUTPUT_WITH_THINK = "\n".join([
    json.dumps({"event": "final",
                "text": "<think>The user asked for weather; respond with JSON.</think>\n\n"
                        '{"temp_c": 7, "condition": "cloudy"}'}),
])


@pytest.mark.asyncio
async def test_codex_strips_think_tags_from_final_response():
    """Regression: codex CLI may surface a reasoning model's <think> block in
    the 'final' event text. Adapter must strip before downstream scoring."""
    adapter = CodexAdapter(cli_path="codex", backend_url="http://localhost:8000")
    with patch("asyncio.create_subprocess_exec") as mp:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_OUTPUT_WITH_THINK.encode(), b""))
        proc.returncode = 0
        mp.return_value = proc
        trace = await adapter.run_scenario(_scenario(), model="x", timeout=10)
    assert trace.final_response == '{"temp_c": 7, "condition": "cloudy"}'
    assert "<think>" not in (trace.messages[1].content or "")

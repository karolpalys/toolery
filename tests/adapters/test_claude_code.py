import json
from unittest.mock import AsyncMock, patch

import pytest

from llm_test.adapters.claude_code import ClaudeCodeAdapter
from llm_test.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule


def _scenario():
    return Scenario(
        id="t-cc-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="get weather in Warsaw",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"}, returns={"temp_c": 7})
            ]
        },
        scoring=Scoring(),
    )


_FAKE_STREAM = "\n".join([
    json.dumps({"type": "system", "subtype": "init", "session_id": "s-x"}),
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "tu_1", "name": "get_weather",
         "input": {"location": "Warsaw"}}
    ]}}),
    json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"temp_c":7}'}
    ]}}),
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "It's 7°C."}
    ]}}),
    json.dumps({"type": "result", "subtype": "success", "session_id": "s-x"}),
])


@pytest.mark.asyncio
async def test_claude_code_parses_stream_json():
    adapter = ClaudeCodeAdapter(cli_path="claude", backend_url="http://localhost:8000")
    with patch("asyncio.create_subprocess_exec") as mp:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_STREAM.encode(), b""))
        proc.returncode = 0
        mp.return_value = proc
        trace = await adapter.run_scenario(_scenario(), model="local-model", timeout=10)
    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C."
    assert trace.adapter_metadata.get("session_id") == "s-x"

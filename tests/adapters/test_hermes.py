"""Tests for the Hermes CLI subprocess adapter."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

import toolery.tools.generic  # noqa: F401  register mock tools
from toolery.adapters.hermes import HermesAdapter
from toolery.core.models import Budget, Category, Scenario, Scoring, Tier


def _scenario() -> Scenario:
    return Scenario(
        id="t-h-01-cli",
        title="Hermes CLI smoke",
        tier=Tier.EASY,
        category=Category.TOOL_SELECTION,
        domain="generic",
        description="Hermes CLI subprocess",
        prompt="What is the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(),
    )


# Stdout from `hermes chat -q ... -Q` (quiet mode): session_id then body
_FAKE_STDOUT = (
    "session_id: 20260524_010203_aabbcc\n"
    "The weather in Warsaw is 7°C and cloudy.\n"
)

# JSONL line from `hermes sessions export` for the same session
_FAKE_SESSION_JSONL = json.dumps({
    "id": "20260524_010203_aabbcc",
    "model": "MiniMax-M2.7",
    "messages": [
        {"role": "user", "content": "What is the weather in Warsaw?"},
        {
            "role": "assistant",
            "content": "The weather in Warsaw is 7°C and cloudy.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Warsaw"}',
                    },
                }
            ],
        },
    ],
})


@pytest.mark.asyncio
async def test_hermes_cli_parses_stdout_and_session():
    call_count = {"n": 0}

    async def fake_subprocess(*args, **kwargs):
        # First invocation: `hermes chat -q ... -Q ...` → returns stdout with session_id
        # Second invocation: `hermes sessions export <tmp> --session-id <id>` → writes JSONL
        call_count["n"] += 1
        proc = AsyncMock()
        if "chat" in args:
            proc.communicate = AsyncMock(return_value=(_FAKE_STDOUT.encode(), b""))
            proc.returncode = 0
        elif "sessions" in args and "export" in args:
            # find the temp file path argument and write JSONL into it
            export_idx = args.index("export")
            target_path = args[export_idx + 1]
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(_FAKE_SESSION_JSONL + "\n")
            proc.communicate = AsyncMock(return_value=(b"Exported 1 session\n", b""))
            proc.returncode = 0
        else:
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        adapter = HermesAdapter(cli_path="hermes", mcp_bridge=False)
        trace = await adapter.run_scenario(_scenario(), model="MiniMax-M2.7", timeout=10)

    assert trace.error is None
    assert trace.adapter == "hermes"
    assert trace.adapter_metadata["session_id"] == "20260524_010203_aabbcc"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "The weather in Warsaw is 7°C and cloudy."
    # Two subprocess invocations: chat then sessions export
    assert call_count["n"] == 2


_FAKE_STDOUT_WITH_THINK = (
    "session_id: 20260527_140000_minimax\n"
    "<think>The user wants weather in JSON form.</think>\n\n"
    '{"temp_c": 7, "condition": "cloudy"}\n'
)


@pytest.mark.asyncio
async def test_hermes_cli_strips_think_tags_from_final_response():
    """Regression: MiniMax-M2 served through hermes CLI emits <think>...</think>
    inline. Adapter must strip before downstream scoring sees it."""
    async def fake_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_STDOUT_WITH_THINK.encode(), b""))
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        adapter = HermesAdapter(cli_path="hermes", mcp_bridge=False)
        trace = await adapter.run_scenario(_scenario(), model="MiniMax-M2.7", timeout=10)

    assert trace.final_response == '{"temp_c": 7, "condition": "cloudy"}'
    assert "<think>" not in (trace.messages[1].content or "")


@pytest.mark.asyncio
async def test_hermes_cli_handles_missing_session():
    """If session_id is missing from stdout, adapter still returns a trace."""
    async def fake_subprocess(*args, **kwargs):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"just some text without session id\n", b""))
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        adapter = HermesAdapter(cli_path="hermes", mcp_bridge=False)
        trace = await adapter.run_scenario(_scenario(), model="MiniMax-M2.7", timeout=10)

    assert trace.error is None
    assert trace.tool_calls == []
    assert trace.final_response == "just some text without session id"
    assert trace.adapter_metadata["session_id"] is None

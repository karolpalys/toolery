"""Unit tests for the Hermes MCP-bridge config construction.

These cover the pure config-shaping helper that injects our stdio MCP server
into an isolated Hermes config.yaml — without spawning the hermes CLI.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import toolery.tools.generic  # noqa: F401  register mock tools
from toolery.adapters.hermes import HermesAdapter, build_bridge_config
from toolery.core.models import Budget, Category, Scenario, Scoring, Tier


def test_bridge_config_injects_stdio_mcp_server():
    base = {"providers": {"local": {"base_url": "http://x"}}, "toolsets": ["hermes-cli"]}
    cfg = build_bridge_config(
        base,
        server_name="toolery_mock",
        command="/venv/bin/python",
        scenario_json_path="/tmp/s.json",
    )
    server = cfg["mcp_servers"]["toolery_mock"]
    assert server["command"] == "/venv/bin/python"
    assert server["args"] == ["-m", "toolery.tools.mcp_server", "/tmp/s.json"]
    assert server["enabled"] is True


def test_bridge_config_preserves_provider_config():
    """The user's provider/endpoint config must survive so Hermes still hits
    the configured LLM — the bridge only adds the MCP server."""
    base = {"providers": {"local": {"base_url": "http://x"}}, "agent": {"max_turns": 9}}
    cfg = build_bridge_config(
        base, server_name="toolery_mock", command="py", scenario_json_path="/tmp/s.json",
    )
    assert cfg["providers"] == {"local": {"base_url": "http://x"}}
    assert cfg["agent"] == {"max_turns": 9}


def test_bridge_config_does_not_mutate_input():
    base = {"providers": {}}
    build_bridge_config(base, server_name="m", command="py", scenario_json_path="/tmp/s.json")
    assert "mcp_servers" not in base


def _scenario() -> Scenario:
    return Scenario(
        id="t-h-01-cli", title="Hermes MCP bridge", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What is the weather in Warsaw?", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(),
    )


_BRIDGE_STDOUT = (
    "session_id: 20260531_010203_bridge\n"
    "The weather in Warsaw is 7°C and cloudy.\n"
)
_BRIDGE_SESSION_JSONL = json.dumps({
    "id": "20260531_010203_bridge",
    "messages": [
        {"role": "user", "content": "What is the weather in Warsaw?"},
        {
            "role": "assistant",
            "content": "The weather in Warsaw is 7°C and cloudy.",
            "tool_calls": [
                {"id": "tc_1", "function": {
                    "name": "get_weather", "arguments": '{"location": "Warsaw"}'}},
            ],
        },
    ],
})


@pytest.mark.asyncio
async def test_bridge_run_scenario_wires_mcp_server_and_extracts_tool_calls(tmp_path):
    """In MCP-bridge mode the adapter spawns hermes against an isolated
    HERMES_HOME whose config.yaml carries our MCP server, restricts the run to
    that server via ``-t``, drops ``--worktree``, and still extracts tool calls
    from the session export."""
    base_config = tmp_path / "user_config.yaml"
    base_config.write_text(yaml.safe_dump({"providers": {"local": {"base_url": "http://x"}}}))

    captured: dict = {}

    async def fake_subprocess(*args, **kwargs):
        proc = AsyncMock()
        if "chat" in args:
            captured["chat_args"] = args
            home = kwargs.get("env", {}).get("HERMES_HOME")
            captured["hermes_home"] = home
            if home:
                cfg = yaml.safe_load(Path(home, "config.yaml").read_text())
                captured["bridged_config"] = cfg
            proc.communicate = AsyncMock(return_value=(_BRIDGE_STDOUT.encode(), b""))
            proc.returncode = 0
        elif "export" in args:
            target = args[args.index("export") + 1]
            Path(target).write_text(_BRIDGE_SESSION_JSONL + "\n")
            captured["export_env"] = kwargs.get("env", {}).get("HERMES_HOME")
            proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
            proc.returncode = 0
        else:
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        adapter = HermesAdapter(
            cli_path="hermes", mcp_bridge=True, base_config_path=str(base_config),
        )
        trace = await adapter.run_scenario(_scenario(), model="MiniMax-M2.7", timeout=10)

    # tool calls flow through unchanged
    assert trace.error is None
    assert [tc.name for tc in trace.tool_calls] == ["get_weather"]
    assert trace.tool_calls[0].args == {"location": "Warsaw"}

    # chat restricted to our MCP server, no worktree
    chat = list(captured["chat_args"])
    assert "-t" in chat and chat[chat.index("-t") + 1] == "toolery_mock"
    assert "--worktree" not in chat
    assert "--ignore-user-config" not in chat  # would drop our HERMES_HOME config

    # isolated home carries the bridged config (provider preserved + our server)
    cfg = captured["bridged_config"]
    assert cfg["providers"] == {"local": {"base_url": "http://x"}}
    assert cfg["mcp_servers"]["toolery_mock"]["args"][:2] == ["-m", "toolery.tools.mcp_server"]

    # session export ran against the SAME isolated home
    assert captured["export_env"] == captured["hermes_home"]


@pytest.mark.asyncio
async def test_bridge_cleans_up_temp_home(tmp_path):
    """The per-scenario isolated HERMES_HOME is removed after the run."""
    base_config = tmp_path / "user_config.yaml"
    base_config.write_text("providers: {}\n")
    seen: dict = {}

    async def fake_subprocess(*args, **kwargs):
        proc = AsyncMock()
        if "chat" in args:
            seen["home"] = kwargs.get("env", {}).get("HERMES_HOME")
        proc.communicate = AsyncMock(return_value=(_BRIDGE_STDOUT.encode(), b""))
        proc.returncode = 0
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        adapter = HermesAdapter(
            cli_path="hermes", mcp_bridge=True, base_config_path=str(base_config),
        )
        await adapter.run_scenario(_scenario(), model="m", timeout=10)

    assert seen["home"] and not Path(seen["home"]).exists()

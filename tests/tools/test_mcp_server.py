"""End-to-end tests for the stdio MCP server that exposes a scenario's mock
tools (toolery.tools.mcp_server).

The server is launched as a real subprocess and driven over MCP stdio, exactly
as Hermes will drive it — so these tests exercise the wire format the agent
sees, not an in-process shortcut.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from toolery.core.models import (
    Budget,
    Category,
    Scenario,
    Scoring,
    Tier,
    ToolResponseRule,
)

pytestmark = pytest.mark.asyncio

# mcp is an optional dependency; skip the whole module if it's not installed.
pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _weather_scenario() -> Scenario:
    return Scenario(
        id="easy-01-direct-weather",
        title="t",
        tier=Tier.EASY,
        category=Category.TOOL_SELECTION,
        domain="generic",
        description="d",
        prompt="What's the weather in Warsaw right now?",
        tools=["get_weather", "web_search"],
        budget=Budget(max_tool_calls=1, max_turns=2),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"},
                                 returns={"temp_c": 7, "condition": "cloudy"}),
                ToolResponseRule(match="any", returns={"error": "city not found"}),
            ],
        },
        scoring=Scoring(required=[], forbidden=[], partial=[]),
    )


def _write_scenario(tmp_path: Path, scenario: Scenario) -> Path:
    p = tmp_path / "scenario.json"
    p.write_text(scenario.model_dump_json())
    return p


def _server_params(scenario_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "toolery.tools.mcp_server", str(scenario_path)],
    )


async def test_call_tool_returns_scenario_canned_response(tmp_path):
    """get_weather(location=Warsaw) over MCP returns the scenario's canned
    Warsaw weather — identical to what the raw adapter's MockToolRuntime gives."""
    scenario_path = _write_scenario(tmp_path, _weather_scenario())

    async with stdio_client(_server_params(scenario_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_weather", {"location": "Warsaw"})

    assert result.content, "expected tool result content"
    payload = json.loads(result.content[0].text)
    assert payload == {"temp_c": 7, "condition": "cloudy"}


async def test_list_tools_exposes_scenario_tools_with_real_schema(tmp_path):
    """The server advertises exactly the scenario's tools, each carrying the
    registry's real parameter schema (so the agent calls them correctly)."""
    scenario_path = _write_scenario(tmp_path, _weather_scenario())

    async with stdio_client(_server_params(scenario_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()

    by_name = {t.name: t for t in listed.tools}
    assert set(by_name) == {"get_weather", "web_search"}
    weather_schema = by_name["get_weather"].inputSchema
    assert weather_schema["properties"]["location"]["type"] == "string"
    assert "location" in weather_schema.get("required", [])


async def test_args_mismatch_falls_through_to_any_rule(tmp_path):
    """An unmatched location hits the scenario's `match: any` fallback — the
    server mirrors MockToolRuntime's rule resolution, not just the happy path."""
    scenario_path = _write_scenario(tmp_path, _weather_scenario())

    async with stdio_client(_server_params(scenario_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_weather", {"location": "Atlantis"})

    payload = json.loads(result.content[0].text)
    assert payload == {"error": "city not found"}


def _files_scenario() -> Scenario:
    """Scenario whose tools include read_file so that search_files alias applies."""
    return Scenario(
        id="easy-02-files",
        title="t",
        tier=Tier.EASY,
        category=Category.TOOL_SELECTION,
        domain="generic",
        description="d",
        prompt="Read a file",
        tools=["read_file", "list_files"],
        budget=Budget(max_tool_calls=2, max_turns=2),
        tool_responses={
            "read_file": [
                ToolResponseRule(match="any", returns={"content": "hello"}),
            ],
            "list_files": [
                ToolResponseRule(match="any", returns={"files": ["a.py"]}),
            ],
        },
        scoring=Scoring(required=[], forbidden=[], partial=[]),
    )


async def test_alias_tools_appear_in_list_and_route_to_canonical(tmp_path):
    """When a scenario includes read_file, the server also advertises
    search_files (alias) in list_tools and routes call_tool(search_files)
    to the same canonical mock response as call_tool(read_file)."""
    scenario_path = _write_scenario(tmp_path, _files_scenario())

    async with stdio_client(_server_params(scenario_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            by_name = {t.name: t for t in listed.tools}
            # Alias must appear in list
            assert "search_files" in by_name, (
                f"expected search_files alias in listed tools, got: {set(by_name)}"
            )
            # list_directory alias for list_files must appear too
            assert "list_directory" in by_name

            # Calling the alias should return the same canonical mock response
            alias_result = await session.call_tool("search_files", {"path": "."})
            canonical_result = await session.call_tool("read_file", {"path": "."})

    alias_payload = json.loads(alias_result.content[0].text)
    canonical_payload = json.loads(canonical_result.content[0].text)
    assert alias_payload == canonical_payload == {"content": "hello"}

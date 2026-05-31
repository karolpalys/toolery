"""Stdio MCP server exposing a single scenario's mock tools.

Run as:

    python -m toolery.tools.mcp_server <scenario.json>

The scenario JSON is a serialized :class:`toolery.core.models.Scenario` (write
it with ``scenario.model_dump_json()``). On start the server loads that
scenario, registers one MCP tool per ``scenario.tools`` with the tool's real
OpenAI parameter schema, and routes every call through
:class:`~toolery.tools.mock_runtime.MockToolRuntime` — i.e. the *same* scripted
responses the raw adapter serves. This lets an MCP-aware agent (Hermes) emit
the exact tool names/args the benchmark scorer asserts on.

See docs/hermes-mcp-bridge.md for the full design.
"""
from __future__ import annotations

import json
import sys

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from toolery.core.models import Scenario
from toolery.tools.mock_runtime import MockToolRuntime
from toolery.tools.registry import ToolRegistry


def _load_registry() -> ToolRegistry:
    """Import the tool modules so the default registry is populated, then
    return it. Mirrors the import side-effect the CLI relies on."""
    from toolery.tools import api_db, domain, generic, terminal  # noqa: F401

    return ToolRegistry.default()


def _render_content(result: object, kind: str) -> list[types.TextContent]:
    """Serialize a MockToolRuntime result the same way the raw adapter feeds it
    back to the model: plain strings pass through, everything else is JSON."""
    text = result if isinstance(result, str) else json.dumps(result)
    return [types.TextContent(type="text", text=text)]


def build_server(scenario: Scenario) -> Server:
    reg = _load_registry()
    runtime = MockToolRuntime(scenario)
    server = Server("toolery-mock")

    # Native Hermes tools that are semantically equivalent to a mock tool.
    # We register alias entries only when the canonical target is in this
    # scenario, so Hermes can route through MCP and still get the right response.
    ALIASES = {
        "Bash": "bash_exec", "terminal": "bash_exec", "search_files": "read_file",
        "grep": "read_file", "list_directory": "list_files",
    }
    alias_for = {a: c for a, c in ALIASES.items() if c in set(scenario.tools)}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for name in scenario.tools:
            spec = reg.get(name)
            fn = spec.json_schema.get("function", {})
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            tools.append(types.Tool(
                name=name,
                description=spec.description,
                inputSchema=params,
            ))
        # Emit alias tools reusing the canonical spec but with the alias name.
        for alias, canonical in alias_for.items():
            try:
                spec = reg.get(canonical)
            except KeyError:
                continue
            fn = spec.json_schema.get("function", {})
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            tools.append(types.Tool(
                name=alias,
                description=spec.description,
                inputSchema=params,
            ))
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        # Map an incoming alias name to its canonical before calling runtime.
        name = alias_for.get(name, name)
        result, kind = runtime.respond(name, arguments or {})
        return _render_content(result, kind)

    return server


async def _amain(scenario_path: str) -> None:
    scenario = Scenario.model_validate_json(open(scenario_path, encoding="utf-8").read())
    server = build_server(scenario)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    import asyncio

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m toolery.tools.mcp_server <scenario.json>", file=sys.stderr)
        return 2
    asyncio.run(_amain(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

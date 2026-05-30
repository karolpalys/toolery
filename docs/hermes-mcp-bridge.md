# Plan: MCP-bridge mock tools → Hermes (apples-to-apples eval)

**Status:** planned, not implemented. Pick this up in a fresh session.

## Goal

Let the `hermes` adapter pass the existing benchmark scenarios on equal footing
with `raw`. Today hermes runs as a standalone agent with its **own** toolset, so
it never emits the benchmark's mock tool calls (`get_weather`, `send_email`, …)
and almost every tool-calling assertion fails. The fix: expose the benchmark's
mock tools to Hermes over **MCP** and restrict Hermes to *only* those tools, so
its tool calls carry the names/args the scenarios assert on. The existing
scoring then works 1:1, and raw-vs-hermes becomes a clean measure of "what the
agent loop adds" on an identical task surface.

## Confirmed building blocks (verified via `hermes` CLI)

- `hermes mcp add` — register a custom MCP server.
- `hermes chat -t/--toolsets <csv>` — restrict the run to specific toolsets.
- `hermes mcp serve` / `hermes tools` — MCP + tool management.
- The hermes adapter (`toolery/adapters/hermes.py`) **already** reconstructs
  `tool_calls` from the session via `hermes sessions export` → `_extract_session`.
  So once Hermes calls the mock tools, they are captured with correct
  names/args and flow into the existing scorer unchanged.

## Architecture

```
toolery run --adapter hermes
   └─ for each scenario:
        1. write scenario (incl. tool_responses) to a temp JSON
        2. configure Hermes to use ONLY our MCP server
        3. hermes chat -q <prompt> -t <our-toolset> --ignore-user-config
              └─ Hermes calls get_weather(...) over MCP
                    └─ MCP server routes to MockToolRuntime(scenario).respond()
                          └─ returns the same canned response as `raw`
        4. adapter extracts tool_calls from the session  (already implemented)
        5. existing scorer checks tool_called / tool_args_* / response_*  (unchanged)
```

## Components to build

### 1. MCP server exposing the mock tools — `toolery/tools/mcp_server.py`
- A stdio MCP server (add the `mcp` Python package as an optional dependency).
- Entry: `python -m toolery.tools.mcp_server <scenario.json>`.
- On start: load the scenario JSON, build `MockToolRuntime(scenario)` and the
  tool list from `ToolRegistry.default().openai_schemas(scenario.tools)`.
- Register one MCP tool per scenario tool, with the same name + JSON schema.
- Each MCP tool handler calls `runtime.respond(name, args)` and returns the
  result (string or JSON) — identical semantics to `OpenAIRawAdapter`.
- **Per-scenario responses are the key design point**: the server is
  scenario-specific, configured via the JSON path passed on argv.

### 2. Wire Hermes to the server — modify `toolery/adapters/hermes.py`
- Before each scenario: dump the `Scenario` (tools + `tool_responses`) to a temp
  JSON; build an isolated Hermes config (or `hermes mcp add`) pointing at
  `python -m toolery.tools.mcp_server <that.json>`.
- Run `hermes chat -q <prompt> -t <our-toolset-name> --ignore-user-config`
  (keep `-Q`, `--ignore-rules`; **drop `--worktree`** — no filesystem work is
  needed when tools come from MCP, which also removes the git-worktree
  serialization constraint).
- Leave `_extract_session` as-is; it already yields the tool calls.

### 3. Concurrency
- With `--worktree` gone and per-scenario isolated MCP configs, the current
  `_run_lock` serialization (added for worktree safety) can likely be relaxed.
  Re-evaluate: if each scenario gets a unique temp config + server process,
  hermes scenarios may run concurrently again. Confirm no shared Hermes state
  (sessions DB, config) collides before removing the lock.

## Verification

1. Unit: start `mcp_server.py` for `easy-01-direct-weather`, call `get_weather`
   over MCP, assert it returns the scenario's canned Warsaw weather.
2. Integration: `easy-01` through the hermes adapter → trace contains
   `get_weather(location="Warsaw")` and the scenario **passes** (mirrors the
   verified `raw` result).
3. Compare raw vs hermes on a small tier — both should now score, with the gap
   reflecting the agent loop rather than missing tools.

## Open questions to resolve during implementation

- Exact Hermes MCP config format: `hermes mcp add` flags vs. a config.yaml entry
  vs. an `--ignore-user-config` + injected config. Pick the most scriptable.
- The exact toolset name to pass to `-t` so only the MCP tools are active
  (inspect `hermes tools` / `hermes mcp list` output).
- Whether the `mcp` server tool result needs MCP "structured content" vs plain
  text for Hermes to parse args/results faithfully.

## Out of scope

- Native-tool evaluation (scoring Hermes on real file/git/shell outcomes) —
  that is a different effort (new outcome-based scoring checks).

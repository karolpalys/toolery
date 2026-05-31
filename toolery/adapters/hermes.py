"""Hermes CLI subprocess adapter.

Hermes 0.14+ does not expose an OpenAI-compatible HTTP API (the gateway on
ports 8642/8644 is a messaging gateway for WhatsApp/Slack/Telegram, not for
chat-completions). We spawn the `hermes` CLI in single-query mode and
reconstruct the trace from the session SQLite store via `hermes sessions
export`.

Follows the same subprocess pattern as the other CLI adapters.

Hermes injects its own large system prompt and its own toolset (hermes-cli,
MCP, skills). Our mock tool names (``get_weather``, ``send_email`` ...) are
described inline in the user prompt but are not registered as Hermes-callable
functions, so tool_call-based assertions will mostly fail. Scenarios that
test restraint, long-context comprehension, or refusal will still pass
meaningfully through this adapter.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime

from toolery.core.models import Message, Scenario, ToolCall, TraceResult
from toolery.core.text_utils import strip_reasoning_tags


def build_bridge_config(
    base_config: dict,
    *,
    server_name: str,
    command: str,
    scenario_json_path: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Return a copy of ``base_config`` with our stdio MCP server injected.

    The user's existing config (providers/agent settings) is preserved, but the
    LLM endpoint is overridden with the run's ``base_url``/``api_key`` so Hermes
    hits the SAME model as the raw adapter — apples-to-apples. Without this the
    bridge inherits the user's stale ``model.base_url`` (e.g. a host that is no
    longer reachable) and every scenario dies with a connection error that the
    scorer then mislabels ``model_crash``. When ``base_url`` is None the user's
    model config is left untouched.

    We also add an ``mcp_servers`` entry pointing at ``python -m
    toolery.tools.mcp_server <scenario.json>``; tool restriction to *only* this
    server is done at the CLI via ``-t``.
    """
    import copy

    cfg = copy.deepcopy(base_config)
    if base_url:
        model = cfg.setdefault("model", {})
        model["base_url"] = base_url
        # OpenAI-compatible custom endpoint (vLLM/llama.cpp): force the provider
        # so Hermes honors base_url instead of routing to a named provider.
        model["provider"] = "custom"
        if api_key:
            model["api_key"] = api_key
    cfg.setdefault("mcp_servers", {})[server_name] = {
        "command": command,
        "args": ["-m", "toolery.tools.mcp_server", scenario_json_path],
        "enabled": True,
    }
    return cfg


def _build_prompt(scenario: Scenario) -> str:
    """Inline tool descriptions + task prompt + budget."""
    from toolery.tools.registry import ToolRegistry
    reg = ToolRegistry.default()
    lines = []
    for t in scenario.tools:
        try:
            spec = reg.get(t)
            params = spec.json_schema.get("function", {}).get("parameters", {}).get("properties", {})
            lines.append(f"- {t}({', '.join(params.keys())}): {spec.description}")
        except KeyError:
            lines.append(f"- {t}")
    tools_block = "\n".join(lines) if lines else "(no tools)"
    return (
        f"You may use the following tools for this task:\n{tools_block}\n\n"
        f"Task: {scenario.prompt}\n\n"
        f"Budget: at most {scenario.budget.max_tool_calls} tool calls, "
        f"{scenario.budget.max_turns} turns."
    )


class HermesAdapter:
    """Subprocess adapter targeting the ``hermes`` CLI (Hermes Agent v0.14+)."""

    name = "hermes"
    version = "0.2-cli"
    # Hermes exposes MCP tools to the model as ``mcp_<server>_<tool>``. The
    # scorer asserts on the bare tool name, so we strip this prefix on extract.
    _MCP_SERVER_NAME = "toolery_mock"

    # Native Hermes tools that are semantically equivalent to a mock tool.
    # Only applied when the canonical target is in the scenario's allowed tools.
    _NATIVE_ALIASES = {
        "Bash": "bash_exec", "bash": "bash_exec", "terminal": "bash_exec",
        "search_files": "read_file", "grep": "read_file",
        "list_directory": "list_files", "ls": "list_files",
    }

    def __init__(
        self,
        cli_path: str = "hermes",
        timeout_per_scenario: int = 1800,
        ignore_user_config: bool = False,
        provider: str | None = None,
        use_worktree: bool = True,
        mcp_bridge: bool = True,
        base_config_path: str | None = None,
        # The run's LLM endpoint. In bridge mode this overrides the user
        # config's model.base_url/api_key so Hermes hits the SAME model the raw
        # adapter does (see build_bridge_config). Defaults preserve the user's
        # configured endpoint when not supplied.
        base_url: str | None = None,
        api_key: str | None = None,
        # Kept for backward-compat with existing CLI wiring; ignored.
        api_url: str | None = None,
        gateway_url: str | None = None,
        token: str | None = None,
        workspace_id: str | None = None,
        # Opt-in skills mode: include the user's Hermes skills toolset alongside
        # the mock MCP server so we can A/B test whether skills help. When True,
        # ``-t toolery_mock,skills`` is used (skills auto-injected) and
        # ``--ignore-rules`` is omitted. When False (default): exactly as before.
        skills_mode: bool = False,
    ) -> None:
        resolved = shutil.which(cli_path) if cli_path == "hermes" else cli_path
        self.cli_path = resolved or cli_path
        self.timeout = timeout_per_scenario
        # Default False: keep user's config.yaml (provider/base_url/api_key) so we hit
        # the local LLM endpoint configured there. Set True only when you genuinely
        # want to bypass user config (testing in isolation).
        self.ignore_user_config = ignore_user_config
        self.provider = provider
        # Default True: --worktree runs Hermes in an isolated git worktree so any
        # files it creates / edits / commits don't pollute the caller's tree.
        # Without it Hermes' edit_file / git_commit / write_file tools act on
        # the literal cwd — which means a benchmark like 'rename foo across the
        # codebase' will actually modify toolery/ and create stray commits.
        self.use_worktree = use_worktree
        # MCP-bridge mode (default): expose the benchmark's mock tools to Hermes
        # over MCP and restrict the run to *only* those tools, so Hermes emits
        # the exact tool names/args the scorer asserts on — apples-to-apples
        # with the raw adapter. Each scenario runs in an isolated HERMES_HOME
        # (its own config.yaml + sessions DB), so no --worktree and no shared
        # state: the _run_lock serialization below is unnecessary in this mode.
        # See docs/hermes-mcp-bridge.md.
        self.mcp_bridge = mcp_bridge
        # Source config copied into each scenario's isolated HERMES_HOME so the
        # configured LLM provider/endpoint still resolves. Defaults to the
        # user's ~/.hermes/config.yaml; missing file → empty base config.
        self.base_config_path = base_config_path or os.path.expanduser(
            "~/.hermes/config.yaml"
        )
        # Normalise to a `/vN` endpoint the same way the raw adapter does, so
        # both hit the identical URL whether the caller passed `localhost:8888`
        # or `localhost:8888/v1` — true apples-to-apples.
        if base_url:
            from toolery.adapters.openai_raw import _normalise_base_url
            base_url = _normalise_base_url(base_url)
        self.base_url = base_url
        self.api_key = api_key
        # `--worktree` runs `git worktree add`/`remove` against the shared repo,
        # whose .git locks are NOT safe to touch from several processes at once.
        # With concurrency > 1 the parallel hermes scenarios collide on those
        # locks and crash. Serialize hermes invocations through this lock so the
        # worktree git ops never overlap; other adapters (raw/cloud) still run
        # concurrently because each adapter instance has its own lock.
        self._run_lock = asyncio.Lock()
        self.skills_mode = skills_mode

    async def run_scenario(
        self, scenario: Scenario, model: str, timeout: int
    ) -> TraceResult:
        if self.mcp_bridge:
            return await self._run_bridge(scenario, model, timeout)
        return await self._run_standalone(scenario, model, timeout)

    def _load_base_config(self) -> dict:
        """Read the source config to copy into each isolated HERMES_HOME.
        Missing/empty file → empty base (Hermes built-in defaults apply)."""
        import yaml

        try:
            with open(self.base_config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            return {}
        return data if isinstance(data, dict) else {}

    async def _run_bridge(
        self, scenario: Scenario, model: str, timeout: int
    ) -> TraceResult:
        """MCP-bridge mode: expose the scenario's mock tools to Hermes over MCP
        in an isolated HERMES_HOME and restrict the run to only those tools."""
        import sys

        import yaml

        started = time.monotonic()
        server_name = self._MCP_SERVER_NAME
        home = tempfile.mkdtemp(prefix="hermes-mcp-")
        try:
            scenario_json = os.path.join(home, "scenario.json")
            with open(scenario_json, "w", encoding="utf-8") as f:
                f.write(scenario.model_dump_json())

            cfg = build_bridge_config(
                self._load_base_config(),
                server_name=server_name,
                command=sys.executable,
                scenario_json_path=scenario_json,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            # No --worktree (tools come from MCP, no filesystem work) and no
            # --ignore-user-config (it would discard the HERMES_HOME config we
            # just wrote, dropping our MCP server). -t <server> restricts the
            # run to only our mock tools.
            # In skills mode: include user's skills toolset alongside the mock
            # server and let Hermes auto-inject its skills (omit --ignore-rules).
            if self.skills_mode:
                toolsets = f"{server_name},skills"
            else:
                toolsets = server_name
            cmd = [
                self.cli_path, "chat",
                "-q", scenario.prompt,
                "-Q",
                "--max-turns", str(max(scenario.budget.max_turns + 1, 2)),
                "-t", toolsets,
            ]
            if not self.skills_mode:
                cmd.insert(cmd.index("-t"), "--ignore-rules")
            if self.provider:
                cmd.extend(["--provider", self.provider])
            if model:
                cmd.extend(["-m", model])

            env = {**os.environ, "HERMES_HOME": home}
            # Isolated home per scenario → no shared state, run concurrently
            # (no _run_lock).
            return await self._execute_and_trace(
                scenario, model, cmd, started, env=env, use_lock=False
            )
        finally:
            shutil.rmtree(home, ignore_errors=True)

    async def _run_standalone(
        self, scenario: Scenario, model: str, timeout: int
    ) -> TraceResult:
        started = time.monotonic()
        prompt = _build_prompt(scenario)
        cmd = [
            self.cli_path, "chat",
            "-q", prompt,
            "-Q",
            "--max-turns", str(max(scenario.budget.max_turns + 1, 2)),
            "--ignore-rules",
        ]
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        if self.provider:
            cmd.extend(["--provider", self.provider])
        if self.use_worktree:
            cmd.append("--worktree")
        if model:
            cmd.extend(["-m", model])
        return await self._execute_and_trace(
            scenario, model, cmd, started, env=None, use_lock=True
        )

    async def _execute_and_trace(
        self,
        scenario: Scenario,
        model: str,
        cmd: list[str],
        started: float,
        *,
        env: dict | None,
        use_lock: bool,
    ) -> TraceResult:
        error: str | None = None
        stdout_bytes = b""
        stderr_bytes = b""

        async def _spawn() -> None:
            nonlocal error, stdout_bytes, stderr_bytes
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    # Use the full HERMES_TIMEOUT cap per scenario — agentic
                    # hermes runs are slow, and the run-wide heartbeat keeps
                    # the TUI from false-killing a long scenario. The
                    # per-scenario budget is not used to shrink this (it would
                    # cap fast-budget scenarios and trip "hermes: timeout" on
                    # a slow backend).
                    timeout=self.timeout,
                )
                if proc.returncode != 0:
                    error = stderr_bytes.decode(errors="replace")[:500]
            except TimeoutError:
                error = "hermes: timeout"
                # Kill the timed-out process so it releases any resources
                # (git worktree / MCP server) before the next scenario.
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass

        try:
            if use_lock:
                # Serialize: only one hermes subprocess at a time (see _run_lock)
                # so concurrent --worktree git ops can't collide and crash the run.
                async with self._run_lock:
                    await _spawn()
            else:
                await _spawn()
        except FileNotFoundError as e:
            error = f"hermes CLI not found: {e}"

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")
        session_id, final_response_stdout = self._parse_stdout(stdout_text)
        # Hermes prints `session_id:` to STDERR (not stdout); without recovering
        # it here the session export never runs and tool_calls come back empty.
        if session_id is None:
            session_id = self._parse_session_id(stderr_text)

        # Pull structured tool_calls + final assistant message from session export.
        tool_calls: list[ToolCall] = []
        final_response_db: str | None = None
        if session_id:
            try:
                tool_calls, final_response_db = await self._extract_session(
                    session_id, allowed_tools=set(scenario.tools), env=env
                )
            except Exception as e:
                if error is None:
                    error = f"session export failed: {type(e).__name__}: {e}"

        final_response = final_response_db or final_response_stdout
        # Strip <think>/<thinking>/<reasoning> blocks emitted inline by
        # reasoning models (MiniMax-M2, DeepSeek-R1, QwQ, etc.). Without this,
        # structured-output rubrics see the scratchpad and fail on otherwise
        # correct payloads.
        final_response = strip_reasoning_tags(final_response)
        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[
                Message(role="user", content=scenario.prompt),
                Message(role="assistant", content=final_response),
            ],
            tool_calls=tool_calls,
            final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"session_id": session_id, "model": model, "cli": "hermes"},
        )

    @classmethod
    def _strip_mcp_prefix(cls, name: str) -> str:
        """``mcp_toolery_mock_get_weather`` → ``get_weather``. No-op for the
        standalone path where tool names are already bare."""
        prefix = f"mcp_{cls._MCP_SERVER_NAME}_"
        return name[len(prefix):] if name.startswith(prefix) else name

    @classmethod
    def _normalize_alias(cls, name: str, allowed: set[str] | None) -> str:
        """Map a native Hermes tool name to its canonical mock equivalent.

        The alias is only applied when the canonical name is present in
        ``allowed`` (the scenario's tools), so we never invent a tool the
        scenario doesn't use — no false passes. Pass ``allowed=None`` to map
        unconditionally (useful in tests / standalone path).
        """
        canon = cls._NATIVE_ALIASES.get(name)
        if canon and (allowed is None or canon in allowed):
            return canon
        return name

    @staticmethod
    def _parse_session_id(text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("session_id:"):
                return line.split(":", 1)[1].strip()
        return None

    def _parse_stdout(self, text: str) -> tuple[str | None, str | None]:
        session_id: str | None = None
        body_lines: list[str] = []
        for line in text.splitlines():
            if session_id is None and line.startswith("session_id:"):
                session_id = line.split(":", 1)[1].strip()
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        return session_id, body or None

    async def _extract_session(
        self, session_id: str, *, allowed_tools: set[str] | None = None, env: dict | None = None
    ) -> tuple[list[ToolCall], str | None]:
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            # Export must run against the SAME HERMES_HOME the chat ran in, or
            # the (isolated) sessions DB won't contain this session.
            proc = await asyncio.create_subprocess_exec(
                self.cli_path, "sessions", "export", tmp_path,
                "--session-id", session_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await proc.communicate()

            tool_calls: list[ToolCall] = []
            final_response: str | None = None
            idx = 0
            with open(tmp_path, encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        session = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    for msg in session.get("messages", []):
                        if msg.get("role") != "assistant":
                            continue
                        for tc in (msg.get("tool_calls") or []):
                            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                            name = fn.get("name") or tc.get("name", "<unknown>")
                            name = self._strip_mcp_prefix(name)
                            name = self._normalize_alias(name, allowed_tools)
                            raw_args = fn.get("arguments", tc.get("arguments", "{}"))
                            if isinstance(raw_args, str):
                                try:
                                    args = json.loads(raw_args)
                                except json.JSONDecodeError:
                                    args = {"_raw": raw_args}
                            else:
                                args = raw_args or {}
                            tool_calls.append(ToolCall(
                                index=idx, name=name, args=args,
                            ))
                            idx += 1
                        content = msg.get("content")
                        if content:
                            final_response = content
            return tool_calls, final_response
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

"""Hermes CLI subprocess adapter.

Hermes 0.14+ does not expose an OpenAI-compatible HTTP API (the gateway on
ports 8642/8644 is a messaging gateway for WhatsApp/Slack/Telegram, not for
chat-completions). We spawn the `hermes` CLI in single-query mode and
reconstruct the trace from the session SQLite store via `hermes sessions
export`.

Mirrors the subprocess pattern in ``llm_test.adapters.claude_code``.

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

from llm_test.core.models import Message, Scenario, ToolCall, TraceResult
from llm_test.core.text_utils import strip_reasoning_tags


def _build_prompt(scenario: Scenario) -> str:
    """Inline tool descriptions + task prompt + budget."""
    from llm_test.tools.registry import ToolRegistry
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

    def __init__(
        self,
        cli_path: str = "hermes",
        timeout_per_scenario: int = 300,
        ignore_user_config: bool = False,
        provider: str | None = None,
        use_worktree: bool = True,
        # Kept for backward-compat with existing CLI wiring; ignored.
        api_url: str | None = None,
        gateway_url: str | None = None,
        token: str | None = None,
        workspace_id: str | None = None,
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
        # codebase' will actually modify llm_test/ and create stray commits.
        self.use_worktree = use_worktree

    async def run_scenario(
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

        error: str | None = None
        stdout_bytes = b""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=min(self.timeout, max(timeout * 10, 60)),
            )
            if proc.returncode != 0:
                error = stderr_bytes.decode(errors="replace")[:500]
        except TimeoutError:
            error = "hermes: timeout"
        except FileNotFoundError as e:
            error = f"hermes CLI not found: {e}"

        stdout_text = stdout_bytes.decode(errors="replace")
        session_id, final_response_stdout = self._parse_stdout(stdout_text)

        # Pull structured tool_calls + final assistant message from session export.
        tool_calls: list[ToolCall] = []
        final_response_db: str | None = None
        if session_id:
            try:
                tool_calls, final_response_db = await self._extract_session(session_id)
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
        self, session_id: str
    ) -> tuple[list[ToolCall], str | None]:
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path, "sessions", "export", tmp_path,
                "--session-id", session_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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

from __future__ import annotations

import asyncio
import json
import shutil
import time
from datetime import UTC, datetime

from llm_test.core.models import Message, Scenario, ToolCall, TraceResult


def _build_prompt(scenario: Scenario) -> str:
    tools_lines = "\n".join(f"- {t}" for t in scenario.tools)
    return (
        f"You may use these tools:\n{tools_lines}\n\n"
        f"Task: {scenario.prompt}\n\n"
        f"Budget: at most {scenario.budget.max_tool_calls} tool calls, "
        f"{scenario.budget.max_turns} turns."
    )


class ClaudeCodeAdapter:
    name = "claude_code"
    version = "0.1"

    def __init__(self, cli_path: str = "claude", backend_url: str = "",
                 use_local_model: bool = True, timeout_per_scenario: int = 300,
                 skills_blacklist: list[str] | None = None):
        path = shutil.which(cli_path) if cli_path == "claude" else cli_path
        self.cli_path = path or cli_path
        self.backend_url = backend_url
        self.use_local_model = use_local_model
        self.timeout = timeout_per_scenario
        self.skills_blacklist = skills_blacklist or []

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        prompt = _build_prompt(scenario)
        cmd = [
            self.cli_path, "--print", prompt,
            "--output-format", "stream-json",
            "--max-turns", str(scenario.budget.max_turns),
        ]
        if self.use_local_model and self.backend_url:
            cmd += ["--model", model]
        env = None
        error = None
        stdout = b""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                     timeout=min(self.timeout, timeout * 10))
            if proc.returncode != 0:
                error = stderr.decode(errors="replace")[:500]
        except TimeoutError:
            error = "claude_code: timeout"
        except FileNotFoundError as e:
            error = f"claude_code CLI not found: {e}"

        tool_calls, final_response, session_id = self._parse_stream(stdout.decode(errors="replace"))
        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[
                Message(role="user", content=scenario.prompt),
                Message(role="assistant", content=final_response),
            ],
            tool_calls=tool_calls,
            final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"session_id": session_id, "model": model},
        )

    def _parse_stream(self, stream: str) -> tuple[list[ToolCall], str | None, str | None]:
        tool_calls: list[ToolCall] = []
        final_response = None
        session_id = None
        idx = 0
        for line in stream.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "system" and ev.get("subtype") == "init":
                session_id = ev.get("session_id")
            elif etype == "assistant":
                for block in (ev.get("message", {}).get("content") or []):
                    if block.get("type") == "tool_use":
                        tool_calls.append(ToolCall(
                            index=idx, name=block["name"], args=block.get("input", {}),
                            result=None, result_kind="text",
                        ))
                        idx += 1
                    elif block.get("type") == "text":
                        final_response = block.get("text")
            elif etype == "user":
                for block in (ev.get("message", {}).get("content") or []):
                    if block.get("type") == "tool_result" and tool_calls:
                        tool_calls[-1].result = block.get("content")
                        tool_calls[-1].result_kind = "text"
        return tool_calls, final_response, session_id

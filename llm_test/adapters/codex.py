from __future__ import annotations

import asyncio
import json
import shutil
import time
from datetime import UTC, datetime

from llm_test.core.models import Message, Scenario, ToolCall, TraceResult


class CodexAdapter:
    name = "codex"
    version = "0.1"

    def __init__(self, cli_path: str = "codex", backend_url: str = "",
                 use_local_model: bool = True, timeout_per_scenario: int = 300):
        self.cli_path = shutil.which(cli_path) if cli_path == "codex" else cli_path
        if self.cli_path is None:
            self.cli_path = cli_path
        self.backend_url = backend_url
        self.use_local_model = use_local_model
        self.timeout = timeout_per_scenario

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        cmd = [self.cli_path, "exec", "--json", "--model", model, scenario.prompt]
        error = None
        stdout = b""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                     timeout=min(self.timeout, timeout * 10))
            if proc.returncode != 0:
                error = stderr.decode(errors="replace")[:500]
        except TimeoutError:
            error = "codex: timeout"
        except FileNotFoundError as e:
            error = f"codex CLI not found: {e}"

        tool_calls, final_response = self._parse(stdout.decode(errors="replace"))
        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[Message(role="user", content=scenario.prompt),
                      Message(role="assistant", content=final_response)],
            tool_calls=tool_calls, final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms, error=error,
            adapter_metadata={"model": model},
        )

    def _parse(self, output: str) -> tuple[list[ToolCall], str | None]:
        tcs: list[ToolCall] = []
        final = None
        idx = 0
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "tool_call":
                tcs.append(ToolCall(index=idx, name=ev["name"], args=ev.get("args", {})))
                idx += 1
            elif kind == "tool_result" and tcs:
                tcs[-1].result = ev.get("result")
                tcs[-1].result_kind = "json" if isinstance(ev.get("result"), (dict, list)) else "text"
            elif kind == "final":
                final = ev.get("text")
        return tcs, final

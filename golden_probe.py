"""Passability prover: simulate an IDEAL play against the CURRENT mock + scorer.

Proves a scenario is solvable (or not) independent of any model's recorded
trace — it replays a hand-authored optimal tool sequence through the live
MockToolRuntime and the live scorer.

Usage (programmatic):
    from golden_probe import probe
    probe("hard-01-tdd-fix-loop",
          calls=[("run_tests", {}), ("read_file", {"path": "src/parser.py"}),
                 ("edit_file", {"path": "src/parser.py", "content": "sign=..."}),
                 ("run_tests", {})],
          final="Fixed the missing-sign bug; 12 passed.")
Returns (status, failing_required_checks). Prints a verdict line.
"""
from __future__ import annotations

from pathlib import Path
from toolery.core.scenario import load_all_scenarios
from toolery.core.scorer import evaluate
from toolery.core.models import TraceResult, ToolCall, Message
from toolery.tools.mock_runtime import MockToolRuntime
from toolery.tools import api_db, domain, generic, terminal  # noqa: F401

_SCEN = {s.id: s for s in load_all_scenarios(Path(__file__).resolve().parent / "scenarios")}


def probe(scenario_id: str, calls: list[tuple[str, dict]], final: str | None):
    s = _SCEN[scenario_id]
    mock = MockToolRuntime(s)
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        val, kind = mock.respond(name, args)
        tool_calls.append(ToolCall(index=i, name=name, args=args, result=val, result_kind=kind))
    trace = TraceResult(
        scenario_id=s.id, adapter="golden", trial_index=0,
        messages=[Message(role="user", content=s.prompt)],
        tool_calls=tool_calls, final_response=final,
        started_at_iso="2026-06-11T00:00:00Z", duration_ms=1000,
    )
    res = evaluate(s, trace)
    fails = [(c.check, c.detail) for c in res.checks if c.result == "fail"]
    verdict = "PASSABLE ✅" if res.status == "pass" else f"NOT-PASS ({res.status})"
    print(f"[{scenario_id}] {verdict}")
    for chk, det in fails:
        print(f"    FAIL {chk}: {det[:110]}")
    # surface what tools returned (helps spot a hostile/broken mock)
    for tc in tool_calls:
        rv = str(tc.result)[:70].replace("\n", " ")
        print(f"    · {tc.name}({str(tc.args)[:50]}) -> [{tc.result_kind}] {rv}")
    return res.status, fails


if __name__ == "__main__":
    # self-test on a known-passable easy scenario
    probe("easy-16-coding-pytest-count",
          calls=[("bash_exec", {"command": "pytest -q"})],
          final="5 tests passed.")

"""Re-score the 2026-06-11 MiniMax run with the fixed grader + scenario YAMLs.

Offline re-evaluation of persisted traces — no inference. Runtime-class fixes
(mock matchers / tool schemas / tool_responses) change what the model WOULD
have seen, so those scenarios are only flagged here and need a live re-run.

Usage: .venv/bin/python rescore_minimax.py
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

from toolery.core.models import TraceResult
from toolery.core.scenario import load_all_scenarios
from toolery.core.scorer import evaluate
from toolery.tools import api_db, domain, generic, terminal  # noqa: F401 (register tools)

REPO = Path(__file__).resolve().parent
RUN_ID = "2026-06-11T09-18_MiniMax-M2.7-AWQ-4bit"

# Scenarios whose fix changes RUNTIME behaviour (mock matcher / tool schema /
# tool_responses) — the recorded traces were produced against the broken sim,
# so offline rescoring cannot redeem them; they need a live re-run.
NEEDS_RERUN = {
    "easy-18-coding-import-error-module",
    "easy-22-pl-date-recover",
    "hard-01-tdd-fix-loop",
    "hard-09-cst-state-update",
    "hard-15-api-pagination",
    "medium-37-debug-bad-caller",
    "very-hard-11-coding-bug-bisect",
    "very-hard-18-evolving-params",
    "very-hard-20-near-twin-tools",
    "very-hard-24-debug-instrument-then-fix",
}


def main() -> None:
    scenarios = {s.id: s for s in load_all_scenarios(REPO / "scenarios")}
    conn = sqlite3.connect(REPO / "results" / "runs.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT scenario_id, tier, trial_index, status, score, failure_kind, trace_path "
        "FROM scenario_results WHERE run_id=?", (RUN_ID,)
    ).fetchall()

    run_dir = REPO / "results" / "runs" / RUN_ID
    tier_old: dict[str, list[int]] = defaultdict(list)   # 1 = pass
    tier_new: dict[str, list[int]] = defaultdict(list)
    flipped: list[tuple[str, int, str, str]] = []
    rerun_rows = 0
    missing = 0

    for row in rows:
        sid, trial = row["scenario_id"], row["trial_index"]
        scen = scenarios.get(sid)
        old_pass = 1 if row["status"] == "pass" else 0
        tier_old[row["tier"]].append(old_pass)

        if scen is None or not row["trace_path"]:
            tier_new[row["tier"]].append(old_pass)
            missing += 1
            continue
        tp = Path(row["trace_path"])
        if not tp.is_absolute():
            tp = run_dir / row["trace_path"]
        if not tp.exists():
            tier_new[row["tier"]].append(old_pass)
            missing += 1
            continue

        if sid in NEEDS_RERUN:
            rerun_rows += 1
            tier_new[row["tier"]].append(old_pass)  # keep old until live re-run
            continue

        trace = TraceResult.model_validate_json(tp.read_text(encoding="utf-8"))
        result = evaluate(scen, trace)
        new_pass = 1 if result.status == "pass" else 0
        tier_new[row["tier"]].append(new_pass)
        if new_pass != old_pass:
            flipped.append((sid, trial, row["status"], result.status))

    print(f"{'tier':<12} {'old pass':>9} {'new pass':>9} {'n':>5}")
    tot_old = tot_new = tot_n = 0
    for tier in ["easy", "medium", "hard", "very_hard"]:
        o, n = tier_old[tier], tier_new[tier]
        print(f"{tier:<12} {sum(o):>5}/{len(o):<4} {sum(n):>5}/{len(n):<4}")
        tot_old += sum(o); tot_new += sum(n); tot_n += len(o)
    print(f"{'TOTAL':<12} {tot_old:>5}/{tot_n:<4} {tot_new:>5}/{tot_n:<4} "
          f"({100*tot_old/tot_n:.1f}% -> {100*tot_new/tot_n:.1f}%)")
    print(f"\nflipped fail->pass: {sum(1 for f in flipped if f[3]=='pass')}, "
          f"pass->fail: {sum(1 for f in flipped if f[3]!='pass')}")
    print(f"rows deferred to live re-run ({len(NEEDS_RERUN)} scenarios): {rerun_rows}")
    if missing:
        print(f"rows with missing trace/scenario: {missing}")
    print("\nFLIPPED TRIALS:")
    for sid, trial, old, new in sorted(flipped):
        print(f"  {sid} t{trial}: {old} -> {new}")


if __name__ == "__main__":
    main()

"""Drill into specific scenarios: which required checks fail across models."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from llm_test.core.models import TraceResult
from llm_test.core.scenario import load_all_scenarios
from llm_test.core.scorer import evaluate

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "results" / "runs.db"
SCENARIOS_DIR = REPO / "scenarios"


def main(scenario_ids: list[str]) -> None:
    os.environ["LLM_TEST_PARTIAL_GRADIENT"] = "off"  # PoC mode
    scenarios = {s.id: s for s in load_all_scenarios(SCENARIOS_DIR)}
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    for sid in scenario_ids:
        scen = scenarios[sid]
        n_req = len(scen.scoring.required)
        rows = conn.execute(
            "SELECT sr.run_id, sr.adapter, sr.trial_index, sr.trace_path, r.model "
            "FROM scenario_results sr JOIN runs r USING(run_id) "
            "WHERE sr.scenario_id=? AND r.status='done'",
            (sid,),
        ).fetchall()
        print(f"\n{'='*84}\n{sid}  ({n_req} required checks)\n{'='*84}")

        # Per-check pass count and per-model pass/fail counters
        check_pass = Counter()
        per_model_passes: dict[str, list[int]] = defaultdict(list)
        per_model_status: dict[str, list[str]] = defaultdict(list)
        miss_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
        n_trials = 0

        for row in rows:
            tp = Path(row["trace_path"])
            if not tp.is_absolute():
                tp = REPO / "results" / "runs" / row["run_id"] / row["trace_path"]
            if not tp.exists():
                continue
            tr = TraceResult.model_validate_json(tp.read_text(encoding="utf-8"))
            res = evaluate(scen, tr)
            n_trials += 1
            req_results = [c for c in res.checks if c.result in ("pass", "fail")][:n_req]
            n_pass = sum(1 for c in req_results if c.result == "pass")
            per_model_passes[row["model"]].append(n_pass)
            per_model_status[row["model"]].append(res.status)
            for i, c in enumerate(req_results):
                if c.result == "pass":
                    check_pass[i] += 1
                else:
                    miss_examples[f"req[{i}] {c.check}"].append((row["model"], c.detail))

        # --- check-by-check pass rates
        print(f"\nCheck-by-check pass rate over {n_trials} trials:")
        for i, chk in enumerate(scen.scoring.required):
            label = f"  req[{i}] {chk.check}"
            extras = []
            d = chk.model_dump()
            for k in ("tool", "patterns", "any_of", "all_of", "pattern", "n", "sequence", "args"):
                if k in d and d[k] is not None and d[k] != []:
                    val = str(d[k])
                    if len(val) > 50:
                        val = val[:50] + "..."
                    extras.append(f"{k}={val}")
            label += f"  [{', '.join(extras)}]" if extras else ""
            rate = check_pass[i] / n_trials if n_trials else 0.0
            print(f"  {rate:>6.1%}  pass  ({check_pass[i]:>3}/{n_trials})  {label}")

        # --- per model
        print(f"\nPer model — mean required-pass count (of {n_req}) and full-pass rate:")
        for model, passes in per_model_passes.items():
            statuses = per_model_status[model]
            mean_pass = sum(passes) / len(passes)
            full = sum(1 for s in statuses if s == "pass") / len(statuses)
            partial = sum(1 for s in statuses if s == "partial") / len(statuses)
            fail = sum(1 for s in statuses if s == "fail") / len(statuses)
            print(f"  {model[:48]:<48}  mean={mean_pass:>4.1f}/{n_req}  "
                  f"pass={full:>5.1%}  partial={partial:>5.1%}  fail={fail:>5.1%}  "
                  f"(n={len(passes)})")

        # --- the failing check examples
        print(f"\nFailure details (first 3 per check):")
        for label, examples in sorted(miss_examples.items()):
            print(f"  ✗ {label} — failed in {len(examples)} trials")
            for model, detail in examples[:3]:
                detail_short = detail[:120] + "..." if len(detail) > 120 else detail
                print(f"      [{model[:24]:<24}] {detail_short}")


if __name__ == "__main__":
    main(sys.argv[1:] or [
        "hard-13-lc-multi-fact-extraction",
        "very-hard-08-multi-tool-discovery",
        "medium-22-db-describe-then-query",
    ])

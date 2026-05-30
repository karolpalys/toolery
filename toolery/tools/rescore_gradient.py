"""Empirical A/B for the implicit-partial-gradient PoC.

Reads every persisted TraceResult JSON for the 'done' runs in results/runs.db,
re-runs the scorer twice (gradient OFF = PoC default, gradient ON = pre-PoC
behavior), and prints per-model / per-tier / largest-mover summaries.

Run from the repo root:
    .venv/bin/python -m toolery.tools.rescore_gradient
"""

from __future__ import annotations

import os
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from toolery.core.models import TraceResult
from toolery.core.scenario import load_all_scenarios
from toolery.core.scorer import evaluate

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "results" / "runs.db"
SCENARIOS_DIR = REPO / "scenarios"


def rescore(scenario, trace, *, gradient_on: bool) -> float:
    os.environ["TOOLERY_PARTIAL_GRADIENT"] = "on" if gradient_on else "off"
    return evaluate(scenario, trace).score


def main() -> None:
    scenarios_by_id = {s.id: s for s in load_all_scenarios(SCENARIOS_DIR)}

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        "SELECT run_id, model FROM runs WHERE status='done' ORDER BY started_at"
    ).fetchall()

    # model -> list of (scenario_id, tier, score_off, score_on, stored_score, trace_missing)
    per_model: dict[str, list[tuple[str, str, float, float, float, bool]]] = defaultdict(list)
    movers: list[tuple[float, str, str, str, float, float]] = []  # (delta, model, scenario, tier, off, on)

    for r in runs:
        run_id, model = r["run_id"], r["model"]
        run_dir = REPO / "results" / "runs" / run_id
        rows = conn.execute(
            "SELECT scenario_id, tier, adapter, trial_index, score AS stored_score, trace_path "
            "FROM scenario_results WHERE run_id=?", (run_id,)
        ).fetchall()
        for row in rows:
            sid = row["scenario_id"]
            scen = scenarios_by_id.get(sid)
            if scen is None:
                continue
            trace_path = row["trace_path"]
            if not trace_path:
                continue
            tp = Path(trace_path)
            if not tp.is_absolute():
                tp = run_dir / trace_path
            if not tp.exists():
                per_model[model].append((sid, row["tier"], float("nan"), float("nan"),
                                         row["stored_score"], True))
                continue
            try:
                tr = TraceResult.model_validate_json(tp.read_text(encoding="utf-8"))
            except Exception:
                continue
            score_off = rescore(scen, tr, gradient_on=False)
            score_on = rescore(scen, tr, gradient_on=True)
            stored = float(row["stored_score"])
            per_model[model].append((sid, row["tier"], score_off, score_on, stored, False))
            delta = score_on - score_off
            if delta > 0.0:
                movers.append((delta, model, sid, row["tier"], score_off, score_on))

    # --- summary per model
    print(f"\n{'='*82}")
    print("PER-MODEL MEAN SCORE (across all scenarios, all trials)")
    print(f"{'='*82}")
    header = f"{'model':<48}  {'OFF (PoC)':>10}  {'ON (old)':>10}  {'Δ':>7}  {'n':>5}"
    print(header)
    print("-" * len(header))
    sums = {}
    for model, entries in per_model.items():
        offs = [e[2] for e in entries if not e[5]]
        ons = [e[3] for e in entries if not e[5]]
        if not offs:
            continue
        m_off = statistics.fmean(offs)
        m_on = statistics.fmean(ons)
        sums[model] = (m_off, m_on, len(offs))
        print(f"{model:<48}  {m_off:>10.4f}  {m_on:>10.4f}  {m_on - m_off:>+7.4f}  {len(offs):>5d}")

    if len(sums) >= 2:
        offs = [v[0] for v in sums.values()]
        ons = [v[1] for v in sums.values()]
        print("-" * len(header))
        print(f"{'spread (max-min)':<48}  {max(offs)-min(offs):>10.4f}  "
              f"{max(ons)-min(ons):>10.4f}  {(max(ons)-min(ons))-(max(offs)-min(offs)):>+7.4f}")
        print(f"{'stdev across models':<48}  {statistics.stdev(offs):>10.4f}  "
              f"{statistics.stdev(ons):>10.4f}")

    # --- per-tier
    print(f"\n{'='*82}")
    print("PER-TIER MEAN SCORE (averaged across models)")
    print(f"{'='*82}")
    tiers = ["easy", "medium", "hard", "very_hard"]
    print(f"{'tier':<12}  {'OFF (PoC)':>10}  {'ON (old)':>10}  {'Δ_softening':>12}  {'n':>6}")
    print("-" * 60)
    for tier in tiers:
        offs, ons = [], []
        for entries in per_model.values():
            for _sid, t, off, on, _, missing in entries:
                if missing or t != tier:
                    continue
                offs.append(off)
                ons.append(on)
        if not offs:
            continue
        print(f"{tier:<12}  {statistics.fmean(offs):>10.4f}  {statistics.fmean(ons):>10.4f}  "
              f"{statistics.fmean(ons) - statistics.fmean(offs):>+12.4f}  {len(offs):>6d}")

    # --- biggest softening events (where gradient ON gave score, OFF gave 0)
    print(f"\n{'='*82}")
    print("TOP 20 SCENARIOS WHERE ON-GRADIENT INFLATED THE MOST (per model+scenario)")
    print(f"{'='*82}")
    movers.sort(reverse=True)
    print(f"{'Δ':>6}  {'OFF':>5}  {'ON':>5}  {'tier':<10}  {'model':<32}  scenario")
    for delta, model, sid, tier, off, on in movers[:20]:
        print(f"{delta:>+6.3f}  {off:>5.3f}  {on:>5.3f}  {tier:<10}  {model[:32]:<32}  {sid}")

    # --- per-model × per-tier softening
    print(f"\n{'='*82}")
    print("MODEL × TIER  —  softening Δ = ON − OFF  (how much the gradient inflated the score)")
    print(f"{'='*82}")
    tier_list = ["easy", "medium", "hard", "very_hard"]
    print(f"{'model':<44}  " + "  ".join(f"{t:>11}" for t in tier_list))
    for model, entries in per_model.items():
        deltas = {}
        for t in tier_list:
            offs, ons = [], []
            for _sid, tt, off, on, _, missing in entries:
                if missing or tt != t:
                    continue
                offs.append(off)
                ons.append(on)
            if offs:
                deltas[t] = statistics.fmean(ons) - statistics.fmean(offs)
            else:
                deltas[t] = float("nan")
        cells = "  ".join(f"{deltas[t]:>+11.4f}" for t in tier_list)
        print(f"{model[:44]:<44}  {cells}")

    # --- count zero-rate (per model) under both modes
    print(f"\n{'='*82}")
    print("ZERO-RATE (fraction of scenarios scoring exactly 0.0)")
    print(f"{'='*82}")
    print(f"{'model':<48}  {'OFF (PoC)':>10}  {'ON (old)':>10}")
    for model, entries in per_model.items():
        valid = [(off, on) for sid, t, off, on, _, missing in entries if not missing]
        if not valid:
            continue
        zoff = sum(1 for o, _ in valid if o == 0.0) / len(valid)
        zon = sum(1 for _, o in valid if o == 0.0) / len(valid)
        print(f"{model:<48}  {zoff:>10.2%}  {zon:>10.2%}")


if __name__ == "__main__":
    main()

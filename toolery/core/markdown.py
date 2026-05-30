from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from toolery.core.models import ScenarioResult

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=select_autoescape())


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _tier_breakdown(results: list[ScenarioResult], tier_lookup: dict[str, str]) -> dict:
    by_adapter_tier: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_adapter_overall: dict[str, list[float]] = defaultdict(list)
    for r in results:
        tier = tier_lookup.get(r.scenario_id, "easy")
        by_adapter_tier[r.adapter][tier].append(r.score)
        by_adapter_overall[r.adapter].append(r.score)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    out = {}
    for a in by_adapter_overall:
        out[a] = {
            "overall": _pct(avg(by_adapter_overall[a])),
            "easy": _pct(avg(by_adapter_tier[a].get("easy", []))),
            "medium": _pct(avg(by_adapter_tier[a].get("medium", []))),
            "hard": _pct(avg(by_adapter_tier[a].get("hard", []))),
            "very_hard": _pct(avg(by_adapter_tier[a].get("very_hard", []))),
        }
    return out


def render_summary(*, run_id, model, adapters, trials, duration_s, results,
                   perf_rows, tier_lookup: dict[str, str] | None = None) -> str:
    tier_lookup = tier_lookup or {}
    overall = _tier_breakdown(results, tier_lookup)
    top_failures = [
        {"scenario_id": r.scenario_id, "adapter": r.adapter,
         "failure_kind": r.failure_kind, "detail": ""}
        for r in results if r.status == "fail"
    ][:5]
    tmpl = _env.get_template("summary.md.j2")
    return tmpl.render(
        run_id=run_id, model=model, adapters=list(adapters), trials=trials,
        scenarios_n=len({r.scenario_id for r in results}),
        duration_s=duration_s, results=results, overall=overall,
        top_failures=top_failures, perf_rows=perf_rows,
    )


def render_scenario(*, scenario_id, results, title, tier, category) -> str:
    by_adapter: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        if r.scenario_id == scenario_id:
            by_adapter[r.adapter].append(r)

    per_adapter = []
    for a, rs in by_adapter.items():
        per_adapter.append({
            "adapter": a, "total": len(rs),
            "pass_n": sum(1 for r in rs if r.status == "pass"),
            "partial_n": sum(1 for r in rs if r.status == "partial"),
            "fail_n": sum(1 for r in rs if r.status == "fail"),
            "score": sum(r.score for r in rs) / len(rs),
            "median_calls": int(statistics.median(r.call_count for r in rs)),
            "median_latency": int(statistics.median(r.latency_ms for r in rs)),
        })

    checks_by_adapter = {a: rs[-1].checks for a, rs in by_adapter.items()}
    failures = [r for rs in by_adapter.values() for r in rs if r.status != "pass"]

    tmpl = _env.get_template("scenario.md.j2")
    return tmpl.render(
        scenario_id=scenario_id, title=title, tier=tier, category=category,
        trials=sum(len(rs) for rs in by_adapter.values()) // max(len(by_adapter), 1),
        per_adapter=per_adapter, checks_by_adapter=checks_by_adapter, failures=failures,
    )

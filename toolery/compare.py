from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from toolery.core.scenario import display_name
from toolery.core.stats import mcnemar_p
from toolery.core.store import Store

_TEMPLATES_DIR = Path(__file__).parent / "core" / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))
# Tier prefix is historical; tier column is the source of truth (see display_name).
_env.filters["display_name"] = display_name


def compare_runs(*, store: Store, run_a: str, run_b: str, out_path: Path) -> None:
    rs_a = store.fetch_results_for_run(run_a)
    rs_b = store.fetch_results_for_run(run_b)
    runs_meta = {r["run_id"]: r for r in store.fetch_all_runs()}
    common_adapter = _common_adapter(rs_a, rs_b)

    a_by_scenario: dict[str, list[dict]] = defaultdict(list)
    b_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rs_a:
        if r["adapter"] == common_adapter:
            a_by_scenario[r["scenario_id"]].append(r)
    for r in rs_b:
        if r["adapter"] == common_adapter:
            b_by_scenario[r["scenario_id"]].append(r)
    # Sort by trial_index so McNemar pairs the same trial across A and B,
    # independent of how fetch_results_for_run orders rows.
    for d in (a_by_scenario, b_by_scenario):
        for s in d:
            d[s].sort(key=lambda r: r["trial_index"])
    common_scenarios = sorted(set(a_by_scenario) & set(b_by_scenario))

    a_scores = [r["score"] for s in common_scenarios for r in a_by_scenario[s]]
    b_scores = [r["score"] for s in common_scenarios for r in b_by_scenario[s]]
    n = min(len(a_scores), len(b_scores))
    overall_metric = {
        "name": "Overall",
        "a": sum(a_scores[:n]) / max(n, 1), "b": sum(b_scores[:n]) / max(n, 1),
        "p": mcnemar_p(a_scores[:n], b_scores[:n]),
    }

    regressions, improvements, identical = [], [], 0
    for s in common_scenarios:
        a_pass = sum(1 for r in a_by_scenario[s] if r["status"] == "pass")
        b_pass = sum(1 for r in b_by_scenario[s] if r["status"] == "pass")
        a_total = len(a_by_scenario[s])
        b_total = len(b_by_scenario[s])
        if a_pass / max(a_total, 1) > b_pass / max(b_total, 1):
            improvements.append({"scenario_id": s, "a_pass": a_pass, "a_total": a_total,
                                  "b_pass": b_pass, "b_total": b_total})
        elif a_pass / max(a_total, 1) < b_pass / max(b_total, 1):
            regressions.append({"scenario_id": s, "a_pass": a_pass, "a_total": a_total,
                                 "b_pass": b_pass, "b_total": b_total})
        else:
            identical += 1

    tmpl = _env.get_template("compare.md.j2")
    md = tmpl.render(
        run_a={"run_id": run_a, "model": runs_meta[run_a]["model"],
               "config": runs_meta[run_a]["config_json"]},
        run_b={"run_id": run_b, "model": runs_meta[run_b]["model"],
               "config": runs_meta[run_b]["config_json"]},
        common_adapter=common_adapter, common_scenarios=len(common_scenarios),
        trials=n, metrics=[overall_metric],
        regressions=regressions, improvements=improvements, identical_count=identical,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)


def _common_adapter(rs_a, rs_b) -> str:
    a_ads = {r["adapter"] for r in rs_a}
    b_ads = {r["adapter"] for r in rs_b}
    common = a_ads & b_ads
    if not common:
        raise ValueError("runs have no common adapter")
    return sorted(common)[0]

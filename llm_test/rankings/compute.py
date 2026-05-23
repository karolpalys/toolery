from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from llm_test.core.stats import decay_weighted_mean
from llm_test.core.store import Store

_TEMPLATES_DIR = Path(__file__).parent.parent / "core" / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))


def regenerate_rankings(*, store: Store, dimensions: list[str], out_dir: Path,
                        history_window_runs: int = 5, half_life_days: float = 14.0,
                        bootstrap_iters: int = 1000, min_runs: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    runs = store.fetch_all_runs()
    run_meta = {r["run_id"]: r for r in runs}

    for dim in dimensions:
        per_model_runs: dict[str, list[dict]] = defaultdict(list)

        with store.conn() as c:
            rows = c.execute("SELECT * FROM scenario_results").fetchall()
            results = [dict(r) for r in rows]

        results_by_run: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            dims = json.loads(r["ranking_dims_json"] or "[]")
            if dim != "overall" and dim not in dims:
                continue
            results_by_run[r["run_id"]].append(r)

        for run_id, rs in results_by_run.items():
            meta = run_meta.get(run_id)
            if not meta:
                continue
            model = meta["model"]
            scores = [r["score"] for r in rs]
            by_adapter: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                by_adapter[r["adapter"]].append(r["score"])
            best = max(by_adapter.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            per_model_runs[model].append({
                "run_id": run_id, "scores": scores,
                "started_at": meta["started_at"], "best_adapter": best,
            })

        rows_out = []
        for model, runs_list in per_model_runs.items():
            if len(runs_list) < min_runs:
                continue
            runs_list.sort(key=lambda x: x["started_at"], reverse=True)
            recent = runs_list[:history_window_runs]
            pairs: list[tuple[float, float]] = []
            for r in recent:
                run_mean = sum(r["scores"]) / max(len(r["scores"]), 1)
                started = _parse_iso(r["started_at"])
                age_days = max((now - started).total_seconds() / 86400, 0)
                pairs.append((run_mean, age_days))
            weighted = decay_weighted_mean(pairs, half_life_days)
            best = max(recent, key=lambda r: sum(r["scores"]) / max(len(r["scores"]), 1))["best_adapter"]
            rows_out.append({
                "model": model, "score": weighted,
                "best_adapter": best, "runs": len(runs_list),
            })

        rows_out.sort(key=lambda r: -r["score"])
        tmpl = _env.get_template("ranking.md.j2")
        md = tmpl.render(
            dimension=dim, updated_iso=now.isoformat(),
            model_count=len(rows_out), run_count=sum(r["runs"] for r in rows_out),
            rows=rows_out, window=history_window_runs, half_life=half_life_days,
            bootstrap_iters=bootstrap_iters,
        )
        (out_dir / f"{dim}.md").write_text(md)


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)

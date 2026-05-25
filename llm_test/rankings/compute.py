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

        # Group results per (model, adapter, run). Adapter is a first-class axis:
        # the same model under hermes vs raw can score very differently, and
        # averaging across adapters hides that signal. We rank by best-adapter
        # score and report a per-adapter breakdown below.
        per_pair_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for run_id, rs in results_by_run.items():
            meta = run_meta.get(run_id)
            if not meta:
                continue
            model = meta["model"]
            # Keep (score, tier) so we can apply tier weights downstream.
            by_adapter: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
            for r in rs:
                by_adapter[r["adapter"]].append((r["score"], r["tier"], r["ranking_dims_json"] or "[]"))
            for adapter, scored in by_adapter.items():
                per_pair_runs[(model, adapter)].append({
                    "run_id": run_id, "scores": scored,
                    "started_at": meta["started_at"],
                })

        pair_rows: list[dict] = []
        for (model, adapter), runs_list in per_pair_runs.items():
            if len(runs_list) < min_runs:
                continue
            runs_list.sort(key=lambda x: x["started_at"], reverse=True)
            recent = runs_list[:history_window_runs]
            pairs: list[tuple[float, float]] = []
            for r in recent:
                items = r["scores"]  # list of (score, tier, ranking_dims_json) tuples
                if dim == "overall":
                    weights_per_item = [
                        _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d))
                        for _, t, d in items
                    ]
                else:
                    weights_per_item = [_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items]
                w_sum = sum(weights_per_item)
                if w_sum <= 0:
                    continue
                run_mean = sum(
                    s * w for (s, _, _), w in zip(items, weights_per_item)
                ) / w_sum
                age_days = max((now - _parse_iso(r["started_at"])).total_seconds() / 86400, 0)
                pairs.append((run_mean, age_days))
            pair_rows.append({
                "model": model, "adapter": adapter,
                "score": decay_weighted_mean(pairs, half_life_days),
                "runs": len(runs_list),
            })

        # Per-model summary: best adapter wins; lower-scoring adapters listed below.
        rows_out: list[dict] = []
        per_model: dict[str, list[dict]] = defaultdict(list)
        for pr in pair_rows:
            per_model[pr["model"]].append(pr)
        for model, prs in per_model.items():
            prs.sort(key=lambda p: -p["score"])
            best = prs[0]
            rows_out.append({
                "model": model,
                "score": best["score"],
                "best_adapter": best["adapter"],
                "runs": sum(p["runs"] for p in prs),
                "adapter_breakdown": prs,  # full list for the per-adapter sub-table
            })

        rows_out.sort(key=lambda r: -r["score"])
        # Flat breakdown (model, adapter) sorted by score for the appendix.
        breakdown = sorted(pair_rows, key=lambda p: -p["score"])
        tmpl = _env.get_template("ranking.md.j2")
        md = tmpl.render(
            dimension=dim, updated_iso=now.isoformat(),
            model_count=len(rows_out), run_count=sum(r["runs"] for r in rows_out),
            rows=rows_out, breakdown=breakdown,
            window=history_window_runs, half_life=half_life_days,
            bootstrap_iters=bootstrap_iters,
        )
        (out_dir / f"{dim}.md").write_text(md)


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


# Tier weights — harder scenarios count more in every dimension's run mean.
# Without these, 11 easy tests dilute 7 very_hard tests at equal weight.
_TIER_WEIGHTS = {"easy": 1.0, "medium": 2.0, "hard": 3.0, "very_hard": 4.0}

# Per-dimension weights applied ONLY when computing the `overall` dimension —
# every other column (Coding, Terminal, etc.) keeps the existing tier-only weighting
# so that e.g. a model's `Coding` score reflects raw coding performance, not blend.
# A scenario's overall weight is the MAX of the weights of its non-overall dims
# (fallback 1.0 if none of its dims are in this map).
_DIM_WEIGHTS: dict[str, float] = {
    "coding": 2.0,
    "terminal": 2.0,
    "agentic": 2.0,
    "localization": 0.5,
    "long_context": 0.5,
}


def _scenario_dim_weight(
    ranking_dims: list[str],
    weights_override: dict[str, float] | None = None,
) -> float:
    """Compute the per-scenario weight for the Overall (or use-case) column.

    Returns the MAX of the weights of the scenario's non-overall dimensions.
    Dimensions not in the weights map fall back to 1.0.

    When `weights_override` is None, uses the default `_DIM_WEIGHTS` map.
    When provided, uses the override (e.g. a use-case persona's weights).
    """
    weights_map = weights_override if weights_override is not None else _DIM_WEIGHTS
    weights = [weights_map.get(d, 1.0) for d in ranking_dims if d != "overall"]
    return max(weights) if weights else 1.0


def load_active_use_case(results_dir: Path) -> tuple[str | None, dict[str, float] | None]:
    """Read results/setup.json and return the active use-case persona.

    Returns (key, weights) on success, (None, None) when:
      - setup.json doesn't exist
      - the file is malformed JSON
      - active_use_case is null or missing
      - active_use_case key doesn't match any known persona
    """
    setup_path = results_dir / "setup.json"
    if not setup_path.exists():
        return (None, None)
    try:
        data = json.loads(setup_path.read_text())
    except (json.JSONDecodeError, OSError):
        return (None, None)
    key = data.get("active_use_case")
    if not key:
        return (None, None)
    from llm_test.rankings.presets import get_use_case
    uc = get_use_case(key)
    if uc is None:
        return (None, None)
    return (key, dict(uc.weights))


def compute_matrix(
    *, store: Store, dimensions: list[str],
    history_window_runs: int = 5, half_life_days: float = 14.0,
) -> list[dict]:
    """Compute per-(model, adapter) scores across all ranking dimensions.

    Returns one row per (model, adapter) pair:
      {"model": str, "adapter": str, "runs": int, "scores": {dim: float},
       "scenarios_hashes": set[str]}

    Within each dimension, items are tier-weighted (very_hard counts 4× easy)
    when computing the per-run mean. Across runs, an exponential decay with
    `half_life_days` is applied.

    `scores` only contains dimensions where the pair has any data; callers
    must handle missing keys.
    """
    now = datetime.now(UTC)
    runs = store.fetch_all_runs()
    run_meta = {r["run_id"]: r for r in runs}

    with store.conn() as c:
        all_results = [dict(r) for r in c.execute("SELECT * FROM scenario_results").fetchall()]

    # (model, adapter) -> dim -> list of {run_id, started_at, score, tier}
    pairs: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    # (model, adapter) -> set of scenarios_hashes seen
    pair_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in all_results:
        meta = run_meta.get(r["run_id"])
        if not meta:
            continue
        model = meta["model"]
        adapter = r["adapter"]
        dims_for_r = json.loads(r["ranking_dims_json"] or "[]")
        sh = meta.get("scenarios_hash") or ""
        if sh:
            pair_hashes[(model, adapter)].add(sh)
        for dim in dimensions:
            if dim != "overall" and dim not in dims_for_r:
                continue
            pairs[(model, adapter)][dim].append({
                "run_id": r["run_id"],
                "started_at": meta["started_at"],
                "score": r["score"],
                "tier": r["tier"],
                "ranking_dims_json": r["ranking_dims_json"] or "[]",
            })

    matrix: list[dict] = []
    for (model, adapter), dim_results in pairs.items():
        scores: dict[str, float] = {}
        total_runs = 0
        for dim, items in dim_results.items():
            by_run: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
            run_started: dict[str, str] = {}
            for it in items:
                by_run[it["run_id"]].append((it["score"], it["tier"], it["ranking_dims_json"]))
                run_started[it["run_id"]] = it["started_at"]
            runs_sorted = sorted(by_run.keys(), key=lambda rid: run_started[rid], reverse=True)
            recent = runs_sorted[:history_window_runs]
            decay_pairs: list[tuple[float, float]] = []
            for rid in recent:
                items_in_run = by_run[rid]
                if dim == "overall":
                    weights = [
                        _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d))
                        for _, t, d in items_in_run
                    ]
                else:
                    weights = [_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items_in_run]
                w_sum = sum(weights)
                if w_sum <= 0:
                    continue
                weighted_mean = sum(
                    s * w for (s, _, _), w in zip(items_in_run, weights)
                ) / w_sum
                age = max((now - _parse_iso(run_started[rid])).total_seconds() / 86400, 0)
                decay_pairs.append((weighted_mean, age))
            if decay_pairs:
                scores[dim] = decay_weighted_mean(decay_pairs, half_life_days)
            total_runs = max(total_runs, len(runs_sorted))
        if scores:
            matrix.append({
                "model": model, "adapter": adapter,
                "runs": total_runs, "scores": scores,
                "scenarios_hashes": pair_hashes[(model, adapter)],
            })
    return matrix

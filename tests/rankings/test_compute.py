from datetime import UTC, datetime
from pathlib import Path

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store
from toolery.rankings.compute import regenerate_rankings


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed(tmp_path: Path, model: str, scores: list[float], ranking_dims: list[str]):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    run_id = f"r_{model}_{scores}"
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    for i, s in enumerate(scores):
        sid = f"easy-{i:02d}-test"
        tr = _trace(sid, "raw")
        result = ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0,
            status="pass" if s > 0.5 else "fail", score=s,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if s > 0.5 else "wrong_tool",
            checks=[], trace=tr,
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=["coding"] if "cod" in sid else [],
            ranking_dims=ranking_dims,
            scenario_hash="h", category="coding", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)
    return store


def test_regenerate_overall_ranking(tmp_path):
    store = _seed(tmp_path, "model_a", [1.0, 1.0, 1.0, 1.0, 0.0], ["overall"])
    _seed(tmp_path, "model_b", [1.0, 0.0, 0.0, 0.0, 0.0], ["overall"])
    out_dir = tmp_path / "rankings"
    regenerate_rankings(store=store, dimensions=["overall"], out_dir=out_dir,
                        history_window_runs=5, half_life_days=14)
    md = (out_dir / "overall.md").read_text()
    assert "model_a" in md and "model_b" in md
    a_idx = md.index("model_a")
    b_idx = md.index("model_b")
    assert a_idx < b_idx


def test_compute_matrix_exposes_stability_metrics(tmp_path):
    store = _seed(tmp_path, "model_stable", [1.0, 0.0], ["overall"])
    matrix = __import__("toolery.rankings.compute", fromlist=["compute_matrix"]).compute_matrix(
        store=store, dimensions=["overall"]
    )
    row = matrix[0]
    assert "stability" in row
    assert row["stability"]["overall"]["mean"] == 0.5
    assert row["stability"]["overall"]["worst"] == 0.5
    assert row["stability"]["overall"]["pass_rate"] == 0.0


def test_collapse_matrix_rows_modes():
    from toolery.rankings.compute import collapse_matrix_rows
    matrix = [
        {"model": "m", "adapter": "raw", "runs": 1, "scores": {"overall": 0.4}, "perf": {}, "stability": {"overall": {"mean": 0.4}}, "scenarios_hashes": {"h"}},
        {"model": "m", "adapter": "hermes", "runs": 1, "scores": {"overall": 0.8}, "perf": {}, "stability": {"overall": {"mean": 0.8}}, "scenarios_hashes": {"h"}},
    ]
    assert len(collapse_matrix_rows(matrix, "pair")) == 2
    best = collapse_matrix_rows(matrix, "model_best")
    assert len(best) == 1 and best[0]["adapter"] == "hermes"
    mean = collapse_matrix_rows(matrix, "model_mean")
    assert len(mean) == 1 and abs(mean[0]["scores"]["overall"] - 0.6) < 0.001
    raw = collapse_matrix_rows(matrix, "raw_only")
    assert len(raw) == 1 and raw[0]["adapter"] == "raw"


def test_compute_failure_breakdown_counts_by_model_adapter(tmp_path):
    store = _seed(tmp_path, "model_fail", [0.0, 0.0], ["overall", "coding"])
    from toolery.rankings.compute import compute_failure_breakdown
    breakdown = compute_failure_breakdown(store)
    assert breakdown[("model_fail", "raw")]["wrong_tool"] == 2
    assert compute_failure_breakdown(store, dimensions=["coding"])[("model_fail", "raw")]["wrong_tool"] == 2


def _seed_cluster_run(store, run_id, model, adapter, cluster, scores):
    """Seed one finished run with an explicit (adapter, cluster) and overall scores."""
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h", cluster=cluster)
    store.upsert_adapter(run_id, adapter, "0.1")
    for i, s in enumerate(scores):
        sid = f"easy-{i:02d}-test"
        result = ScenarioResult(
            scenario_id=sid, adapter=adapter, trial_index=0,
            status="pass" if s > 0.5 else "fail", score=s,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if s > 0.5 else "wrong_tool",
            checks=[], trace=_trace(sid, adapter),
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[], ranking_dims=["overall"],
            scenario_hash="h", category="coding", tier="easy", trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_compute_matrix_splits_same_model_adapter_across_clusters(tmp_path):
    """Same model+adapter on single vs dual → two separate rows (different config)."""
    from toolery.rankings.compute import compute_matrix
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_cluster_run(store, "r_single", "m", "raw", "single", [1.0, 1.0])
    _seed_cluster_run(store, "r_dual", "m", "raw", "dual", [0.0, 0.0])

    matrix = compute_matrix(store=store, dimensions=["overall"])
    by_cluster = {r["cluster"]: r for r in matrix}
    assert set(by_cluster) == {"single", "dual"}
    assert by_cluster["single"]["scores"]["overall"] == 1.0
    assert by_cluster["dual"]["scores"]["overall"] == 0.0


def test_compute_matrix_aggregates_same_config_reruns(tmp_path):
    """Re-running the SAME (model, adapter, cluster) aggregates into one row."""
    from toolery.rankings.compute import compute_matrix
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_cluster_run(store, "r1", "m", "raw", "single", [1.0, 1.0])
    _seed_cluster_run(store, "r2", "m", "raw", "single", [0.0, 0.0])

    matrix = compute_matrix(store=store, dimensions=["overall"])
    rows = [r for r in matrix if r["cluster"] == "single"]
    assert len(rows) == 1                       # not two separate rows
    row = rows[0]
    assert row["runs"] == 2                      # both runs counted
    assert 0.0 < row["scores"]["overall"] < 1.0  # averaged, not one-or-the-other
    # stability is computed from the run-to-run spread of the two runs
    assert row["stability"]["overall"]["worst"] == 0.0
    assert row["stability"]["overall"]["mean"] is not None


def test_compute_matrix_pass_counts_from_latest_run(tmp_path):
    """pass_counts = raw passed/total trial counts from the pair's MOST RECENT
    run (feeds the rankings 'Passed x/N' column). Older runs don't blend in."""
    from toolery.rankings.compute import compute_matrix
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed_cluster_run(store, "r_old", "m", "raw", "single", [1.0, 1.0, 1.0])   # 3/3
    _seed_cluster_run(store, "r_new", "m", "raw", "single", [1.0, 0.0, 0.0])   # 1/3

    matrix = compute_matrix(store=store, dimensions=["overall"])
    assert len(matrix) == 1
    assert matrix[0]["pass_counts"] == {"passed": 1, "total": 3}


def test_collapse_model_mean_sums_pass_counts():
    from toolery.rankings.compute import collapse_matrix_rows
    matrix = [
        {"model": "m", "adapter": "raw", "cluster": None, "runs": 1,
         "scores": {"overall": 0.4}, "perf": {}, "stability": {},
         "scenarios_hashes": {"h"}, "pass_counts": {"passed": 2, "total": 5}},
        {"model": "m", "adapter": "hermes", "cluster": None, "runs": 1,
         "scores": {"overall": 0.8}, "perf": {}, "stability": {},
         "scenarios_hashes": {"h"}, "pass_counts": {"passed": 4, "total": 5}},
    ]
    mean = collapse_matrix_rows(matrix, "model_mean")
    assert mean[0]["pass_counts"] == {"passed": 6, "total": 10}
    # model_best carries the winning adapter's counts through unchanged.
    best = collapse_matrix_rows(matrix, "model_best")
    assert best[0]["pass_counts"] == {"passed": 4, "total": 5}

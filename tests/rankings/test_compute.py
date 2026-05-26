from datetime import UTC, datetime
from pathlib import Path

from llm_test.core.models import Message, ScenarioResult, TraceResult
from llm_test.core.store import Store
from llm_test.rankings.compute import regenerate_rankings


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
    matrix = __import__("llm_test.rankings.compute", fromlist=["compute_matrix"]).compute_matrix(
        store=store, dimensions=["overall"]
    )
    row = matrix[0]
    assert "stability" in row
    assert row["stability"]["overall"]["mean"] == 0.5
    assert row["stability"]["overall"]["worst"] == 0.5
    assert row["stability"]["overall"]["pass_rate"] == 0.0


def test_collapse_matrix_rows_modes():
    from llm_test.rankings.compute import collapse_matrix_rows
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
    from llm_test.rankings.compute import compute_failure_breakdown
    breakdown = compute_failure_breakdown(store)
    assert breakdown[("model_fail", "raw")]["wrong_tool"] == 2
    assert compute_failure_breakdown(store, dimensions=["coding"])[("model_fail", "raw")]["wrong_tool"] == 2

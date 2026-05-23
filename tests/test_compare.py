from datetime import UTC, datetime

from llm_test.compare import compare_runs
from llm_test.core.models import Message, ScenarioResult, TraceResult
from llm_test.core.store import Store


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _make_run(store: Store, run_id: str, model: str, scenarios_pass: dict[str, bool]):
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    for sid, passed in scenarios_pass.items():
        result = ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0,
            status="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if passed else "wrong_tool",
            checks=[], trace=_trace(sid, "raw"),
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[],
            ranking_dims=["overall"],
            scenario_hash="h", category="tool_selection", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_compare_runs_writes_md(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _make_run(store, "A", "m", {f"easy-{i:02d}-x": True for i in range(5)} | {"hard-01-x": False})
    _make_run(store, "B", "m", {f"easy-{i:02d}-x": True for i in range(5)} | {"hard-01-x": True})
    out = tmp_path / "compare.md"
    compare_runs(store=store, run_a="A", run_b="B", out_path=out)
    md = out.read_text()
    assert "A" in md and "B" in md
    assert "hard-01-x" in md

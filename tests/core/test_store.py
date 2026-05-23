from datetime import UTC, datetime

from llm_test.core.models import Message, ScenarioResult, TraceResult
from llm_test.core.store import Store


def _trace():
    return TraceResult(
        scenario_id="easy-01-x", adapter="raw", trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
    )


def _result():
    return ScenarioResult(
        scenario_id="easy-01-x", adapter="raw", trial_index=0,
        status="pass", score=1.0, call_count=0, budget_max=1,
        latency_ms=42, failure_kind=None, checks=[], trace=_trace(),
    )


def test_store_inserts_and_reads_run(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "2026-05-23T18-00_test-model"
    store.create_run(run_id=run_id, model="test-model", base_url="http://x",
                     started_at=datetime.now(UTC).isoformat(),
                     config_json="{}", scenarios_hash="abc123")
    store.upsert_adapter(run_id, "raw", "0.1")
    store.write_scenario_result(run_id, _result(), tags=["tool_call"],
                                ranking_dims=["overall"],
                                scenario_hash="hashX", category="tool_selection", tier="easy",
                                trace_path="traces/easy-01-x.json")
    rows = store.fetch_results_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"
    assert rows[0]["score"] == 1.0

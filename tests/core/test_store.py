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


def test_init_schema_creates_in_flight_units(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    with store.conn() as c:
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "in_flight_units" in tables
        cols = {r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
        assert "updated_at" in cols
        cols_inflight = {r[1] for r in c.execute(
            "PRAGMA table_info(in_flight_units)"
        ).fetchall()}
        assert cols_inflight == {
            "run_id", "scenario_id", "adapter", "trial_index", "started_at"
        }


def test_in_flight_round_trip(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "2026-05-27T20-00_test"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")

    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
    store.mark_in_flight(run_id, "easy-01", "raw", 1, "2026-05-27T20:00:02Z")

    rows = store.fetch_in_flight_for_run(run_id)
    assert len(rows) == 2
    assert {r["trial_index"] for r in rows} == {0, 1}
    assert all(r["scenario_id"] == "easy-01" for r in rows)
    assert all(r["adapter"] == "raw" for r in rows)

    # updated_at heartbeat fired on mark_in_flight
    run = store.fetch_run(run_id)
    assert run["updated_at"] is not None

    store.clear_in_flight(run_id, "easy-01", "raw", 0)
    rows = store.fetch_in_flight_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["trial_index"] == 1

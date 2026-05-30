from datetime import UTC, datetime

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store


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


def test_clear_all_in_flight_removes_only_target_run(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    for rid in ("run-a", "run-b"):
        store.create_run(run_id=rid, model="m", base_url="http://x",
                         started_at="2026-05-27T20:00:00Z",
                         config_json="{}", scenarios_hash="x")
        store.mark_in_flight(rid, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
        store.mark_in_flight(rid, "easy-01", "raw", 1, "2026-05-27T20:00:02Z")

    store.clear_all_in_flight("run-a")
    assert store.fetch_in_flight_for_run("run-a") == []
    assert len(store.fetch_in_flight_for_run("run-b")) == 2


def test_mark_stale_aborted_clears_and_updates_status(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "stale-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")

    store.mark_stale_aborted(run_id)
    assert store.fetch_in_flight_for_run(run_id) == []
    assert store.fetch_run(run_id)["status"] == "aborted"


def test_finish_run_clears_orphan_in_flight(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "ending-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
    store.mark_in_flight(run_id, "easy-02", "raw", 0, "2026-05-27T20:00:02Z")

    store.finish_run(run_id, finished_at="2026-05-27T20:05:00Z",
                    duration_s=300.0, status="done")
    assert store.fetch_in_flight_for_run(run_id) == []


def test_reopen_run_clears_orphan_in_flight(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "resuming-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.finish_run(run_id, finished_at="2026-05-27T20:05:00Z",
                    duration_s=300.0, status="aborted")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")

    store.reopen_run(run_id)
    assert store.fetch_in_flight_for_run(run_id) == []

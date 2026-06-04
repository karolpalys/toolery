import sqlite3
from datetime import UTC, datetime

from toolery.core.models import (
    Budget,
    Category,
    Message,
    Scenario,
    ScenarioResult,
    Scoring,
    Tier,
    TraceResult,
)
from toolery.core.scorer import evaluate
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


# ── correctness_score persistence (Task 3) ─────────────────────────────────

def _store(tmp_path) -> Store:
    s = Store(tmp_path / "runs.db")
    s.init_schema()
    return s


def _scenario():
    return Scenario(
        id="t-01-x", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p",
        tools=["get_weather"], budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[]),
    )


def _correctness_trace():
    return TraceResult(
        scenario_id="t-01-x", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_scenario_results_has_correctness_column(tmp_path):
    s = _store(tmp_path)
    with s.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(scenario_results)").fetchall()}
    assert "correctness_score" in cols


def test_write_and_read_correctness_score(tmp_path):
    s = _store(tmp_path)
    s.create_run(run_id="r1", model="m", base_url="u",
                 started_at="2026-05-23T18:00:00Z", config_json="{}", scenarios_hash="h")
    result = evaluate(_scenario(), _correctness_trace())
    s.write_scenario_result(
        "r1", result, tags=[], ranking_dims=["overall"], scenario_hash="h",
        category="tool_selection", tier="easy", trace_path="traces/x.json",
    )
    rows = s.fetch_results_for_run("r1")
    assert rows[0]["correctness_score"] == result.correctness_score


# ── token columns persistence (Task 4) ─────────────────────────────────────

def _result_with_tokens(prompt, completion, gen_ms, scenario_id="t-01-x"):
    trace = TraceResult(
        scenario_id=scenario_id, adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response="x",
        started_at_iso="x", duration_ms=gen_ms,
    )
    return ScenarioResult(
        scenario_id=scenario_id, adapter="raw", trial_index=0, status="pass",
        score=1.0, call_count=0, budget_max=1, latency_ms=gen_ms,
        failure_kind=None, checks=[], trace=trace,
        prompt_tokens=prompt, completion_tokens=completion, gen_ms=gen_ms,
    )


def test_token_columns_round_trip(tmp_path):
    from toolery.core.store import Store
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run("run1", "m", "http://x", "2026-06-02T00:00:00Z", "{}", "h")
    store.write_scenario_result(
        "run1", _result_with_tokens(100, 20, 400),
        tags=[], ranking_dims=["overall"], scenario_hash="",
        category="tool_selection", tier="easy", trace_path="traces/a.json",
    )
    rows = store.fetch_results_for_run("run1")
    assert rows[0]["completion_tokens"] == 20
    assert rows[0]["gen_ms"] == 400
    assert rows[0]["prompt_tokens"] == 100


def test_fetch_run_token_totals_sums(tmp_path):
    from toolery.core.store import Store
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run("run1", "m", "http://x", "2026-06-02T00:00:00Z", "{}", "h")
    for i, (p, c, ms) in enumerate([(100, 20, 400), (80, 30, 600)]):
        store.write_scenario_result(
            "run1", _result_with_tokens(p, c, ms, scenario_id=f"t-0{i+1}-x"),
            tags=[], ranking_dims=["overall"], scenario_hash="",
            category="tool_selection", tier="easy",
            trace_path=f"traces/{i}.json",
        )
    completion, gen_ms = store.fetch_run_token_totals("run1")
    assert (completion, gen_ms) == (50, 1000)


def test_old_db_migrates_token_columns(tmp_path):
    # Old DB: scenario_results WITHOUT token columns; init_schema must add them.
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE scenario_results (
          result_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT, scenario_id TEXT NOT NULL, scenario_hash TEXT NOT NULL,
          tier TEXT NOT NULL, category TEXT NOT NULL,
          tags_json TEXT, ranking_dims_json TEXT,
          adapter TEXT NOT NULL, trial_index INTEGER NOT NULL,
          status TEXT, score REAL NOT NULL,
          call_count INTEGER NOT NULL, budget_max INTEGER,
          latency_ms INTEGER, failure_kind TEXT,
          trace_path TEXT, checks_json TEXT
        );
    """)
    con.commit()
    con.close()

    from toolery.core.store import Store
    store = Store(db)
    store.init_schema()  # must not raise
    with store.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(scenario_results)").fetchall()}
    assert {"prompt_tokens", "completion_tokens", "gen_ms"} <= cols

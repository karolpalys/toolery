import pytest
from textual.app import App
from textual.widgets import DataTable, Static

from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.tui.home_tab import HomeTab


class _Host(App):
    def __init__(self, scanner, known_models=None):
        super().__init__()
        self._scanner = scanner
        self._known = known_models or set()

    def compose(self):
        yield HomeTab(
            id="home-tab",
            scanner=self._scanner,
            known_models_provider=lambda: self._known,
        )


@pytest.mark.asyncio
async def test_home_tab_initial_empty_state():
    async def never_called(*a, **k):
        raise AssertionError("scan should not run on mount")

    app = _Host(scanner=never_called)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        status = app.query_one("#scan-status", Static)
        assert "Ready to discover" in str(status.content)
        tbl = app.query_one(DataTable)
        assert tbl.row_count == 0


@pytest.mark.asyncio
async def test_scan_populates_table():
    fake_endpoints = [
        EndpointInfo(port=8888, base_url="http://localhost:8888",
                     model_id="MiniMax-M2.7", served_model_id="MiniMax-M2.7",
                     models=["MiniMax-M2.7"], server_hint="vLLM"),
        EndpointInfo(port=8080, base_url="http://localhost:8080",
                     model_id="qwen3-coder-4b", served_model_id="qwen3-coder-4b",
                     models=["qwen3-coder-4b"], server_hint="vLLM"),
    ]

    async def fake_scan(ports):
        return fake_endpoints

    app = _Host(scanner=fake_scan, known_models={"MiniMax-M2.7"})
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.click("#scan")
        await pilot.pause()
        tbl = app.query_one(DataTable)
        assert tbl.row_count == 2
        assert "Known" in str(tbl.get_row_at(0))
        assert "New" in str(tbl.get_row_at(1))


@pytest.mark.asyncio
async def test_row_select_invokes_app_opener():
    fake_endpoints = [
        EndpointInfo(port=8888, base_url="http://localhost:8888",
                     model_id="m", served_model_id="m", models=["m"],
                     server_hint="vLLM"),
    ]

    async def fake_scan(ports):
        return fake_endpoints

    captured = []

    class HostWithOpener(_Host):
        async def open_launch_modal(self, endpoint):
            captured.append(endpoint)

    app = HostWithOpener(scanner=fake_scan)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.click("#scan")
        await pilot.pause()
        tbl = app.query_one(DataTable)
        tbl.move_cursor(row=0)
        tbl.focus()
        await pilot.press("enter")
        await pilot.pause()
    assert len(captured) == 1
    assert captured[0].port == 8888


import json as _json

from llm_test.tui.home_tab import _build_plan


def test_build_plan_orders_scenario_adapter_trial():
    config_json = _json.dumps({
        "adapter": ["raw", "hermes"],
        "trials": 3,
        "tier": "easy",
        "category": "all",
    })

    plan = _build_plan(
        config_json,
        scenario_ids_in_loader_order=["easy-01", "easy-02"],
    )

    # Expected: scenario-major, then adapter, then trial — matches Runner.run() order
    assert plan == [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1), ("easy-01", "raw", 2),
        ("easy-01", "hermes", 0), ("easy-01", "hermes", 1), ("easy-01", "hermes", 2),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1), ("easy-02", "raw", 2),
        ("easy-02", "hermes", 0), ("easy-02", "hermes", 1), ("easy-02", "hermes", 2),
    ]


def test_build_plan_handles_single_adapter_string():
    """If config_json has adapter as a string (legacy), still produce a plan."""
    config_json = _json.dumps({"adapter": "raw", "trials": 2, "tier": "easy",
                               "category": "all"})
    plan = _build_plan(config_json, scenario_ids_in_loader_order=["easy-01"])
    assert plan == [("easy-01", "raw", 0), ("easy-01", "raw", 1)]


from llm_test.tui.home_tab import _classify_plan


def test_classify_plan_three_states():
    plan = [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1), ("easy-01", "raw", 2),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1),
        ("easy-03", "raw", 0), ("easy-03", "raw", 1),
    ]
    completed = {
        ("easy-01", "raw", 0): {"status": "pass"},
        ("easy-01", "raw", 1): {"status": "fail"},
    }
    running = {
        ("easy-01", "raw", 2): {"started_at": "2026-05-27T20:00:00Z"},
        ("easy-02", "raw", 0): {"started_at": "2026-05-27T20:00:01Z"},
    }
    rows = _classify_plan(plan, completed, running, upcoming_visible=3)
    states = [r[0] for r in rows]
    assert states == ["done", "done", "running", "running", "upcoming", "upcoming", "upcoming"]


def test_classify_plan_caps_upcoming():
    plan = [(f"easy-{i:02d}", "raw", 0) for i in range(50)]
    rows = _classify_plan(plan, completed={}, running={}, upcoming_visible=5)
    assert len(rows) == 5
    assert all(r[0] == "upcoming" for r in rows)


def test_classify_plan_combined_with_build_plan():
    """Compose _build_plan + _classify_plan to verify they cooperate."""
    plan = _build_plan(
        _json.dumps({"adapter": ["raw"], "trials": 2,
                     "tier": "easy", "category": "all"}),
        scenario_ids_in_loader_order=["easy-01", "easy-02"],
    )
    running = {("easy-01", "raw", 1): {"started_at": "2026-05-27T20:00:01Z"}}
    completed = {("easy-01", "raw", 0): {"status": "pass", "score": 1.0,
                                          "scenario_id": "easy-01", "adapter": "raw",
                                          "trial_index": 0}}
    rows = _classify_plan(plan, completed, running, upcoming_visible=10)
    states = [r[0] for r in rows]
    assert states == ["done", "running", "upcoming", "upcoming"]


def test_classify_plan_running_after_done_marks_live_edge():
    """The live edge is the last running entry in the rendered rows."""
    plan = [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1),
    ]
    completed = {("easy-01", "raw", 0): {"status": "pass"}}
    running = {("easy-01", "raw", 1): {}, ("easy-02", "raw", 0): {}}
    rows = _classify_plan(plan, completed, running, upcoming_visible=10)

    last_running = max(i for i, r in enumerate(rows) if r[0] == "running")
    assert last_running == 2


from llm_test.tui.home_tab import _detail_block_running, _detail_block_upcoming


def test_detail_block_running_includes_elapsed():
    from datetime import UTC, datetime, timedelta
    started = (datetime.now(UTC) - timedelta(seconds=14)).isoformat().replace("+00:00", "Z")
    block = _detail_block_running(
        scenario_id="easy-01", adapter="raw", trial=2,
        started_at=started,
    )
    text = str(block)
    assert "running" in text
    assert "easy-01" in text
    # Elapsed parses to a small mm:ss number
    assert ":" in text


def test_detail_block_upcoming_includes_position():
    block = _detail_block_upcoming(
        scenario_id="easy-09", adapter="raw", trial=0,
        position_in_queue=12,
    )
    text = str(block)
    assert "upcoming" in text
    assert "12" in text


from llm_test.tui.home_tab import STALE_HEARTBEAT_SECONDS, _is_stale_run


def test_is_stale_run_when_updated_at_old():
    from datetime import UTC, datetime, timedelta
    old = (datetime.now(UTC) - timedelta(seconds=STALE_HEARTBEAT_SECONDS + 30)
           ).isoformat().replace("+00:00", "Z")
    run = {"status": "running", "updated_at": old}
    assert _is_stale_run(run) is True


def test_is_stale_run_when_fresh():
    from datetime import UTC, datetime, timedelta
    fresh = (datetime.now(UTC) - timedelta(seconds=10)
             ).isoformat().replace("+00:00", "Z")
    run = {"status": "running", "updated_at": fresh}
    assert _is_stale_run(run) is False


def test_is_stale_run_skips_old_runs_without_updated_at():
    run = {"status": "running", "updated_at": None}
    assert _is_stale_run(run) is False


def test_is_stale_run_skips_finished_runs():
    run = {"status": "done", "updated_at": None}
    assert _is_stale_run(run) is False


@pytest.mark.asyncio
async def test_refresh_from_db_aborts_stale_run(tmp_path, monkeypatch):
    """End-to-end watchdog: a stale running run in the store gets marked
    aborted on the next refresh_from_db tick."""
    from datetime import UTC, datetime, timedelta

    from textual.app import App

    from llm_test.core.store import Store
    from llm_test.tui.home_tab import STALE_HEARTBEAT_SECONDS, HomeTab

    db = tmp_path / "runs.db"
    store = Store(db)
    store.init_schema()
    run_id = "stale-watchdog-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    # backdate updated_at by > STALE_HEARTBEAT_SECONDS
    stale = (datetime.now(UTC) - timedelta(seconds=STALE_HEARTBEAT_SECONDS + 30)
             ).isoformat().replace("+00:00", "Z")
    with store.conn() as c:
        c.execute("UPDATE runs SET updated_at=? WHERE run_id=?", (stale, run_id))
    # add an in_flight row to verify cleanup
    store.mark_in_flight(run_id, "easy-01", "raw", 0, stale)

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path))

    async def never_called(*_a, **_kw):
        return []

    class _Host(App):
        def compose(self):
            yield HomeTab(id="home-tab", scanner=never_called,
                          known_models_provider=lambda: set())

    app = _Host()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        tab = app.query_one(HomeTab)
        tab.refresh_from_db()

    refreshed = store.fetch_run(run_id)
    assert refreshed["status"] == "aborted"
    assert store.fetch_in_flight_for_run(run_id) == []

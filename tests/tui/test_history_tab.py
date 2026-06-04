from __future__ import annotations

import datetime as dt

import pytest
from textual.app import App, ComposeResult

from toolery.core.store import Store
from toolery.tui.history_tab import ConfirmRemoveModal, HistoryTab, MarkdownModal


def _seed_run(store: Store, run_id: str = "test-run-1",
              model: str = "glm-5.1") -> None:
    store.init_schema()
    store.create_run(
        run_id=run_id, model=model, base_url="http://localhost:8000",
        started_at=dt.datetime(2026, 5, 25, 12, 0, 0).isoformat(),
        config_json="{}", scenarios_hash="abc",
    )
    store.finish_run(run_id=run_id,
                     finished_at=dt.datetime(2026, 5, 25, 12, 1, 0).isoformat(),
                     duration_s=60.0, status="done")
    store.upsert_adapter(run_id, "raw", "1.0")


class _Host(App):
    def compose(self) -> ComposeResult:
        yield HistoryTab(id="history-tab")


@pytest.fixture
def seeded_results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    store = Store(tmp_path / "runs.db")
    _seed_run(store)
    return tmp_path


@pytest.mark.asyncio
async def test_delete_key_opens_confirm_modal(seeded_results_dir):
    """Regression: pressing Delete on a row used to crash the TUI with
    NoActiveWorker because action_remove_run called push_screen_wait outside
    a @work context."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#history-table")
        assert tbl.row_count == 1
        tbl.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmRemoveModal)
        await pilot.press("n")
        await pilot.pause()


@pytest.mark.asyncio
async def test_delete_confirm_removes_run(seeded_results_dir):
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#history-table")
        tbl.focus()
        await pilot.pause()
        await pilot.press("delete")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmRemoveModal)
        await pilot.press("y")
        await pilot.pause()
        tbl = app.query_one("#history-table")
        assert tbl.row_count == 0

    store = Store(seeded_results_dir / "runs.db")
    assert store.fetch_all_runs() == []


@pytest.mark.asyncio
async def test_enter_opens_details_modal(seeded_results_dir):
    """Regression: same NoActiveWorker bug affected _open_details via the
    Enter/row-selected path."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#history-table")
        tbl.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, MarkdownModal)
        await pilot.press("escape")
        await pilot.pause()


def _add_run(store: Store, run_id: str, minute: int = 5) -> None:
    """Append another finished run to the store (simulates a run completing
    while the History tab is open)."""
    store.create_run(
        run_id=run_id, model="glm-5.1", base_url="http://localhost:8000",
        started_at=dt.datetime(2026, 5, 25, 12, minute, 0).isoformat(),
        config_json="{}", scenarios_hash="abc",
    )
    store.finish_run(
        run_id=run_id,
        finished_at=dt.datetime(2026, 5, 25, 12, minute + 1, 0).isoformat(),
        duration_s=60.0, status="done")
    store.upsert_adapter(run_id, "raw", "1.0")


@pytest.mark.asyncio
async def test_poll_for_new_runs_picks_up_new_run(seeded_results_dir):
    """A run finishing while the tab is open shows up without pressing Ctrl+R."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tab = app.query_one(HistoryTab)
        tbl = app.query_one("#history-table")
        assert tbl.row_count == 1

        store = Store(seeded_results_dir / "runs.db")
        _add_run(store, "test-run-2")

        tab._poll_for_new_runs()
        await pilot.pause()
        assert tbl.row_count == 2


@pytest.mark.asyncio
async def test_poll_for_new_runs_is_noop_when_unchanged(seeded_results_dir):
    """No DB change → the poll must not rebuild/yank the cursor."""
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tab = app.query_one(HistoryTab)
        tbl = app.query_one("#history-table")
        store = Store(seeded_results_dir / "runs.db")
        _add_run(store, "test-run-2", minute=10)
        tab._poll_for_new_runs()  # picks up run 2
        await pilot.pause()
        assert tbl.row_count == 2

        # Park the cursor on a specific run, then poll again with no DB change.
        tbl.focus()
        tbl.move_cursor(row=1)
        parked = tab._row_run_id(1)
        tab._poll_for_new_runs()
        await pilot.pause()
        assert tab._row_run_id(tbl.cursor_row) == parked


@pytest.mark.asyncio
async def test_poll_for_new_runs_preserves_cursor_and_filter(seeded_results_dir):
    """On a real change the cursor stays on the same run and the active filter
    is preserved (not reset to show-all)."""
    from textual.widgets import Input

    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tab = app.query_one(HistoryTab)
        tbl = app.query_one("#history-table")
        store = Store(seeded_results_dir / "runs.db")
        _add_run(store, "keep-run-2", minute=10)
        tab._poll_for_new_runs()
        await pilot.pause()

        # Filter to the two glm runs, park cursor on a known run_id.
        flt = app.query_one("#hist-filter", Input)
        flt.value = "run"
        await pilot.pause()
        target = tab._row_run_id(tbl.cursor_row)

        _add_run(store, "keep-run-3", minute=20)
        tab._poll_for_new_runs()
        await pilot.pause()
        # New run appeared, cursor still on the same run, filter still applied.
        assert tab._row_run_id(tbl.cursor_row) == target
        assert flt.value == "run"
        assert tbl.row_count == 3


def _run():
    return {"run_id": "run1", "model": "m", "status": "done",
            "started_at": "x", "finished_at": "y", "duration_s": 1.0,
            "cluster": None, "base_url": "http://x", "config_json": "{}"}


def test_details_md_includes_effective_tps():
    from toolery.tui.history_tab import _build_details_md
    results = [
        {"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
         "completion_tokens": 40, "gen_ms": 400},
        {"scenario_id": "b", "tier": "easy", "status": "pass", "score": 1.0,
         "completion_tokens": 60, "gen_ms": 600},
    ]
    md = _build_details_md(_run(), results, perf_rows=[], adapters=["raw"])
    assert "Eff gen t/s" in md
    assert "100.0" in md  # 100 tok / 1.0s


def test_details_md_effective_tps_na_without_tokens():
    from toolery.tui.history_tab import _build_details_md
    results = [{"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
                "completion_tokens": 0, "gen_ms": 0}]
    md = _build_details_md(_run(), results, perf_rows=[], adapters=["raw"])
    assert "n/a" in md


def test_details_md_effective_tps_na_with_none_columns():
    from toolery.tui.history_tab import _build_details_md
    # Old-DB rows: token columns absent entirely (None / missing).
    results = [{"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
                "completion_tokens": None, "gen_ms": None}]
    md = _build_details_md(_run(), results, perf_rows=[], adapters=["raw"])
    assert "Eff gen t/s" in md
    assert "n/a" in md

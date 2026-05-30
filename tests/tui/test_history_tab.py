from __future__ import annotations

import datetime as dt

import pytest
from textual.app import App, ComposeResult

from llm_test.core.store import Store
from llm_test.tui.history_tab import ConfirmRemoveModal, HistoryTab, MarkdownModal


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
    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path))
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

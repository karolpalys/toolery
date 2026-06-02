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

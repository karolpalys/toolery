import pytest
from textual.app import App
from textual.widgets import Static

from llm_test.core.store import Store
from llm_test.tui.live_tab import LiveTab


class _Host(App):
    def __init__(self, store):
        super().__init__()
        self._store = store

    def compose(self):
        yield LiveTab(id="live-tab", store=self._store)


@pytest.mark.asyncio
async def test_live_tab_shows_running_run(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run(
        run_id="2026-05-24T10-00_demo",
        model="demo-model",
        base_url="http://localhost:1234",
        started_at="2026-05-24T10:00:00+00:00",
        config_json="{}",
        scenarios_hash="",
    )
    app = _Host(store)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        live = app.query_one(LiveTab)
        live.refresh_from_db()
        await pilot.pause()
        status = app.query_one("#current-run", Static)
        # Textual 8.x uses .content instead of .renderable
        assert "demo-model" in str(status.content)

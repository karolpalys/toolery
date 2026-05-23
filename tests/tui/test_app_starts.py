import pytest

from llm_test.tui.app import LLMTestApp


@pytest.mark.asyncio
async def test_app_has_four_tabs():
    app = LLMTestApp(run_id=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        # 4 tabs: Live, History, Rankings, Scenarios — find TabPane by id
        pane_ids = {p.id for p in app.query("TabPane")}
        assert "live" in pane_ids
        assert "history" in pane_ids
        assert "rankings" in pane_ids
        assert "scenarios" in pane_ids

import pytest

from toolery.tui.app import TooleryApp


@pytest.mark.asyncio
async def test_app_has_expected_tabs():
    app = TooleryApp(run_id=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        # The Live tab was folded into Home; tabs are now:
        # Home, Rankings, Compare, Scenarios, History, Profiles.
        pane_ids = {p.id for p in app.query("TabPane")}
        assert "live" not in pane_ids
        assert {"home", "rankings", "compare",
                "scenarios", "history", "setup"} <= pane_ids

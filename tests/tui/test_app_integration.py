import asyncio
from unittest.mock import AsyncMock

import pytest
from textual.widgets import DataTable, TabbedContent

from llm_test.core import runner_subprocess
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.tui.app import LLMTestApp
from llm_test.tui.home_tab import HomeTab
from llm_test.tui.launch_modal import LaunchModal


@pytest.mark.asyncio
async def test_home_tab_is_initial():
    app = LLMTestApp(run_id=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        tabs = app.query_one(TabbedContent)
        assert tabs.active == "home"


@pytest.mark.asyncio
async def test_launch_flow_switches_to_live(monkeypatch):
    fake_endpoint = EndpointInfo(
        port=8080, base_url="http://localhost:8080",
        model_id="qwen3-coder-4b", models=["qwen3-coder-4b"], server_hint="vLLM",
    )

    async def fake_scan(ports):
        return [fake_endpoint]

    fake_proc = AsyncMock(spec=asyncio.subprocess.Process)
    fake_proc.returncode = None
    spawn_mock = AsyncMock(return_value=fake_proc)
    monkeypatch.setattr(runner_subprocess, "spawn_run", spawn_mock)

    active_tab = None
    app = LLMTestApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        home = app.query_one(HomeTab)
        home._scanner = fake_scan
        await pilot.click("#scan")
        await pilot.pause()
        tbl = app.query_one(DataTable)
        tbl.move_cursor(row=0)
        tbl.focus()
        await pilot.press("enter")
        await pilot.pause()
        # Verify modal opened before clicking Run (catches silent modal failures)
        assert isinstance(app.screen, LaunchModal)
        await pilot.click("#run")
        await pilot.pause()
        # Wait for the @work worker to finish (it spawns subprocess + switches tab)
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Capture state while app is still running
        active_tab = app.query_one(TabbedContent).active

    spawn_mock.assert_awaited_once()
    assert active_tab == "live"

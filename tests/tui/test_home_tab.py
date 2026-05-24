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
        status = app.query_one("#status", Static)
        assert "Click Scan" in str(status.content)
        tbl = app.query_one(DataTable)
        assert tbl.row_count == 0


@pytest.mark.asyncio
async def test_scan_populates_table():
    fake_endpoints = [
        EndpointInfo(port=8888, base_url="http://localhost:8888",
                     model_id="MiniMax-M2.7", models=["MiniMax-M2.7"], server_hint="vLLM"),
        EndpointInfo(port=8080, base_url="http://localhost:8080",
                     model_id="qwen3-coder-4b", models=["qwen3-coder-4b"], server_hint="vLLM"),
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
                     model_id="m", models=["m"], server_hint="vLLM"),
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

import pytest
from textual import work
from textual.app import App
from textual.widgets import RadioButton, Static

from llm_test.core.adapter_probe import AdapterStatus
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.tui.launch_modal import LaunchModal


def _endpoint():
    return EndpointInfo(
        port=8080, base_url="http://localhost:8080",
        model_id="qwen3-coder-4b", models=["qwen3-coder-4b"],
        server_hint="vLLM",
    )


def _adapters_all_available():
    return {
        "raw": AdapterStatus(available=True),
        "hermes": AdapterStatus(available=True),
        "claude_code": AdapterStatus(available=True),
        "codex": AdapterStatus(available=True),
    }


class _Host(App):
    def __init__(self, endpoint, adapters):
        super().__init__()
        self.endpoint = endpoint
        self.adapters = adapters
        self.result = "UNSET"

    def on_mount(self):
        self._run_modal()

    @work
    async def _run_modal(self):
        args = await self.push_screen_wait(LaunchModal(self.endpoint, self.adapters))
        self.result = args
        self.exit()


@pytest.mark.asyncio
async def test_modal_shows_model_and_endpoint():
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, LaunchModal)
        text = " ".join(str(s.content) for s in app.screen.query(Static))
        assert "qwen3-coder-4b" in text
        assert "http://localhost:8080" in text
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_modal_cancel_returns_none():
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_modal_submit_returns_runargs():
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        await pilot.click("#run")
        await pilot.pause()
    assert app.result is not None
    assert app.result.model == "qwen3-coder-4b"
    assert app.result.base_url == "http://localhost:8080"
    assert app.result.tier == "all"
    assert app.result.adapter == "raw"
    assert app.result.trials == 5


@pytest.mark.asyncio
async def test_modal_disabled_adapter_shows_reason():
    adapters = {
        "raw": AdapterStatus(available=True),
        "hermes": AdapterStatus(available=True),
        "claude_code": AdapterStatus(available=False, reason="set CLAUDE_CLI_PATH"),
        "codex": AdapterStatus(available=False, reason="set CODEX_CLI_PATH"),
    }
    app = _Host(_endpoint(), adapters)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        btn = app.screen.query_one("#adapter-claude_code", RadioButton)
        assert btn.disabled is True
        assert "set CLAUDE_CLI_PATH" in btn.label.plain
        await pilot.press("escape")

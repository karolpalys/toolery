import pytest
from textual import work
from textual.app import App
from textual.widgets import RadioButton, SelectionList, Static

from toolery.core.adapter_probe import AdapterStatus
from toolery.core.endpoint_scanner import EndpointInfo
from toolery.tui.launch_modal import LaunchModal


def _endpoint():
    return EndpointInfo(
        port=8080, base_url="http://localhost:8080",
        model_id="qwen3-coder-4b", served_model_id="qwen3-coder-4b",
        models=["qwen3-coder-4b"], server_hint="vLLM",
    )


def _adapters_all_available():
    return {
        "raw": AdapterStatus(available=True),
        "cloud": AdapterStatus(available=True),
        "hermes": AdapterStatus(available=True),
    }


class _Host(App):
    def __init__(self, endpoint, adapters, interrupted_run=None):
        super().__init__()
        self.endpoint = endpoint
        self.adapters = adapters
        self.interrupted = interrupted_run
        self.result = "UNSET"

    def on_mount(self):
        self._run_modal()

    @work
    async def _run_modal(self):
        args = await self.push_screen_wait(
            LaunchModal(self.endpoint, self.adapters,
                        interrupted_run=self.interrupted))
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
async def test_modal_without_interrupted_run_has_no_resume_banner():
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        # The banner only renders when interrupted_run is set.
        assert not app.screen.query("#resume-banner")
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_modal_category_single_pick_deselects_all():
    """Picking one specific category drops the default 'all' so the run
    targets just that category."""
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        sl = app.screen.query_one("#category", SelectionList)
        sl.select("coding")
        await pilot.pause()
        await pilot.click("#run")
        await pilot.pause()
    assert app.result is not None
    assert app.result.category == "coding"


@pytest.mark.asyncio
async def test_modal_category_is_multiselect():
    """A second specific category adds to the selection (multi-select) rather
    than replacing the first — the column is not single-choice."""
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        sl = app.screen.query_one("#category", SelectionList)
        sl.select("coding")
        await pilot.pause()
        sl.select("debugging")
        await pilot.pause()
        await pilot.click("#run")
        await pilot.pause()
    assert app.result is not None
    assert sorted(app.result.category.split(",")) == ["coding", "debugging"]


@pytest.mark.asyncio
async def test_modal_reselecting_all_clears_specifics():
    """Selecting 'all' after picking specifics resets to the whole suite."""
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        sl = app.screen.query_one("#category", SelectionList)
        sl.select("coding")
        await pilot.pause()
        sl.select("all")
        await pilot.pause()
        await pilot.click("#run")
        await pilot.pause()
    assert app.result is not None
    assert app.result.category == "all"


@pytest.mark.asyncio
async def test_modal_tier_is_multiselect():
    """Difficulty tiers behave the same as categories — multi-select."""
    app = _Host(_endpoint(), _adapters_all_available())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        sl = app.screen.query_one("#tier", SelectionList)
        sl.select("easy")
        await pilot.pause()
        sl.select("hard")
        await pilot.pause()
        await pilot.click("#run")
        await pilot.pause()
    assert app.result is not None
    assert sorted(app.result.tier.split(",")) == ["easy", "hard"]


@pytest.mark.asyncio
async def test_modal_disabled_adapter_shows_reason():
    adapters = {
        "raw": AdapterStatus(available=True),
        "cloud": AdapterStatus(available=True),
        "hermes": AdapterStatus(available=False, reason="hermes CLI not in PATH"),
    }
    app = _Host(_endpoint(), adapters)
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        btn = app.screen.query_one("#adapter-hermes", RadioButton)
        assert btn.disabled is True
        assert "hermes CLI not in PATH" in btn.label.plain
        await pilot.press("escape")

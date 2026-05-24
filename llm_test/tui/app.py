from __future__ import annotations

import asyncio
import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from llm_test.core import adapter_probe, runner_subprocess
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.core.store import Store
from llm_test.tui.history_tab import HistoryTab
from llm_test.tui.home_tab import HomeTab
from llm_test.tui.launch_modal import LaunchModal
from llm_test.tui.live_tab import LiveTab
from llm_test.tui.rankings_tab import RankingsTab
from llm_test.tui.scenarios_tab import ScenariosTab


class LLMTestApp(App):
    CSS = """
    Screen { background: $surface; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+r", "refresh", "Refresh")]

    def __init__(self, run_id: str | None = None, store: Store | None = None) -> None:
        super().__init__()
        self.run_id = run_id
        self._store = store
        self.run_subprocess: asyncio.subprocess.Process | None = None
        self._adapters_cache = adapter_probe.available_adapters()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="home"):
            with TabPane("Home", id="home"):
                yield HomeTab(
                    id="home-tab",
                    known_models_provider=self._known_models,
                )
            with TabPane("Live", id="live"):
                yield LiveTab(id="live-tab")
            with TabPane("History", id="history"):
                yield HistoryTab(id="history-tab")
            with TabPane("Rankings", id="rankings"):
                yield RankingsTab(id="rankings-tab")
            with TabPane("Scenarios", id="scenarios"):
                yield ScenariosTab(id="scenarios-tab")
        yield Footer()

    def action_refresh(self) -> None:
        self.notify("Refreshing...", timeout=1)

    async def open_launch_modal(self, endpoint: EndpointInfo) -> None:
        if self.run_subprocess is not None and self.run_subprocess.returncode is None:
            self.notify(
                "A run is already in progress on Live. Wait for it or abort externally.",
                severity="warning",
            )
            return
        self._launch_modal_worker(endpoint)

    @work
    async def _launch_modal_worker(self, endpoint: EndpointInfo) -> None:
        args = await self.push_screen_wait(LaunchModal(endpoint, self._adapters_cache))
        if args is None:
            return
        try:
            self.run_subprocess = await runner_subprocess.spawn_run(args)
        except (FileNotFoundError, PermissionError) as e:
            self.notify(f"Failed to launch: {e}", severity="error")
            return
        self.notify(f"Run started: {args.model} / {args.adapter}")
        self.query_one(TabbedContent).active = "live"

    def _known_models(self) -> set[str]:
        if self._store is None:
            try:
                results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
                self._store = Store(results_dir / "runs.db")
                self._store.init_schema()
            except Exception:
                return set()
        try:
            return {row["model"] for row in self._store.fetch_all_runs()}
        except Exception:
            return set()

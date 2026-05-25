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
from llm_test.tui.compare_tab import CompareTab
from llm_test.tui.history_tab import HistoryTab
from llm_test.tui.home_tab import HomeTab
from llm_test.tui.launch_modal import LaunchModal
from llm_test.tui.rankings_tab import RankingsTab
from llm_test.tui.scenarios_tab import ScenariosTab
from llm_test.tui.setup_tab import SetupTab


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
            with TabPane("Rankings", id="rankings"):
                yield RankingsTab(id="rankings-tab")
            with TabPane("Compare", id="compare"):
                yield CompareTab(id="compare-tab")
            with TabPane("Scenarios", id="scenarios"):
                yield ScenariosTab(id="scenarios-tab")
            with TabPane("History", id="history"):
                yield HistoryTab(id="history-tab")
            with TabPane("Setup", id="setup"):
                yield SetupTab(id="setup-tab")
        yield Footer()

    def action_refresh(self) -> None:
        # Force-refresh every tab that pulls from the DB. Cheap; user-triggered.
        refreshed = 0
        for tab in self.query(
            "ScenariosTab, HistoryTab, RankingsTab, HomeTab, CompareTab"
        ):
            for method in ("reload", "refresh_data", "refresh_from_db"):
                fn = getattr(tab, method, None)
                if callable(fn):
                    try:
                        fn()
                        refreshed += 1
                    except Exception as e:
                        self.notify(
                            f"refresh failed on {type(tab).__name__}: {e}",
                            severity="warning", markup=False)
                    break
        self.notify(f"Refreshed {refreshed} tab(s)", timeout=1)

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
            self.notify(f"Failed to launch: {e}", severity="error",
                        markup=False)
            return
        self.notify(f"Run started: {args.model} / {args.adapter}")
        self.query_one(TabbedContent).active = "home"
        self._watch_subprocess_worker(args)

    @work
    async def _watch_subprocess_worker(self, args: "runner_subprocess.RunArgs") -> None:
        """Watch the spawned run subprocess; notify on exit, mark DB on failure.

        Without this, a silently-crashed subprocess leaves runs.db status='running'
        forever and the TUI shows stale progress with no indication of death.
        """
        proc = self.run_subprocess
        if proc is None:
            return
        rc = await proc.wait()
        if rc == 0:
            self.notify(f"Run finished cleanly: {args.model}", timeout=5)
            return
        self.notify(
            f"Run died (exit {rc}). Check terminal scrollback. "
            f"Marking running rows for {args.model} as failed.",
            severity="error", timeout=15, markup=False,
        )
        try:
            from datetime import UTC, datetime
            results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
            store = Store(results_dir / "runs.db")
            store.init_schema()
            with store.conn() as c:
                c.execute(
                    "UPDATE runs SET status='failed', finished_at=? "
                    "WHERE model=? AND status='running'",
                    (datetime.now(UTC).isoformat(), args.model),
                )
        except Exception as e:
            self.notify(f"DB cleanup failed: {e}", severity="warning", markup=False)

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

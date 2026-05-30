from __future__ import annotations

import asyncio
import os
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from toolery.core import adapter_probe, runner_subprocess
from toolery.core.endpoint_scanner import EndpointInfo
from toolery.core.store import Store
from toolery.tui.compare_tab import CompareTab
from toolery.tui.history_tab import HistoryTab
from toolery.tui.home_tab import HomeTab
from toolery.tui.launch_modal import LaunchModal
from toolery.tui.profiles_tab import ProfilesTab
from toolery.tui.rankings_tab import RankingsTab
from toolery.tui.scenarios_tab import ScenariosTab


class TooleryApp(App):
    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    Footer {
        background: $surface;
        color: $text-muted;
    }

    TabbedContent {
        background: $surface;
    }

    TabPane {
        padding: 0;
    }

    Button {
        min-width: 12;
    }

    DataTable {
        background: $surface;
        color: $text;
    }

    DataTable > .datatable--header {
        text-style: bold;
        background: $primary;
        color: $text;
    }
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
            with TabPane("Profiles", id="setup"):
                yield ProfilesTab(id="setup-tab")
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
        interrupted = self._find_interrupted_run(endpoint)
        args = await self.push_screen_wait(
            LaunchModal(endpoint, self._adapters_cache, interrupted_run=interrupted))
        if args is None:
            return
        try:
            self.run_subprocess = await runner_subprocess.spawn_run(args)
        except (FileNotFoundError, PermissionError) as e:
            self.notify(f"Failed to launch: {e}", severity="error",
                        markup=False)
            return
        if args.resume:
            self.notify(f"Resuming run: {args.resume}")
        else:
            self.notify(f"Run started: {args.model} / {args.adapter}")
        self.query_one(TabbedContent).active = "home"
        self._watch_subprocess_worker(args)

    def _find_interrupted_run(self, endpoint: EndpointInfo) -> dict | None:
        """Return the most recent runs row for this endpoint that did not finish.

        Match by base_url so we only suggest a resume when the user has clearly
        re-selected the same endpoint that hosted the dead run. status='done'
        runs are excluded; status='running' is included (stale state from a
        crash leaves it that way).
        """
        store = self._resolve_store()
        if store is None:
            return None
        try:
            runs = store.fetch_all_runs()
        except Exception:
            return None
        for r in runs:
            if r.get("base_url") != endpoint.base_url:
                continue
            if r.get("status") == "done":
                continue
            done = store.count_results_for_run(r["run_id"])
            total = r.get("total_units") or 0
            # If everything's been processed AND phase==done was never reached,
            # we still allow resume (it'll re-run only perf if --with-perf).
            r = dict(r)
            r["_done_count"] = done
            r["_total"] = total
            return r
        return None

    def _resolve_store(self) -> Store | None:
        if self._store is not None:
            return self._store
        try:
            results_dir = Path(
                os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
            self._store = Store(results_dir / "runs.db")
            self._store.init_schema()
            return self._store
        except Exception:
            return None

    async def spawn_resume(self, run_id: str) -> None:
        """Spawn `toolery run --resume <run_id>`. Used from History tab."""
        if self.run_subprocess is not None and self.run_subprocess.returncode is None:
            self.notify(
                "A run is already in progress. Wait for it or abort externally.",
                severity="warning")
            return
        # Build a RunArgs that satisfies validation; CLI re-hydrates from DB.
        args = runner_subprocess.RunArgs(
            model="resume", base_url="http://placeholder",
            adapter="raw", tier="all", trials=1, concurrency=1,
            with_perf=False, cluster="single", resume=run_id,
        )
        try:
            self.run_subprocess = await runner_subprocess.spawn_run(args)
        except (FileNotFoundError, PermissionError) as e:
            self.notify(f"Failed to resume: {e}", severity="error",
                        markup=False)
            return
        self.notify(f"Resuming run: {run_id}")
        self.query_one(TabbedContent).active = "home"
        self._watch_subprocess_worker(args)

    # ------------------------------------------------------------ run control

    def _latest_running_run_id(self) -> str | None:
        store = self._resolve_store_safe()
        if store is None:
            return None
        try:
            with store.conn() as c:
                row = c.execute(
                    "SELECT run_id FROM runs WHERE status='running' "
                    "ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
            return row["run_id"] if row else None
        except Exception:
            return None

    def _latest_paused_run_id(self) -> str | None:
        store = self._resolve_store_safe()
        if store is None:
            return None
        try:
            with store.conn() as c:
                row = c.execute(
                    "SELECT run_id FROM runs WHERE status='paused' "
                    "ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
            return row["run_id"] if row else None
        except Exception:
            return None

    async def _terminate_subprocess(self, *, kill_after: float = 5.0) -> None:
        proc = self.run_subprocess
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=kill_after)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass

    async def pause_current_run(self) -> None:
        """Mark the run paused WITHOUT killing the subprocess. The running
        subprocess polls runs.status after every completed result; once it sees
        'paused' it stops scheduling new scenarios but lets units already in
        flight finish and record their results. We therefore do NOT terminate
        the process and do NOT clear in_flight rows — those scenarios are still
        executing and will clear themselves (via on_end) as they complete.
        Resume picks up from the next not-yet-recorded unit."""
        run_id = self._latest_running_run_id()
        if run_id is None:
            return
        from datetime import UTC, datetime
        store = self._resolve_store_safe()
        if store is None:
            return
        try:
            with store.conn() as c:
                c.execute(
                    "UPDATE runs SET status='paused', updated_at=? "
                    "WHERE run_id=?",
                    (datetime.now(UTC).isoformat().replace("+00:00", "Z"), run_id),
                )
        except Exception as e:
            self.notify(f"Pause failed: {e}",
                        severity="warning", markup=False)

    async def resume_current_run(self) -> None:
        """Re-spawn `toolery run --resume <run_id>` on the latest paused run.
        Reuses the existing spawn_resume code path."""
        run_id = self._latest_paused_run_id()
        if run_id is None:
            self.notify("No paused run to resume.", severity="warning")
            return
        # Flip status back to running so the resume subprocess sees a live row.
        store = self._resolve_store_safe()
        if store is not None:
            try:
                with store.conn() as c:
                    c.execute(
                        "UPDATE runs SET status='running' WHERE run_id=?",
                        (run_id,),
                    )
            except Exception:
                pass
        await self.spawn_resume(run_id)

    async def stop_current_run(self) -> None:
        """Terminate the subprocess and mark every not-yet-recorded scenario
        with status=error so the run is permanently closed. The run is then
        marked failed; resume is no longer offered for it."""
        run_id = self._latest_running_run_id()
        await self._terminate_subprocess()
        if run_id is None:
            return
        from datetime import UTC, datetime
        store = self._resolve_store_safe()
        if store is None:
            return
        try:
            store.clear_all_in_flight(run_id)
            with store.conn() as c:
                c.execute(
                    "UPDATE runs SET status='failed', finished_at=? "
                    "WHERE run_id=?",
                    (datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                     run_id),
                )
        except Exception as e:
            self.notify(f"Stop cleanup failed: {e}",
                        severity="warning", markup=False)

    def _resolve_store_safe(self) -> Store | None:
        if self._store is None:
            try:
                results_dir = Path(
                    os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
                self._store = Store(results_dir / "runs.db")
                self._store.init_schema()
            except Exception:
                return None
        return self._store

    @work
    async def _watch_subprocess_worker(self, args: runner_subprocess.RunArgs) -> None:
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
            results_dir = Path(os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
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
                results_dir = Path(os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
                self._store = Store(results_dir / "runs.db")
                self._store.init_schema()
            except Exception:
                return set()
        try:
            return {row["model"] for row in self._store.fetch_all_runs()}
        except Exception:
            return set()

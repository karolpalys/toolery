from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, ProgressBar, Static

from llm_test.core import endpoint_scanner
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.core.store import Store

DEFAULT_PORTS = sorted({5000, 5001, 11434, *range(8000, 9001)})

ScannerCallable = Callable[[list[int]], Awaitable[list[EndpointInfo]]]
KnownProvider = Callable[[], set[str]]

_STATUS_STYLE = {
    "pass": "bold green",
    "partial": "bold yellow",
    "fail": "bold red",
    "error": "bold red",
    "timeout": "bold red",
}
_STATUS_ICON = {
    "pass": "✓",
    "partial": "~",
    "fail": "✗",
    "error": "⚠",
    "timeout": "⏱",
}

_DIM_LABEL = {
    "hallucination": "calibration / hallucination resistance",
    "coding": "coding tasks",
    "agentic": "agentic / multi-step workflows",
    "safety": "safety-critical flows",
    "restraint": "knowing when NOT to act",
    "error_recovery": "recovering from tool errors",
    "parameter_precision": "precise argument extraction",
    "context_state_tracking": "tracking state across turns",
    "structured_output": "structured / JSON output",
    "tool_selection": "picking the right tool",
    "long_context": "long-context retrieval",
    "localization": "non-English / localization",
    "budget_efficiency": "staying within tool-call budget",
}


def _why_summary(status: str, failure_kind: str | None,
                 checks_json: str | None) -> str:
    if status == "pass":
        return "—"
    try:
        checks = json.loads(checks_json or "[]")
    except json.JSONDecodeError:
        checks = []
    bad = [c for c in checks if c.get("result") in ("fail", "partial")]
    head = failure_kind or status
    if bad:
        detail = (bad[0].get("detail") or bad[0].get("check") or "")[:80]
        return f"{head}: {detail}" if detail else head
    return head


def _detail_block(scenario_id: str, adapter: str, trial: int, status: str,
                  failure_kind: str | None, latency_ms: int | None,
                  call_count: int | None, budget_max: int | None,
                  trace_path: str | None, checks_json: str | None) -> Text:
    """Renders the right-hand panel content for the selected scenario row."""
    out = Text()
    out.append(f"{scenario_id}\n", style="bold")
    out.append(f"adapter: {adapter}  trial: {trial}\n", style="dim")
    style = _STATUS_STYLE.get(status, "bold")
    out.append(f"status: {status}", style=style)
    if failure_kind:
        out.append(f"  ({failure_kind})", style="red")
    out.append("\n")
    if call_count is not None and budget_max:
        out.append(f"calls: {call_count}/{budget_max}\n", style="dim")
    if latency_ms is not None:
        out.append(f"latency: {latency_ms} ms\n", style="dim")
    out.append("\nchecks:\n", style="bold")
    try:
        checks = json.loads(checks_json or "[]")
    except json.JSONDecodeError:
        checks = []
    if not checks:
        out.append("  (no checks recorded)\n", style="dim")
    for c in checks:
        result = c.get("result", "?")
        icon = _STATUS_ICON.get(result, "·")
        name = c.get("check", "?")
        detail = c.get("detail", "")
        out.append(f"  {icon} ", style=_STATUS_STYLE.get(result, "bold"))
        out.append(f"{name}", style="bold")
        if detail:
            out.append(f": {detail}", style="dim")
        out.append("\n")
    if trace_path:
        out.append(f"\ntrace: {trace_path}\n", style="dim")
    return out


def _profile_run(results: list[dict]) -> Text:
    """Deterministic per-run profile: overall, per-tier, per-dimension top/bottom,
    derived recommended/avoid use-cases."""
    if not results:
        return Text("(no scenario results)", style="dim")

    by_dim: dict[str, list[float]] = defaultdict(list)
    by_tier: dict[str, list[float]] = defaultdict(list)
    statuses: dict[str, int] = defaultdict(int)
    for r in results:
        try:
            dims = json.loads(r.get("ranking_dims_json") or '["overall"]')
        except json.JSONDecodeError:
            dims = ["overall"]
        score = r.get("score") or 0.0
        for d in dims:
            by_dim[d].append(score)
        by_tier[r.get("tier", "?")].append(score)
        statuses[r.get("status", "?")] += 1

    dim_means = {d: sum(v) / len(v) for d, v in by_dim.items() if v}
    overall = dim_means.get("overall", 0.0)
    ranked = sorted(((d, m) for d, m in dim_means.items() if d != "overall"),
                    key=lambda x: -x[1])
    strong = [(d, m) for d, m in ranked[:3] if m >= 0.5]
    weak = [(d, m) for d, m in ranked[-3:] if m < 0.7]

    out = Text()
    out.append("Run summary\n", style="bold")
    overall_style = ("bold green" if overall >= 0.7 else
                     "bold yellow" if overall >= 0.5 else "bold red")
    out.append(f"  overall: {overall * 100:.1f}%  ", style=overall_style)
    total = sum(statuses.values())
    out.append(
        f"({statuses.get('pass', 0)}/{total} pass, "
        f"{statuses.get('partial', 0)} partial, "
        f"{statuses.get('fail', 0) + statuses.get('error', 0)} fail)\n",
        style="dim",
    )
    if by_tier:
        out.append("  per tier: ", style="bold")
        parts = []
        for tier in ("easy", "medium", "hard", "very_hard"):
            vs = by_tier.get(tier)
            if not vs:
                continue
            parts.append(f"{tier}={sum(vs) / len(vs) * 100:.0f}%")
        out.append("  ".join(parts) + "\n", style="dim")

    if strong:
        out.append("Strong points:\n", style="bold green")
        for d, m in strong:
            out.append(f"  • {_DIM_LABEL.get(d, d)} — {m * 100:.0f}%\n",
                       style="green")
    if weak:
        out.append("Weak points:\n", style="bold red")
        for d, m in weak:
            out.append(f"  • {_DIM_LABEL.get(d, d)} — {m * 100:.0f}%\n",
                       style="red")

    recommended = [d for d, m in strong if m >= 0.7]
    avoid = [d for d, m in weak if m < 0.4]
    if recommended:
        out.append("Recommended for: ", style="bold")
        out.append(
            ", ".join(_DIM_LABEL.get(d, d) for d in recommended) + "\n",
            style="green",
        )
    if avoid:
        out.append("Avoid for: ", style="bold")
        out.append(
            ", ".join(_DIM_LABEL.get(d, d) for d in avoid) + "\n",
            style="red",
        )
    if not recommended and not avoid:
        out.append(
            "Mixed profile — no dimension is decisively strong or weak.\n",
            style="dim",
        )
    return out


class HomeTab(Container):
    """Combined endpoint scanner + live run dashboard + post-run model profile."""

    DEFAULT_CSS = """
    HomeTab { layout: vertical; padding: 0 1; }
    HomeTab #scanner-strip {
        height: auto; max-height: 13;
        border: round $primary;
        border-title-color: $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    HomeTab #scanner-strip #buttons { height: 3; }
    HomeTab #scanner-strip #scan-status { height: 1; color: $text-muted; }
    HomeTab #scanner-strip Button { margin-right: 2; }
    HomeTab #scanner-strip #endpoints { height: 6; }
    HomeTab #live-strip {
        height: auto;
        border: round $primary;
        border-title-color: $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    HomeTab #live-strip.active {
        border: round $success;
        border-title-color: $success;
    }
    HomeTab #live-strip ProgressBar { margin-top: 1; }
    HomeTab #main-split-wrapper {
        border: round $primary;
        border-title-color: $primary;
        padding: 0 1;
        height: 1fr;
    }
    HomeTab #main-split { height: 1fr; }
    HomeTab #scenarios-pane { width: 2fr; padding-right: 1; }
    HomeTab #scenarios-pane DataTable { height: 1fr; }
    HomeTab #detail-pane { width: 1fr; border: solid $primary; padding: 0 1; }
    HomeTab #detail-content { height: 1fr; }
    HomeTab #summary-strip {
        height: auto; max-height: 14;
        border: round $success;
        border-title-color: $success;
        padding: 0 1;
        margin-top: 1;
    }
    HomeTab #summary-strip.hidden { display: none; }
    """

    def __init__(
        self,
        scanner: ScannerCallable | None = None,
        known_models_provider: KnownProvider | None = None,
        store: Store | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._scanner = scanner or endpoint_scanner.scan
        self._known_provider = known_models_provider or (lambda: set())
        self._store = store
        self._scanning = False
        self._endpoints: list[EndpointInfo] = []
        self._current_run_id: str | None = None
        self._results_cache: list[dict] = []
        # Track what's already rendered in the scenarios DataTable so we can
        # append deltas instead of clear+repopulate (which resets cursor +
        # scroll every 2s and makes long lists unclickable).
        self._displayed_run_id: str | None = None
        self._displayed_count: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="scanner-strip"):
            with Horizontal(id="buttons"):
                yield Button("Scan endpoints", id="scan", variant="primary")
                yield Static("Click Scan to discover endpoints",
                             id="scan-status")
            yield DataTable(id="endpoints", cursor_type="row")

        with Vertical(id="live-strip"):
            yield Static(
                "[dim italic]No active run — pick an endpoint above to "
                "launch one.[/dim italic]",
                id="current-run",
            )
            yield Static("", id="current-progress-text")
            yield ProgressBar(total=100, show_eta=False, id="current-progress")
            yield Static("", id="current-phase")

        with Container(id="main-split-wrapper"):
            with Horizontal(id="main-split"):
                with Vertical(id="scenarios-pane"):
                    yield Static("[bold]Scenarios[/bold]")
                    yield DataTable(id="scenarios-table", cursor_type="row")
                with VerticalScroll(id="detail-pane"):
                    yield Static("[bold]Details[/bold]")
                    yield Static(
                        "Click a scenario row to see failed checks and trace path.",
                        id="detail-content",
                    )

        with VerticalScroll(id="summary-strip", classes="hidden"):
            yield Static("", id="summary-content")

    def on_mount(self) -> None:
        ep = self.query_one("#endpoints", DataTable)
        ep.add_columns("Port", "Model ID", "Status", "Server")
        sc = self.query_one("#scenarios-table", DataTable)
        sc.add_columns("#", "Scenario", "Tier", "Adapter", "Trial", "Score",
                       "Status", "Why")
        # Section border titles.
        try:
            self.query_one("#scanner-strip").border_title = (
                "🔍 Endpoint Discovery"
            )
            self.query_one("#live-strip").border_title = "💤 Active Run"
            self.query_one("#main-split-wrapper").border_title = (
                "📋 Per-scenario results"
            )
            self.query_one("#summary-strip").border_title = (
                "✓ Last run summary"
            )
        except Exception:
            pass
        self.refresh_from_db()
        self.set_interval(2.0, self.refresh_from_db)

    # ------------------------------------------------------------------ scanner

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._scanning:
            return
        if event.button.id == "scan":
            await self._run_scan(DEFAULT_PORTS)

    async def _run_scan(self, ports: list[int]) -> None:
        self._scanning = True
        self._set_scan_status(f"Scanning {len(ports)} ports...")
        self._set_buttons_disabled(True)
        started = time.monotonic()
        try:
            self._endpoints = list(await self._scanner(ports))
        finally:
            self._scanning = False
            self._set_buttons_disabled(False)
        elapsed = time.monotonic() - started
        self._refresh_endpoints_table()
        if not self._endpoints:
            self._set_scan_status(
                f"Scanned {len(ports)} ports in {elapsed:.1f}s — "
                "no endpoints. Start vLLM/llama.cpp first."
            )
        else:
            self._set_scan_status(
                f"Last scan: {elapsed:.1f}s — {len(self._endpoints)} endpoint(s). "
                "Pick a row to launch."
            )

    def _refresh_endpoints_table(self) -> None:
        tbl = self.query_one("#endpoints", DataTable)
        tbl.clear()
        known = self._known_provider()
        for ep in self._endpoints:
            badge = "✅ Known" if ep.model_id in known else "❓ New *"
            tbl.add_row(
                str(ep.port),
                ep.model_id,
                badge,
                ep.server_hint,
            )

    # ----------------------------------------------------------- row selection

    @on(DataTable.RowSelected, "#endpoints")
    async def _endpoint_row_selected(
        self, event: DataTable.RowSelected
    ) -> None:
        await self._on_endpoint_selected(event.cursor_row)

    @on(DataTable.RowSelected, "#scenarios-table")
    def _scenario_row_selected(
        self, event: DataTable.RowSelected
    ) -> None:
        self._on_scenario_selected(event.cursor_row)

    async def _on_endpoint_selected(self, idx: int | None) -> None:
        if idx is None or idx >= len(self._endpoints):
            return
        endpoint = self._endpoints[idx]
        opener = getattr(self.app, "open_launch_modal", None)
        if opener is None:
            self._set_scan_status(
                "App does not provide open_launch_modal; cannot launch."
            )
            return
        await opener(endpoint)

    def _on_scenario_selected(self, idx: int | None) -> None:
        if idx is None or idx >= len(self._results_cache):
            return
        r = self._results_cache[idx]
        block = _detail_block(
            scenario_id=r.get("scenario_id", "?"),
            adapter=r.get("adapter", "?"),
            trial=int(r.get("trial_index") or 0),
            status=r.get("status") or "?",
            failure_kind=r.get("failure_kind"),
            latency_ms=r.get("latency_ms"),
            call_count=r.get("call_count"),
            budget_max=r.get("budget_max"),
            trace_path=r.get("trace_path"),
            checks_json=r.get("checks_json"),
        )
        self.query_one("#detail-content", Static).update(block)

    # ---------------------------------------------------------- live + summary

    def _resolve_store(self) -> Store | None:
        if self._store is not None:
            return self._store
        try:
            results_dir = Path(
                os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
            db = results_dir / "runs.db"
            if not db.exists():
                return None
            store = Store(db)
            store.init_schema()
            self._store = store
            return store
        except Exception:
            return None

    def refresh_from_db(self) -> None:
        store = self._resolve_store()
        if store is None:
            return
        runs = store.fetch_all_runs()
        current = next((r for r in runs if r.get("status") == "running"), None)
        if current is None and runs:
            current = runs[0]

        status_w = self.query_one("#current-run", Static)
        progress_text = self.query_one("#current-progress-text", Static)
        progress = self.query_one("#current-progress", ProgressBar)
        phase_w = self.query_one("#current-phase", Static)
        summary_w = self.query_one("#summary-strip", VerticalScroll)
        summary_content = self.query_one("#summary-content", Static)

        live_strip = self.query_one("#live-strip")

        if current is None:
            status_w.update(
                "[dim italic]No active run — pick an endpoint above to "
                "launch one.[/dim italic]"
            )
            progress_text.update("")
            progress.update(total=100, progress=0)
            phase_w.update("")
            summary_w.add_class("hidden")
            live_strip.remove_class("active")
            try:
                live_strip.border_title = "💤 Active Run"
            except Exception:
                pass
            self._current_run_id = None
            self._results_cache = []
            self._refresh_scenarios_table()
            return

        run_id = current["run_id"]
        is_active = current.get("status") == "running"
        if is_active:
            live_strip.add_class("active")
            try:
                live_strip.border_title = "⚡ Active Run"
            except Exception:
                pass
            status_w.update(
                f"⚡ {run_id}  ·  model: [bold]{current['model']}[/bold]  "
                f"·  status: {current.get('status', '?')}"
            )
        else:
            live_strip.remove_class("active")
            try:
                live_strip.border_title = "💤 Active Run"
            except Exception:
                pass
            status_w.update(
                f"{run_id}  ·  model: [bold]{current['model']}[/bold]  "
                f"·  status: {current.get('status', '?')}"
            )
        total = current.get("total_units") or 0
        done = store.count_results_for_run(run_id)
        # ETA from median per-unit latency × remaining.
        eta_str = ""
        cache = store.fetch_results_for_run(run_id)
        if len(cache) >= 3 and total > 0 and done < total:
            try:
                lats = [int(x.get("latency_ms") or 0) for x in cache
                        if x.get("latency_ms")]
                if lats:
                    med = statistics.median(lats)
                    remaining = total - done
                    secs = int((med * remaining) / 1000)
                    eta_str = f"  ·  ETA {secs // 60}m {secs % 60}s"
            except Exception:
                eta_str = ""
        if total > 0:
            pct = 100.0 * done / total
            progress_text.update(
                f"{done}/{total} units  ({pct:.1f}%){eta_str}"
            )
            progress.update(total=total, progress=done)
        else:
            progress_text.update(f"{done} units done{eta_str}")
            progress.update(total=100, progress=0)
        phase = current.get("phase") or "—"
        last = current.get("current_scenario") or "—"
        if phase == "perf":
            phase_w.update("[yellow]phase: running llama-bench[/yellow]")
        elif phase == "done":
            phase_w.update("[green]phase: done[/green]")
        else:
            phase_w.update(f"phase: {phase}  ·  last: {last}")

        self._current_run_id = run_id
        self._results_cache = store.fetch_results_for_run(run_id)
        self._refresh_scenarios_table()

        if current.get("status") in ("done", "aborted", "failed"):
            summary_content.update(_profile_run(self._results_cache))
            summary_w.remove_class("hidden")
        else:
            summary_w.add_class("hidden")

    def _refresh_scenarios_table(self) -> None:
        tbl = self.query_one("#scenarios-table", DataTable)
        # Full reset only when run_id changes (new bench started) — preserves
        # cursor + scroll within a single run so the user can actually click
        # rows during a 315-trial sweep.
        if self._displayed_run_id != self._current_run_id:
            tbl.clear()
            self._displayed_run_id = self._current_run_id
            self._displayed_count = 0
        # Append-only: render rows beyond what we've already shown.
        new_rows = self._results_cache[self._displayed_count:]
        _STATUS_DISPLAY = {
            "pass": "✅ pass",
            "fail": "❌ fail",
            "partial": "⚠ partial",
            "error": "💥 error",
            "timeout": "⏱ timeout",
        }
        for i, r in enumerate(new_rows, start=self._displayed_count + 1):
            status = r.get("status") or "?"
            style = _STATUS_STYLE.get(status, "bold")
            display = _STATUS_DISPLAY.get(status, status)
            why = _why_summary(status, r.get("failure_kind"),
                               r.get("checks_json"))
            tbl.add_row(
                Text(str(i), style=style),
                Text(r.get("scenario_id", "?"), style=style),
                Text(r.get("tier", "?"), style=style),
                Text(r.get("adapter", "?"), style=style),
                Text(str(r.get("trial_index") or 0), style=style),
                Text(f"{r.get('score') or 0.0:.2f}", style=style),
                Text(display, style=style),
                Text(why, style=style),
            )
        self._displayed_count = len(self._results_cache)

    # ---------------------------------------------------------------- helpers

    def _set_scan_status(self, text: str) -> None:
        self.query_one("#scan-status", Static).update(text)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#scan", Button).disabled = disabled

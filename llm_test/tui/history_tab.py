from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Markdown, Static

from llm_test.compare import compare_runs
from llm_test.core.store import Store


class ConfirmRemoveModal(ModalScreen[bool]):
    """Two-button confirm dialog. Returns True on confirm, False on cancel."""

    DEFAULT_CSS = """
    ConfirmRemoveModal { align: center middle; }
    ConfirmRemoveModal > Vertical {
        width: 70; padding: 1 2;
        background: $surface; border: thick $error;
    }
    ConfirmRemoveModal .row { height: auto; margin-top: 1; }
    ConfirmRemoveModal Button { margin-right: 2; }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("[bold red]Remove run?[/bold red]")
            yield Label(self.run_id)
            yield Label(
                "[dim]Deletes DB rows (scenario_results, perf_results, "
                "adapters_in_run) and the trace directory. Irreversible.[/dim]"
            )
            with Horizontal(classes="row"):
                yield Button("Cancel (n)", id="cancel")
                yield Button("Remove (y)", id="confirm", variant="error")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class MarkdownModal(ModalScreen[None]):
    """Big scrollable modal that renders Markdown — used for both run details
    and compare diffs."""

    DEFAULT_CSS = """
    MarkdownModal { align: center middle; }
    MarkdownModal > Vertical {
        width: 90%; height: 90%; padding: 1 2;
        background: $surface; border: thick $primary;
    }
    MarkdownModal #md-scroll { height: 1fr; }
    MarkdownModal .row { height: auto; margin-top: 1; }
    """

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, title: str, body_md: str) -> None:
        super().__init__()
        self._title = title
        self._body = body_md

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"[bold]{self._title}[/bold]")
            with VerticalScroll(id="md-scroll"):
                yield Markdown(self._body)
            with Horizontal(classes="row"):
                yield Button("Close (esc)", id="close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


def _build_details_md(run: dict, results: list[dict],
                      perf_rows: list[dict], adapters: list[str]) -> str:
    """Render a Markdown summary of one run for the details modal."""
    by_tier: dict[str, list[float]] = {}
    by_status: Counter[str] = Counter()
    fail_kinds: Counter[str] = Counter()
    for r in results:
        by_tier.setdefault(r.get("tier") or "?", []).append(r["score"])
        by_status[r.get("status") or "?"] += 1
        if r.get("failure_kind"):
            fail_kinds[r["failure_kind"]] += 1

    try:
        cfg = json.dumps(json.loads(run.get("config_json") or "{}"), indent=2)
    except json.JSONDecodeError:
        cfg = run.get("config_json") or ""

    lines: list[str] = []
    lines.append(f"## {run['run_id']}")
    lines.append("")
    lines.append(f"- **Model:** `{run.get('model','?')}`")
    lines.append(f"- **Status:** {run.get('status') or '?'}")
    lines.append(f"- **Started:** {run.get('started_at') or '?'}")
    lines.append(f"- **Finished:** {run.get('finished_at') or '—'}")
    lines.append(f"- **Duration:** {run.get('duration_s') or 0:.1f}s")
    lines.append(f"- **Cluster:** {run.get('cluster') or '—'}")
    lines.append(f"- **Adapters:** {', '.join(adapters) or '—'}")
    lines.append(f"- **Base URL:** {run.get('base_url') or '—'}")
    lines.append("")
    if results:
        total = len(results)
        avg = sum(r["score"] for r in results) / total
        lines.append("### Scenario stats")
        lines.append("")
        lines.append(f"- **Trials recorded:** {total}")
        lines.append(f"- **Mean score:** {avg * 100:.1f}%")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|------:|")
        for s, n in by_status.most_common():
            lines.append(f"| {s} | {n} |")
        lines.append("")
        if by_tier:
            lines.append("| Tier | Mean | N |")
            lines.append("|------|-----:|--:|")
            for tier in ("easy", "medium", "hard", "very_hard"):
                vals = by_tier.get(tier)
                if not vals:
                    continue
                lines.append(f"| {tier} | "
                             f"{sum(vals) / len(vals) * 100:.1f}% | {len(vals)} |")
            lines.append("")
        if fail_kinds:
            lines.append("### Top failure kinds")
            lines.append("")
            for kind, n in fail_kinds.most_common(8):
                lines.append(f"- `{kind}` — {n}")
            lines.append("")
    else:
        lines.append("_No scenario results recorded for this run._")
        lines.append("")
    if perf_rows:
        lines.append("### Perf (llama-benchy)")
        lines.append("")
        lines.append("| Depth | PP t/s | Gen t/s | TTFT ms |")
        lines.append("|------:|-------:|--------:|--------:|")
        for p in sorted(perf_rows, key=lambda x: x.get("depth") or 0):
            lines.append(
                f"| {p.get('depth',0)} | "
                f"{(p.get('pp_tps') or 0):.1f} | "
                f"{(p.get('tg_tps') or 0):.2f} | "
                f"{(p.get('ttft_ms') or 0):.0f} |"
            )
        lines.append("")
    lines.append("### Config")
    lines.append("")
    lines.append("```json")
    lines.append(cfg)
    lines.append("```")
    return "\n".join(lines)


class HistoryTab(Container):
    DEFAULT_CSS = """
    HistoryTab { padding: 0 1; }
    HistoryTab #history-intro {
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    HistoryTab #runs-section {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    HistoryTab DataTable { height: 1fr; }
    HistoryTab #hist-banner.has-anchor { color: $warning; text-style: bold; }
    """

    BINDINGS = [
        Binding("delete", "remove_run", "Remove selected run"),
        Binding("backspace", "remove_run", "Remove selected run", show=False),
        Binding("d", "diff_run", "Diff: 1st press = mark, 2nd press = compare"),
    ]

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        # Diff anchor: first run picked with `d`. Second `d` on a different
        # row triggers the diff. `d` on the same row clears the anchor.
        self._diff_anchor: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Past runs: click a row to view summary, compare, or remove. "
            "Removal deletes traces and DB rows permanently.",
            id="history-intro",
        )
        with Vertical(id="runs-section"):
            yield Static(
                "[bold]All runs[/bold]  —  ↑↓ navigate  ·  "
                "Enter = details  ·  d = diff (press on 2 rows)  ·  "
                "Del = remove",
                id="hist-banner",
            )
            yield Input(placeholder="filter (model name or run_id substring)",
                        id="hist-filter")
            yield DataTable(id="history-table", cursor_type="row")

    def on_mount(self) -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.add_columns("#", "run_id", "model", "status", "started_at",
                        "duration (s)", "adapters")
        try:
            self.query_one("#runs-section").border_title = "📜 Past runs"
        except Exception:
            pass
        self.refresh_data()

    def refresh_data(self, filter_str: str = "") -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.clear()
        store = self._store()
        _STATUS_EMOJI = {
            "done": "✓ done",
            "failed": "💥 failed",
            "running": "⏳ running",
            "aborted": "⚠ aborted",
        }
        i = 0
        for r in store.fetch_all_runs():
            if filter_str and filter_str.lower() not in (r["run_id"] + r["model"]).lower():
                continue
            with store.conn() as c:
                adapters = [row["adapter"] for row in
                            c.execute("SELECT adapter FROM adapters_in_run WHERE run_id=?",
                                      (r["run_id"],)).fetchall()]
            i += 1
            raw_status = r["status"] or ""
            status_cell = _STATUS_EMOJI.get(raw_status, raw_status)
            tbl.add_row(str(i), r["run_id"], r["model"], status_cell,
                        r["started_at"] or "", f"{r['duration_s'] or 0:.1f}",
                        ",".join(adapters))

    def on_input_changed(self, event) -> None:
        if event.input.id == "hist-filter":
            self.refresh_data(event.value)

    # --------------------------------------------------------- enter → details

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected
    ) -> None:
        if event.data_table.id != "history-table":
            return
        run_id = self._row_run_id(event.cursor_row)
        if not run_id:
            return
        self._open_details(run_id)

    @work
    async def _open_details(self, run_id: str) -> None:
        store = self._store()
        runs = [r for r in store.fetch_all_runs() if r["run_id"] == run_id]
        if not runs:
            self.app.notify(f"Run not found: {run_id}",
                            severity="error", markup=False)
            return
        run = runs[0]
        results = store.fetch_results_for_run(run_id)
        with store.conn() as c:
            perf_rows = [dict(r) for r in c.execute(
                "SELECT * FROM perf_results WHERE run_id=?", (run_id,)).fetchall()]
            adapters = [row["adapter"] for row in c.execute(
                "SELECT adapter FROM adapters_in_run WHERE run_id=?",
                (run_id,)).fetchall()]
        md = _build_details_md(run, results, perf_rows, adapters)
        await self.app.push_screen_wait(
            MarkdownModal(f"Run details — {run_id}", md))

    # --------------------------------------------------------------- d → diff

    @work
    async def action_diff_run(self) -> None:
        tbl = self.query_one("#history-table", DataTable)
        if tbl.row_count == 0 or tbl.cursor_row is None:
            self.app.notify("No row selected", severity="warning",
                            markup=False)
            return
        run_id = self._row_run_id(tbl.cursor_row)
        if not run_id:
            return
        # First `d` — set anchor.
        if self._diff_anchor is None:
            self._diff_anchor = run_id
            self._set_anchor_banner(run_id)
            self.app.notify(
                f"Diff anchor set: {run_id}\nPress 'd' on another run "
                "to compare, or 'd' again here to clear.",
                markup=False)
            return
        # Same row pressed again — clear.
        if self._diff_anchor == run_id:
            self._diff_anchor = None
            self._set_anchor_banner(None)
            self.app.notify("Diff anchor cleared", markup=False)
            return
        # Two different runs — run compare.
        anchor = self._diff_anchor
        self._diff_anchor = None
        self._set_anchor_banner(None)
        try:
            md = self._build_diff_md(anchor, run_id)
        except Exception as e:
            self.app.notify(f"Diff failed: {e}",
                            severity="error", markup=False)
            return
        await self.app.push_screen_wait(
            MarkdownModal(f"Diff — {anchor}  vs  {run_id}", md))

    def _build_diff_md(self, run_a: str, run_b: str) -> str:
        results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        safe = lambda s: s.replace("/", "_")
        out_path = (results_dir / "compare"
                    / f"{safe(run_a)}__vs__{safe(run_b)}.md")
        store = self._store()
        compare_runs(store=store, run_a=run_a, run_b=run_b, out_path=out_path)
        return out_path.read_text()

    def _set_anchor_banner(self, run_id: str | None) -> None:
        banner = self.query_one("#hist-banner", Static)
        if run_id:
            banner.update(
                f"[bold yellow]Diff anchor:[/bold yellow] {run_id}  ·  "
                "press 'd' on another row to compare (or 'd' here to clear)"
            )
            banner.add_class("has-anchor")
        else:
            banner.update(
                "[bold]All runs[/bold]  —  ↑↓ navigate  ·  "
                "Enter = details  ·  d = diff (press on 2 rows)  ·  "
                "Del = remove"
            )
            banner.remove_class("has-anchor")

    # -------------------------------------------------------------- remove flow

    @work
    async def action_remove_run(self) -> None:
        tbl = self.query_one("#history-table", DataTable)
        if tbl.row_count == 0 or tbl.cursor_row is None:
            self.app.notify("No row selected to remove",
                            severity="warning", markup=False)
            return
        run_id = self._row_run_id(tbl.cursor_row)
        if not run_id:
            return
        confirmed = await self.app.push_screen_wait(
            ConfirmRemoveModal(run_id))
        if not confirmed:
            return
        store = self._store()
        rows = [r for r in store.fetch_all_runs() if r["run_id"] == run_id]
        if rows and rows[0].get("status") == "running":
            self.app.notify(
                f"Refusing to remove an active run ({run_id}). "
                "Abort the subprocess first.",
                severity="error", markup=False)
            return
        try:
            self._do_remove(run_id)
        except Exception as e:
            self.app.notify(f"Remove failed: {e}",
                            severity="error", markup=False)
            return
        # If the removed run was the diff anchor, clear it.
        if self._diff_anchor == run_id:
            self._diff_anchor = None
            self._set_anchor_banner(None)
        self.app.notify(f"Removed run: {run_id}", markup=False)
        self.refresh_data()

    def _do_remove(self, run_id: str) -> None:
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        store = self._store()
        with store.conn() as c:
            c.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        trace_dir = results_dir / "runs" / run_id
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
            parent = trace_dir.parent
            if parent != results_dir / "runs" and not any(parent.iterdir()):
                parent.rmdir()

    # -------------------------------------------------------------- utilities

    def _row_run_id(self, row_idx: int | None) -> str | None:
        if row_idx is None:
            return None
        tbl = self.query_one("#history-table", DataTable)
        try:
            # Column 0 is "#" counter; run_id is column 1.
            return str(tbl.get_row_at(row_idx)[1])
        except Exception:
            return None

    def _store(self) -> Store:
        results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        store = Store(results_dir / "runs.db")
        store.init_schema()
        return store

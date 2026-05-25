from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Static

from llm_test.rankings.presets import USE_CASES, get_use_case


# Dim → short label (≤8 chars to fit in a 10-wide column with breathing room).
_DIM_LABEL = {
    "coding": "coding",
    "terminal": "terminal",
    "agentic": "agentic",
    "safety": "safety",
    "restraint": "restraint",
    "error_recovery": "err_rec",
    "parameter_precision": "params",
    "context_state_tracking": "state",
    "structured_output": "struct",
    "tool_selection": "toolSel",
    "long_context": "longCtx",
    "localization": "loc",
    "budget_efficiency": "budget",
    "hallucination": "hallucin",
}

_DIM_ORDER = [
    "coding", "terminal", "agentic", "safety",
    "restraint", "error_recovery", "parameter_precision",
    "context_state_tracking", "structured_output", "tool_selection",
    "long_context", "localization", "budget_efficiency", "hallucination",
]
_GROUP_SIZE = 7
_COL_WIDTH = 11


def _format_weights_block(weights: dict[str, float] | None) -> str:
    if not weights:
        return "[dim]No persona selected — all dimensions count 1.0 in Overall.[/dim]"
    groups = [_DIM_ORDER[i:i + _GROUP_SIZE]
              for i in range(0, len(_DIM_ORDER), _GROUP_SIZE)]
    lines: list[str] = []
    for g_idx, group in enumerate(groups):
        if g_idx > 0:
            lines.append("")
        name_row = "".join(f"{_DIM_LABEL[d]:<{_COL_WIDTH}}" for d in group)
        lines.append(f"[bold]{name_row}[/bold]")
        vals = []
        for d in group:
            w = weights.get(d, 1.0)
            raw = f"{w:.2f}".rstrip("0").rstrip(".")
            if w >= 2.0:
                cell = f"[green]{raw}[/green]"
            elif w <= 0.5:
                cell = f"[red]{raw}[/red]"
            else:
                cell = raw
            pad = _COL_WIDTH - len(raw)
            vals.append(cell + " " * pad)
        lines.append("".join(vals))
    return "\n".join(lines)


class SetupTab(Container):
    """Pick a use-case persona; preview weights + see model ranking under that persona.

    The global Rankings tab is NEVER modified — this tab is a standalone viewer.
    """

    DEFAULT_CSS = """
    SetupTab { padding: 1; }
    SetupTab #setup-header { text-style: bold; margin-bottom: 1; }
    SetupTab #persona-row { height: auto; margin-bottom: 1; }
    SetupTab #persona-row Button { margin-right: 1; min-width: 12; }
    SetupTab #weights-block {
        height: auto; padding: 1 2;
        border: round $primary; margin-bottom: 1;
    }
    SetupTab #apply-row { height: auto; margin-top: 1; margin-bottom: 1; }
    SetupTab #setup-status { padding-left: 1; margin-bottom: 1; }
    SetupTab #uc-rank-title { text-style: bold; margin-top: 1; }
    SetupTab #uc-rank-table {
        height: auto; max-height: 18;
        border: thick $primary;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results")
        )
        self._previewed_key: str | None = None
        self._active_key: str | None = None

    def compose(self) -> ComposeResult:
        self._active_key = self._read_active_use_case()
        self._previewed_key = self._active_key
        with Vertical():
            yield Static(
                "Pick a use-case to preview its weights. Click [bold]Apply[/bold] to "
                "compute the model ranking for that persona (shown below). "
                "[dim]The global Rankings tab is never affected — that one always uses "
                "the standard weights (coding/terminal/agentic ×2, localization/long_context ×0.5).[/dim]",
                id="setup-header",
            )
            with Horizontal(id="persona-row"):
                yield Button("None", id="uc-none", variant=self._variant_for("none"))
                for uc in USE_CASES:
                    yield Button(uc.name, id=f"uc-{uc.key}",
                                 variant=self._variant_for(uc.key))
            yield Static(self._render_weights(), id="weights-block")
            with Horizontal(id="apply-row"):
                yield Button("Apply", id="apply", variant="primary")
            yield Static(self._status_text(), id="setup-status")
            yield Static("[dim]Click Apply to compute the ranking for the previewed persona.[/dim]",
                         id="uc-rank-title")
            yield DataTable(
                id="uc-rank-table",
                zebra_stripes=True,
                cursor_type="row",
            )

    # ---- helpers ----

    def _variant_for(self, key: str) -> str:
        if self._previewed_key is None and key == "none":
            return "success"
        if self._previewed_key == key:
            return "success"
        return "default"

    def _render_weights(self) -> str:
        if self._previewed_key is None:
            return _format_weights_block(None)
        uc = get_use_case(self._previewed_key)
        if uc is None:
            return f"[red]Unknown persona '{self._previewed_key}'[/red]"
        return _format_weights_block(uc.weights)

    def _read_active_use_case(self) -> str | None:
        p = self._results_dir / "setup.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text()).get("active_use_case")
        except (json.JSONDecodeError, OSError):
            return None

    def _status_text(self) -> str:
        previewing = self._previewed_key or "none"
        active = self._active_key or "none"
        return (f"[dim]Previewing: [bold]{previewing}[/bold]    ·    "
                f"Active in setup.json: [bold]{active}[/bold][/dim]")

    def _save_active_use_case(self, key: str | None) -> None:
        p = self._results_dir / "setup.json"
        if key is None:
            p.unlink(missing_ok=True)
            return
        self._results_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"version": 1, "active_use_case": key}, indent=2))

    # ---- events ----

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "apply":
            self._handle_apply()
            return
        if bid.startswith("uc-"):
            key = bid.removeprefix("uc-")
            self._previewed_key = None if key == "none" else key
            self._refresh_persona_buttons()
            self._refresh_weights_block()
            self._refresh_status()

    def _handle_apply(self) -> None:
        key = self._previewed_key
        self._save_active_use_case(key)
        self._active_key = key
        if key is None:
            self.app.notify("Cleared — no persona ranking computed.")
            self.query_one("#uc-rank-title", Static).update(
                "[dim]No persona selected.[/dim]")
            self.query_one("#uc-rank-table", DataTable).clear(columns=True)
        else:
            self.app.notify(f"Computing ranking for '{key}'...")
            self._render_ranking_table(key)
        self._refresh_status()

    # ---- redraw helpers ----

    def _refresh_persona_buttons(self) -> None:
        for btn in self.query("#persona-row Button"):
            bid = btn.id or ""
            if not bid.startswith("uc-"):
                continue
            btn.variant = self._variant_for(bid.removeprefix("uc-"))

    def _refresh_weights_block(self) -> None:
        self.query_one("#weights-block", Static).update(self._render_weights())

    def _refresh_status(self) -> None:
        self.query_one("#setup-status", Static).update(self._status_text())

    # ---- inline ranking table ----

    def _render_ranking_table(self, key: str) -> None:
        """Compute use-case ranking inline and populate the DataTable below Apply.

        This does NOT modify the global Rankings tab or any rankings .md files.
        """
        from llm_test.core.store import Store
        from llm_test.rankings.compute import compute_matrix
        uc = get_use_case(key)
        if uc is None:
            return
        db = self._results_dir / "runs.db"
        title = self.query_one("#uc-rank-title", Static)
        tbl = self.query_one("#uc-rank-table", DataTable)
        tbl.clear(columns=True)
        if not db.exists():
            title.update("[yellow]No runs database yet — run a test first.[/yellow]")
            return
        store = Store(db)
        store.init_schema()
        try:
            matrix = compute_matrix(
                store=store, dimensions=["overall"],
                use_case_weights=dict(uc.weights),
            )
        except Exception as e:
            title.update(f"[red]compute failed: {e}[/red]")
            return
        if not matrix:
            title.update("[yellow]No results yet — run a test first.[/yellow]")
            return

        # One row per model: best adapter under the use-case score.
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in matrix:
            by_model[r["model"]].append(r)
        rows = [
            max(prs, key=lambda r: r["scores"].get("use_case",
                                                   r["scores"].get("overall", -1.0)))
            for prs in by_model.values()
        ]
        rows.sort(key=lambda r: -r["scores"].get("use_case",
                                                 r["scores"].get("overall", -1.0)))

        tbl.add_column("#", key="rank")
        tbl.add_column("Model", key="model")
        tbl.add_column("Adapter", key="adapter")
        tbl.add_column(f"UC:{uc.name} score", key="uc_score")
        tbl.add_column("Overall (global)", key="overall")
        tbl.add_column("Runs", key="runs")

        for i, r in enumerate(rows, start=1):
            uc_score = r["scores"].get("use_case")
            overall = r["scores"].get("overall")
            tbl.add_row(
                Text(str(i), style="bold"),
                Text(r["model"], style="bold"),
                Text(r["adapter"]),
                Text(f"{uc_score * 100:.1f}%" if uc_score is not None else "—",
                     style="bold green"),
                Text(f"{overall * 100:.1f}%" if overall is not None else "—",
                     style="dim"),
                Text(str(r.get("runs", 0))),
            )

        title.update(
            f"[bold]Model ranking for [green]{uc.name}[/green][/bold]   "
            f"[dim]({len(rows)} models, best adapter per model · "
            f"this table is local to Setup, the Rankings tab is unaffected)[/dim]"
        )

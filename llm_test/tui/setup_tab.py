from __future__ import annotations

import json
import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

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

# Display order matches the Rankings table grouping — first 7, then the rest.
_DIM_ORDER = [
    "coding", "terminal", "agentic", "safety",
    "restraint", "error_recovery", "parameter_precision",
    "context_state_tracking", "structured_output", "tool_selection",
    "long_context", "localization", "budget_efficiency", "hallucination",
]
_GROUP_SIZE = 7   # dims per row of names/values

_COL_WIDTH = 11   # chars per column (8 label + breathing)


def _format_weights_block(weights: dict[str, float] | None) -> str:
    """Render 14 dims into 2 groups × 2 rows (names + values), Rich-coloured."""
    if not weights:
        return "[dim]Click a persona button above to preview its weights.[/dim]"
    groups = [_DIM_ORDER[i:i + _GROUP_SIZE]
              for i in range(0, len(_DIM_ORDER), _GROUP_SIZE)]
    lines: list[str] = []
    for g_idx, group in enumerate(groups):
        if g_idx > 0:
            lines.append("")
        name_row = "".join(f"{_DIM_LABEL[d]:<{_COL_WIDTH}}" for d in group)
        lines.append(f"[bold]{name_row}[/bold]")
        val_cells = []
        for d in group:
            w = weights.get(d, 1.0)
            cell = f"{w:<{_COL_WIDTH}.2f}".rstrip("0").rstrip(".").ljust(_COL_WIDTH)
            # Re-pad: format strips trailing zeros (3.00 → 3) and shortens the
            # cell, then ljust restores the column width.
            if w >= 2.0:
                cell = f"[green]{cell.rstrip()}[/green]" + " " * (_COL_WIDTH - len(cell.rstrip()))
            elif w <= 0.5:
                cell = f"[red]{cell.rstrip()}[/red]" + " " * (_COL_WIDTH - len(cell.rstrip()))
            val_cells.append(cell)
        lines.append("".join(val_cells))
    return "\n".join(lines)


class SetupTab(Container):
    """Pick a use-case persona to drive an additional ranking column."""

    DEFAULT_CSS = """
    SetupTab { padding: 1; }
    SetupTab #setup-header { text-style: bold; margin-bottom: 1; }
    SetupTab #persona-row {
        height: auto;
        margin-bottom: 1;
    }
    SetupTab #persona-row Button {
        margin-right: 1;
        min-width: 12;
    }
    SetupTab #weights-block {
        height: auto;
        padding: 1 2;
        border: round $primary;
        margin-bottom: 1;
    }
    SetupTab #apply-row {
        height: auto;
        margin-top: 1;
    }
    SetupTab #setup-status { margin-top: 1; padding-left: 1; }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results")
        )
        # _previewed_key: what's shown in weights block (None == nothing previewed,
        # or "none" == None-button previewed).
        # _active_key: what's persisted in setup.json (None == cleared).
        self._previewed_key: str | None = None
        self._active_key: str | None = None

    def compose(self) -> ComposeResult:
        self._active_key = self._read_active_use_case()
        self._previewed_key = self._active_key  # preview shows active on mount
        with Vertical():
            yield Static(
                "Pick a use-case to [bold]preview[/bold] its dimension weights. "
                "Click [bold]Apply[/bold] to persist the selection — this creates an extra "
                "[bold]UC:<Name>[/bold] column in Rankings (Overall is unaffected).",
                id="setup-header",
            )
            with Horizontal(id="persona-row"):
                yield Button("None", id="uc-none",
                             variant=self._variant_for("none"))
                for uc in USE_CASES:
                    yield Button(uc.name, id=f"uc-{uc.key}",
                                 variant=self._variant_for(uc.key))
            yield Static(self._render_weights(), id="weights-block")
            with Horizontal(id="apply-row"):
                yield Button("Apply", id="apply", variant="primary")
            yield Static(self._status_text(), id="setup-status")

    # ---- state helpers ----

    def _variant_for(self, key: str) -> str:
        """Button colour: success when this key is currently previewed."""
        if self._previewed_key is None and key == "none":
            return "success"
        if self._previewed_key == key:
            return "success"
        return "default"

    def _render_weights(self) -> str:
        if self._previewed_key is None or self._previewed_key == "none":
            return "[dim]No persona previewed — all dimensions count 1.0 in Overall.[/dim]"
        uc = get_use_case(self._previewed_key)
        if uc is None:
            return f"[red]Unknown persona '{self._previewed_key}'[/red]"
        return _format_weights_block(uc.weights)

    def _read_active_use_case(self) -> str | None:
        setup_path = self._results_dir / "setup.json"
        if not setup_path.exists():
            return None
        try:
            data = json.loads(setup_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return data.get("active_use_case")

    def _status_text(self) -> str:
        previewing = (
            "none" if self._previewed_key is None
            else self._previewed_key
        )
        active = self._active_key or "none"
        return (
            f"[dim]Previewing: [bold]{previewing}[/bold]"
            f"    ·    Active in setup.json: [bold]{active}[/bold][/dim]"
        )

    def _save_active_use_case(self, key: str | None) -> None:
        setup_path = self._results_dir / "setup.json"
        if key is None:
            setup_path.unlink(missing_ok=True)
            return
        self._results_dir.mkdir(parents=True, exist_ok=True)
        setup_path.write_text(
            json.dumps({"version": 1, "active_use_case": key}, indent=2)
        )

    # ---- event handling ----

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
        key = self._previewed_key  # None means cleared
        if key is None:
            self._save_active_use_case(None)
            self.app.notify("Use-case cleared")
        else:
            self._save_active_use_case(key)
            self.app.notify(f"Use-case '{key}' applied — regenerating rankings...")
        self._active_key = key
        self._regenerate_rankings()
        self._refresh_status()
        self._switch_focus_to_rankings()

    # ---- redraw helpers ----

    def _refresh_persona_buttons(self) -> None:
        for btn in self.query("#persona-row Button"):
            bid = btn.id or ""
            if not bid.startswith("uc-"):
                continue
            key = bid.removeprefix("uc-")
            btn.variant = self._variant_for(key)

    def _refresh_weights_block(self) -> None:
        self.query_one("#weights-block", Static).update(self._render_weights())

    def _refresh_status(self) -> None:
        self.query_one("#setup-status", Static).update(self._status_text())

    # ---- regen + focus (unchanged from previous impl) ----

    def _regenerate_rankings(self) -> None:
        from llm_test.core.store import Store
        from llm_test.rankings.compute import (
            load_active_use_case, regenerate_rankings,
        )
        db = self._results_dir / "runs.db"
        if not db.exists():
            return
        store = Store(db)
        store.init_schema()
        uc_key, uc_weights = load_active_use_case(self._results_dir)
        dims = [
            "overall", "coding", "agentic", "safety", "restraint",
            "long_context", "budget_efficiency", "hallucination",
            "error_recovery", "parameter_precision",
            "context_state_tracking", "structured_output",
            "tool_selection", "localization", "terminal",
        ]
        try:
            regenerate_rankings(
                store=store, dimensions=dims,
                out_dir=self._results_dir / "rankings",
                use_case_weights=uc_weights, use_case_key=uc_key,
            )
        except Exception as e:
            self.app.notify(f"Regen failed: {e}", severity="error")

    def _switch_focus_to_rankings(self) -> None:
        from textual.widgets import TabbedContent
        try:
            tabs = self.app.query_one(TabbedContent)
            tabs.active = "rankings"
        except Exception:
            pass
        try:
            from llm_test.tui.rankings_tab import RankingsTab
            rt = self.app.query_one(RankingsTab)
            if hasattr(rt, "reload"):
                rt.reload()
        except Exception:
            pass

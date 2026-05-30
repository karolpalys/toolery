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
    "debugging": "debug",
    "terminal": "terminal",
    "agentic": "agentic",
    "safety": "safety",
    "adversarial_robustness": "adv_rob",
    "restraint": "restraint",
    "error_recovery": "err_rec",
    "parameter_precision": "params",
    "context_state_tracking": "state",
    "structured_output": "struct",
    "tool_selection": "toolSel",
    "instruction_following": "instr",
    "long_context": "longCtx",
    "localization": "loc",
    "budget_efficiency": "budget",
    "hallucination": "hallucin",
}

_DIM_ORDER = [
    "coding", "debugging", "terminal", "agentic", "safety",
    "adversarial_robustness", "restraint", "error_recovery",
    "parameter_precision", "context_state_tracking", "structured_output",
    "tool_selection", "instruction_following", "long_context",
    "localization", "budget_efficiency", "hallucination",
]


def _format_weights_block(weights: dict[str, float] | None) -> str:
    if not weights:
        parts = [f"[bold]{_DIM_LABEL[d]}[/bold]=1" for d in _DIM_ORDER]
        return "  |  ".join(parts)
    parts: list[str] = []
    for d in _DIM_ORDER:
        w = weights.get(d, 1.0)
        raw = f"{w:.2f}".rstrip("0").rstrip(".")
        if w >= 2.0:
            value = f"[green]{raw}[/green]"
        elif w <= 0.5:
            value = f"[red]{raw}[/red]"
        else:
            value = raw
        parts.append(f"[bold]{_DIM_LABEL[d]}[/bold]={value}")
    return "  |  ".join(parts)


class ProfilesTab(Container):
    """Pick a use-case persona; preview weights + see model ranking under that persona.

    The global Rankings tab is NEVER modified — this tab is a standalone viewer.
    """

    DEFAULT_CSS = """
    ProfilesTab {
        layout: vertical;
        padding: 1 2;
        background: $surface;
    }

    ProfilesTab #setup-intro {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    ProfilesTab #ranking-section,
    ProfilesTab #selector-section,
    ProfilesTab #weights-section,
    ProfilesTab #sparks-section {
        border: round $primary;
        border-title-color: $primary;
        background: $surface;
        padding: 0 1;
    }

    ProfilesTab #selector-section {
        height: 4;
        margin-bottom: 1;
    }

    ProfilesTab #persona-row {
        width: 1fr;
        height: 3;
    }

    ProfilesTab #persona-row Button {
        margin-right: 1;
        min-width: 10;
    }

    ProfilesTab #apply-row {
        width: auto;
        height: 3;
    }

    ProfilesTab #weights-section {
        height: 5;
        margin-bottom: 1;
    }

    ProfilesTab #weights-block {
        height: auto;
        padding: 1 1;
        color: $text;
    }

    ProfilesTab #sparks-section {
        height: 4;
        margin-bottom: 1;
    }

    ProfilesTab #sparks-row {
        width: 1fr;
        height: 3;
    }

    ProfilesTab #adapter-row {
        width: 1fr;
        height: 3;
        margin-left: 2;
    }

    ProfilesTab #adapter-row Button {
        margin-right: 1;
        min-width: 8;
    }

    ProfilesTab #sparks-row Button {
        margin-right: 1;
        min-width: 6;
    }

    ProfilesTab #ranking-section {
        height: 1fr;
        margin-bottom: 1;
    }

    ProfilesTab #uc-rank-title {
        height: auto;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    ProfilesTab #uc-rank-table {
        height: 1fr;
    }

    ProfilesTab #setup-status {
        height: 1;
        color: $text-muted;
        padding-left: 1;
    }
    """

    # SPARKS button id suffix → cluster value in DB (None = no filter, all clusters)
    _SPARKS_TO_CLUSTER = {
        "all": None,
        "1": "single",
        "2": "dual",
        "3": "triple",
        "4": "quad",
        "8": "octa",
    }

    # Adapter button id suffix → adapter value in DB. `all` clears the filter.
    _ADAPTER_TO_DB = {
        "all": None,
        "cloud": "cloud",
        "hermes": "hermes",
    }

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results")
        )
        self._previewed_key: str | None = None
        self._active_key: str | None = None
        self._sparks_key: str = "all"  # SPARKS filter (default: ALL clusters)
        self._adapter_key: str = "all"  # adapter filter (default: ALL adapters)

    def compose(self) -> ComposeResult:
        self._active_key = self._read_active_use_case()
        self._previewed_key = self._active_key
        yield Static(
            "Choose a usage profile and apply it to compute a local ranking.",
            id="setup-intro",
        )
        # Top: use-case profiles + weight preview (user-controlled inputs).
        with Horizontal(id="selector-section"):
            with Horizontal(id="persona-row"):
                yield Button("None", id="uc-none", variant=self._variant_for("none"))
                for uc in USE_CASES:
                    yield Button(uc.name, id=f"uc-{uc.key}",
                                 variant=self._variant_for(uc.key))
            with Horizontal(id="apply-row"):
                yield Button("Apply profile", id="apply", variant="primary")
        with Vertical(id="weights-section"):
            yield Static(self._render_weights(), id="weights-block")
        # SPARKS + adapter filters side-by-side — both narrow which runs
        # contribute to the ranking table below.
        with Horizontal(id="sparks-section"):
            with Horizontal(id="sparks-row"):
                for key in ("all", "1", "2", "3", "4", "8"):
                    label = "ALL" if key == "all" else key
                    yield Button(label, id=f"sparks-{key}",
                                 variant=self._sparks_variant_for(key))
            with Horizontal(id="adapter-row"):
                for key in ("all", "cloud", "hermes"):
                    label = "ALL" if key == "all" else key
                    yield Button(label, id=f"adapter-filter-{key}",
                                 variant=self._adapter_variant_for(key))
        # Ranking table below (the output the user inspects).
        with Vertical(id="ranking-section"):
            yield Static("[dim]Apply a profile to compute this ranking.[/dim]",
                         id="uc-rank-title")
            yield DataTable(
                id="uc-rank-table",
                zebra_stripes=True,
                cursor_type="row",
            )
        yield Static(self._status_text(), id="setup-status")

    def on_mount(self) -> None:
        try:
            self.query_one("#selector-section").border_title = "Use-case profiles"
            self.query_one("#weights-section").border_title = "Weight preview"
            self.query_one("#sparks-section").border_title = (
                "SPARKS (cluster nodes) — adapter (run harness)"
            )
            self.query_one("#ranking-section").border_title = "Profile ranking"
        except Exception:
            pass

    # ---- helpers ----

    def _variant_for(self, key: str) -> str:
        if self._previewed_key is None and key == "none":
            return "success"
        if self._previewed_key == key:
            return "success"
        return "default"

    def _sparks_variant_for(self, key: str) -> str:
        """Highlight the active SPARKS filter button."""
        return "success" if self._sparks_key == key else "default"

    def _adapter_variant_for(self, key: str) -> str:
        """Highlight the active adapter filter button."""
        return "success" if self._adapter_key == key else "default"

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
        sparks = "ALL" if self._sparks_key == "all" else self._sparks_key
        return (f"[dim]Previewing: [bold]{previewing}[/bold]    ·    "
                f"Active in setup.json: [bold]{active}[/bold]    ·    "
                f"SPARKS: [bold]{sparks}[/bold][/dim]")

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
            return
        if bid.startswith("sparks-"):
            key = bid.removeprefix("sparks-")
            if key not in self._SPARKS_TO_CLUSTER:
                return
            self._sparks_key = key
            self._refresh_sparks_buttons()
            if self._active_key is not None:
                self._render_ranking_table(self._active_key)
            self._refresh_status()
            return
        if bid.startswith("adapter-filter-"):
            key = bid.removeprefix("adapter-filter-")
            if key not in self._ADAPTER_TO_DB:
                return
            self._adapter_key = key
            self._refresh_adapter_buttons()
            if self._active_key is not None:
                self._render_ranking_table(self._active_key)
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

    def _refresh_sparks_buttons(self) -> None:
        for btn in self.query("#sparks-row Button"):
            bid = btn.id or ""
            if not bid.startswith("sparks-"):
                continue
            btn.variant = self._sparks_variant_for(bid.removeprefix("sparks-"))

    def _refresh_adapter_buttons(self) -> None:
        for btn in self.query("#adapter-row Button"):
            bid = btn.id or ""
            if not bid.startswith("adapter-filter-"):
                continue
            btn.variant = self._adapter_variant_for(
                bid.removeprefix("adapter-filter-"))

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
        cluster_filter = self._SPARKS_TO_CLUSTER.get(self._sparks_key)
        adapter_filter = self._ADAPTER_TO_DB.get(self._adapter_key)
        try:
            matrix = compute_matrix(
                store=store, dimensions=["overall"],
                use_case_weights=dict(uc.weights),
                cluster_filter=cluster_filter,
                adapter_filter=adapter_filter,
            )
        except Exception as e:
            title.update(f"[red]compute failed: {e}[/red]")
            return
        if not matrix:
            sparks_label = ("ALL" if self._sparks_key == "all"
                            else f"{self._sparks_key} ({cluster_filter})")
            title.update(
                f"[yellow]No results for SPARKS={sparks_label} — "
                "run a test with this cluster topology first.[/yellow]"
            )
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

        sparks_label = ("ALL clusters" if self._sparks_key == "all"
                        else f"SPARKS={self._sparks_key} ({cluster_filter})")
        title.update(
            f"[bold]Model ranking for [green]{uc.name}[/green][/bold]   "
            f"[dim]({len(rows)} models · {sparks_label} · "
            f"best adapter per model · local to Setup tab)[/dim]"
        )

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, SelectionList, Static

from toolery.core.store import Store
from toolery.rankings.compute import compute_matrix
from toolery.tui.rankings_tab import (
    _DIMENSIONS,
    _HEADERS,
    _PERF_COLS,
    _PERF_HEADERS,
    _aggregate_perf,
)

# Winner cells need to be unmistakable — plain `bold` is too subtle in most
# terminals against the surrounding numbers. Black-on-green flips both bg
# and weight so the eye snaps to it. Losers go `dim` so the contrast is
# 2-sided (winner louder + others quieter).
_WINNER_STYLE = "bold black on green"
_LOSER_STYLE = "dim"


def _fmt_score(value: float | None, is_winner: bool) -> Text:
    if value is None:
        return Text(" — ", style="dim")
    text = f" {value * 100:.1f}% "  # pad so the background block has breathing room
    return Text(text, style=_WINNER_STYLE if is_winner else _LOSER_STYLE)


def _fmt_perf(value: float | None, is_winner: bool) -> Text:
    if value is None:
        return Text(" — ", style="dim")
    text = f" {value:.1f} "
    return Text(text, style=_WINNER_STYLE if is_winner else _LOSER_STYLE)


def _winners_by(rows: list[dict], value_fn) -> set[int]:
    """Indices of rows that hold the column max. Ties → all tied indices win
    (so two models on identical 92.3% are both highlighted)."""
    best = float("-inf")
    winners: set[int] = set()
    for i, r in enumerate(rows):
        v = value_fn(r)
        if v is None:
            continue
        if v > best:
            best = v
            winners = {i}
        elif v == best:
            winners.add(i)
    return winners


class CompareTab(Container):
    """User-curated comparison matrix. Same data as Rankings, but limited to
    models the user explicitly picks. Per-column winner is rendered bold;
    every other value is plain weight — no medal icons."""

    DEFAULT_CSS = """
    CompareTab {
        layout: vertical;
        padding: 1 2;
        background: $surface;
    }

    CompareTab #compare-intro {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    CompareTab #picker-section,
    CompareTab #actions-section,
    CompareTab #matrix-section {
        border: round $primary;
        border-title-color: $primary;
        background: $surface;
        padding: 0 1;
    }

    CompareTab #picker-section {
        height: 10;
        margin-bottom: 1;
    }

    CompareTab #model-picker {
        height: 1fr;
    }

    CompareTab #actions-section {
        height: 4;
        margin-bottom: 1;
    }

    CompareTab #actions-section Button {
        margin-right: 2;
    }

    CompareTab #compare-status {
        height: 1;
        color: $text-muted;
        padding-left: 1;
        margin-bottom: 1;
    }

    CompareTab #matrix-section {
        height: 1fr;
    }

    CompareTab #compare-matrix {
        height: 1fr;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._matrix_cache: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static(
            "Select models for a focused head-to-head view. Winners are highlighted per metric.",
            id="compare-intro",
        )
        with Vertical(id="picker-section"):
            yield SelectionList[str](id="model-picker")
        with Horizontal(id="actions-section"):
            yield Button("Compare", id="build", variant="primary")
            yield Button("Reset", id="clear")
        yield Static(
            "[dim italic]Pick models above, then compare.[/dim italic]",
            id="compare-status",
        )
        with Vertical(id="matrix-section"):
            yield DataTable(id="compare-matrix", zebra_stripes=True,
                            cursor_type="row")

    def on_mount(self) -> None:
        tbl = self.query_one("#compare-matrix", DataTable)
        tbl.add_column("#", key="rank")
        for key, header in (("model", "Model"), ("adapter", "Adapter")):
            tbl.add_column(header, key=key)
        for dim in _DIMENSIONS:
            tbl.add_column(_HEADERS[dim], key=f"dim:{dim}")
        for p in _PERF_COLS:
            tbl.add_column(_PERF_HEADERS[p], key=f"perf:{p}")
        tbl.add_column("Runs", key="runs")
        tbl.add_column("Cluster", key="cluster")
        try:
            self.query_one("#picker-section").border_title = "Models"
            self.query_one("#actions-section").border_title = "Actions"
            self.query_one("#matrix-section").border_title = "Head-to-head matrix"
        except Exception:
            pass
        self._populate_picker()

    def on_show(self) -> None:
        # Re-fill the model picker when the tab becomes visible — picks up
        # models that landed from new runs in another tab.
        self._populate_picker()

    # ---------------------------------------------------------------- helpers

    def _store(self) -> Store | None:
        results_dir = Path(
            os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
        db = results_dir / "runs.db"
        if not db.exists():
            return None
        store = Store(db)
        store.init_schema()
        return store

    def _populate_picker(self) -> None:
        picker = self.query_one("#model-picker", SelectionList)
        store = self._store()
        if store is None:
            self.query_one("#compare-status", Static).update(
                "[yellow]No runs database yet — run a test first.[/yellow]"
            )
            return
        models = sorted({r["model"] for r in store.fetch_all_runs()})
        prev = set(picker.selected)
        picker.clear_options()
        if models:
            picker.add_options([(m, m, m in prev) for m in models])
        self.query_one("#compare-status", Static).update(
            f"[dim]{len(models)} models in DB · {len(prev)} currently selected[/dim]"
        )

    # ---------------------------------------------------------------- actions

    @on(Button.Pressed, "#build")
    def _on_build(self) -> None:
        picker = self.query_one("#model-picker", SelectionList)
        sel = set(picker.selected)
        status = self.query_one("#compare-status", Static)
        if len(sel) < 1:
            status.update("[yellow]Select at least 1 model first.[/yellow]")
            return
        store = self._store()
        if store is None:
            return
        matrix = compute_matrix(store=store, dimensions=_DIMENSIONS)
        perf_agg = _aggregate_perf(store)
        for r in matrix:
            # cluster is part of the matrix row now; perf matches on it too.
            key = (r["model"], r["adapter"], r.get("cluster"))
            r["perf"] = perf_agg.get(key, {})

        # One row per selected model: same rule as Rankings (best-overall adapter).
        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in matrix:
            if row["model"] in sel:
                by_model[row["model"]].append(row)
        rows = [
            max(prs, key=lambda r: r["scores"].get("overall", -1.0))
            for prs in by_model.values()
        ]
        rows.sort(key=lambda r: -(r["scores"].get("overall") or 0))
        self._matrix_cache = rows
        self._render_table()

        missing = sel - {r["model"] for r in rows}
        note = (f"  ·  [yellow]no data: {', '.join(sorted(missing))}[/yellow]"
                if missing else "")
        status.update(
            f"[dim]Comparing {len(rows)} model(s){note}[/dim]"
        )

    @on(Button.Pressed, "#clear")
    def _on_clear(self) -> None:
        self.query_one("#model-picker", SelectionList).deselect_all()
        self._matrix_cache = []
        self.query_one("#compare-matrix", DataTable).clear()
        self.query_one("#compare-status", Static).update(
            "[dim italic]Pick models above, then compare.[/dim italic]"
        )

    # ---------------------------------------------------------------- render

    def _render_table(self) -> None:
        tbl = self.query_one("#compare-matrix", DataTable)
        tbl.clear()
        rows = self._matrix_cache

        # Per-column winner set (ties → all tied rows are highlighted).
        winners_dim: dict[str, set[int]] = {
            dim: _winners_by(rows, lambda r, d=dim: r["scores"].get(d))
            for dim in _DIMENSIONS
        }
        winners_perf: dict[str, set[int]] = {
            p: _winners_by(rows, lambda r, p=p: r["perf"].get(p))
            for p in _PERF_COLS
        }

        for i, r in enumerate(rows):
            cells: list[Text] = [Text(str(i + 1)), Text(r["model"]), Text(r["adapter"])]
            for dim in _DIMENSIONS:
                cells.append(_fmt_score(r["scores"].get(dim),
                                        i in winners_dim[dim]))
            for p in _PERF_COLS:
                cells.append(_fmt_perf(r["perf"].get(p),
                                       i in winners_perf[p]))
            cells.append(Text(str(r.get("runs", 0))))
            cluster = r.get("cluster")
            if cluster == "quad":
                cells.append(Text("⚡⚡⚡ quad", style="cyan"))
            elif cluster == "triple":
                cells.append(Text("⚡⚡ triple", style="cyan"))
            elif cluster == "dual":
                cells.append(Text("⚡ dual", style="cyan"))
            elif cluster == "single":
                cells.append(Text("• single"))
            else:
                cells.append(Text("—", style="dim"))
            tbl.add_row(*cells, height=2)

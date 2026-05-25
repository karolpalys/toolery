from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

from rich.text import Text
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import DataTable, Static

from llm_test.core.scenario import load_all_scenarios
from llm_test.core.store import Store
from llm_test.rankings.compute import compute_matrix

# Score-column order, left-to-right.
_DIMENSIONS = [
    "overall",
    "hallucination",
    "coding",
    "agentic",
    "safety",
    "restraint",
    "error_recovery",
    "parameter_precision",
    "context_state_tracking",
    "structured_output",
    "tool_selection",
    "long_context",
    "localization",
    "budget_efficiency",
    "terminal",
]

# Short column headers — full names would be too wide with 14 score cols + perf.
_HEADERS = {
    "overall": "Overall",
    "hallucination": "Calibr.",
    "coding": "Coding",
    "agentic": "Agentic",
    "safety": "Safety",
    "restraint": "Restraint",
    "error_recovery": "ErrRec",
    "parameter_precision": "Params",
    "context_state_tracking": "State",
    "structured_output": "Struct",
    "tool_selection": "ToolSel",
    "long_context": "LongCtx",
    "localization": "L10n",
    "budget_efficiency": "Budget",
    "terminal": "Term",
}

# Perf columns rendered after the score cols.
_PERF_COLS = ["pp_tps", "tg_tps"]
_PERF_HEADERS = {"pp_tps": "PP t/s", "tg_tps": "Gen t/s"}

# One-line description per score column — rendered below the table so the
# user does not have to keep the README open. Overall is skipped (it's just
# the tier-weighted aggregate of every scored scenario).
_LEGEND: list[tuple[str, str]] = [
    ("hallucination",          "Calibrated uncertainty — refuses to fabricate when ungrounded"),
    ("coding",                 "TDD loops, multi-file refactors, file ops, git discipline"),
    ("agentic",                "Multi-step planning, conditional chains, parallel fan-out"),
    ("safety",                 "Prompt-injection resistance via adversarial tool results"),
    ("restraint",              "Refuses to call a tool when the answer is already in context"),
    ("error_recovery",         "Recovery from timeouts, 429s, malformed responses, partial fails"),
    ("parameter_precision",    "Parameter precision — ISO codes, DST, numeric bounds, MIME types"),
    ("context_state_tracking", "Reuses prior tool results across turns instead of re-fetching"),
    ("structured_output",      "Non-JSON structured output — CSV, YAML, markdown tables"),
    ("tool_selection",         "Picks the right tool when distractors are present"),
    ("long_context",           "Needle-in-haystack retrieval from long contexts"),
    ("localization",           "Non-English prompts and responses"),
    ("budget_efficiency",      "Completes complex tasks within tight tool-call budgets"),
    ("terminal",               "Shell commands, CLI output parsing, ANSI/TTY handling, processes & safety"),
]

# Top-3 podium icons — medal emojis prepended to the value. Previous coloured
# backgrounds were less visible than emoji on most terminal themes.
_PODIUM_ICON = {1: "🥇", 2: "🥈", 3: "🥉"}


def _fmt_score(value: float | None, podium_rank: int | None) -> Text:
    if value is None:
        return Text("—", style="bold dim")
    icon = _PODIUM_ICON.get(podium_rank, "")
    text = f"{icon} {value * 100:.1f}%" if icon else f"{value * 100:.1f}%"
    return Text(text, style="bold")


def _fmt_perf(value: float | None, podium_rank: int | None) -> Text:
    if value is None:
        return Text("—", style="bold dim")
    icon = _PODIUM_ICON.get(podium_rank, "")
    text = f"{icon} {value:.1f}" if icon else f"{value:.1f}"
    return Text(text, style="bold")


def _meta_cell(text: str, dim_style: str = "bold") -> Text:
    return Text(text, style=dim_style)


def _current_scenarios_hash() -> str | None:
    """Hash of the currently-loaded scenarios — same fn as cli.py uses on save."""
    try:
        xs = load_all_scenarios(Path("scenarios"))
    except Exception:
        return None
    if not xs:
        return None
    return hashlib.sha256(
        ",".join(sorted(s.id for s in xs)).encode("utf-8")
    ).hexdigest()[:16]


def _latest_cluster(store: Store) -> dict[tuple[str, str], str]:
    """Per (model, adapter), pick the cluster topology of the most recent run.
    Older runs without a cluster value report None and are skipped."""
    out: dict[tuple[str, str], str] = {}
    with store.conn() as c:
        rows = c.execute("""
            SELECT r.model, ar.adapter, r.cluster, r.started_at
            FROM runs r
            JOIN adapters_in_run ar ON ar.run_id = r.run_id
            WHERE r.cluster IS NOT NULL
            ORDER BY r.started_at DESC
        """).fetchall()
    for row in rows:
        key = (row["model"], row["adapter"])
        if key not in out:
            out[key] = row["cluster"]
    return out


def _aggregate_perf(store: Store, history_window_runs: int = 5) -> dict[tuple[str, str], dict[str, float]]:
    """Return per-(model, adapter) average pp_tps + tg_tps from perf_results.

    Aggregation: for each pair, take the last N runs (by started_at), for each
    run mean across depths, then mean across runs (no decay — perf is a hardware
    measurement, not a model-quality one).
    """
    out: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"pp_tps": [], "tg_tps": []}
    )
    with store.conn() as c:
        rows = c.execute("""
            SELECT r.run_id, r.model, ar.adapter, p.pp_tps, p.tg_tps
            FROM perf_results p
            JOIN runs r ON r.run_id = p.run_id
            JOIN adapters_in_run ar ON ar.run_id = r.run_id
            ORDER BY r.started_at DESC
        """).fetchall()
    for row in rows:
        key = (row["model"], row["adapter"])
        if row["pp_tps"] is not None:
            out[key]["pp_tps"].append(row["pp_tps"])
        if row["tg_tps"] is not None:
            out[key]["tg_tps"].append(row["tg_tps"])
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for key, perf in out.items():
        d: dict[str, float] = {}
        for k, vals in perf.items():
            if vals:
                d[k] = sum(vals) / len(vals)
        if d:
            agg[key] = d
    return agg


class RankingsTab(Container):
    """Rankings matrix — clickable headers sort by that column (desc)."""

    DEFAULT_CSS = """
    RankingsTab { padding: 0 1; }
    RankingsTab #rankings-intro {
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    RankingsTab #matrix-section {
        height: auto;
        max-height: 25;
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }
    RankingsTab #legend-section {
        height: 1fr;
        border: round $primary;
        padding: 1 2;
    }
    RankingsTab #rank-matrix {
        height: auto;
        max-height: 23;
        text-style: bold;
    }
    /* header_height=2 makes the header twice as tall (terminals can't scale
       fonts; this is the analogue). Bold + reverse + a strong background
       differentiates it from the data rows. */
    RankingsTab #rank-matrix > .datatable--header {
        text-style: bold reverse;
        background: $accent;
        color: $text;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        # column_key -> sort accessor (row dict → float). Set in reload().
        self._sort_keys: dict[str, callable] = {}
        self._sort_by: str | None = "overall"   # default sort
        self._sort_desc: bool = True
        self._rows_cache: list[dict] = []

    def compose(self):
        yield Static(
            "Tier-weighted ranking across all scenarios. "
            "Best adapter per model shown; click headers to sort.",
            id="rankings-intro",
        )
        with Vertical(id="matrix-section"):
            yield Static("", id="rank-summary")
            yield DataTable(
                id="rank-matrix",
                zebra_stripes=True,
                cursor_type="row",
                header_height=2,
                cell_padding=2,
            )
        with VerticalScroll(id="legend-section"):
            yield Static(self._legend_body(), id="rank-legend-body")

    @staticmethod
    def _legend_body() -> str:
        pad = max(len(_HEADERS[k]) for k, _ in _LEGEND)
        lines = [
            f"  [bold cyan]{_HEADERS[k]:<{pad}}[/bold cyan]  {desc}"
            for k, desc in _LEGEND
        ]
        return "\n".join(lines)

    def on_mount(self) -> None:
        self.reload()
        try:
            self.query_one("#matrix-section").border_title = "🏆 Rankings matrix"
            self.query_one("#legend-section").border_title = "📖 Column legend"
        except Exception:
            pass
        # Poll the DB so completed runs surface without needing the user to
        # restart the TUI. Cheap: one SELECT per dimension + perf join.
        self.set_interval(5.0, self.reload)

    def on_show(self) -> None:
        """Force-refresh when the tab becomes visible — picks up changes that
        landed while the user was looking at another tab."""
        self.reload()

    # -- data loading --
    def reload(self) -> None:
        tbl = self.query_one("#rank-matrix", DataTable)
        tbl.clear(columns=True)
        summary = self.query_one("#rank-summary", Static)

        # Always register columns + accessors so headers are visible even
        # before any run is recorded.
        for key, header in [("rank", "#"), ("model", "Model"), ("adapter", "Adapter")]:
            tbl.add_column(header, key=key)
        for dim in _DIMENSIONS:
            tbl.add_column(_HEADERS[dim], key=f"dim:{dim}")
        for p in _PERF_COLS:
            tbl.add_column(_PERF_HEADERS[p], key=f"perf:{p}")
        tbl.add_column("Runs", key="runs")
        tbl.add_column("Set", key="set")
        tbl.add_column("Cluster", key="cluster")
        self._sort_keys = {
            "model": lambda r: r["model"].lower(),
            "adapter": lambda r: r["adapter"],
            "runs": lambda r: -(r.get("runs") or 0),
            "set": lambda r: 1 if r.get("stale") else 0,
            # dual ranks before single ranks before unknown — purely display.
            "cluster": lambda r: {"dual": 0, "single": 1}.get(r.get("cluster"), 2),
        }
        for dim in _DIMENSIONS:
            d = dim
            self._sort_keys[f"dim:{d}"] = lambda r, d=d: -(r["scores"].get(d, -1.0))
        for p in _PERF_COLS:
            self._sort_keys[f"perf:{p}"] = lambda r, p=p: -(r["perf"].get(p, -1.0))

        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        db = results_dir / "runs.db"
        if not db.exists():
            summary.update("[dim italic]💤 No runs recorded yet. Use the Home tab to start a benchmark, then return here.[/dim italic]")
            self._rows_cache = []
            return
        store = Store(db)
        store.init_schema()

        matrix = compute_matrix(store=store, dimensions=_DIMENSIONS)
        if not matrix:
            summary.update("[dim italic]💤 Database is empty. Use the Home tab to start a benchmark.[/dim italic]")
            self._rows_cache = []
            return

        perf_agg = _aggregate_perf(store)
        cluster_agg = _latest_cluster(store)
        canonical_hash = _current_scenarios_hash()

        for r in matrix:
            key = (r["model"], r["adapter"])
            r["perf"] = perf_agg.get(key, {})
            r["cluster"] = cluster_agg.get(key)   # None when not recorded
            hashes = r.get("scenarios_hashes") or set()
            r["stale"] = bool(canonical_hash and hashes and canonical_hash not in hashes)

        # One row per model: pick the adapter with the best 'overall' score.
        by_model: dict[str, list[dict]] = defaultdict(list)
        for row in matrix:
            by_model[row["model"]].append(row)
        rows: list[dict] = [
            max(prs, key=lambda r: r["scores"].get("overall", -1.0))
            for prs in by_model.values()
        ]
        self._rows_cache = rows
        self._populate_rows()

        sources = sum(r["runs"] for r in rows)
        stale_n = sum(1 for r in rows if r.get("stale"))
        stale_note = (f"  ·  [yellow]{stale_n} pair(s) tested on an older "
                      f"scenario set — marked ⚠ in 'Set' column[/yellow]"
                      if stale_n else "")
        summary.update(
            f"[dim]{len(rows)} models · {sources} runs · "
            f"showing best-overall adapter per model · "
            f"{len(_DIMENSIONS)} score dims + 2 perf{stale_note}[/dim]"
        )

    def _populate_rows(self) -> None:
        """Re-sort + re-render. Reads self._rows_cache + self._sort_by/_desc."""
        tbl = self.query_one("#rank-matrix", DataTable)
        # Clear data rows only — keep columns.
        tbl.clear()
        rows = list(self._rows_cache)

        # Default sort = overall desc. User clicks override.
        sort_key = self._sort_by or "dim:overall"
        accessor = self._sort_keys.get(sort_key, self._sort_keys["dim:overall"])
        rows.sort(key=accessor, reverse=False if self._sort_desc else True)
        # Accessors return *negative* for "higher-is-better" columns so the
        # default ascending sort by negative score yields top-down ordering.

        # Per-column podium ranks: recomputed for current visible ordering.
        podium_score: dict[tuple[int, str], int] = {}
        for dim in _DIMENSIONS:
            scored = [(i, r["scores"].get(dim)) for i, r in enumerate(rows)
                      if r["scores"].get(dim) is not None]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (i, _v) in enumerate(scored[:3], start=1):
                podium_score[(i, dim)] = rank
        podium_perf: dict[tuple[int, str], int] = {}
        for p in _PERF_COLS:
            scored = [(i, r["perf"].get(p)) for i, r in enumerate(rows)
                      if r["perf"].get(p) is not None]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (i, _v) in enumerate(scored[:3], start=1):
                podium_perf[(i, p)] = rank

        for i, r in enumerate(rows):
            cells: list[Text] = [
                _meta_cell(str(i + 1)),
                _meta_cell(r["model"]),
                _meta_cell(r["adapter"]),
            ]
            for dim in _DIMENSIONS:
                cells.append(_fmt_score(r["scores"].get(dim),
                                        podium_score.get((i, dim))))
            for p in _PERF_COLS:
                cells.append(_fmt_perf(r["perf"].get(p),
                                       podium_perf.get((i, p))))
            cells.append(_meta_cell(str(r.get("runs", 0))))
            cells.append(Text("⚠", style="bold yellow") if r.get("stale")
                         else Text("✓", style="bold dim green"))
            # Cluster cell — bold colour-coded so single/dual stands out at a glance.
            cluster = r.get("cluster")
            if cluster == "dual":
                cells.append(Text("⚡ dual", style="bold cyan"))
            elif cluster == "single":
                cells.append(Text("• single", style="bold"))
            else:
                cells.append(Text("—", style="bold dim"))
            # height=2 → ≈1.5× visual size in the terminal.
            tbl.add_row(*cells, height=2)

    # -- click-to-sort --
    def on_data_table_header_selected(self, event) -> None:
        """User clicked a column header — sort by that column (toggle dir)."""
        column_key = (
            event.column_key.value if hasattr(event.column_key, "value")
            else str(event.column_key)
        )
        # Toggle direction when clicking the same column again.
        if self._sort_by == column_key:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_by = column_key
            self._sort_desc = True
        self._populate_rows()

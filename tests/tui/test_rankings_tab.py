from __future__ import annotations

from datetime import UTC, datetime

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Static

from toolery.core.models import Message, ScenarioResult, TraceResult
from toolery.core.store import Store
from toolery.tui.rankings_tab import (
    _HEADERS,
    _LEGEND,
    _PERF_HEADERS,
    RankingsTab,
)


def _trace(sid, adapter):
    return TraceResult(scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")], tool_calls=[],
        final_response="ok", started_at_iso="2026-05-01T00:00:00Z",
        duration_ms=10, error=None)


def _seed(store, run_id, model, adapter, cluster, scores):
    store.create_run(run_id=run_id, model=model, base_url="x",
        started_at=datetime.now(UTC).isoformat(), config_json="{}",
        scenarios_hash="h", cluster=cluster)
    store.upsert_adapter(run_id, adapter, "0.1")
    for i, s in enumerate(scores):
        sid = f"easy-{i:02d}-test"
        r = ScenarioResult(scenario_id=sid, adapter=adapter, trial_index=0,
            status="pass" if s > 0.5 else "fail", score=s, call_count=1,
            budget_max=1, latency_ms=10, failure_kind=None, checks=[], trace=_trace(sid, adapter))
        store.write_scenario_result(run_id=run_id, result=r, tags=[],
            ranking_dims=["overall"], scenario_hash="h", category="coding",
            tier="easy", trace_path="x.json")
    store.finish_run(run_id, datetime.now(UTC).isoformat(), 1.0)


def test_legend_documents_every_table_column():
    # _LEGEND is a list of (display-header, description) pairs covering every
    # column the matrix renders: meta + Overall + capability dims + perf + meta.
    legend_headers = [header for header, _ in _LEGEND]
    # No column is documented twice.
    assert len(legend_headers) == len(set(legend_headers))
    legend_set = set(legend_headers)
    # Every aggregate/capability header (including Overall) is documented.
    assert set(_HEADERS.values()) <= legend_set
    # Perf columns are documented too.
    assert set(_PERF_HEADERS.values()) <= legend_set
    # Meta columns are documented.
    assert {"#", "Model", "Adapter"} <= legend_set


def test_render_signature_ignores_score_drift_tracks_real_changes():
    """The render signature must NOT move on time-decay drift of the score
    floats — including those (even rounded) made ~2 cells cross a rounding
    boundary per 5s tick, rebuilding the table every poll and killing the row
    selection. Real data changes (a new run) surface in discrete fields."""
    base = {
        "model": "m", "adapter": "raw", "runs": 3,
        "stale": False, "cluster": "dual",
        "scores": {"overall": 0.523456, "coding": 0.811111},
        "perf": {"tg_tps": 1234.5},
        "stability": {"overall": {"mean": 0.5, "worst": 0.4}},
        "pass_counts": {"passed": 100, "total": 143},
    }
    sig_base = RankingsTab._signature_for_row(base)

    # Pure decay micro-drift — signature unchanged.
    drifted = {**base, "scores": {"overall": 0.523457, "coding": 0.811112}}
    assert RankingsTab._signature_for_row(drifted) == sig_base

    # Even a display-visible score move alone is decay/cosmetic and must NOT
    # trigger a rebuild — only discrete new-data signals do.
    visible = {**base, "scores": {"overall": 0.990000, "coding": 0.100000}}
    assert RankingsTab._signature_for_row(visible) == sig_base

    # A new run bumps the run count → signature changes.
    assert RankingsTab._signature_for_row({**base, "runs": 4}) != sig_base
    # The most-recent-run pass tally changing → signature changes.
    assert RankingsTab._signature_for_row(
        {**base, "pass_counts": {"passed": 101, "total": 143}}
    ) != sig_base
    # A fresh perf bench (non-decayed mean) → signature changes.
    assert RankingsTab._signature_for_row(
        {**base, "perf": {"tg_tps": 1300.0}}
    ) != sig_base


class _Host(App):
    def compose(self) -> ComposeResult:
        yield RankingsTab(id="rt")


@pytest.mark.asyncio
async def test_rankings_tab_renders_table_and_legend(tmp_path, monkeypatch):
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    Store(tmp_path / "runs.db").init_schema()
    app = _Host()
    async with app.run_test(size=(150, 60)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#rank-matrix")
        # The matrix flexes to fill its pane (height: 1fr) and scrolls internally.
        assert tbl.header_height == 2
        assert tbl.cell_padding == 2
        legend_body = app.query_one("#rank-legend-body", Static)
        text = str(legend_body.render())
        assert "Calibr." in text
        assert "Budget" in text
        assert "Calibrated uncertainty" in text
        # The legend now documents every column, Overall included.
        assert "Overall" in text


@pytest.mark.asyncio
async def test_meta_columns_are_frozen(tmp_path, monkeypatch):
    """The first two columns (# rank, Model) stay pinned while the rest of the
    wide matrix scrolls horizontally, so the user never loses row identity."""
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    Store(tmp_path / "runs.db").init_schema()
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#rank-matrix", DataTable)
        # Adapter (3rd column) is intentionally left scrollable.
        assert tbl.fixed_columns == 2


@pytest.mark.asyncio
async def test_row_selection_survives_reload(tmp_path, monkeypatch):
    """Selecting a model's row and then a 5s poll rebuild must keep the
    highlight on the SAME model, not snap back to the top row."""
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "ra", "alpha", "raw", "dual", [1.0, 1.0, 1.0])
    _seed(store, "rb", "bravo", "raw", "dual", [1.0, 0.0, 1.0])
    _seed(store, "rc", "charlie", "raw", "dual", [0.0, 0.0, 1.0])
    app = _Host()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        tab = app.query_one(RankingsTab)
        tbl = app.query_one("#rank-matrix", DataTable)
        assert tbl.row_count >= 3
        tbl.move_cursor(row=2, animate=False)
        await pilot.pause()
        before = tbl.coordinate_to_cell_key(tbl.cursor_coordinate).row_key.value
        # Force the rebuild path a real new run would take.
        tab._last_render_sig = None
        tab.reload()
        await pilot.pause()
        after = tbl.coordinate_to_cell_key(tbl.cursor_coordinate).row_key.value
        assert tbl.cursor_row == 2, "cursor must not snap back to the top row"
        assert after == before, "selection must follow the same model identity"


@pytest.mark.asyncio
async def test_frozen_columns_keep_zebra_not_flat_fixed(tmp_path, monkeypatch):
    """Pinned columns must look like the rest of the table: the per-row zebra
    stripe, NOT Textual's flat `datatable--fixed` highlight (which read as a
    blue block on the first two columns)."""
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    Store(tmp_path / "runs.db").init_schema()
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tbl = app.query_one("#rank-matrix", DataTable)
        flat_fixed = tbl.get_component_styles("datatable--fixed").rich_style
        for ri in (0, 1):  # an even-row and an odd-row index
            zebra = tbl._get_row_style(ri, flat_fixed)
            # Fixed columns (#=0, Model=1) render with the per-row zebra style…
            assert tbl._effective_base_style(ri, 0, flat_fixed) == zebra
            assert tbl._effective_base_style(ri, 1, flat_fixed) == zebra
            # …scrollable columns are untouched.
            assert tbl._effective_base_style(ri, 5, zebra) == zebra
        # The frozen columns are no longer the flat (blue) fixed highlight.
        assert tbl._effective_base_style(0, 0, flat_fixed) != flat_fixed


@pytest.mark.asyncio
async def test_horizontal_scroll_survives_reload(tmp_path, monkeypatch):
    """The 5s poll occasionally flips the render signature (time-decay drift)
    and triggers a rebuild. That rebuild must NOT yank the user's horizontal
    scroll back to the leftmost column."""
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    _seed(store, "r1", "mymodel", "raw", "dual", [1.0, 0.0, 1.0])
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        tab = app.query_one(RankingsTab)
        tbl = app.query_one("#rank-matrix", DataTable)
        # The matrix is far wider than the 120-col viewport, so it scrolls.
        assert tbl.virtual_size.width > tbl.size.width
        tbl.scroll_to(x=30, y=0, animate=False)
        await pilot.pause()
        target = tbl.scroll_x
        assert target > 0
        # Force the data-changed rebuild path (what the periodic signature
        # flip hits) and confirm the horizontal offset is preserved.
        tab._last_render_sig = None
        tab.reload()
        await pilot.pause()
        assert tbl.scroll_x == pytest.approx(target, abs=1.0)

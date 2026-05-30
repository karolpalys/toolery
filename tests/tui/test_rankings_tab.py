from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from toolery.core.store import Store
from toolery.tui.rankings_tab import (
    _HEADERS,
    _LEGEND,
    _PERF_HEADERS,
    RankingsTab,
)


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


def test_render_signature_ignores_decay_drift():
    """Scores are time-decayed, so a pair's float drifts by ~1e-6 every poll.

    Regression: that micro-drift must NOT change the render signature, or the
    5s poll rebuilds the table every tick and resets the user's horizontal
    scroll. A user-visible change (>= display precision) still must differ.
    """
    base = {
        "model": "m", "adapter": "raw", "runs": 3,
        "stale": False, "cluster": "dual",
        "scores": {"overall": 0.523456, "coding": 0.811111},
        "perf": {"tg_tps": 1234.5},
        "stability": {"overall": {"mean": 0.5, "worst": 0.4}},
    }
    drifted = {
        **base,
        "scores": {"overall": 0.523457, "coding": 0.811112},  # +1e-6
    }
    sig_base = RankingsTab._signature_for_row(base)
    assert RankingsTab._signature_for_row(drifted) == sig_base

    visible = {**base, "scores": {"overall": 0.530000, "coding": 0.811111}}
    assert RankingsTab._signature_for_row(visible) != sig_base


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

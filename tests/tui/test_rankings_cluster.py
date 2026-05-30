from datetime import UTC, datetime

import pytest
from textual.app import App
from textual.widgets import DataTable

from llm_test.core.models import Message, ScenarioResult, TraceResult
from llm_test.core.store import Store
from llm_test.tui.rankings_tab import RankingsTab


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


class _Host(App):
    def compose(self):
        yield RankingsTab(id="rk")


@pytest.mark.asyncio
async def test_rankings_shows_separate_rows_per_cluster(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path))
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    # All five topologies, incl. 4× quad and 8× octa.
    _seed(store, "r_single", "mymodel", "raw", "single", [1.0, 1.0])
    _seed(store, "r_dual",   "mymodel", "raw", "dual",   [0.0, 0.0])
    _seed(store, "r_triple", "mymodel", "raw", "triple", [1.0, 0.0])
    _seed(store, "r_quad",   "mymodel", "raw", "quad",   [1.0, 1.0])
    _seed(store, "r_octa",   "mymodel", "raw", "octa",   [0.0, 1.0])

    app = _Host()
    async with app.run_test(size=(220, 60)) as pilot:
        await pilot.pause()
        tab = app.query_one(RankingsTab)
        # pair mode = one row per (model, adapter, cluster)
        tab._mode = "pair"
        tab._last_render_sig = None
        tab.reload()
        await pilot.pause()
        rows = tab._rows_cache
        clusters = sorted(r["cluster"] for r in rows if r["model"] == "mymodel")
        assert clusters == ["dual", "octa", "quad", "single", "triple"], f"got {clusters}"
        assert len([r for r in rows if r["model"] == "mymodel"]) == 5

        # quad and octa render with their proper Cluster-column label (not "—").
        tbl = app.query_one("#rank-matrix", DataTable)
        col_keys = [str(c.key.value) for c in tbl.columns.values()]
        cidx = col_keys.index("cluster")
        rendered = []
        for row_key in tbl.rows:
            cell = tbl.get_row(row_key)[cidx]
            rendered.append(cell.plain if hasattr(cell, "plain") else str(cell))
        joined = " ".join(rendered)
        assert "quad" in joined and "octa" in joined, joined
        assert "—" not in joined  # every seeded row has a real topology


def test_runargs_accepts_quad_and_octa():
    """RunArgs must validate 4× quad and 8× octa, or the TUI can't launch them."""
    from llm_test.core.runner_subprocess import RunArgs
    for c in ("single", "dual", "triple", "quad", "octa"):
        args = RunArgs(model="m", base_url="x", adapter="raw",
                       trials=1, concurrency=1, cluster=c)
        assert args.cluster == c

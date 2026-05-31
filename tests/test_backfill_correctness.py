from toolery.cli import _backfill_correctness_run
from toolery.core.models import (
    Budget, Category, Message, Scenario, Scoring, ScoringCheck, Tier, ToolCall, TraceResult,
)
from toolery.core.scorer import evaluate
from toolery.core.store import Store


def _scenario():
    return Scenario(
        id="bf-01", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[ScoringCheck.model_validate(
            {"check": "tool_called", "tool": "get_weather"})]),
    )


def _budget_violated_correct_trace():
    # 2 calls > budget 1 → headline fail/budget_violated, but required passes.
    return TraceResult(
        scenario_id="bf-01", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"}),
                    ToolCall(index=1, name="get_weather", args={"location": "Warsaw"})],
        final_response="ok", started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_backfill_populates_correctness_from_trace(tmp_path):
    results_dir = tmp_path
    run_id = "2026-05-31T00-00_m"
    run_dir = results_dir / "runs" / run_id / "traces"
    run_dir.mkdir(parents=True)
    trace = _budget_violated_correct_trace()
    (run_dir / "bf-01__hermes__t0.json").write_text(trace.model_dump_json())

    store = Store(results_dir / "runs.db")
    store.init_schema()
    store.create_run(run_id=run_id, model="m", base_url="u",
                     started_at="2026-05-31T00:00:00Z", config_json="{}", scenarios_hash="h")
    res = evaluate(_scenario(), trace)
    assert res.failure_kind == "budget_violated" and res.score == 0.0
    store.write_scenario_result(run_id, res, tags=[], ranking_dims=["overall"],
                                scenario_hash="h", category="tool_selection", tier="easy",
                                trace_path="traces/bf-01__hermes__t0.json")
    with store.conn() as c:
        c.execute("UPDATE scenario_results SET correctness_score=NULL")

    scenarios = {"bf-01": _scenario()}
    updated, skipped = _backfill_correctness_run(store, run_id, results_dir, scenarios)

    assert updated == 1 and skipped == 0
    row = store.fetch_results_for_run(run_id)[0]
    assert row["correctness_score"] == 1.0  # solved, only budget failed

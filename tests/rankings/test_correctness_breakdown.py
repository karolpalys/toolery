from toolery.core.models import (
    Budget,
    Category,
    Message,
    Scenario,
    Scoring,
    ScoringCheck,
    Tier,
    ToolCall,
    TraceResult,
)
from toolery.core.scorer import evaluate
from toolery.core.store import Store
from toolery.rankings.compute import compute_correctness_breakdown


def _scenario():
    return Scenario(
        id="cb-01", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[ScoringCheck.model_validate(
            {"check": "tool_called", "tool": "get_weather"})]),
    )


def _trace(n_calls):
    return TraceResult(
        scenario_id="cb-01", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[ToolCall(index=i, name="get_weather", args={"location": "Warsaw"})
                    for i in range(n_calls)],
        final_response="ok", started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_correctness_breakdown_reports_score_and_correctness(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="u",
                     started_at="2026-05-23T18:00:00Z", config_json="{}", scenarios_hash="h")
    # budget-violated-correct: 2 calls > budget 1 → score 0, correctness 1.0
    res = evaluate(_scenario(), _trace(2))
    store.write_scenario_result("r1", res, tags=[], ranking_dims=["overall"],
                                scenario_hash="h", category="tool_selection", tier="easy",
                                trace_path="traces/x.json")

    out = compute_correctness_breakdown(store)
    row = out[("m", "hermes")]
    assert row["n"] == 1
    assert row["score_mean"] == 0.0
    assert row["correctness_mean"] == 1.0
    assert row["solved_not_scored"] == 1

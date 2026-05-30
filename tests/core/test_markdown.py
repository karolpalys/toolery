from toolery.core.markdown import render_scenario, render_summary
from toolery.core.models import CheckResult, Message, ScenarioResult, TraceResult


def _result(adapter="raw", status="pass", score=1.0):
    tr = TraceResult(
        scenario_id="easy-01-x", adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
    )
    return ScenarioResult(
        scenario_id="easy-01-x", adapter=adapter, trial_index=0,
        status=status, score=score, call_count=0, budget_max=1,
        latency_ms=42, failure_kind=None,
        checks=[CheckResult(check="tool_called", result="pass", detail="get_weather called")],
        trace=tr,
    )


def test_render_summary_has_score_table():
    results = [_result("raw"), _result("hermes", status="fail", score=0.0)]
    md = render_summary(
        run_id="2026-05-23T18-00_x", model="x", adapters=["raw", "hermes"],
        trials=1, duration_s=10.0, results=results, perf_rows=[],
    )
    assert "# Run: 2026-05-23T18-00_x" in md
    assert "raw" in md and "hermes" in md
    assert "Overall" in md


def test_render_scenario_per_adapter_block():
    results = [_result("raw"), _result("hermes")]
    md = render_scenario(scenario_id="easy-01-x", results=results, title="t",
                         tier="easy", category="tool_selection")
    assert "easy-01-x" in md
    assert "raw" in md and "hermes" in md

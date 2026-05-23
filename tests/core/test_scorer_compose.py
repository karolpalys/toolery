from llm_test.core.models import (
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
from llm_test.core.scorer import evaluate


def _scenario(scoring: Scoring) -> Scenario:
    return Scenario(
        id="test-01-x", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p",
        tools=["get_weather", "web_search"],
        budget=Budget(max_tool_calls=2, max_turns=2, timeout_seconds=30),
        scoring=scoring,
    )


def _trace(*calls, response="ok"):
    return TraceResult(
        scenario_id="test-01-x", adapter="mock", trial_index=0,
        messages=[Message(role="assistant", content=response)],
        tool_calls=[ToolCall(index=i, name=n, args=a) for i, (n, a) in enumerate(calls)],
        final_response=response,
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_evaluate_pass_all_required():
    scoring = Scoring(
        required=[
            ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"}),
            ScoringCheck.model_validate({"check": "tool_args_contain", "tool": "get_weather",
                                         "args": {"location": "Warsaw"}}),
        ],
        forbidden=[ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"})],
        partial=[],
    )
    s = _scenario(scoring)
    tr = _trace(("get_weather", {"location": "Warsaw"}))
    result = evaluate(s, tr)
    assert result.status == "pass"
    assert result.score == 1.0
    assert result.failure_kind is None


def test_evaluate_fail_on_required():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        forbidden=[], partial=[],
    )
    tr = _trace(("web_search", {"query": "warsaw weather"}))
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "fail"
    assert r.score == 0.0


def test_evaluate_fail_on_forbidden():
    scoring = Scoring(
        required=[],
        forbidden=[ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"})],
        partial=[],
    )
    tr = _trace(("web_search", {"query": "x"}))
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "fail"
    assert r.failure_kind == "forbidden_action"


def test_evaluate_partial_score():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        forbidden=[],
        partial=[
            ScoringCheck.model_validate({"check": "response_contains", "patterns": ["cloud"]}),
            ScoringCheck.model_validate({"check": "call_count_at_most", "n": 1}),
        ],
    )
    tr = _trace(("get_weather", {}), response="It's sunny.")
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "partial"
    assert 0.0 < r.score < 1.0

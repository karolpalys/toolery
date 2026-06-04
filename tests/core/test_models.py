import pytest
from pydantic import ValidationError

from toolery.core.models import (
    Budget,
    Category,
    Scenario,
    ScenarioResult,
    Tier,
    TraceResult,
    TurnUsage,
    effective_tps,
)


def _trace_with_usage(usage):
    return TraceResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response="done",
        started_at_iso="2026-06-02T00:00:00Z", duration_ms=1000,
        usage=usage,
    )


def test_traceresult_usage_defaults_empty():
    # Old trace JSON predates the usage field; it must still parse.
    t = TraceResult.model_validate_json(
        '{"scenario_id":"t-01-x","adapter":"raw","trial_index":0,'
        '"messages":[],"tool_calls":[],"started_at_iso":"x","duration_ms":5}'
    )
    assert t.usage == []
    assert t.token_totals() == (0, 0, 0)


def test_token_totals_sums_across_turns():
    t = _trace_with_usage([
        TurnUsage(turn_index=0, prompt_tokens=100, completion_tokens=20, latency_ms=400),
        TurnUsage(turn_index=1, prompt_tokens=130, completion_tokens=30, latency_ms=600),
    ])
    assert t.token_totals() == (230, 50, 1000)


def test_effective_tps_basic():
    # 50 completion tokens over 1.0s = 50 tok/s
    assert effective_tps(50, 1000) == 50.0


def test_effective_tps_zero_time_is_none():
    assert effective_tps(0, 0) is None
    assert effective_tps(10, 0) is None


def test_effective_tps_none_when_no_completion_tokens():
    # Server omitted usage but the request was still timed: 0 tokens over 400ms
    # must read as n/a (None), not 0.0.
    assert effective_tps(0, 400) is None


def test_scenario_result_token_fields_default_zero():
    r = ScenarioResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0, status="pass",
        score=1.0, call_count=0, budget_max=1, latency_ms=5, failure_kind=None,
        checks=[], trace=_trace_with_usage([]),
    )
    assert (r.prompt_tokens, r.completion_tokens, r.gen_ms) == (0, 0, 0)


def test_scenario_minimal_valid():
    s = Scenario(
        id="easy-01-test",
        title="t",
        tier=Tier.EASY,
        category=Category.TOOL_SELECTION,
        domain="generic",
        description="d",
        prompt="p",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring={"required": [], "forbidden": [], "partial": [],
                 "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
    )
    assert s.id == "easy-01-test"
    assert s.tier == Tier.EASY
    assert s.context_prefill_tokens == 0


def test_scenario_id_must_be_kebab():
    with pytest.raises(ValidationError):
        Scenario(
            id="easy 01 test",  # spaces not allowed
            title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
            domain="generic", description="d", prompt="p", tools=[],
            budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
            scoring={"required": [], "forbidden": [], "partial": [],
                     "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
        )


def test_tracerresult_roundtrip():
    tr = TraceResult(
        scenario_id="x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response=None,
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
        adapter_metadata={},
    )
    assert tr.duration_ms == 42


@pytest.mark.parametrize("bad_id", [
    "weather",         # single segment
    "EASY-01",         # uppercase
    "easy_01",         # underscore
    "1easy-01",        # leading digit
    "-easy-01",        # leading hyphen
    "easy-01-",        # trailing hyphen
    "easy--01",        # double hyphen
])
def test_scenario_id_rejects_invalid_kebab(bad_id):
    with pytest.raises(ValidationError):
        Scenario(
            id=bad_id, title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
            domain="generic", description="d", prompt="p", tools=[],
            budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
            scoring={"required": [], "forbidden": [], "partial": [],
                     "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
        )


@pytest.mark.parametrize("good_id", [
    "easy-01",                # two segments
    "easy-01-direct-weather", # multi-segment
    "very-hard-04-multi",     # tier with hyphen
])
def test_scenario_id_accepts_valid_kebab(good_id):
    s = Scenario(
        id=good_id, title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=[],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring={"required": [], "forbidden": [], "partial": [],
                 "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
    )
    assert s.id == good_id

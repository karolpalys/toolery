import pytest
from pydantic import ValidationError

from llm_test.core.models import Budget, Category, Scenario, Tier, TraceResult


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

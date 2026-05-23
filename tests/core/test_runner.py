import pytest

from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import Budget, Category, Scenario, Scoring, ScoringCheck, Tier, ToolCall
from llm_test.core.runner import Runner


def _scenario():
    return Scenario(
        id="easy-01-direct-weather", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="p", tools=["get_weather", "web_search"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring=Scoring(
            required=[
                ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"}),
            ],
            forbidden=[
                ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"}),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_runner_single_scenario_single_adapter_pass():
    plans = {"easy-01-direct-weather": ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
        final_response="ok",
    )}
    mock = MockAdapter(plans=plans)
    runner = Runner(adapters={"mock": mock}, trials=3, model="x")
    results = await runner.run([_scenario()])
    assert len(results) == 3
    assert all(r.status == "pass" for r in results)


@pytest.mark.asyncio
async def test_runner_multiple_adapters():
    s = _scenario()
    plan_good = ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="get_weather", args={})], final_response="ok"
    )
    plan_bad = ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="web_search", args={"query": "x"})], final_response="ok"
    )
    runner = Runner(
        adapters={"good": MockAdapter({s.id: plan_good}), "bad": MockAdapter({s.id: plan_bad})},
        trials=2, model="x",
    )
    results = await runner.run([s])
    assert len(results) == 4
    by_adapter = {a: [r for r in results if r.adapter == a] for a in ("good", "bad")}
    assert all(r.status == "pass" for r in by_adapter["good"])
    assert all(r.status == "fail" for r in by_adapter["bad"])

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


@pytest.mark.asyncio
async def test_runner_fires_on_start_and_on_end_callbacks():
    s = _scenario()
    plan = ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
        final_response="ok",
    )
    starts: list[tuple[str, str, int, str]] = []
    ends: list[tuple[str, str, int]] = []

    def _on_start(scenario_id, adapter_name, trial_index, started_at):
        starts.append((scenario_id, adapter_name, trial_index, started_at))

    def _on_end(scenario_id, adapter_name, trial_index):
        ends.append((scenario_id, adapter_name, trial_index))

    runner = Runner(
        adapters={"mock": MockAdapter({s.id: plan})},
        trials=2, model="x",
        on_start=_on_start, on_end=_on_end,
    )
    results = await runner.run([s])

    assert len(results) == 2
    # Each trial fires exactly one start and one end
    assert len(starts) == 2
    assert len(ends) == 2
    # Started_at is ISO and looks like a UTC timestamp
    for _, _, _, started_at in starts:
        assert "T" in started_at and ("Z" in started_at or "+00:00" in started_at)
    # Trial indices cover {0, 1} in both lists
    assert {s_[2] for s_ in starts} == {0, 1}
    assert {e[2] for e in ends} == {0, 1}

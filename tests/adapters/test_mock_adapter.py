import pytest

from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import Budget, Category, Scenario, Scoring, Tier, ToolCall


@pytest.mark.asyncio
async def test_mock_adapter_replays_plan():
    plan = ScenarioPlan(
        tool_calls=[
            ToolCall(index=0, name="get_weather", args={"location": "Warsaw"}),
        ],
        final_response="It's 7°C and cloudy.",
    )
    scenario = Scenario(
        id="easy-01-direct-weather", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring=Scoring(),
    )
    adapter = MockAdapter(plans={"easy-01-direct-weather": plan})
    trace = await adapter.run_scenario(scenario, model="mock-model", timeout=10)
    assert trace.scenario_id == "easy-01-direct-weather"
    assert trace.adapter == "mock"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.final_response == "It's 7°C and cloudy."
    assert trace.error is None

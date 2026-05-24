from pathlib import Path

import pytest

from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import ToolCall
from llm_test.core.runner import Runner
from llm_test.core.scenario import load_all_scenarios


@pytest.mark.asyncio
async def test_e2e_three_scenarios_with_mock():
    scenarios = load_all_scenarios(Path("scenarios"))
    assert len(scenarios) >= 3

    plans = {
        "easy-01-direct-weather": ScenarioPlan(
            tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
            final_response="It's 7°C and cloudy in Warsaw.",
        ),
        "easy-02-read-before-write": ScenarioPlan(
            tool_calls=[
                ToolCall(index=0, name="read_file", args={"path": "/workspace/notes.md"}),
                ToolCall(index=1, name="write_file",
                         args={"path": "/workspace/notes.md",
                               "content": "# Notes\n\nExisting content here.\nTODO: review\n"}),
            ],
            final_response="Done.",
        ),
        "easy-03-refuse-trivial-math": ScenarioPlan(
            tool_calls=[], final_response="The answer is 4."
        ),
    }
    runner = Runner(adapters={"mock": MockAdapter(plans)}, trials=1, model="mock")
    results = await runner.run([s for s in scenarios if s.id in plans])
    assert len(results) == 3
    by_id = {r.scenario_id: r for r in results}
    assert by_id["easy-01-direct-weather"].status == "pass"
    assert by_id["easy-02-read-before-write"].status == "pass"
    assert by_id["easy-03-refuse-trivial-math"].status == "pass"

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from llm_test.core.models import Message, Scenario, ToolCall, TraceResult


class Adapter(Protocol):
    name: str
    version: str

    async def run_scenario(
        self, scenario: Scenario, model: str, timeout: int
    ) -> TraceResult: ...


@dataclass
class ScenarioPlan:
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str | None = None
    error: str | None = None


class MockAdapter:
    name = "mock"
    version = "0.1"

    def __init__(self, plans: dict[str, ScenarioPlan]) -> None:
        self.plans = plans

    async def run_scenario(self, scenario, model, timeout) -> TraceResult:
        plan = self.plans.get(scenario.id, ScenarioPlan())
        started = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[
                Message(role="user", content=scenario.prompt),
                Message(role="assistant", content=plan.final_response),
            ],
            tool_calls=plan.tool_calls,
            final_response=plan.final_response,
            started_at_iso=started,
            duration_ms=1,
            error=plan.error,
            adapter_metadata={"plan": True},
        )

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC

from llm_test.adapters.base import Adapter
from llm_test.core.models import Scenario, ScenarioResult
from llm_test.core.scorer import evaluate


@dataclass
class Runner:
    adapters: dict[str, Adapter]
    trials: int = 1
    model: str = "model"
    concurrency: int = 4

    async def _run_one(self, scenario: Scenario, adapter_name: str, adapter: Adapter,
                       trial_index: int) -> ScenarioResult:
        try:
            trace = await asyncio.wait_for(
                adapter.run_scenario(scenario, self.model, scenario.budget.timeout_seconds),
                timeout=scenario.budget.timeout_seconds + 5,
            )
        except TimeoutError:
            from datetime import datetime

            from llm_test.core.models import TraceResult
            trace = TraceResult(
                scenario_id=scenario.id, adapter=adapter_name, trial_index=trial_index,
                messages=[], tool_calls=[], final_response=None,
                started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                duration_ms=scenario.budget.timeout_seconds * 1000,
                error="timeout",
            )
        trace = trace.model_copy(update={"adapter": adapter_name, "trial_index": trial_index})
        return evaluate(scenario, trace)

    async def run(self, scenarios: Iterable[Scenario]) -> list[ScenarioResult]:
        sem = asyncio.Semaphore(self.concurrency)

        async def bounded(coro):
            async with sem:
                return await coro

        tasks = []
        for s in scenarios:
            for adapter_name, adapter in self.adapters.items():
                for t in range(self.trials):
                    tasks.append(bounded(self._run_one(s, adapter_name, adapter, t)))
        return await asyncio.gather(*tasks)

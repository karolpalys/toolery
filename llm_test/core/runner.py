from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC

from llm_test.adapters.base import Adapter
from llm_test.core.models import Scenario, ScenarioResult
from llm_test.core.scorer import evaluate

ResultCallback = Callable[[ScenarioResult], Awaitable[None] | None]


@dataclass
class Runner:
    adapters: dict[str, Adapter]
    trials: int = 1
    model: str = "model"
    concurrency: int = 4
    skip: set[tuple[str, str, int]] | None = None

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

    async def run(self, scenarios: Iterable[Scenario],
                  on_result: ResultCallback | None = None) -> list[ScenarioResult]:
        """Execute all (scenario × adapter × trial) units concurrently.

        If on_result is provided, it is invoked for each ScenarioResult as soon as
        the unit completes (in completion order, not submission order). The
        callback may be sync or async. Exceptions in the callback are swallowed
        so a single bad observer cannot abort the whole run.
        """
        sem = asyncio.Semaphore(self.concurrency)

        async def bounded(coro):
            async with sem:
                return await coro

        skip = self.skip or set()
        tasks: list[asyncio.Task[ScenarioResult]] = []
        for s in scenarios:
            for adapter_name, adapter in self.adapters.items():
                for t in range(self.trials):
                    if (s.id, adapter_name, t) in skip:
                        continue
                    tasks.append(asyncio.create_task(
                        bounded(self._run_one(s, adapter_name, adapter, t))))

        results: list[ScenarioResult] = []
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
            if on_result is not None:
                try:
                    out = on_result(r)
                    if asyncio.iscoroutine(out):
                        await out
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "on_result callback failed for %s", r.scenario_id)
        return results

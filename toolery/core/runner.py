from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from toolery.adapters.base import Adapter
from toolery.core.models import Scenario, ScenarioResult
from toolery.core.scorer import evaluate

ResultCallback = Callable[[ScenarioResult], Awaitable[None] | None]
StartCallback = Callable[[str, str, int, str], Awaitable[None] | None]
EndCallback = Callable[[str, str, int], Awaitable[None] | None]

_log = logging.getLogger(__name__)

# Sentinel returned by a unit that was gated off by should_stop() before it
# started. Distinct from None (a valid-ish callback return) so the gather loop
# can tell "skipped, never ran" apart from a real ScenarioResult.
_SKIP = object()


async def _maybe_call(cb, *args) -> None:
    if cb is None:
        return
    try:
        out = cb(*args)
        if asyncio.iscoroutine(out):
            await out
    except Exception:
        _log.exception("callback %s failed", getattr(cb, "__name__", cb))


@dataclass
class Runner:
    adapters: dict[str, Adapter]
    trials: int = 1
    model: str = "model"
    concurrency: int = 4
    skip: set[tuple[str, str, int]] | None = None
    on_start: StartCallback | None = None
    on_end: EndCallback | None = None
    # Multiplier on each scenario's timeout_seconds. The budgets are tuned for
    # fast local serving; a cloud/reasoning adapter (high RTT + long CoT) gets
    # killed mid-answer at scale 1.0. Bump for remote endpoints (e.g. 4.0).
    timeout_scale: float = 1.0

    async def _run_one(self, scenario: Scenario, adapter_name: str, adapter: Adapter,
                       trial_index: int) -> ScenarioResult:
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        await _maybe_call(self.on_start, scenario.id, adapter_name, trial_index, started_at)
        eff_timeout = int(scenario.budget.timeout_seconds * self.timeout_scale)
        try:
            try:
                trace = await asyncio.wait_for(
                    adapter.run_scenario(scenario, self.model, eff_timeout),
                    timeout=eff_timeout + 5,
                )
            except TimeoutError:
                from toolery.core.models import TraceResult
                trace = TraceResult(
                    scenario_id=scenario.id, adapter=adapter_name, trial_index=trial_index,
                    messages=[], tool_calls=[], final_response=None,
                    started_at_iso=started_at,
                    duration_ms=eff_timeout * 1000,
                    error="timeout",
                )
            trace = trace.model_copy(update={"adapter": adapter_name, "trial_index": trial_index})
            return evaluate(scenario, trace)
        finally:
            await _maybe_call(self.on_end, scenario.id, adapter_name, trial_index)

    async def run(self, scenarios: Iterable[Scenario],
                  on_result: ResultCallback | None = None,
                  should_stop: Callable[[], bool] | None = None,
                  ) -> list[ScenarioResult]:
        """Execute all (scenario × adapter × trial) units concurrently.

        If on_result is provided, it is invoked for each ScenarioResult as soon as
        the unit completes (in completion order, not submission order). The
        callback may be sync or async. Exceptions in the callback are swallowed
        so a single bad observer cannot abort the whole run.

        If should_stop is provided, it is consulted after a unit acquires its
        concurrency slot but before it starts running. Once it returns True, no
        further units begin — yet units already in flight finish normally and
        record their results. This is the graceful-pause path: it drains rather
        than cancels, so a paused run can resume from the next not-yet-run unit.
        """
        sem = asyncio.Semaphore(self.concurrency)

        async def bounded(coro):
            async with sem:
                if should_stop is not None and should_stop():
                    # Stop scheduling new work: discard the not-yet-started
                    # coroutine (so on_start never fires / no in_flight row)
                    # and signal skip. In-flight units past this gate keep
                    # running to completion.
                    coro.close()
                    return _SKIP
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
            if r is _SKIP:
                continue
            results.append(r)
            await _maybe_call(on_result, r)
        return results

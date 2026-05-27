# Live Concurrency View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the post-mortem-style "Scenario Results" table with a live concurrency view showing planned trials (gray), running trials (purple), and completed trials (green/orange/red) in a single deterministic ordering matching the runner's submission order.

**Architecture:** Add a small `in_flight_units` table that the runner writes on task start and clears on task finish. The TUI reconstructs the full plan from `runs.config_json` + scenario loader, joins it with completed and in-flight rows, and renders the three-state table with smart-follow auto-scroll. Heartbeat is implicit (every `in_flight_units` write updates `runs.updated_at`); the TUI uses it for stale-run detection (auto-abort after 5 minutes of silence).

**Tech Stack:** Python 3.11+, SQLite via stdlib, Textual TUI framework, pytest + pytest-asyncio.

**Reference spec:** `docs/superpowers/specs/2026-05-27-live-concurrency-view-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `llm_test/core/store.py` | Persistence layer. Add `in_flight_units` table, `updated_at` column, 5 new methods, cleanup hooks. | Modify |
| `llm_test/core/runner.py` | Test execution engine. Add optional `on_start`/`on_end` callbacks; wrap `_run_one` to fire them around the existing logic. | Modify |
| `llm_test/cli.py` | Run orchestration. Wire `_on_start`/`_on_end` to `Store.mark_in_flight`/`clear_in_flight`; defensive cleanup at startup. | Modify |
| `llm_test/tui/home_tab.py` | Evaluation workspace. Build plan, render three-state rows, smart-follow scroll, expanded detail pane, stale-run detection. | Modify |
| `tests/core/test_store.py` | Unit tests for new Store methods. | Modify |
| `tests/core/test_runner.py` | Unit tests for callback wiring. | Modify |
| `tests/tui/test_home_tab.py` | TUI tests for plan reconstruction + three-state render. | Modify |

---

## Task 1: Store schema — `in_flight_units` table + `runs.updated_at`

**Files:**
- Modify: `llm_test/core/store.py:10-55` (SCHEMA constant + `_MIGRATIONS` list)
- Test: `tests/core/test_store.py` (new test `test_init_schema_creates_in_flight_units`)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_store.py`:

```python
def test_init_schema_creates_in_flight_units(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    with store.conn() as c:
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "in_flight_units" in tables
        cols = {r[1] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
        assert "updated_at" in cols
        cols_inflight = {r[1] for r in c.execute(
            "PRAGMA table_info(in_flight_units)"
        ).fetchall()}
        assert cols_inflight == {
            "run_id", "scenario_id", "adapter", "trial_index", "started_at"
        }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_store.py::test_init_schema_creates_in_flight_units -v`
Expected: FAIL — `in_flight_units not in tables`.

- [ ] **Step 3: Add table DDL to SCHEMA constant**

In `llm_test/core/store.py`, extend the `SCHEMA` constant (after the `perf_results` table, before the trailing index definitions):

```python
CREATE TABLE IF NOT EXISTS in_flight_units (
  run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  scenario_id TEXT NOT NULL,
  adapter     TEXT NOT NULL,
  trial_index INTEGER NOT NULL,
  started_at  TEXT NOT NULL,
  PRIMARY KEY (run_id, scenario_id, adapter, trial_index)
);
CREATE INDEX IF NOT EXISTS idx_in_flight_run ON in_flight_units(run_id);
```

- [ ] **Step 4: Add `updated_at` to `_MIGRATIONS`**

Append to `_MIGRATIONS` list in `llm_test/core/store.py:50-55`:

```python
    "ALTER TABLE runs ADD COLUMN updated_at TEXT",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/test_store.py::test_init_schema_creates_in_flight_units -v`
Expected: PASS.

- [ ] **Step 6: Run full store test suite as regression check**

Run: `pytest tests/core/test_store.py -v`
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add llm_test/core/store.py tests/core/test_store.py
git commit -m "feat(store): add in_flight_units table and runs.updated_at column"
```

---

## Task 2: Store API — `mark_in_flight`, `clear_in_flight`, `fetch_in_flight_for_run`

**Files:**
- Modify: `llm_test/core/store.py` (add 3 methods)
- Test: `tests/core/test_store.py` (new test `test_in_flight_round_trip`)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_store.py`:

```python
def test_in_flight_round_trip(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "2026-05-27T20-00_test"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")

    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
    store.mark_in_flight(run_id, "easy-01", "raw", 1, "2026-05-27T20:00:02Z")

    rows = store.fetch_in_flight_for_run(run_id)
    assert len(rows) == 2
    assert {r["trial_index"] for r in rows} == {0, 1}
    assert all(r["scenario_id"] == "easy-01" for r in rows)
    assert all(r["adapter"] == "raw" for r in rows)

    # updated_at heartbeat fired on mark_in_flight
    run = store.fetch_run(run_id)
    assert run["updated_at"] is not None

    store.clear_in_flight(run_id, "easy-01", "raw", 0)
    rows = store.fetch_in_flight_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["trial_index"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_store.py::test_in_flight_round_trip -v`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'mark_in_flight'`.

- [ ] **Step 3: Implement the three methods**

Add to `Store` class in `llm_test/core/store.py`, just below `fetch_results_for_run` (around line 168):

```python
def mark_in_flight(self, run_id: str, scenario_id: str, adapter: str,
                   trial_index: int, started_at: str) -> None:
    """Record that (scenario_id, adapter, trial_index) just entered _run_one.
    Also updates runs.updated_at as an implicit heartbeat — same transaction."""
    with self.conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO in_flight_units"
            "(run_id, scenario_id, adapter, trial_index, started_at) "
            "VALUES (?,?,?,?,?)",
            (run_id, scenario_id, adapter, trial_index, started_at),
        )
        c.execute("UPDATE runs SET updated_at=? WHERE run_id=?",
                  (started_at, run_id))

def clear_in_flight(self, run_id: str, scenario_id: str, adapter: str,
                    trial_index: int) -> None:
    """Remove the in-flight marker (task completed/failed/timed out).
    Bumps runs.updated_at in the same transaction."""
    from datetime import UTC, datetime
    now = datetime.now(UTC).isoformat()
    with self.conn() as c:
        c.execute(
            "DELETE FROM in_flight_units WHERE run_id=? AND scenario_id=? "
            "AND adapter=? AND trial_index=?",
            (run_id, scenario_id, adapter, trial_index),
        )
        c.execute("UPDATE runs SET updated_at=? WHERE run_id=?", (now, run_id))

def fetch_in_flight_for_run(self, run_id: str) -> list[dict]:
    with self.conn() as c:
        rows = c.execute(
            "SELECT * FROM in_flight_units WHERE run_id=? ORDER BY started_at",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_store.py::test_in_flight_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/store.py tests/core/test_store.py
git commit -m "feat(store): mark_in_flight/clear_in_flight/fetch_in_flight_for_run with implicit heartbeat"
```

---

## Task 3: Store API — `clear_all_in_flight` and `mark_stale_aborted`

**Files:**
- Modify: `llm_test/core/store.py` (add 2 methods)
- Test: `tests/core/test_store.py` (new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_store.py`:

```python
def test_clear_all_in_flight_removes_only_target_run(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    for rid in ("run-a", "run-b"):
        store.create_run(run_id=rid, model="m", base_url="http://x",
                         started_at="2026-05-27T20:00:00Z",
                         config_json="{}", scenarios_hash="x")
        store.mark_in_flight(rid, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
        store.mark_in_flight(rid, "easy-01", "raw", 1, "2026-05-27T20:00:02Z")

    store.clear_all_in_flight("run-a")
    assert store.fetch_in_flight_for_run("run-a") == []
    assert len(store.fetch_in_flight_for_run("run-b")) == 2


def test_mark_stale_aborted_clears_and_updates_status(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "stale-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")

    store.mark_stale_aborted(run_id)
    assert store.fetch_in_flight_for_run(run_id) == []
    assert store.fetch_run(run_id)["status"] == "aborted"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_store.py::test_clear_all_in_flight_removes_only_target_run tests/core/test_store.py::test_mark_stale_aborted_clears_and_updates_status -v`
Expected: both FAIL with `AttributeError`.

- [ ] **Step 3: Implement the two methods**

Add to `Store` class in `llm_test/core/store.py`, right below the methods added in Task 2:

```python
def clear_all_in_flight(self, run_id: str) -> None:
    """Bulk wipe of in-flight rows for a run. Used by finish_run, reopen_run,
    CLI startup defensive cleanup, and stale detection."""
    with self.conn() as c:
        c.execute("DELETE FROM in_flight_units WHERE run_id=?", (run_id,))

def mark_stale_aborted(self, run_id: str) -> None:
    """Atomically mark a run as aborted and clear any orphan in-flight rows.
    Used by the TUI when heartbeat goes silent for STALE_HEARTBEAT_SECONDS."""
    with self.conn() as c:
        c.execute("DELETE FROM in_flight_units WHERE run_id=?", (run_id,))
        c.execute(
            "UPDATE runs SET status='aborted', phase='done' WHERE run_id=?",
            (run_id,),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_store.py::test_clear_all_in_flight_removes_only_target_run tests/core/test_store.py::test_mark_stale_aborted_clears_and_updates_status -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/store.py tests/core/test_store.py
git commit -m "feat(store): clear_all_in_flight + mark_stale_aborted for cleanup paths"
```

---

## Task 4: Store — wire cleanup into `finish_run` and `reopen_run`

**Files:**
- Modify: `llm_test/core/store.py:105-142` (`finish_run` + `reopen_run`)
- Test: `tests/core/test_store.py` (new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_store.py`:

```python
def test_finish_run_clears_orphan_in_flight(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "ending-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")
    store.mark_in_flight(run_id, "easy-02", "raw", 0, "2026-05-27T20:00:02Z")

    store.finish_run(run_id, finished_at="2026-05-27T20:05:00Z",
                    duration_s=300.0, status="done")
    assert store.fetch_in_flight_for_run(run_id) == []


def test_reopen_run_clears_orphan_in_flight(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "resuming-run"
    store.create_run(run_id=run_id, model="m", base_url="http://x",
                     started_at="2026-05-27T20:00:00Z",
                     config_json="{}", scenarios_hash="x")
    store.finish_run(run_id, finished_at="2026-05-27T20:05:00Z",
                    duration_s=300.0, status="aborted")
    store.mark_in_flight(run_id, "easy-01", "raw", 0, "2026-05-27T20:00:01Z")

    store.reopen_run(run_id)
    assert store.fetch_in_flight_for_run(run_id) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/core/test_store.py::test_finish_run_clears_orphan_in_flight tests/core/test_store.py::test_reopen_run_clears_orphan_in_flight -v`
Expected: both FAIL — orphan rows remain.

- [ ] **Step 3: Modify `finish_run`**

In `llm_test/core/store.py`, replace the existing `finish_run` body to add the cleanup:

```python
def finish_run(self, run_id, finished_at, duration_s, status: str = "done") -> None:
    with self.conn() as c:
        c.execute(
            "UPDATE runs SET finished_at=?, duration_s=?, status=?, phase='done' "
            "WHERE run_id=?",
            (finished_at, duration_s, status, run_id),
        )
        c.execute("DELETE FROM in_flight_units WHERE run_id=?", (run_id,))
```

- [ ] **Step 4: Modify `reopen_run`**

In `llm_test/core/store.py`, replace the existing `reopen_run` body:

```python
def reopen_run(self, run_id: str) -> None:
    """Reset a finished/aborted run back to 'running' so resume can append more results.
    Also clears any stale in_flight_units that might survive from a crashed prior session."""
    with self.conn() as c:
        c.execute(
            "UPDATE runs SET status='running', finished_at=NULL, duration_s=NULL, "
            "phase='scenarios' WHERE run_id=?",
            (run_id,),
        )
        c.execute("DELETE FROM in_flight_units WHERE run_id=?", (run_id,))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_store.py -v`
Expected: all tests PASS, including the new ones.

- [ ] **Step 6: Commit**

```bash
git add llm_test/core/store.py tests/core/test_store.py
git commit -m "feat(store): clear orphan in_flight_units on finish_run and reopen_run"
```

---

## Task 5: Runner — optional `on_start` / `on_end` callbacks

**Files:**
- Modify: `llm_test/core/runner.py`
- Test: `tests/core/test_runner.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_runner.py::test_runner_fires_on_start_and_on_end_callbacks -v`
Expected: FAIL — `Runner.__init__() got an unexpected keyword argument 'on_start'`.

- [ ] **Step 3: Extend `Runner` dataclass and wrap `_run_one`**

Replace the contents of `llm_test/core/runner.py` with:

```python
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_test.adapters.base import Adapter
from llm_test.core.models import Scenario, ScenarioResult
from llm_test.core.scorer import evaluate

ResultCallback = Callable[[ScenarioResult], Awaitable[None] | None]
StartCallback = Callable[[str, str, int, str], Awaitable[None] | None]
EndCallback = Callable[[str, str, int], Awaitable[None] | None]

_log = logging.getLogger(__name__)


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

    async def _run_one(self, scenario: Scenario, adapter_name: str, adapter: Adapter,
                       trial_index: int) -> ScenarioResult:
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        await _maybe_call(self.on_start, scenario.id, adapter_name, trial_index, started_at)
        try:
            try:
                trace = await asyncio.wait_for(
                    adapter.run_scenario(scenario, self.model, scenario.budget.timeout_seconds),
                    timeout=scenario.budget.timeout_seconds + 5,
                )
            except TimeoutError:
                from llm_test.core.models import TraceResult
                trace = TraceResult(
                    scenario_id=scenario.id, adapter=adapter_name, trial_index=trial_index,
                    messages=[], tool_calls=[], final_response=None,
                    started_at_iso=started_at,
                    duration_ms=scenario.budget.timeout_seconds * 1000,
                    error="timeout",
                )
            trace = trace.model_copy(update={"adapter": adapter_name, "trial_index": trial_index})
            return evaluate(scenario, trace)
        finally:
            await _maybe_call(self.on_end, scenario.id, adapter_name, trial_index)

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
            await _maybe_call(on_result, r)
        return results
```

Note: `_maybe_call` now centralizes the swallow-and-log behavior that used to live inline for `on_result`.

- [ ] **Step 4: Run runner tests as regression check**

Run: `pytest tests/core/test_runner.py -v`
Expected: all tests PASS, including the new callback test.

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/runner.py tests/core/test_runner.py
git commit -m "feat(runner): optional on_start/on_end callbacks wrapping _run_one"
```

---

## Task 6: CLI — wire in-flight callbacks + defensive startup cleanup

**Files:**
- Modify: `llm_test/cli.py:152-196` (around `Runner(...)` construction)
- Test: `tests/test_cli.py` (unit test that intercepts Runner construction and verifies callbacks)

The CLI `run` command requires a live LLM endpoint, so a true end-to-end test is impractical. Instead we monkeypatch `Runner` to a no-op stub that captures the callbacks the CLI passes in, then exercise the callbacks directly against a real `Store` to verify they do the right writes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_cli_run_wires_in_flight_callbacks(tmp_path, monkeypatch):
    """Patch Runner so we can capture the on_start/on_end callbacks the CLI
    constructs, then drive them directly and assert the resulting DB state."""
    from typer.testing import CliRunner
    from llm_test.cli import app as cli_app
    from llm_test.core.store import Store
    import llm_test.cli as cli_module

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: [])

    captured: dict = {}

    class _StubRunner:
        def __init__(self, *, adapters, trials, model, concurrency,
                     on_start=None, on_end=None, **kwargs):
            captured["on_start"] = on_start
            captured["on_end"] = on_end

        async def run(self, scenarios, on_result=None):
            return []

    monkeypatch.setattr(cli_module, "Runner", _StubRunner)

    cli_runner = CliRunner()
    result = cli_runner.invoke(cli_app, [
        "run",
        "--model", "any-model",
        "--adapter", "raw",
        "--tier", "easy",
        "--trials", "1",
        "--concurrency", "1",
        "--base-url", "http://localhost:0",
    ])
    # The CLI may exit non-zero because no scenarios match — that's fine. We
    # just need to have reached the Runner-construction point.
    assert captured.get("on_start") is not None, result.output
    assert captured.get("on_end") is not None, result.output

    # Drive the callbacks directly against the real store
    db = tmp_path / "results" / "runs.db"
    store = Store(db)
    runs = store.fetch_all_runs()
    assert len(runs) >= 1
    run_id = runs[0]["run_id"]

    captured["on_start"]("easy-XX", "raw", 0, "2026-05-27T20:00:00Z")
    assert len(store.fetch_in_flight_for_run(run_id)) == 1
    captured["on_end"]("easy-XX", "raw", 0)
    assert store.fetch_in_flight_for_run(run_id) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_run_wires_in_flight_callbacks -v`
Expected: FAIL — `captured["on_start"]` is `None` (CLI doesn't pass callbacks yet).

- [ ] **Step 3: Wire callbacks in CLI**

In `llm_test/cli.py`, locate the block around line 176 where `runner = Runner(...)` is constructed (inside the `if not perf_only:` branch). Insert just before it:

```python
# Defensive cleanup at startup: if a prior crashed session left orphan in_flight
# rows for this run_id (resume path), wipe them now before any task starts.
store.clear_all_in_flight(run_id)

def _on_start(scenario_id: str, adapter_name: str, trial_index: int,
              started_at: str) -> None:
    store.mark_in_flight(run_id, scenario_id, adapter_name, trial_index, started_at)

def _on_end(scenario_id: str, adapter_name: str, trial_index: int) -> None:
    store.clear_in_flight(run_id, scenario_id, adapter_name, trial_index)
```

Then replace the existing `Runner(...)` line with:

```python
runner = Runner(
    adapters=adapters, trials=trials, model=api_model, concurrency=concurrency,
    on_start=_on_start, on_end=_on_end,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_test/cli.py tests/test_cli.py
git commit -m "feat(cli): wire in_flight callbacks + defensive startup cleanup"
```

---

## Task 7: TUI — plan reconstruction helper (`_build_plan`)

**Files:**
- Modify: `llm_test/tui/home_tab.py` (add helper + state field)
- Test: `tests/tui/test_home_tab.py` (new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/tui/test_home_tab.py`:

```python
import json as _json
from llm_test.tui.home_tab import _build_plan


def test_build_plan_orders_scenario_adapter_trial():
    config_json = _json.dumps({
        "adapter": ["raw", "hermes"],
        "trials": 3,
        "tier": "easy",
        "category": "all",
    })

    class _Stub:
        def __init__(self, sid): self.id = sid

    plan = _build_plan(
        config_json,
        scenario_ids_in_loader_order=["easy-01", "easy-02"],
    )

    # Expected: scenario-major, then adapter, then trial — matches Runner.run() order
    assert plan == [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1), ("easy-01", "raw", 2),
        ("easy-01", "hermes", 0), ("easy-01", "hermes", 1), ("easy-01", "hermes", 2),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1), ("easy-02", "raw", 2),
        ("easy-02", "hermes", 0), ("easy-02", "hermes", 1), ("easy-02", "hermes", 2),
    ]


def test_build_plan_handles_single_adapter_string():
    """If config_json has adapter as a string (legacy), still produce a plan."""
    config_json = _json.dumps({"adapter": "raw", "trials": 2, "tier": "easy",
                               "category": "all"})
    plan = _build_plan(config_json, scenario_ids_in_loader_order=["easy-01"])
    assert plan == [("easy-01", "raw", 0), ("easy-01", "raw", 1)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tui/test_home_tab.py::test_build_plan_orders_scenario_adapter_trial tests/tui/test_home_tab.py::test_build_plan_handles_single_adapter_string -v`
Expected: both FAIL — `ImportError: cannot import name '_build_plan'`.

- [ ] **Step 3: Implement `_build_plan`**

In `llm_test/tui/home_tab.py`, add this near the top of the module (after the constants block around line 56, before `_why_summary`):

```python
def _build_plan(config_json: str, *,
                scenario_ids_in_loader_order: list[str]
                ) -> list[tuple[str, str, int]]:
    """Reconstruct the full submission order Runner.run() would have used.

    Order: scenario_loader_order × config["adapter"] × range(config["trials"]).
    """
    try:
        cfg = json.loads(config_json or "{}")
    except json.JSONDecodeError:
        cfg = {}
    adapters_field = cfg.get("adapter", [])
    if isinstance(adapters_field, str):
        adapters = [adapters_field]
    else:
        adapters = list(adapters_field)
    trials = int(cfg.get("trials", 0) or 0)
    plan: list[tuple[str, str, int]] = []
    for sid in scenario_ids_in_loader_order:
        for ad in adapters:
            for t in range(trials):
                plan.append((sid, ad, t))
    return plan
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tui/test_home_tab.py -v -k _build_plan`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_test/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): _build_plan reconstructs Runner submission order"
```

---

## Task 8: TUI — three-state render with signature short-circuit

**Files:**
- Modify: `llm_test/tui/home_tab.py` (rewrite `_refresh_scenarios_table` and surrounding state fields)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Add module-level constants**

In `llm_test/tui/home_tab.py`, add near the top (right after the existing `_STATUS_ICON` block, around line 41):

```python
UPCOMING_VISIBLE = 10
FOLLOW_THRESHOLD_ROWS = 5
STALE_HEARTBEAT_SECONDS = 300
```

- [ ] **Step 2: Add helper to classify plan entries**

Add a new helper in `llm_test/tui/home_tab.py`, just below `_build_plan` from Task 7:

```python
def _classify_plan(plan: list[tuple[str, str, int]],
                   completed: dict[tuple[str, str, int], dict],
                   running: dict[tuple[str, str, int], dict],
                   upcoming_visible: int = UPCOMING_VISIBLE,
                   ) -> list[tuple[str, tuple, dict | None]]:
    """Walk the plan and tag each entry as ('done'|'running'|'upcoming', key, payload).

    Trims upcoming rows to at most `upcoming_visible` past the live edge.
    """
    rows: list[tuple[str, tuple, dict | None]] = []
    upcoming_count = 0
    for key in plan:
        if key in completed:
            rows.append(("done", key, completed[key]))
        elif key in running:
            rows.append(("running", key, running[key]))
        elif upcoming_count < upcoming_visible:
            rows.append(("upcoming", key, None))
            upcoming_count += 1
    return rows
```

- [ ] **Step 3: Write the failing test for `_classify_plan`**

Append to `tests/tui/test_home_tab.py`:

```python
from llm_test.tui.home_tab import _classify_plan


def test_classify_plan_three_states():
    plan = [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1), ("easy-01", "raw", 2),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1),
        ("easy-03", "raw", 0), ("easy-03", "raw", 1),
    ]
    completed = {
        ("easy-01", "raw", 0): {"status": "pass"},
        ("easy-01", "raw", 1): {"status": "fail"},
    }
    running = {
        ("easy-01", "raw", 2): {"started_at": "2026-05-27T20:00:00Z"},
        ("easy-02", "raw", 0): {"started_at": "2026-05-27T20:00:01Z"},
    }
    rows = _classify_plan(plan, completed, running, upcoming_visible=3)
    states = [r[0] for r in rows]
    assert states == ["done", "done", "running", "running", "upcoming", "upcoming", "upcoming"]


def test_classify_plan_caps_upcoming():
    plan = [(f"easy-{i:02d}", "raw", 0) for i in range(50)]
    rows = _classify_plan(plan, completed={}, running={}, upcoming_visible=5)
    assert len(rows) == 5
    assert all(r[0] == "upcoming" for r in rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tui/test_home_tab.py -v -k _classify_plan`
Expected: PASS.

- [ ] **Step 5: Rewire `_refresh_scenarios_table` to use plan-based rendering**

In `llm_test/tui/home_tab.py`, replace the entire body of `_refresh_scenarios_table` (around `home_tab.py:602-636`) with:

```python
def _refresh_scenarios_table(self) -> None:
    tbl = self.query_one("#scenarios-table", DataTable)

    # Resolve plan once per run
    if self._displayed_run_id != self._current_run_id:
        tbl.clear()
        self._displayed_run_id = self._current_run_id
        self._plan = self._resolve_plan_for_current_run()
        self._last_signature = None

    completed = {
        (r["scenario_id"], r["adapter"], r["trial_index"]): r
        for r in self._results_cache
    }
    running = self._fetch_running_units()
    rows = _classify_plan(self._plan, completed, running)

    signature = (len(completed), frozenset(running.keys()),
                 sum(1 for r in rows if r[0] == "upcoming"))
    if signature == self._last_signature:
        return
    self._last_signature = signature

    tbl.clear()
    for state, key, payload in rows:
        tbl.add_row(*self._format_row(state, key, payload))
    self._maybe_autoscroll()
```

Replace the helper instance fields by extending `HomeTab.__init__` (or whatever initializer pattern exists; if there's no `__init__`, add one). Within the existing class body, locate where `_displayed_run_id`, `_displayed_count`, `_results_cache` are declared and replace them with:

```python
self._displayed_run_id: str | None = None
self._results_cache: list[dict] = []
self._plan: list[tuple[str, str, int]] = []
self._last_signature: tuple | None = None
self._follow_mode: bool = True
```

Add the helper methods to the same class:

```python
def _resolve_plan_for_current_run(self) -> list[tuple[str, str, int]]:
    store = self._resolve_store()
    if store is None or self._current_run_id is None:
        return []
    run = store.fetch_run(self._current_run_id)
    if not run:
        return []
    return self._plan_from_config(run.get("config_json") or "{}")

def _plan_from_config(self, config_json: str) -> list[tuple[str, str, int]]:
    """Load scenarios using the same filtering CLI uses, then reconstruct the plan."""
    from llm_test.core.scenario import load_all_scenarios
    try:
        cfg = json.loads(config_json or "{}")
    except json.JSONDecodeError:
        cfg = {}
    tier = cfg.get("tier", "all")
    category = cfg.get("category", "all")
    scenarios_dir = Path(os.environ.get("LLM_TEST_SCENARIOS_DIR", "scenarios"))
    try:
        xs = load_all_scenarios(scenarios_dir)
    except Exception:
        return []
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    if category != "all":
        xs = [s for s in xs if s.category.value == category]
    return _build_plan(config_json, scenario_ids_in_loader_order=[s.id for s in xs])

def _fetch_running_units(self) -> dict[tuple[str, str, int], dict]:
    store = self._resolve_store()
    if store is None or self._current_run_id is None:
        return {}
    return {
        (r["scenario_id"], r["adapter"], r["trial_index"]): r
        for r in store.fetch_in_flight_for_run(self._current_run_id)
    }
```

`_format_row` and `_maybe_autoscroll` are stubs for now — implement them as:

```python
def _format_row(self, state: str, key: tuple[str, str, int],
                payload: dict | None) -> tuple:
    """Stub — Task 9 fills this in with proper colors."""
    scenario_id, adapter, trial_index = key
    return (
        Text(state),
        Text(scenario_id),
        Text("?"),                       # tier — filled by Task 9
        Text(adapter),
        Text(str(trial_index)),
        Text(f"{(payload or {}).get('score') or 0.0:.2f}"),
        Text(state),
        Text("—"),
    )

def _maybe_autoscroll(self) -> None:
    """Stub — Task 10 fills this in."""
    pass
```

- [ ] **Step 6: Write integration test for the refresh**

Append to `tests/tui/test_home_tab.py`:

```python
@pytest.mark.asyncio
async def test_home_tab_renders_three_states(tmp_path):
    """End-to-end-ish: mount HomeTab with a stub store; verify the table renders
    done/running/upcoming rows in plan order without duplicates."""
    from llm_test.core.store import Store

    db = tmp_path / "runs.db"
    store = Store(db)
    store.init_schema()
    run_id = "test-run"
    store.create_run(
        run_id=run_id, model="m", base_url="http://x",
        started_at="2026-05-27T20:00:00Z",
        config_json=_json.dumps({"adapter": ["raw"], "trials": 3,
                                  "tier": "easy", "category": "all"}),
        scenarios_hash="x", total_units=15,
    )
    # We can't run the full scenario loader here in isolation; this test mostly
    # asserts the helpers compose without raising. UI smoke is verified manually.
    plan = _build_plan(
        _json.dumps({"adapter": ["raw"], "trials": 2,
                     "tier": "easy", "category": "all"}),
        scenario_ids_in_loader_order=["easy-01", "easy-02"],
    )
    running = {("easy-01", "raw", 1): {"started_at": "2026-05-27T20:00:01Z"}}
    completed = {("easy-01", "raw", 0): {"status": "pass", "score": 1.0,
                                          "scenario_id": "easy-01", "adapter": "raw",
                                          "trial_index": 0}}
    rows = _classify_plan(plan, completed, running, upcoming_visible=10)
    states = [r[0] for r in rows]
    assert states == ["done", "running", "upcoming", "upcoming"]
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/tui/test_home_tab.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add llm_test/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): rewrite scenarios table render to three-state plan-based"
```

---

## Task 9: TUI — row styling (purple/gray + status colors + emojis)

**Files:**
- Modify: `llm_test/tui/home_tab.py` (`_format_row` and styling tables)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Add full styling tables**

In `llm_test/tui/home_tab.py`, replace `_STATUS_STYLE` and `_STATUS_ICON` blocks (around lines 27-40) with:

```python
_STATUS_STYLE = {
    "pass": "green",
    "partial": "orange3",
    "fail": "red",
    "error": "bold red",
    "timeout": "red dim",
    "running": "magenta",
    "upcoming": "grey50",
}
_STATUS_ICON = {
    "pass": "✅",
    "partial": "⚠",
    "fail": "❌",
    "error": "💥",
    "timeout": "⏱",
    "running": "⟳",
    "upcoming": "⌛",
}
_STATUS_DISPLAY = {
    "pass": "✅ pass",
    "partial": "⚠ partial",
    "fail": "❌ fail",
    "error": "💥 error",
    "timeout": "⏱ timeout",
    "running": "⟳ running",
    "upcoming": "⌛ upcoming",
}
```

- [ ] **Step 2: Replace `_format_row` stub with the real implementation**

In `llm_test/tui/home_tab.py`, replace the `_format_row` stub from Task 8 with:

```python
def _format_row(self, state: str, key: tuple[str, str, int],
                payload: dict | None) -> tuple:
    scenario_id, adapter, trial_index = key
    if state == "done":
        status = payload.get("status") or "?"
        style = _STATUS_STYLE.get(status, "bold")
        display = _STATUS_DISPLAY.get(status, status)
        why = _why_summary(status, payload.get("failure_kind"),
                           payload.get("checks_json"))
        tier = payload.get("tier") or "?"
        score = f"{payload.get('score') or 0.0:.2f}"
    elif state == "running":
        style = _STATUS_STYLE["running"]
        display = _STATUS_DISPLAY["running"]
        why = "—"
        tier = "—"
        score = "—"
    else:  # upcoming
        style = _STATUS_STYLE["upcoming"]
        display = _STATUS_DISPLAY["upcoming"]
        why = "—"
        tier = "—"
        score = "—"
    idx = len(self.query_one("#scenarios-table", DataTable).rows) + 1
    return (
        Text(str(idx), style=style),
        Text(scenario_id, style=style),
        Text(str(tier), style=style),
        Text(adapter, style=style),
        Text(str(trial_index), style=style),
        Text(score, style=style),
        Text(display, style=style),
        Text(why, style=style),
    )
```

- [ ] **Step 3: Verify it still renders**

Run: `pytest tests/tui/test_home_tab.py -v`
Expected: PASS.

- [ ] **Step 4: Manual smoke**

Launch the TUI with a small run and confirm purple/gray/colored rows render correctly. If you cannot launch live, defer to Task 12's manual verification step.

- [ ] **Step 5: Commit**

```bash
git add llm_test/tui/home_tab.py
git commit -m "feat(tui): purple/gray styling for running/upcoming rows"
```

---

## Task 10: TUI — smart-follow auto-scroll

**Files:**
- Modify: `llm_test/tui/home_tab.py` (`_maybe_autoscroll` + scroll listeners)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Implement `_maybe_autoscroll` and helpers**

Replace the `_maybe_autoscroll` stub from Task 8 with this real implementation. Also add the two helpers it depends on. In `llm_test/tui/home_tab.py`, inside the `HomeTab` class:

```python
def _maybe_autoscroll(self) -> None:
    if not self._follow_mode:
        return
    tbl = self.query_one("#scenarios-table", DataTable)
    if tbl.row_count == 0:
        return
    live_edge = self._find_live_edge_index(tbl)
    # Keep ~FOLLOW_THRESHOLD_ROWS upcoming visible below the live edge
    target = min(live_edge + FOLLOW_THRESHOLD_ROWS, tbl.row_count - 1)
    try:
        tbl.move_cursor(row=target, animate=False)
    except Exception:
        # Some Textual versions raise on move_cursor with no prior cursor; ignore.
        pass

def _find_live_edge_index(self, tbl: DataTable) -> int:
    """Last running row index, or last done row index if no running rows."""
    last_done = -1
    last_running = -1
    for i, row_key in enumerate(tbl.rows):
        row_text = str(tbl.get_row_at(i)[6])  # status column index
        if "running" in row_text:
            last_running = i
        elif "upcoming" not in row_text:
            last_done = i
    return last_running if last_running >= 0 else max(last_done, 0)

def _cursor_near_bottom(self, tbl: DataTable) -> bool:
    if tbl.cursor_row is None:
        return True
    return (tbl.row_count - tbl.cursor_row) <= FOLLOW_THRESHOLD_ROWS
```

- [ ] **Step 2: Add a scroll-event handler that toggles follow mode**

Append inside the `HomeTab` class:

```python
@on(DataTable.RowHighlighted, "#scenarios-table")
def _on_scenario_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
    tbl = self.query_one("#scenarios-table", DataTable)
    # If user navigated near bottom → resume follow. If they scrolled up far → release.
    self._follow_mode = self._cursor_near_bottom(tbl)
```

- [ ] **Step 3: Write a unit test for `_find_live_edge_index` semantics via `_classify_plan`**

Append to `tests/tui/test_home_tab.py`:

```python
def test_classify_plan_running_after_done_marks_live_edge():
    """The live edge is the last running entry in the rendered rows."""
    plan = [
        ("easy-01", "raw", 0), ("easy-01", "raw", 1),
        ("easy-02", "raw", 0), ("easy-02", "raw", 1),
    ]
    completed = {("easy-01", "raw", 0): {"status": "pass"}}
    running = {("easy-01", "raw", 1): {}, ("easy-02", "raw", 0): {}}
    rows = _classify_plan(plan, completed, running, upcoming_visible=10)

    # Index of last 'running' state in rows:
    last_running = max(i for i, r in enumerate(rows) if r[0] == "running")
    assert last_running == 2
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/tui/test_home_tab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_test/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): smart-follow auto-scroll keyed to live edge"
```

---

## Task 11: TUI — detail pane for running / upcoming states

**Files:**
- Modify: `llm_test/tui/home_tab.py` (`_detail_block` + `_on_scenario_selected`)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/tui/test_home_tab.py`:

```python
from llm_test.tui.home_tab import _detail_block_running, _detail_block_upcoming


def test_detail_block_running_includes_elapsed():
    from datetime import UTC, datetime, timedelta
    started = (datetime.now(UTC) - timedelta(seconds=14)).isoformat().replace("+00:00", "Z")
    block = _detail_block_running(
        scenario_id="easy-01", adapter="raw", trial=2,
        started_at=started,
    )
    text = str(block)
    assert "running" in text
    assert "easy-01" in text
    # Elapsed parses to a small mm:ss number
    assert ":" in text


def test_detail_block_upcoming_includes_position():
    block = _detail_block_upcoming(
        scenario_id="easy-09", adapter="raw", trial=0,
        position_in_queue=12,
    )
    text = str(block)
    assert "upcoming" in text
    assert "12" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tui/test_home_tab.py -v -k detail_block`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement the two helpers**

In `llm_test/tui/home_tab.py`, add right below the existing `_detail_block` function:

```python
def _detail_block_running(scenario_id: str, adapter: str, trial: int,
                          started_at: str) -> Text:
    from datetime import UTC, datetime
    out = Text()
    out.append(f"{scenario_id}\n", style="bold")
    out.append(f"adapter: {adapter}  trial: {trial}\n", style="dim")
    out.append("status: ⟳ running\n", style="magenta")
    out.append(f"started: {started_at}\n", style="dim")
    try:
        started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        elapsed = datetime.now(UTC) - started_dt
        secs = int(elapsed.total_seconds())
        mm, ss = divmod(max(secs, 0), 60)
        out.append(f"elapsed: {mm:02d}:{ss:02d}\n", style="dim")
    except Exception:
        out.append("elapsed: ?\n", style="dim")
    return out


def _detail_block_upcoming(scenario_id: str, adapter: str, trial: int,
                           position_in_queue: int) -> Text:
    out = Text()
    out.append(f"{scenario_id}\n", style="bold")
    out.append(f"adapter: {adapter}  trial: {trial}\n", style="dim")
    out.append("status: ⌛ upcoming\n", style="grey50")
    out.append(f"position in queue: {position_in_queue}\n", style="dim")
    return out
```

- [ ] **Step 4: Wire the new helpers into `_on_scenario_selected`**

In `llm_test/tui/home_tab.py`, replace the body of `_on_scenario_selected` (around line 463) with logic that branches by state. Since the row index now refers to mixed-state rendered rows (not just completed results), the method needs to consult `self._plan` and current state:

```python
def _on_scenario_selected(self, idx: int | None) -> None:
    if idx is None:
        return
    completed = {
        (r["scenario_id"], r["adapter"], r["trial_index"]): r
        for r in self._results_cache
    }
    running = self._fetch_running_units()
    rows = _classify_plan(self._plan, completed, running)
    if idx >= len(rows):
        return
    state, key, payload = rows[idx]
    scenario_id, adapter, trial_index = key
    if state == "done":
        block = _detail_block(
            scenario_id=scenario_id, adapter=adapter, trial=trial_index,
            status=payload.get("status") or "?",
            failure_kind=payload.get("failure_kind"),
            latency_ms=payload.get("latency_ms"),
            call_count=payload.get("call_count"),
            budget_max=payload.get("budget_max"),
            trace_path=payload.get("trace_path"),
            checks_json=payload.get("checks_json"),
        )
    elif state == "running":
        block = _detail_block_running(
            scenario_id=scenario_id, adapter=adapter, trial=trial_index,
            started_at=payload.get("started_at") or "?",
        )
    else:  # upcoming
        # position_in_queue = how many items before this one are still upcoming or
        # waiting. We approximate as: index of this entry in self._plan minus
        # (completed + running) count.
        plan_index = self._plan.index(key)
        position = max(plan_index - (len(completed) + len(running)), 0)
        block = _detail_block_upcoming(
            scenario_id=scenario_id, adapter=adapter, trial=trial_index,
            position_in_queue=position,
        )
    self.query_one("#detail-content", Static).update(block)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/tui/test_home_tab.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add llm_test/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): detail pane for running and upcoming states"
```

---

## Task 12: TUI — stale-run detection (heartbeat watchdog)

**Files:**
- Modify: `llm_test/tui/home_tab.py` (extend `refresh_from_db`)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/tui/test_home_tab.py`:

```python
from llm_test.tui.home_tab import _is_stale_run, STALE_HEARTBEAT_SECONDS


def test_is_stale_run_when_updated_at_old():
    from datetime import UTC, datetime, timedelta
    old = (datetime.now(UTC) - timedelta(seconds=STALE_HEARTBEAT_SECONDS + 30)
           ).isoformat().replace("+00:00", "Z")
    run = {"status": "running", "updated_at": old}
    assert _is_stale_run(run) is True


def test_is_stale_run_when_fresh():
    from datetime import UTC, datetime, timedelta
    fresh = (datetime.now(UTC) - timedelta(seconds=10)
             ).isoformat().replace("+00:00", "Z")
    run = {"status": "running", "updated_at": fresh}
    assert _is_stale_run(run) is False


def test_is_stale_run_skips_old_runs_without_updated_at():
    run = {"status": "running", "updated_at": None}
    assert _is_stale_run(run) is False


def test_is_stale_run_skips_finished_runs():
    run = {"status": "done", "updated_at": None}
    assert _is_stale_run(run) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tui/test_home_tab.py -v -k _is_stale_run`
Expected: FAIL — `_is_stale_run` not defined.

- [ ] **Step 3: Implement `_is_stale_run`**

In `llm_test/tui/home_tab.py`, add as a module-level helper just below the constants block:

```python
def _is_stale_run(run: dict) -> bool:
    """A run is stale if status='running' and updated_at is older than
    STALE_HEARTBEAT_SECONDS. Old runs without updated_at are never stale."""
    from datetime import UTC, datetime
    if run.get("status") != "running":
        return False
    updated_at = run.get("updated_at")
    if not updated_at:
        return False
    try:
        updated_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (datetime.now(UTC) - updated_dt).total_seconds() > STALE_HEARTBEAT_SECONDS
```

- [ ] **Step 4: Wire the watchdog into `refresh_from_db`**

In `llm_test/tui/home_tab.py`, inside `refresh_from_db`, just after the existing `current = next((r for r in runs if r.get("status") == "running"), None)` block (around line 504), add:

```python
if current is not None and _is_stale_run(current):
    store.mark_stale_aborted(current["run_id"])
    current = None
```

This forces the TUI to treat a silent run as aborted within ~2 refresh ticks of the timeout.

- [ ] **Step 5: Run tests**

Run: `pytest tests/tui/test_home_tab.py -v`
Expected: all PASS.

- [ ] **Step 6: Manual end-to-end verification**

Launch the TUI and start a small run (e.g., `--tier easy --trials 2 --concurrency 4`). Confirm:
- 4 purple rows appear during the run.
- Gray "upcoming" rows below them, capped at 10.
- Rows turn green/orange/red as trials complete, in plan order.
- Selecting a purple row shows elapsed timer.
- Selecting a gray row shows position-in-queue.
- Scrolling up freezes the view; scrolling back near the bottom resumes follow.

- [ ] **Step 7: Commit**

```bash
git add llm_test/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): stale-run detection auto-aborts silent runs after 5 minutes"
```

---

## Self-review checklist (run AFTER all tasks complete)

- [ ] Full test suite passes: `pytest -v`
- [ ] No `_displayed_count` references remain in `home_tab.py` (we replaced append-only logic)
- [ ] `git log feat/ranking-balance` shows 12 atomic commits matching the task IDs
- [ ] Manual smoke run: purple → green/red transitions visible, smart-follow works, 10 upcoming gray rows persist
- [ ] Old runs (before this PR) still render — no crashes when `updated_at IS NULL` or `in_flight_units` empty

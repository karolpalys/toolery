# Live concurrency view for Evaluation workspace — design

**Status:** approved (design phase)
**Date:** 2026-05-27
**Author:** brainstorm session (rahueme + Claude)

## Problem

The current "Scenario Results" table inside the Evaluation workspace renders only completed trials and only in DB insertion order. During a live run this produces three usability gaps:

1. The order is **completion order** (effectively random due to LLM latency variance and `asyncio.as_completed`). The reader cannot tell whether scenarios are being processed sequentially or how concurrency is behaving.
2. There is **no signal of work-in-progress**. A trial is either invisible or a finalized row — no intermediate state.
3. There is **no visibility of the plan**. The user cannot tell how much remains, what's next, or which scenarios are queued.

The view essentially behaves like a post-mortem log streamed live, instead of a window into the running test.

## Goal

Turn the table into a live concurrency view that shows the planned units, the currently running units, and the completed units, in a single deterministic ordering. The reader should be able to glance at the panel and immediately see:

- where the run currently is,
- how many slots are concurrently active,
- which trials passed/failed/erred,
- what's coming next.

Out of scope:
- Per-trial real-time progress bars (no per-token streaming).
- Rearranging the top-level workspace layout (`Endpoints`, `Run status`, `Selected Scenario` panes stay as they are).
- Modifying ranking / scoring / persistence semantics.

## Decisions (locked in during brainstorm)

- **Approach B** chosen: precise in-flight tracking via a dedicated DB table written by the runner. Rejected alternatives: stateless inference from completed set (Approach A — too approximate, no timer), pre-populating `scenario_results` with `pending` rows (Approach C — schema migration of NOT NULL columns too invasive).
- **Smart-follow auto-scroll** (chat-app behavior). Auto-follow live edge when cursor is near bottom; release when user scrolls up; resume when user scrolls back to bottom.
- **Static ⟳ glyph** for running rows (no animated spinner — avoids per-tick re-render overhead).
- **Heartbeat-based stale detection** kept in scope, motivated by prior hard-hang incidents on the spark1 node during `with_perf` runs.
- **Old runs (no `in_flight_units` data)** are shown without purple rows. No fallback inference path.
- **Single ordering**: `(scenario_loader_order, adapter_order_from_config, trial_index)` — matches runner submission order in `Runner.run`.

## Architecture

### Data model

New table:

```sql
CREATE TABLE IF NOT EXISTS in_flight_units (
  run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  scenario_id TEXT NOT NULL,
  adapter     TEXT NOT NULL,
  trial_index INTEGER NOT NULL,
  started_at  TEXT NOT NULL,                          -- ISO-8601 UTC
  PRIMARY KEY (run_id, scenario_id, adapter, trial_index)
);
CREATE INDEX IF NOT EXISTS idx_in_flight_run ON in_flight_units(run_id);
```

Schema addition to `runs`:

```sql
ALTER TABLE runs ADD COLUMN updated_at TEXT;          -- last heartbeat, NULL for old rows
```

Migration: both statements are idempotent (`CREATE TABLE IF NOT EXISTS`, `ALTER TABLE ADD COLUMN`). Old runs remain readable; `updated_at IS NULL` means "stale detection skipped for this run".

### Runner instrumentation (`llm_test/core/runner.py`)

`Runner` dataclass gains two optional callbacks:

```python
on_start: Callable[[str, str, int, str], Awaitable[None] | None] | None = None
on_end:   Callable[[str, str, int], Awaitable[None] | None] | None = None
```

`_run_one` is wrapped:

```python
async def _run_one(self, scenario, adapter_name, adapter, trial_index):
    started_at = datetime.now(UTC).isoformat()
    await _maybe_call(self.on_start, scenario.id, adapter_name, trial_index, started_at)
    try:
        # existing logic: wait_for + adapter.run_scenario + evaluate
        ...
        return evaluate(scenario, trace)
    finally:
        await _maybe_call(self.on_end, scenario.id, adapter_name, trial_index)
```

`_maybe_call` swallows callback exceptions and logs them via the existing logger pattern (`on_result` callback already does this).

Callbacks are optional. `Runner` with no callbacks behaves exactly like today — keeps unit tests untouched.

### Store API additions (`llm_test/core/store.py`)

```python
def mark_in_flight(self, run_id, scenario_id, adapter, trial_index, started_at): ...
def clear_in_flight(self, run_id, scenario_id, adapter, trial_index): ...
def fetch_in_flight_for_run(self, run_id) -> list[dict]: ...
def clear_all_in_flight(self, run_id): ...           # used by finish_run, reopen_run, CLI startup
def mark_stale_aborted(self, run_id): ...            # status='aborted' + clear_all_in_flight, one txn
```

Heartbeat is **implicit**: `mark_in_flight` and `clear_in_flight` each `UPDATE runs SET updated_at = <now>` within the same transaction. No separate `heartbeat()` call needed at the call-site — every task boundary touches `updated_at` for free.

`finish_run` and `reopen_run` both call `clear_all_in_flight(run_id)` for orphan cleanup.

### CLI wiring (`llm_test/cli.py`)

```python
def _on_start(scenario_id, adapter_name, trial_index, started_at):
    store.mark_in_flight(run_id, scenario_id, adapter_name, trial_index, started_at)

def _on_end(scenario_id, adapter_name, trial_index):
    store.clear_in_flight(run_id, scenario_id, adapter_name, trial_index)

runner = Runner(adapters=adapters, trials=trials, model=api_model,
                concurrency=concurrency, on_start=_on_start, on_end=_on_end)
```

(`updated_at` is updated implicitly inside `mark_in_flight` / `clear_in_flight`.)

Defensive cleanup at CLI startup for the same `run_id` (covers resume path):
```python
store.clear_all_in_flight(run_id)
```

### TUI render (`llm_test/tui/home_tab.py`)

**Plan reconstruction**, cached per `run_id`:

```python
def _build_plan(self, run_id) -> list[tuple[str, str, int]]:
    cfg = json.loads(self._current_run_meta["config_json"])
    scenarios = load_scenarios(tier=cfg["tier"], category=cfg.get("category", "all"))
    plan = []
    for s in scenarios:                  # loader-deterministic order
        for adapter in cfg["adapter"]:   # config-declared order
            for t in range(cfg["trials"]):
                plan.append((s.id, adapter, t))
    return plan
```

Cache invalidated on `_current_run_id` change.

**Refresh tick** (`set_interval(2.0, ...)` stays):

```python
def _refresh_scenarios_table(self):
    completed = {(r["scenario_id"], r["adapter"], r["trial_index"]): r
                 for r in self._results_cache}
    running   = {(r["scenario_id"], r["adapter"], r["trial_index"]): r
                 for r in store.fetch_in_flight_for_run(self._current_run_id)}

    rows = []
    upcoming_count = 0
    for key in self._plan:
        if key in completed:
            rows.append(("done", completed[key]))
        elif key in running:
            rows.append(("running", running[key]))
        elif upcoming_count < UPCOMING_VISIBLE:   # constant = 10
            rows.append(("upcoming", key))
            upcoming_count += 1
        # else: not rendered — beyond the upcoming window

    signature = self._signature_for(rows)
    if signature == self._last_signature:
        return                                    # no-op tick, skip rebuild
    self._rebuild_table(rows)
    self._last_signature = signature
    self._maybe_autoscroll()
```

`_signature_for` hashes `(len(completed), frozenset(running keys), upcoming_count)` — cheap, stable, detects all meaningful state changes.

**Row styling**:

| Status      | Glyph | Style                    |
|-------------|-------|--------------------------|
| pass        | ✅    | `green`                  |
| partial     | ⚠     | `orange3`                |
| fail        | ❌    | `red`                    |
| error       | 💥    | `red bold`               |
| timeout     | ⏱     | `red dim`                |
| **running** | ⟳     | `magenta` (purple)       |
| **upcoming**| ⌛    | `grey50`                 |

Score / Latency / Why are `—` for running/upcoming.

**Smart-follow scroll**:

```python
def _maybe_autoscroll(self):
    if not self._follow_mode:
        return
    live_edge = self._find_live_edge()           # last running row, or last completed if no running
    target = min(live_edge + 5, self._row_count - 1)  # keep ~5 upcoming visible below
    self._table.scroll_to_row(target, animate=False)

def _on_scroll_changed(self):
    if self._cursor_far_from_bottom():
        self._follow_mode = False
    elif self._cursor_at_bottom():
        self._follow_mode = True
```

Constants at `home_tab.py` module top: `UPCOMING_VISIBLE = 10`, `FOLLOW_THRESHOLD_ROWS = 5`, `STALE_HEARTBEAT_SECONDS = 300`.

### Detail pane

Existing `_detail_block` extended with two new states:

**Running**:
```
<scenario_id>
─────────────
adapter: <a>   trial: <t>
status:  ⟳ running
started: <iso UTC>
elapsed: <mm:ss>          ← computed at render time
```

**Upcoming**:
```
<scenario_id>
─────────────
adapter: <a>   trial: <t>
status: ⌛ upcoming
position in queue: <N>    ← index in self._plan minus (completed + running)
```

Trace / Checks sections are conditionally suppressed when the row is not `done`.

### Failure modes

| Scenario                         | Handling                                                     |
|----------------------------------|---------------------------------------------------------------|
| CLI subprocess SIGKILL / OOM     | Stale detection: TUI sees `runs.status='running'` and `now - updated_at > STALE_HEARTBEAT_SECONDS` (constant, default 300s = 5 min). Calls `mark_stale_aborted(run_id)` which atomically clears in-flight and sets `status='aborted'`. 300s is comfortably above any single scenario timeout in the suite (max is currently ~120s) but tight enough that user sees the recovery within a couple of TUI refresh ticks. |
| Graceful abort from TUI          | SIGTERM → runner finally-block → `finish_run(status='aborted')` → `clear_all_in_flight(run_id)`. |
| Resume (`reopen_run`)            | CLI startup calls `clear_all_in_flight(run_id)` before any task starts. |
| Old runs (no `updated_at`, no `in_flight_units`) | Rendered without purple rows. Stale detection skipped. View degrades to completed + upcoming only. |
| Multiple concurrent runs (cluster mode) | Isolated by `run_id` PK. No interference. |
| Trial timeout                    | `_run_one` reaches finally-block normally → `on_end` clears in-flight row. No special path. |
| Callback DB write fails          | `_maybe_call` swallows + logs. Trial proceeds; row may leak as orphan and be cleaned by stale detection or next `finish_run`. |

## Files changed

| File                              | Change                                                         |
|-----------------------------------|----------------------------------------------------------------|
| `llm_test/core/store.py`          | New table DDL, new methods, `updated_at` column on `runs`, cleanup hooks in `finish_run`/`reopen_run`. |
| `llm_test/core/runner.py`         | Add `on_start`/`on_end` callbacks, wrap `_run_one`.            |
| `llm_test/cli.py`                 | Wire `_on_start`/`_on_end`, defensive cleanup at startup.      |
| `llm_test/tui/home_tab.py`        | Plan reconstruction, three-source render, smart-follow scroll, detail-pane states for running/upcoming. |

## Testing strategy

- **Unit tests** for new `Store` methods: insert/delete/fetch round-trip, cleanup-on-finish, heartbeat update.
- **Runner test** with synthetic adapter that signals start/end ordering: verifies callbacks fire in the expected sequence with concurrency=2.
- **TUI snapshot test** (Textual snapshot pytest plugin if available, otherwise asserting `_build_plan` + `_refresh_scenarios_table` output shapes) covering three table states: only-completed, mixed, all-upcoming.
- **Manual verification** on a real run: start a 5-trial × 4-concurrency run, confirm 4 purple rows appear, 10 gray rows follow, completed rows accumulate above, smart-follow keeps live edge visible.

## Open questions

None — all design decisions resolved during brainstorm.

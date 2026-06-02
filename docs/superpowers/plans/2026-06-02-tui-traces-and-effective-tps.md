# TUI Tool-Call Traces + Effective tokens/s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-run tool-call traces readable in the TUI (compact inline + full-screen modal), and capture an effective tokens/s number from the tool-test requests so a run has a throughput figure even when `llama-benchy` is skipped.

**Architecture:** Token usage is captured per assistant turn in the OpenAI-compatible HTTP adapter (`OpenAIRawAdapter`; `CloudAdapter` subclasses it, so it inherits the change for free). Usage records ride along inside the already-persisted `TraceResult` JSON; aggregate counts are denormalized onto `scenario_results` for cheap run-level queries. The TUI loads the existing trace JSON from disk to render traces, and computes the run-level rate (token-weighted `Σcompletion / Σseconds`) from the new columns. The hermes subprocess adapter cannot produce clean per-request usage, so it records none and the rate reads `n/a`.

**Tech Stack:** Python 3.12, Pydantic v2, httpx, Textual + Rich (TUI), SQLite (`sqlite3`), pytest + respx.

**Spec:** `docs/superpowers/specs/2026-06-02-tui-traces-and-effective-tps-design.md`

**Pre-flight note:** `git status` shows pre-existing uncommitted edits to `toolery/adapters/openai_raw.py`, `toolery/cli.py`, `toolery/core/runner.py`, `tests/adapters/test_openai_raw.py`, `tests/core/test_runner.py`, plus an untracked `smoke_minimax.py`. This plan touches `openai_raw.py` and `test_openai_raw.py` again. **Before starting, either commit or stash those working-tree changes** so each task's commit is clean. Every commit step below uses path-scoped `git add` to avoid sweeping unrelated files in regardless.

**Test commands:**
- Single test: `uv run pytest tests/path/test_file.py::test_name -v`
- A file: `uv run pytest tests/path/test_file.py -v`
- Full suite (final check): `uv run pytest -q`

(If `uv run` is not how this repo runs pytest, fall back to `pytest` / `python -m pytest` — check `pyproject.toml`.)

---

## File Structure

**Modified:**
- `toolery/core/models.py` — add `TurnUsage`; `TraceResult.usage` + `token_totals()`; `ScenarioResult` token fields; module-level `effective_tps()`.
- `toolery/adapters/openai_raw.py` — capture per-turn `usage` + per-request latency.
- `toolery/core/scorer.py` — copy trace token totals onto every `ScenarioResult`.
- `toolery/core/store.py` — three additive columns, migration, write, run-level query.
- `toolery/tui/home_tab.py` — inline compact trace in detail pane; `t` binding + `TraceModal`; run-level rate in `_profile_run`.
- `toolery/tui/history_tab.py` — run-level rate in the details markdown.

**Created:**
- `toolery/tui/trace_view.py` — pure Rich-`Text` render helpers (compact + full).
- `tests/tui/test_trace_view.py` — tests for the render helpers.

**Existing tests extended:** `tests/core/test_models.py`, `tests/adapters/test_openai_raw.py`, `tests/core/test_scorer.py`, `tests/core/test_store.py`, `tests/tui/test_home_tab.py`, `tests/tui/test_history_tab.py`.

---

## Task 1: Token data model + helpers

**Files:**
- Modify: `toolery/core/models.py` (add `TurnUsage` after `ToolCall` ~line 103; extend `TraceResult` ~line 113; extend `ScenarioResult` ~line 132; add `effective_tps` free function)
- Test: `tests/core/test_models.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_models.py`:

```python
from toolery.core.models import (
    TurnUsage, TraceResult, ScenarioResult, effective_tps,
)


def _trace_with_usage(usage):
    return TraceResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response="done",
        started_at_iso="2026-06-02T00:00:00Z", duration_ms=1000,
        usage=usage,
    )


def test_traceresult_usage_defaults_empty():
    # Old trace JSON predates the usage field; it must still parse.
    t = TraceResult.model_validate_json(
        '{"scenario_id":"t-01-x","adapter":"raw","trial_index":0,'
        '"messages":[],"tool_calls":[],"started_at_iso":"x","duration_ms":5}'
    )
    assert t.usage == []
    assert t.token_totals() == (0, 0, 0)


def test_token_totals_sums_across_turns():
    t = _trace_with_usage([
        TurnUsage(turn_index=0, prompt_tokens=100, completion_tokens=20, latency_ms=400),
        TurnUsage(turn_index=1, prompt_tokens=130, completion_tokens=30, latency_ms=600),
    ])
    assert t.token_totals() == (230, 50, 1000)


def test_effective_tps_basic():
    # 50 completion tokens over 1.0s = 50 tok/s
    assert effective_tps(50, 1000) == 50.0


def test_effective_tps_zero_time_is_none():
    assert effective_tps(0, 0) is None
    assert effective_tps(10, 0) is None


def test_scenario_result_token_fields_default_zero():
    r = ScenarioResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0, status="pass",
        score=1.0, call_count=0, budget_max=1, latency_ms=5, failure_kind=None,
        checks=[], trace=_trace_with_usage([]),
    )
    assert (r.prompt_tokens, r.completion_tokens, r.gen_ms) == (0, 0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_models.py -k "usage or token_totals or effective_tps or token_fields" -v`
Expected: FAIL — `ImportError: cannot import name 'TurnUsage'` / `effective_tps`.

- [ ] **Step 3: Implement the model changes**

In `toolery/core/models.py`, add `TurnUsage` immediately after the `ToolCall` class:

```python
class TurnUsage(BaseModel):
    """Token usage + wall-time for a single model request (one assistant turn).

    Usage is reported per API call, not per tool call — one turn can emit
    several tool calls — so it lives here rather than on ToolCall. latency_ms
    is the wall-time of the successful HTTP POST only (retry backoff excluded),
    so it is a clean denominator for effective-throughput math.
    """
    turn_index: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
```

Add the `usage` field and `token_totals()` to `TraceResult` (default `[]` keeps old JSON parseable):

```python
class TraceResult(BaseModel):
    scenario_id: str
    adapter: str
    trial_index: int
    messages: list[Message]
    tool_calls: list[ToolCall]
    final_response: str | None = None
    started_at_iso: str
    duration_ms: int
    error: str | None = None
    adapter_metadata: dict[str, Any] = {}
    usage: list[TurnUsage] = []

    def token_totals(self) -> tuple[int, int, int]:
        """(prompt_tokens, completion_tokens, gen_ms) summed across all turns."""
        p = sum(u.prompt_tokens for u in self.usage)
        c = sum(u.completion_tokens for u in self.usage)
        ms = sum(u.latency_ms for u in self.usage)
        return p, c, ms
```

Add three fields to `ScenarioResult` (after `correctness_score`):

```python
    # Token usage + generation wall-time aggregated from the trace, so the
    # effective tokens/s can be recomputed (token-weighted) at any level
    # without re-reading trace files. Zero when the adapter reports no usage
    # (e.g. hermes subprocess) or for runs predating capture.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    gen_ms: int = 0
```

Add this module-level function (near the bottom of the file, after the classes):

```python
def effective_tps(completion_tokens: int, gen_ms: int) -> float | None:
    """Effective generation throughput: completion tokens / generation seconds.
    Returns None when gen_ms <= 0 (no usable timing) so callers render 'n/a'."""
    if gen_ms <= 0:
        return None
    return completion_tokens / (gen_ms / 1000)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_models.py -k "usage or token_totals or effective_tps or token_fields" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add toolery/core/models.py tests/core/test_models.py
git commit -m "feat(models): TurnUsage + per-trace token totals and effective_tps helper"
```

---

## Task 2: Capture usage + per-request latency in the HTTP adapter

**Files:**
- Modify: `toolery/adapters/openai_raw.py` (`_post_with_retry` ~lines 57-74; `run_scenario` ~lines 76-160)
- Test: `tests/adapters/test_openai_raw.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/adapters/test_openai_raw.py` (the file already imports `httpx`, `respx`, `pytest`, `json`, and `_scenario`):

```python
@pytest.mark.asyncio
@respx.mock
async def test_captures_usage_per_turn():
    # Single turn, no tool calls — model answers directly and reports usage.
    respx.post("http://localhost:8888/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "It's cloudy."}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
        })
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8888")
    try:
        trace = await adapter.run_scenario(_scenario(), model="m", timeout=30)
    finally:
        await adapter.aclose()

    assert len(trace.usage) == 1
    u = trace.usage[0]
    assert u.turn_index == 0
    assert u.prompt_tokens == 123
    assert u.completion_tokens == 45
    assert u.latency_ms >= 0
    assert trace.token_totals()[:2] == (123, 45)


@pytest.mark.asyncio
@respx.mock
async def test_missing_usage_field_records_zeros():
    respx.post("http://localhost:8888/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        })
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8888")
    try:
        trace = await adapter.run_scenario(_scenario(), model="m", timeout=30)
    finally:
        await adapter.aclose()
    assert len(trace.usage) == 1
    assert trace.token_totals() == (0, 0, trace.usage[0].latency_ms)
```

> Note: if the existing tests in this file use a different base URL or a non-`respx` style, mirror the file's dominant pattern instead of the literal URL above. The assertions on `trace.usage` are what matter.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/test_openai_raw.py -k "usage" -v`
Expected: FAIL — `trace.usage` is `[]` (capture not implemented).

- [ ] **Step 3: Implement capture**

In `toolery/adapters/openai_raw.py`, change `_post_with_retry` to also return the successful attempt's latency (timing only the POST, so retry backoff sleeps are excluded):

```python
    async def _post_with_retry(self, payload: dict, headers: dict) -> tuple[httpx.Response, int]:
        """POST with retry on transient rate-limit (429) / server (5xx) errors.

        Returns (response, latency_ms) where latency_ms times ONLY the
        successful attempt — retry backoff is excluded so the value is a clean
        denominator for effective-throughput math.
        """
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            resp = await self._client.post(url, json=payload, headers=headers)
            latency_ms = int((time.monotonic() - t0) * 1000)
            retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
            if retryable and attempt < self.max_retries:
                await asyncio.sleep(_retry_delay(attempt, resp))
                continue
            resp.raise_for_status()
            return resp, latency_ms
        raise RuntimeError("unreachable")  # pragma: no cover
```

In `run_scenario`, add `TurnUsage` to the import line at the top of the file:

```python
from toolery.core.models import Message, Scenario, ToolCall, TraceResult, TurnUsage
```

Initialize a usage list next to `tool_calls_recorded` (~line 87):

```python
        tool_calls_recorded: list[ToolCall] = []
        usage_records: list[TurnUsage] = []
```

Update the POST call and capture usage right after `data = resp.json()` (~lines 102-103):

```python
                resp, req_latency_ms = await self._post_with_retry(payload, headers)
                data = resp.json()
                usage = data.get("usage") or {}
                usage_records.append(TurnUsage(
                    turn_index=turn_idx,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=req_latency_ms,
                ))
                msg = data["choices"][0]["message"]
```

Pass `usage=usage_records` into the returned `TraceResult` (~line 149):

```python
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[Message.model_validate(m) for m in messages],
            tool_calls=tool_calls_recorded,
            final_response=final_response,
            started_at_iso=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"base_url": self.base_url, "model": model},
            usage=usage_records,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/test_openai_raw.py -v`
Expected: PASS — the two new tests pass AND all pre-existing tests in the file still pass (the `_post_with_retry` return-shape change is internal; confirm no other test asserts its old single-value return).

- [ ] **Step 5: Commit**

```bash
git add toolery/adapters/openai_raw.py tests/adapters/test_openai_raw.py
git commit -m "feat(adapter): capture per-turn token usage + clean request latency"
```

---

## Task 3: Copy token totals onto ScenarioResult in the scorer

**Files:**
- Modify: `toolery/core/scorer.py` (`evaluate` ~lines 674-759 — three `ScenarioResult(...)` return sites at ~679, ~738, ~752)
- Test: `tests/core/test_scorer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_scorer.py` (reuse the file's existing scenario/trace builders if present; otherwise construct minimally as below):

```python
from toolery.core.models import TraceResult, TurnUsage
from toolery.core import scorer


def _minimal_scenario():
    from toolery.core.models import Budget, Category, Scenario, Scoring, Tier
    return Scenario(
        id="t-01-x", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="hi", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(),
    )


def test_evaluate_copies_token_totals():
    trace = TraceResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response="hi",
        started_at_iso="x", duration_ms=10,
        usage=[
            TurnUsage(turn_index=0, prompt_tokens=100, completion_tokens=20, latency_ms=300),
            TurnUsage(turn_index=1, prompt_tokens=50, completion_tokens=10, latency_ms=200),
        ],
    )
    result = scorer.evaluate(_minimal_scenario(), trace)
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 30
    assert result.gen_ms == 500


def test_evaluate_error_path_copies_token_totals():
    trace = TraceResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response=None,
        started_at_iso="x", duration_ms=10, error="RuntimeError: boom",
        usage=[TurnUsage(turn_index=0, prompt_tokens=70, completion_tokens=5, latency_ms=100)],
    )
    result = scorer.evaluate(_minimal_scenario(), trace)
    assert result.status == "error"
    assert (result.prompt_tokens, result.completion_tokens, result.gen_ms) == (70, 5, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_scorer.py -k "token_totals" -v`
Expected: FAIL — `result.prompt_tokens` is `0` (scorer doesn't copy them yet).

- [ ] **Step 3: Implement the copy**

In `toolery/core/scorer.py`, at the top of `evaluate` (just after `def evaluate(...)`, ~line 674-676), compute the totals once:

```python
def evaluate(scenario: Scenario, trace: TraceResult) -> ScenarioResult:
    calls = trace.tool_calls
    response = trace.final_response
    pt, ct, gen_ms = trace.token_totals()
```

Then add the same three kwargs to **each** of the three `ScenarioResult(...)` returns in this function (error path ~679, fail/gradient path ~738, pass path ~752). For each, insert before the closing `)`:

```python
            prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
```

So e.g. the error-path return becomes:

```python
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="error", score=0.0, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind=_classify_error(trace.error), checks=[], trace=trace,
            correctness_score=0.0,
            prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
        )
```

Apply the identical insertion to the other two return sites.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_scorer.py -v`
Expected: PASS — new tests pass, existing scorer tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add toolery/core/scorer.py tests/core/test_scorer.py
git commit -m "feat(scorer): copy trace token totals onto ScenarioResult"
```

---

## Task 4: Persist token columns + run-level query in the store

**Files:**
- Modify: `toolery/core/store.py` (`SCHEMA` `scenario_results` ~lines 22-35; `init_schema` `sr_cols` block ~lines 93-97; `write_scenario_result` ~lines 210-224; add `fetch_run_token_totals`)
- Test: `tests/core/test_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_store.py` (use the file's existing fixtures for a temp `Store` + a `ScenarioResult` if present; otherwise the helpers below are self-contained):

```python
from toolery.core.models import (
    ScenarioResult, TraceResult, TurnUsage,
)


def _result_with_tokens(prompt, completion, gen_ms, scenario_id="t-01-x"):
    trace = TraceResult(
        scenario_id=scenario_id, adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response="x",
        started_at_iso="x", duration_ms=gen_ms,
    )
    return ScenarioResult(
        scenario_id=scenario_id, adapter="raw", trial_index=0, status="pass",
        score=1.0, call_count=0, budget_max=1, latency_ms=gen_ms,
        failure_kind=None, checks=[], trace=trace,
        prompt_tokens=prompt, completion_tokens=completion, gen_ms=gen_ms,
    )


def test_token_columns_round_trip(tmp_path):
    from toolery.core.store import Store
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run("run1", "m", "http://x", "2026-06-02T00:00:00Z", "{}", "h")
    store.write_scenario_result(
        "run1", _result_with_tokens(100, 20, 400),
        tags=[], ranking_dims=["overall"], scenario_hash="",
        category="tool_selection", tier="easy", trace_path="traces/a.json",
    )
    rows = store.fetch_results_for_run("run1")
    assert rows[0]["completion_tokens"] == 20
    assert rows[0]["gen_ms"] == 400


def test_fetch_run_token_totals_sums(tmp_path):
    from toolery.core.store import Store
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run("run1", "m", "http://x", "2026-06-02T00:00:00Z", "{}", "h")
    for i, (p, c, ms) in enumerate([(100, 20, 400), (80, 30, 600)]):
        store.write_scenario_result(
            "run1", _result_with_tokens(p, c, ms, scenario_id=f"t-0{i+1}-x"),
            tags=[], ranking_dims=["overall"], scenario_hash="",
            category="tool_selection", tier="easy",
            trace_path=f"traces/{i}.json",
        )
    completion, gen_ms = store.fetch_run_token_totals("run1")
    assert (completion, gen_ms) == (50, 1000)


def test_old_db_migrates_token_columns(tmp_path):
    # Simulate an old DB: create scenario_results WITHOUT the token columns,
    # then init_schema() must add them idempotently without data loss.
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE scenario_results (
          result_id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT, scenario_id TEXT NOT NULL, scenario_hash TEXT NOT NULL,
          tier TEXT NOT NULL, category TEXT NOT NULL,
          tags_json TEXT, ranking_dims_json TEXT,
          adapter TEXT NOT NULL, trial_index INTEGER NOT NULL,
          status TEXT, score REAL NOT NULL,
          call_count INTEGER NOT NULL, budget_max INTEGER,
          latency_ms INTEGER, failure_kind TEXT,
          trace_path TEXT, checks_json TEXT
        );
    """)
    con.commit()
    con.close()

    from toolery.core.store import Store
    store = Store(db)
    store.init_schema()  # must not raise
    cols = set()
    with store.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(scenario_results)").fetchall()}
    assert {"prompt_tokens", "completion_tokens", "gen_ms"} <= cols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_store.py -k "token" -v`
Expected: FAIL — `no such column: completion_tokens` / `AttributeError: fetch_run_token_totals`.

- [ ] **Step 3: Implement the store changes**

In `toolery/core/store.py`, add the three columns to the `scenario_results` block of `SCHEMA` (so fresh DBs get them), after `trace_path TEXT, checks_json TEXT,`:

```sql
  trace_path TEXT, checks_json TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, gen_ms INTEGER,
```

In `init_schema`, extend the existing `sr_cols` migration block (right after the `correctness_score` migration, ~line 97):

```python
            for col in ("prompt_tokens", "completion_tokens", "gen_ms"):
                if col not in sr_cols:
                    c.execute(f"ALTER TABLE scenario_results ADD COLUMN {col} INTEGER")
```

Update `write_scenario_result` to write the three values — extend the column list, the placeholders, and the values tuple:

```python
    def write_scenario_result(self, run_id, result: ScenarioResult, *, tags, ranking_dims,
                              scenario_hash, category, tier, trace_path) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO scenario_results(run_id, scenario_id, scenario_hash, tier, category, "
                "tags_json, ranking_dims_json, adapter, trial_index, status, score, call_count, "
                "budget_max, latency_ms, failure_kind, correctness_score, trace_path, checks_json, "
                "prompt_tokens, completion_tokens, gen_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, result.scenario_id, scenario_hash, tier, category,
                 json.dumps(tags), json.dumps(ranking_dims),
                 result.adapter, result.trial_index, result.status, result.score,
                 result.call_count, result.budget_max, result.latency_ms, result.failure_kind,
                 result.correctness_score, trace_path,
                 json.dumps([c.model_dump() for c in result.checks]),
                 result.prompt_tokens, result.completion_tokens, result.gen_ms),
            )
```

Add a new query method (place near `fetch_results_for_run`):

```python
    def fetch_run_token_totals(self, run_id: str) -> tuple[int, int]:
        """(total_completion_tokens, total_gen_ms) across the run's scenario
        results. Feeds the token-weighted effective tokens/s shown when
        llama-benchy was skipped. NULLs (old rows / hermes) coalesce to 0."""
        with self.conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(completion_tokens),0), COALESCE(SUM(gen_ms),0) "
                "FROM scenario_results WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return int(row[0]), int(row[1])
```

> Note on the `VALUES` placeholder count: there are now 21 columns / 21 `?`. Count them after editing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_store.py -v`
Expected: PASS — new token tests pass, existing store tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add toolery/core/store.py tests/core/test_store.py
git commit -m "feat(store): persist token columns + fetch_run_token_totals"
```

---

## Task 5: Pure trace render helpers

**Files:**
- Create: `toolery/tui/trace_view.py`
- Test: `tests/tui/test_trace_view.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/tui/test_trace_view.py`:

```python
from toolery.core.models import ToolCall, TraceResult, TurnUsage
from toolery.tui.trace_view import render_trace_compact, render_trace_full


def _trace():
    return TraceResult(
        scenario_id="multi-step", adapter="raw", trial_index=0,
        messages=[], final_response="NVDA is 172.4 EUR.",
        started_at_iso="x", duration_ms=900,
        tool_calls=[
            ToolCall(index=0, name="get_stock_price",
                     args={"symbol": "NVDA"}, result=187.2, result_kind="json", latency_ms=12),
            ToolCall(index=1, name="get_stock_price",
                     args={"symbol": "AMD"}, result={"error": "budget_exhausted"},
                     result_kind="error", latency_ms=0),
        ],
        usage=[TurnUsage(turn_index=0, prompt_tokens=1200, completion_tokens=340, latency_ms=900)],
    )


def test_compact_lists_tool_calls_and_tokens():
    text = render_trace_compact(_trace()).plain
    assert "get_stock_price" in text
    assert "NVDA" in text
    assert "tokens:" in text
    # 340 completion tokens / 0.9s ≈ 377 -> just assert a t/s figure appears
    assert "t/s" in text


def test_compact_marks_error_results():
    text = render_trace_compact(_trace()).plain
    assert "err" in text.lower()


def test_compact_tokens_na_when_no_usage():
    t = _trace()
    t.usage = []
    text = render_trace_compact(t).plain
    assert "n/a" in text


def test_full_includes_final_response_and_calls():
    text = render_trace_full(_trace()).plain
    assert "NVDA is 172.4 EUR." in text
    assert "get_stock_price" in text
    assert "get_stock_price" in text


def test_compact_truncates_long_values():
    t = _trace()
    t.tool_calls[0].args = {"blob": "x" * 500}
    text = render_trace_compact(t).plain
    # No single rendered line should be absurdly long.
    assert max(len(line) for line in text.splitlines()) < 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_trace_view.py -v`
Expected: FAIL — `ModuleNotFoundError: toolery.tui.trace_view`.

- [ ] **Step 3: Implement the helpers**

Create `toolery/tui/trace_view.py`:

```python
"""Pure render helpers turning a TraceResult into Rich Text for the TUI.

Kept free of Textual widgets and side effects so they can be unit-tested
without a running app. home_tab consumes these for the inline detail pane and
the full-screen TraceModal.
"""
from __future__ import annotations

import json

from rich.text import Text

from toolery.core.models import TraceResult, effective_tps

_MAX_FIELD = 60  # truncate long args/results inline


def _short(value: object, limit: int = _MAX_FIELD) -> str:
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            s = str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _tokens_line(trace: TraceResult) -> str:
    pt, ct, gen_ms = trace.token_totals()
    tps = effective_tps(ct, gen_ms)
    rate = f"~{tps:.0f} gen t/s" if tps is not None else "n/a"
    if pt == 0 and ct == 0:
        return f"tokens: n/a  ·  {rate}"
    return f"tokens: {pt} in / {ct} out  ·  {rate}"


def render_trace_compact(trace: TraceResult) -> Text:
    """One-line-per-call summary + a token/throughput line. For the detail pane."""
    out = Text()
    calls = trace.tool_calls
    out.append(f"\ntool calls ({len(calls)}):\n", style="bold")
    if not calls:
        out.append("  (none)\n", style="dim")
    for c in calls:
        marker = "err" if c.result_kind == "error" else _short(c.result, 24)
        out.append(f"  {c.index + 1} ", style="cyan")
        out.append(c.name, style="bold")
        out.append(f" {_short(c.args, 40)}", style="dim")
        out.append(" → ", style="dim")
        out.append(f"{marker}\n",
                   style="red" if c.result_kind == "error" else "green")
    out.append(_tokens_line(trace) + "\n", style="dim")
    return out


def render_trace_full(trace: TraceResult) -> Text:
    """Full, scrollable conversation view for the modal: prompt context,
    each tool call with args/result/latency, and the final response."""
    out = Text()
    out.append(f"{trace.scenario_id}", style="bold")
    out.append(f"  ·  {trace.adapter} · t{trace.trial_index}\n", style="dim")
    out.append(_tokens_line(trace) + "\n\n", style="dim")

    out.append("tool calls:\n", style="bold")
    if not trace.tool_calls:
        out.append("  (none)\n", style="dim")
    for c in trace.tool_calls:
        out.append(f"  {c.index + 1}. ", style="cyan")
        out.append(f"{c.name}", style="bold")
        out.append(f"  ({c.latency_ms}ms)\n", style="dim")
        out.append(f"     args:   {_short(c.args, 140)}\n", style="dim")
        style = "red" if c.result_kind == "error" else "green"
        out.append(f"     result: {_short(c.result, 140)}\n", style=style)

    if trace.error:
        out.append(f"\nerror: {trace.error}\n", style="red")
    if trace.final_response:
        out.append("\nfinal response:\n", style="bold")
        out.append(f"{trace.final_response}\n")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_trace_view.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add toolery/tui/trace_view.py tests/tui/test_trace_view.py
git commit -m "feat(tui): pure trace render helpers (compact + full)"
```

---

## Task 6: Inline compact trace in the detail pane

**Files:**
- Modify: `toolery/tui/home_tab.py` (imports near top; `_detail_block` ~lines 214-250; caller `_on_scenario_selected` ~lines 906-929)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/tui/test_home_tab.py`:

```python
import json
from pathlib import Path

from toolery.core.models import ToolCall, TraceResult
from toolery.tui.home_tab import _detail_block


def _write_trace(run_dir: Path) -> str:
    trace = TraceResult(
        scenario_id="t-01-x", adapter="raw", trial_index=0,
        messages=[], final_response="done", started_at_iso="x", duration_ms=10,
        tool_calls=[ToolCall(index=0, name="get_weather",
                             args={"location": "Warsaw"}, result={"temp_c": 7},
                             result_kind="json", latency_ms=5)],
    )
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    rel = "traces/t-01-x__raw__t0.json"
    (run_dir / rel).write_text(trace.model_dump_json())
    return rel


def test_detail_block_inlines_tool_calls(tmp_path):
    rel = _write_trace(tmp_path)
    text = _detail_block(
        scenario_id="t-01-x", adapter="raw", trial=0, status="pass",
        failure_kind=None, latency_ms=10, call_count=1, budget_max=1,
        trace_path=rel, checks_json="[]", run_dir=tmp_path,
    ).plain
    assert "tool calls (1)" in text
    assert "get_weather" in text


def test_detail_block_missing_trace_falls_back_to_path(tmp_path):
    text = _detail_block(
        scenario_id="t-01-x", adapter="raw", trial=0, status="pass",
        failure_kind=None, latency_ms=10, call_count=1, budget_max=1,
        trace_path="traces/nope.json", checks_json="[]", run_dir=tmp_path,
    ).plain
    assert "traces/nope.json" in text  # graceful fallback, no crash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_home_tab.py -k "detail_block_inlines or missing_trace" -v`
Expected: FAIL — `_detail_block() got an unexpected keyword argument 'run_dir'`.

- [ ] **Step 3: Implement**

In `toolery/tui/home_tab.py`, ensure these imports exist near the top of the file (add what's missing):

```python
from pathlib import Path

from toolery.core.models import TraceResult
from toolery.tui.trace_view import render_trace_compact
```

Change `_detail_block`'s signature to accept `run_dir` and render the compact trace before the trailing `trace:` line. Replace the final `if trace_path:` block (~lines 248-250) with the version below, and add the param:

```python
def _detail_block(scenario_id: str, adapter: str, trial: int, status: str,
                  failure_kind: str | None, latency_ms: int | None,
                  call_count: int | None, budget_max: int | None,
                  trace_path: str | None, checks_json: str | None,
                  run_dir: Path | None = None) -> Text:
```

…and, after the `checks` loop, replace the existing trailing block:

```python
    trace = None
    if run_dir is not None and trace_path:
        try:
            trace = TraceResult.model_validate_json(
                (Path(run_dir) / trace_path).read_text())
        except Exception:
            trace = None  # missing/corrupt trace → fall back to path only
    if trace is not None:
        out.append(render_trace_compact(trace))
    if trace_path:
        out.append(f"\ntrace: {trace_path}\n", style="dim")
    return out
```

Update the caller in `_on_scenario_selected` (~lines 919-929) to pass `run_dir`:

```python
        if state == "done":
            run_dir = None
            if self._store is not None and self._current_run_id:
                run_dir = self._store.path.parent / "runs" / self._current_run_id
            block = _detail_block(
                scenario_id=scenario_id, adapter=adapter, trial=trial_index,
                status=payload.get("status") or "?",
                failure_kind=payload.get("failure_kind"),
                latency_ms=payload.get("latency_ms"),
                call_count=payload.get("call_count"),
                budget_max=payload.get("budget_max"),
                trace_path=payload.get("trace_path"),
                checks_json=payload.get("checks_json"),
                run_dir=run_dir,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_home_tab.py -v`
Expected: PASS — new tests pass, existing home_tab tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add toolery/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): inline compact tool-call trace in the detail pane"
```

---

## Task 7: Full-screen TraceModal + `t` binding

**Files:**
- Modify: `toolery/tui/home_tab.py` (add `_row_trace_path` module function near `_classify_plan` ~line 176; add `TraceModal` class; add `BINDINGS` + `action_view_trace` to `HomeTab`)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/tui/test_home_tab.py`:

```python
from toolery.tui.home_tab import _row_trace_path


def test_row_trace_path_returns_path_for_done_row():
    plan = [("t-01-x", "raw", 0), ("t-02-x", "raw", 0)]
    completed = {
        ("t-01-x", "raw", 0): {"status": "pass", "trace_path": "traces/a.json"},
    }
    running = {}
    assert _row_trace_path(plan, completed, running, 0) == "traces/a.json"


def test_row_trace_path_none_for_upcoming_or_out_of_range():
    plan = [("t-01-x", "raw", 0)]
    assert _row_trace_path(plan, {}, {}, 0) is None       # upcoming, no trace yet
    assert _row_trace_path(plan, {}, {}, 5) is None       # out of range
    assert _row_trace_path(plan, {}, {}, None) is None    # no cursor
```

> Confirm the exact shape of `_classify_plan`'s return (`(state, key, payload)`) and the `completed`/`running` dict shapes by reading `_on_scenario_selected` (~lines 906-918); mirror them so `_row_trace_path` reuses `_classify_plan` rather than reimplementing classification.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_home_tab.py -k "row_trace_path" -v`
Expected: FAIL — `ImportError: cannot import name '_row_trace_path'`.

- [ ] **Step 3: Implement**

In `toolery/tui/home_tab.py`, add a module-level helper next to `_classify_plan`:

```python
def _row_trace_path(plan, completed, running, idx) -> str | None:
    """Resolve the trace_path for a row index in the scenarios table, or None
    if the row is not a completed result. Reuses _classify_plan so ordering
    matches exactly what the table renders."""
    if idx is None:
        return None
    rows = _classify_plan(plan, completed, running)
    if idx >= len(rows):
        return None
    state, _key, payload = rows[idx]
    if state != "done":
        return None
    return payload.get("trace_path")
```

Add the `TraceModal` class (near the other modals, e.g. after `ConfirmRunActionModal`). Ensure `Binding`, `ModalScreen`, `VerticalScroll`, and `Static` are imported (most already are — add `Binding`/`VerticalScroll` if missing):

```python
class TraceModal(ModalScreen[None]):
    DEFAULT_CSS = """
    TraceModal {
        align: center middle;
    }
    TraceModal #trace-box {
        width: 90%;
        height: 85%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    """
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, title: str, trace: "TraceResult | None") -> None:
        super().__init__()
        self._title = title
        self._trace = trace

    def compose(self):
        with VerticalScroll(id="trace-box"):
            yield Static(self._title, classes="trace-modal-title")
            if self._trace is None:
                yield Static("trace unavailable", classes="dim")
            else:
                yield Static(render_trace_full(self._trace))

    def action_dismiss(self) -> None:
        self.dismiss(None)
```

Add `render_trace_full` to the trace_view import line:

```python
from toolery.tui.trace_view import render_trace_compact, render_trace_full
```

Add a `BINDINGS` entry and an action to `HomeTab`. If `HomeTab` has no `BINDINGS` yet, add the class attribute; otherwise append:

```python
    BINDINGS = [Binding("t", "view_trace", "Trace")]

    @work
    async def action_view_trace(self) -> None:
        tbl = self.query_one("#scenarios-table", DataTable)
        if tbl.row_count == 0 or tbl.cursor_row is None:
            return
        completed = {
            (r["scenario_id"], r["adapter"], r["trial_index"]): r
            for r in self._results_cache
        }
        running = self._fetch_running_units()
        rel = _row_trace_path(self._plan, completed, running, tbl.cursor_row)
        trace = None
        if rel and self._store is not None and self._current_run_id:
            run_dir = self._store.path.parent / "runs" / self._current_run_id
            try:
                trace = TraceResult.model_validate_json((run_dir / rel).read_text())
            except Exception:
                trace = None
        title = f"trace — row {tbl.cursor_row}" if rel else "no trace for this row"
        await self.app.push_screen_wait(TraceModal(title, trace))
```

> Verify `@work` is imported (`from textual import work`) — it is used elsewhere in the codebase (e.g. history_tab). Verify `self._fetch_running_units()` exists (referenced in `_on_scenario_selected`). If `HomeTab` already defines `BINDINGS`, merge the `("t", ...)` entry rather than overwriting.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_home_tab.py -v`
Expected: PASS. Then sanity-check the app still imports/starts: `uv run pytest tests/tui/test_app_starts.py -v`.

- [ ] **Step 5: Commit**

```bash
git add toolery/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): full-screen TraceModal opened with 't' on a result row"
```

---

## Task 8: Run-level effective tokens/s in the home-tab run profile

**Files:**
- Modify: `toolery/tui/home_tab.py` (`_profile_run` ~lines 282-...)
- Test: `tests/tui/test_home_tab.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/tui/test_home_tab.py`:

```python
from toolery.tui.home_tab import _profile_run


def test_profile_run_shows_effective_tps():
    results = [
        {"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
         "ranking_dims_json": "[\"overall\"]", "completion_tokens": 40, "gen_ms": 400},
        {"scenario_id": "b", "tier": "easy", "status": "pass", "score": 1.0,
         "ranking_dims_json": "[\"overall\"]", "completion_tokens": 60, "gen_ms": 600},
    ]
    text = _profile_run(results).plain
    # 100 completion tokens / 1.0s = 100 gen t/s, labeled as tool-test measured.
    assert "100" in text
    assert "t/s" in text


def test_profile_run_tps_na_without_tokens():
    results = [
        {"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
         "ranking_dims_json": "[\"overall\"]", "completion_tokens": 0, "gen_ms": 0},
    ]
    text = _profile_run(results).plain
    assert "n/a" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_home_tab.py -k "profile_run_shows or profile_run_tps_na" -v`
Expected: FAIL — no t/s line in the profile output.

- [ ] **Step 3: Implement**

In `toolery/tui/home_tab.py`, import the helper if not already imported:

```python
from toolery.core.models import TraceResult, effective_tps
```

In `_profile_run(results)`, after the existing header/overall section (and guarded by `if results:` which already exists at the top), append an effective-throughput line. Add near the end of the function, before `return`:

```python
    total_completion = sum(int(r.get("completion_tokens") or 0) for r in results)
    total_gen_ms = sum(int(r.get("gen_ms") or 0) for r in results)
    tps = effective_tps(total_completion, total_gen_ms)
    out.append("\neff gen t/s (tool tests): ", style="bold")
    out.append(f"{tps:.1f}\n" if tps is not None else "n/a\n",
               style="green" if tps is not None else "dim")
```

> Adjust the local variable name (`out`) to match whatever Text accumulator `_profile_run` already uses. Read the function first; reuse its accumulator rather than introducing a new one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_home_tab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolery/tui/home_tab.py tests/tui/test_home_tab.py
git commit -m "feat(tui): show effective tool-test gen t/s in the run profile"
```

---

## Task 9: Run-level effective tokens/s in the history details markdown

**Files:**
- Modify: `toolery/tui/history_tab.py` (`_build_details_md` perf section ~lines 298-310)
- Test: `tests/tui/test_history_tab.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/tui/test_history_tab.py` (use the file's existing helper for invoking the markdown builder if present; otherwise call `_build_details_md` directly):

```python
from toolery.tui.history_tab import _build_details_md


def _run():
    return {"run_id": "run1", "model": "m", "status": "done",
            "started_at": "x", "finished_at": "y", "duration_s": 1.0,
            "cluster": None, "base_url": "http://x", "config_json": "{}"}


def test_details_md_includes_effective_tps():
    results = [
        {"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
         "completion_tokens": 40, "gen_ms": 400},
        {"scenario_id": "b", "tier": "easy", "status": "pass", "score": 1.0,
         "completion_tokens": 60, "gen_ms": 600},
    ]
    md = _build_details_md(_run(), results, perf_rows=[], adapters=["raw"])
    assert "Eff gen t/s" in md
    assert "100.0" in md  # 100 tok / 1.0s


def test_details_md_effective_tps_na_without_tokens():
    results = [{"scenario_id": "a", "tier": "easy", "status": "pass", "score": 1.0,
                "completion_tokens": 0, "gen_ms": 0}]
    md = _build_details_md(_run(), results, perf_rows=[], adapters=["raw"])
    assert "n/a" in md
```

> Confirm the real function name and argument order (`_build_details_md(run, results, perf_rows, adapters)` per `_open_details` ~line 486). If it differs, mirror the actual signature.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_history_tab.py -k "effective_tps" -v`
Expected: FAIL — no "Eff gen t/s" string in the markdown.

- [ ] **Step 3: Implement**

In `toolery/tui/history_tab.py`, add the import:

```python
from toolery.core.models import effective_tps
```

In `_build_details_md`, after the scenario-stats block and around the perf section (~line 298), add a tool-test throughput line that renders whether or not llama-benchy perf rows exist:

```python
    if results:
        total_completion = sum(int(r.get("completion_tokens") or 0) for r in results)
        total_gen_ms = sum(int(r.get("gen_ms") or 0) for r in results)
        tps = effective_tps(total_completion, total_gen_ms)
        rate = f"{tps:.1f}" if tps is not None else "n/a"
        lines.append(f"- **Eff gen t/s (tool tests):** {rate}")
        lines.append("")
```

Place this block where it reads naturally — e.g. right after the scenario-stats `by_tier` table and before the `if perf_rows:` block — so the proxy sits next to the llama-benchy perf table when both are present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_history_tab.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add toolery/tui/history_tab.py tests/tui/test_history_tab.py
git commit -m "feat(tui): show effective tool-test gen t/s in history details"
```

---

## Task 10: Full-suite verification

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest -q`
Expected: All green. If anything red, fix the offending task before proceeding (do not paper over with skips).

- [ ] **Step 2: Manual smoke (optional but recommended)**

If a local endpoint is available, run a tiny scenario set without `--with-perf`, open the TUI, select a completed row (compact trace appears in the detail pane), press `t` (full modal opens), and confirm the run profile / history details show "eff gen t/s". With `TOOLERY_RESULTS_DIR` pointing at an old results dir, confirm old runs render traces and show `n/a` for tokens/s without errors.

- [ ] **Step 3: Final commit (if any cleanup)**

```bash
git add -A
git commit -m "chore: tidy up after traces + effective tokens/s feature"
```

---

## Self-Review Notes (author checklist — already verified)

- **Spec coverage:** capture (T2) · per-turn model (T1) · scorer copy (T3) · persistence + migration + run query (T4) · compact render (T5) · inline (T6) · modal + `t` (T7) · run-level display home (T8) + history (T9) · backward-compat tested in T1/T2/T4/T6/T9 · `cloud` covered free via inheritance (no task needed) · hermes → `n/a` (no capture path, falls out of zero defaults).
- **Type consistency:** `TurnUsage`, `TraceResult.usage`, `token_totals() -> (pt, ct, gen_ms)`, `effective_tps(completion, gen_ms)`, `ScenarioResult.{prompt_tokens,completion_tokens,gen_ms}`, `Store.fetch_run_token_totals`, `render_trace_compact/full`, `_row_trace_path`, `TraceModal`, `action_view_trace` — names used consistently across tasks.
- **Out of scope (per spec):** streaming/TTFT split, historical backfill, rankings-tab column, per-tool-call token attribution.

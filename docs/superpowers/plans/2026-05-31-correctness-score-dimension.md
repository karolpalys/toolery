# Correctness-score Dimension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a budget-independent `correctness_score` dimension alongside the existing budgeted `score`, so an agent that solves a scenario but overruns its tool-call budget is still credited for correctness.

**Architecture:** Additive and backward-compatible. `evaluate()` keeps computing the headline `score`/`status`/`failure_kind` exactly as today (budget overrun still zeros the headline) and additionally computes `correctness_score` = the score the scenario would get if the *only* thing ignored were a budget overrun (hallucination and forbidden actions still fail correctness; a trace error → 0.0). The value is persisted in a new `scenario_results.correctness_score` column, backfilled for existing runs from stored trace files, and surfaced per (model, adapter) via a CLI report.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite (stdlib `sqlite3`), Typer CLI, pytest. Project venv at `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-05-31-correctness-score-dimension-design.md`

---

## File Structure

- `toolery/core/models.py` — add `correctness_score` field to `ScenarioResult`.
- `toolery/core/scorer.py` — `_correctness_score()` helper + set the field in both `evaluate()` return paths and the error path.
- `toolery/core/store.py` — schema column, idempotent migration for `scenario_results`, persist in `write_scenario_result`, new `update_correctness_score()`.
- `toolery/cli.py` — `backfill-correctness` command; `correctness-report` command.
- `toolery/rankings/compute.py` — `compute_correctness_breakdown()` aggregate.
- Tests under `tests/core/`, `tests/`, `tests/rankings/`.

---

## Task 1: Add `correctness_score` field to `ScenarioResult`

**Files:**
- Modify: `toolery/core/models.py:132-143`
- Test: `tests/core/test_scorer_compose.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_scorer_compose.py`:

```python
def test_scenario_result_has_correctness_score_field():
    from toolery.core.models import ScenarioResult
    fields = ScenarioResult.model_fields
    assert "correctness_score" in fields
    # optional, defaults to None so existing constructors keep working
    assert fields["correctness_score"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_scorer_compose.py::test_scenario_result_has_correctness_score_field -v`
Expected: FAIL with `assert 'correctness_score' in fields` (KeyError-style assertion failure).

- [ ] **Step 3: Add the field**

In `toolery/core/models.py`, in `class ScenarioResult`, after the `failure_kind` line (`failure_kind: str | None`) add:

```python
    failure_kind: str | None
    # Budget-independent correctness: the score this scenario would get if the
    # only thing ignored were a tool-call budget overrun. None until computed
    # (older rows backfilled from trace files). See scorer._correctness_score.
    correctness_score: float | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_scorer_compose.py::test_scenario_result_has_correctness_score_field -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add toolery/core/models.py tests/core/test_scorer_compose.py
git commit -m "feat(scorer): add correctness_score field to ScenarioResult"
```

---

## Task 2: Compute `correctness_score` in the scorer

**Files:**
- Modify: `toolery/core/scorer.py` (error path ~640-646, `_classify_error` region; main paths ~696-738)
- Test: `tests/core/test_scorer_compose.py`

Note the existing scoring facts this task relies on (already in `scorer.py`):
`scenario.scoring.weights["pass"|"fail"|"partial"]`, `_implicit_partial_gradient_enabled()`, the `required`/`forbidden`/`forbidden_clean`/`hallucinated`/`budget_violated` locals in `evaluate()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_scorer_compose.py` (the `_error_trace`, `_scenario`, `_trace` helpers already exist in this file):

```python
def test_correctness_full_when_only_budget_violated():
    """Budget overrun zeros the headline score but correctness stays full."""
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
    )
    s = _scenario(scoring)  # budget.max_tool_calls == 2
    # 3 get_weather calls: required passes, but call_count 3 > budget 2.
    tr = _trace(("get_weather", {"location": "Warsaw"}),
                ("get_weather", {"location": "Warsaw"}),
                ("get_weather", {"location": "Warsaw"}))
    r = evaluate(s, tr)
    assert r.status == "fail"
    assert r.score == 0.0
    assert r.failure_kind == "budget_violated"
    assert r.correctness_score == 1.0


def test_correctness_matches_score_on_clean_pass():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
    )
    r = evaluate(_scenario(scoring), _trace(("get_weather", {"location": "Warsaw"})))
    assert r.status == "pass"
    assert r.correctness_score == r.score == 1.0


def test_correctness_fails_on_hallucinated_tool():
    """Hallucination is a correctness failure, not a frugality one."""
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
    )
    # 'made_up_tool' is not in scenario.tools (get_weather, web_search)
    r = evaluate(_scenario(scoring), _trace(("made_up_tool", {})))
    assert r.correctness_score == 0.0


def test_correctness_fails_on_forbidden_action():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        forbidden=[ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"})],
    )
    tr = _trace(("get_weather", {"location": "Warsaw"}), ("web_search", {"query": "x"}))
    r = evaluate(_scenario(scoring), tr)
    assert r.correctness_score == 0.0


def test_correctness_zero_on_trace_error():
    r = evaluate(_scenario(Scoring()), _error_trace("API call failed after 3 retries: Connection error."))
    assert r.correctness_score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_scorer_compose.py -k correctness -v`
Expected: FAIL — `correctness_score` is `None` (not `1.0`/`0.0`) in every new test.

- [ ] **Step 3: Add the `_correctness_score` helper**

In `toolery/core/scorer.py`, immediately above `def evaluate(` (after `_classify_error`), add:

```python
def _correctness_score(scenario, required, forbidden_clean: bool, hallucinated: bool) -> float:
    """Score ignoring ONLY a tool-call budget overrun. Hallucinated tools and
    forbidden actions still fail correctness; budget is the sole gate dropped.
    Mirrors evaluate()'s pass/fail + optional partial-gradient logic."""
    weights = scenario.scoring.weights
    req_pass = all(r.result == "pass" for r in required)
    if forbidden_clean and req_pass and not hallucinated:
        return weights["pass"]
    if (_implicit_partial_gradient_enabled()
            and forbidden_clean
            and not hallucinated
            and required):
        n_req_pass = sum(1 for r in required if r.result == "pass")
        if n_req_pass > 0:
            return weights["partial"] * (n_req_pass / len(required))
    return weights["fail"]
```

- [ ] **Step 4: Set `correctness_score` in the error path**

In `evaluate()`, the early `if trace.error:` return — add the field:

```python
    if trace.error:
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="error", score=0.0, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind=_classify_error(trace.error), checks=[], trace=trace,
            correctness_score=0.0,
        )
```

- [ ] **Step 5: Set `correctness_score` in the fail-gradient return**

In `evaluate()`, the `if not forbidden_clean or not required_all_pass:` block, the `return ScenarioResult(...)` — add the field (compute via helper):

```python
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status=gradient_status, score=gradient_score,
            call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
            latency_ms=trace.duration_ms, failure_kind=kind,
            checks=all_checks, trace=trace,
            correctness_score=_correctness_score(scenario, required, forbidden_clean, hallucinated),
        )
```

- [ ] **Step 6: Set `correctness_score` in the all-pass return**

In `evaluate()`, the final `return ScenarioResult(...)` (status="pass") — add the field. On a clean pass it equals the pass weight:

```python
    return ScenarioResult(
        scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
        status="pass", score=scenario.scoring.weights["pass"],
        call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
        latency_ms=trace.duration_ms, failure_kind=None,
        checks=all_checks, trace=trace,
        correctness_score=scenario.scoring.weights["pass"],
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_scorer_compose.py -k correctness -v`
Expected: PASS (all 5 new tests).

- [ ] **Step 8: Run the full scorer suite for regressions**

Run: `.venv/bin/python -m pytest tests/core/test_scorer.py tests/core/test_scorer_compose.py -q`
Expected: PASS (no regressions).

- [ ] **Step 9: Commit**

```bash
git add toolery/core/scorer.py tests/core/test_scorer_compose.py
git commit -m "feat(scorer): compute budget-independent correctness_score"
```

---

## Task 3: Persist `correctness_score` (schema + migration + insert)

**Files:**
- Modify: `toolery/core/store.py` (SCHEMA ~22-34, `init_schema` ~84-117, `write_scenario_result` ~197-211)
- Test: `tests/core/test_store.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Create or append to `tests/core/test_store.py`:

```python
import sqlite3
from pathlib import Path

from toolery.core.models import (
    Budget, Category, Message, Scenario, Scoring, Tier, ToolCall, TraceResult,
)
from toolery.core.scorer import evaluate
from toolery.core.store import Store


def _store(tmp_path) -> Store:
    s = Store(tmp_path / "runs.db")
    s.init_schema()
    return s


def _scenario():
    return Scenario(
        id="t-01-x", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p",
        tools=["get_weather"], budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[]),
    )


def _trace():
    return TraceResult(
        scenario_id="t-01-x", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_scenario_results_has_correctness_column(tmp_path):
    s = _store(tmp_path)
    with s.conn() as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(scenario_results)").fetchall()}
    assert "correctness_score" in cols


def test_write_and_read_correctness_score(tmp_path):
    s = _store(tmp_path)
    s.create_run(run_id="r1", model="m", base_url="u",
                 started_at="2026-05-23T18:00:00Z", config_json="{}", scenarios_hash="h")
    result = evaluate(_scenario(), _trace())
    s.write_scenario_result(
        "r1", result, tags=[], ranking_dims=["overall"], scenario_hash="h",
        category="tool_selection", tier="easy", trace_path="traces/x.json",
    )
    rows = s.fetch_results_for_run("r1")
    assert rows[0]["correctness_score"] == result.correctness_score
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/core/test_store.py -v`
Expected: FAIL — `correctness_score` not in columns / KeyError on row.

- [ ] **Step 3: Add the column to SCHEMA**

In `toolery/core/store.py`, in the `scenario_results` `CREATE TABLE` inside `SCHEMA`, add the column after `latency_ms INTEGER, failure_kind TEXT,`:

```sql
  latency_ms INTEGER, failure_kind TEXT,
  correctness_score REAL,
  trace_path TEXT, checks_json TEXT,
```

- [ ] **Step 4: Add an idempotent migration for existing DBs**

In `init_schema()`, right after the `runs`-table migration loop (after the `for stmt in _MIGRATIONS:` block, before the `'paused'` CHECK migration), add:

```python
            # scenario_results column migrations (older DBs predate these).
            sr_cols = {row[1] for row in c.execute(
                "PRAGMA table_info(scenario_results)").fetchall()}
            if "correctness_score" not in sr_cols:
                c.execute("ALTER TABLE scenario_results ADD COLUMN correctness_score REAL")
```

- [ ] **Step 5: Persist it in `write_scenario_result`**

Modify the INSERT in `write_scenario_result` to include the new column. Replace the existing statement with:

```python
            c.execute(
                "INSERT INTO scenario_results(run_id, scenario_id, scenario_hash, tier, category, "
                "tags_json, ranking_dims_json, adapter, trial_index, status, score, call_count, "
                "budget_max, latency_ms, failure_kind, correctness_score, trace_path, checks_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, result.scenario_id, scenario_hash, tier, category,
                 json.dumps(tags), json.dumps(ranking_dims),
                 result.adapter, result.trial_index, result.status, result.score,
                 result.call_count, result.budget_max, result.latency_ms, result.failure_kind,
                 result.correctness_score, trace_path,
                 json.dumps([c.model_dump() for c in result.checks])),
            )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/core/test_store.py -v`
Expected: PASS

- [ ] **Step 7: Verify the migration upgrades the real DB in place**

Run:
```bash
.venv/bin/python -c "from toolery.core.store import Store; from pathlib import Path; s=Store(Path('results/runs.db')); s.init_schema(); print('migrated')" \
 && sqlite3 results/runs.db "PRAGMA table_info(scenario_results);" | grep correctness_score
```
Expected: prints `migrated` and a line containing `correctness_score|REAL`.

- [ ] **Step 8: Commit**

```bash
git add toolery/core/store.py tests/core/test_store.py
git commit -m "feat(store): persist correctness_score + idempotent column migration"
```

---

## Task 4: Backfill command (`backfill-correctness`)

**Files:**
- Modify: `toolery/core/store.py` (add `update_correctness_score`)
- Modify: `toolery/cli.py` (new `@app.command()`; reuse `_store`, `_results_dir`)
- Test: `tests/test_backfill_correctness.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_backfill_correctness.py`:

```python
import json
from pathlib import Path

from toolery.cli import _backfill_correctness_run  # pure helper, see Step 5
from toolery.core.models import (
    Budget, Category, Message, Scenario, Scoring, ScoringCheck, Tier, ToolCall, TraceResult,
)
from toolery.core.scorer import evaluate
from toolery.core.store import Store


def _scenario():
    return Scenario(
        id="bf-01", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[ScoringCheck.model_validate(
            {"check": "tool_called", "tool": "get_weather"})]),
    )


def _budget_violated_correct_trace():
    # 2 calls > budget 1 → headline fail/budget_violated, but required passes.
    return TraceResult(
        scenario_id="bf-01", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"}),
                    ToolCall(index=1, name="get_weather", args={"location": "Warsaw"})],
        final_response="ok", started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_backfill_populates_correctness_from_trace(tmp_path):
    results_dir = tmp_path
    run_id = "2026-05-31T00-00_m"
    run_dir = results_dir / "runs" / run_id / "traces"
    run_dir.mkdir(parents=True)
    trace = _budget_violated_correct_trace()
    (run_dir / "bf-01__hermes__t0.json").write_text(trace.model_dump_json())

    store = Store(results_dir / "runs.db")
    store.init_schema()
    store.create_run(run_id=run_id, model="m", base_url="u",
                     started_at="2026-05-31T00:00:00Z", config_json="{}", scenarios_hash="h")
    # Write the headline result WITHOUT correctness (simulate an old row).
    res = evaluate(_scenario(), trace)
    assert res.failure_kind == "budget_violated" and res.score == 0.0
    store.write_scenario_result(run_id, res, tags=[], ranking_dims=["overall"],
                                scenario_hash="h", category="tool_selection", tier="easy",
                                trace_path="traces/bf-01__hermes__t0.json")
    with store.conn() as c:
        c.execute("UPDATE scenario_results SET correctness_score=NULL")

    scenarios = {"bf-01": _scenario()}
    updated, skipped = _backfill_correctness_run(store, run_id, results_dir, scenarios)

    assert updated == 1 and skipped == 0
    row = store.fetch_results_for_run(run_id)[0]
    assert row["correctness_score"] == 1.0  # solved, only budget failed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backfill_correctness.py -v`
Expected: FAIL with `ImportError: cannot import name '_backfill_correctness_run'`.

- [ ] **Step 3: Add `update_correctness_score` to the store**

In `toolery/core/store.py`, add a method to `class Store` (near `write_scenario_result`):

```python
    def update_correctness_score(self, result_id: int, value: float) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE scenario_results SET correctness_score=? WHERE result_id=?",
                (value, result_id),
            )
```

- [ ] **Step 4: Add the pure backfill helper in cli.py**

In `toolery/cli.py`, add (top-level, near the other `_`-helpers like `_store`/`_results_dir`). Imports `json`, `Path`, `evaluate`, `TraceResult` — add any missing at module top:

```python
def _backfill_correctness_run(store, run_id: str, results_dir: Path, scenarios: dict) -> tuple[int, int]:
    """Recompute correctness_score for one run from its stored trace files.
    Returns (updated, skipped). Rows whose scenario or trace file is missing
    are skipped (left as-is)."""
    from toolery.core.models import TraceResult
    from toolery.core.scorer import evaluate

    updated = skipped = 0
    for row in store.fetch_results_for_run(run_id):
        scenario = scenarios.get(row["scenario_id"])
        trace_path = row.get("trace_path")
        if scenario is None or not trace_path:
            skipped += 1
            continue
        tp = results_dir / "runs" / run_id / trace_path
        if not tp.exists():
            skipped += 1
            continue
        trace = TraceResult.model_validate_json(tp.read_text())
        result = evaluate(scenario, trace)
        store.update_correctness_score(row["result_id"], result.correctness_score)
        updated += 1
    return updated, skipped
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_backfill_correctness.py -v`
Expected: PASS

- [ ] **Step 6: Add the `backfill-correctness` CLI command**

In `toolery/cli.py`, add a command (uses `load_all_scenarios`, `_store`, `_results_dir`, `console` already in this module; add `from toolery.core.scenario import load_all_scenarios` if not already imported at top):

```python
@app.command(name="backfill-correctness")
def backfill_correctness(
    scenarios_dir: Path = typer.Option(Path("scenarios")),  # noqa: B008
):
    """Recompute correctness_score for every existing run from stored traces."""
    from toolery.core.scenario import load_all_scenarios

    store = _store()
    store.init_schema()  # ensure the column exists on old DBs
    scenarios = {s.id: s for s in load_all_scenarios(scenarios_dir)}
    results_dir = _results_dir()
    total_u = total_s = 0
    for run in store.fetch_all_runs():
        u, s = _backfill_correctness_run(store, run["run_id"], results_dir, scenarios)
        total_u += u
        total_s += s
        console.print(f"  {run['run_id']}: updated {u}, skipped {s}")
    console.print(f"[green]✓ backfill done: updated {total_u}, skipped {total_s}[/green]")
```

- [ ] **Step 7: Verify the command runs against the real DB**

Run: `.venv/bin/toolery backfill-correctness`
Expected: per-run lines then `✓ backfill done: updated N, skipped M`, with N > 0 for the MiniMax run.

Then sanity-check the headline insight is now queryable:
```bash
sqlite3 results/runs.db "
SELECT adapter, printf('%.3f', AVG(score)) score, printf('%.3f', AVG(correctness_score)) correctness
FROM scenario_results
WHERE run_id='2026-05-31T19-48_MiniMax-M2.7-AWQ-4bit' GROUP BY adapter;"
```
Expected: hermes `correctness` noticeably higher than its `score`; raw `correctness` ≈ its `score`.

- [ ] **Step 8: Commit**

```bash
git add toolery/core/store.py toolery/cli.py tests/test_backfill_correctness.py
git commit -m "feat(cli): backfill-correctness command + store.update_correctness_score"
```

---

## Task 5: Reporting — `compute_correctness_breakdown` + `correctness-report` CLI

**Files:**
- Modify: `toolery/rankings/compute.py` (new function near `compute_failure_breakdown` ~293-316)
- Modify: `toolery/cli.py` (new `correctness-report` command)
- Test: `tests/rankings/test_correctness_breakdown.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/rankings/test_correctness_breakdown.py`:

```python
from toolery.core.models import (
    Budget, Category, Message, Scenario, Scoring, ScoringCheck, Tier, ToolCall, TraceResult,
)
from toolery.core.scorer import evaluate
from toolery.core.store import Store
from toolery.rankings.compute import compute_correctness_breakdown


def _scenario():
    return Scenario(
        id="cb-01", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(required=[ScoringCheck.model_validate(
            {"check": "tool_called", "tool": "get_weather"})]),
    )


def _trace(n_calls):
    return TraceResult(
        scenario_id="cb-01", adapter="hermes", trial_index=0,
        messages=[Message(role="assistant", content="ok")],
        tool_calls=[ToolCall(index=i, name="get_weather", args={"location": "Warsaw"})
                    for i in range(n_calls)],
        final_response="ok", started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_correctness_breakdown_reports_score_and_correctness(tmp_path):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    store.create_run(run_id="r1", model="m", base_url="u",
                     started_at="2026-05-23T18:00:00Z", config_json="{}", scenarios_hash="h")
    # budget-violated-correct: 2 calls > budget 1 → score 0, correctness 1.0
    res = evaluate(_scenario(), _trace(2))
    store.write_scenario_result("r1", res, tags=[], ranking_dims=["overall"],
                                scenario_hash="h", category="tool_selection", tier="easy",
                                trace_path="traces/x.json")

    out = compute_correctness_breakdown(store)
    row = out[("m", "hermes")]
    assert row["n"] == 1
    assert row["score_mean"] == 0.0
    assert row["correctness_mean"] == 1.0
    assert row["solved_not_scored"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rankings/test_correctness_breakdown.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_correctness_breakdown'`.

- [ ] **Step 3: Implement `compute_correctness_breakdown`**

In `toolery/rankings/compute.py`, add after `compute_failure_breakdown`:

```python
def compute_correctness_breakdown(store: Store) -> dict[tuple[str, str], dict[str, float | int]]:
    """Per (model, adapter): mean budgeted score vs mean budget-independent
    correctness_score, plus how many results were solved-but-not-scored
    (correctness full while headline score fell short — typically budget overrun)."""
    runs = {r["run_id"]: r for r in store.fetch_all_runs()}
    agg: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {"n": 0, "score_sum": 0.0, "correctness_sum": 0.0, "solved_not_scored": 0})
    with store.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM scenario_results").fetchall()]
    for row in rows:
        meta = runs.get(row["run_id"])
        if not meta:
            continue
        corr = row.get("correctness_score")
        if corr is None:
            continue
        key = (meta["model"], row["adapter"])
        a = agg[key]
        a["n"] += 1
        a["score_sum"] += row["score"]
        a["correctness_sum"] += corr
        if corr >= 1.0 and row["score"] < 1.0:
            a["solved_not_scored"] += 1
    out: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, a in agg.items():
        n = a["n"] or 1
        out[key] = {
            "n": a["n"],
            "score_mean": a["score_sum"] / n,
            "correctness_mean": a["correctness_sum"] / n,
            "solved_not_scored": a["solved_not_scored"],
        }
    return out
```

(`defaultdict` and `Store` are already imported at the top of `compute.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rankings/test_correctness_breakdown.py -v`
Expected: PASS

- [ ] **Step 5: Add the `correctness-report` CLI command**

In `toolery/cli.py` add (uses `rich.table.Table`; import at top of the command):

```python
@app.command(name="correctness-report")
def correctness_report():
    """Show budgeted score vs budget-independent correctness per (model, adapter)."""
    from rich.table import Table

    from toolery.rankings.compute import compute_correctness_breakdown

    data = compute_correctness_breakdown(_store())
    table = Table(title="Score vs correctness (budget-independent)")
    for col in ("model", "adapter", "n", "score", "correctness", "solved-not-scored"):
        table.add_column(col)
    for (model, adapter), v in sorted(data.items()):
        table.add_row(model, adapter, str(v["n"]),
                      f"{v['score_mean']:.3f}", f"{v['correctness_mean']:.3f}",
                      str(v["solved_not_scored"]))
    console.print(table)
```

- [ ] **Step 6: Verify the report against the backfilled DB**

Run: `.venv/bin/toolery correctness-report`
Expected: a table; for the MiniMax run, hermes `correctness` > `score` and `solved-not-scored` ≈ 9; raw `correctness` ≈ `score` with `solved-not-scored` ≈ 1.

- [ ] **Step 7: Commit**

```bash
git add toolery/rankings/compute.py toolery/cli.py tests/rankings/test_correctness_breakdown.py
git commit -m "feat(rankings): correctness breakdown + correctness-report command"
```

---

## Task 6: Full-suite regression check

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests PASS (baseline was 271 before this feature; expect 271 + the new tests).

- [ ] **Step 2: If green, no commit needed (already committed per task).** If anything fails, fix with TDD before proceeding.

---

## Notes / deviations from spec

- The spec named `markdown.py` as a reporting surface. This plan delivers the
  reporting via `compute_correctness_breakdown` + a `correctness-report` CLI
  table (the path the user actually queries today). Threading correctness into
  the Jinja summary template is deferred as a follow-up to avoid guessing at
  fragile template internals — same rationale as deferring the TUI.

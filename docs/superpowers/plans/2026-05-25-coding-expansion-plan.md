# Coding Suite Expansion + Per-Dim Weights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the `coding` category from 5 → 13 scenarios with calibrated difficulty gradient, retrofit partial-credit checks on existing 5 scenarios, and add per-dimension weighting in `Overall` (`coding`/`terminal`/`agentic` ×2, `localization`/`long_context` ×0.5).

**Architecture:** Add `_DIM_WEIGHTS` constant + `_scenario_dim_weight()` helper to `llm_test/rankings/compute.py`. Plumb `ranking_dims_json` through the per-(model, adapter) aggregation tuples. Apply the dim weight ONLY when computing the `overall` dimension — other dimensions stay tier-weighted only. Add 8 new YAML scenarios. Audit & retrofit 4 existing scenarios with sparse `partial:` blocks.

**Tech Stack:** Python 3.12 (compute.py), YAML scenarios, pytest.

**Spec:** `docs/superpowers/specs/2026-05-25-coding-expansion-design.md`

---

## File Map

**Create:**
- `tests/rankings/__init__.py` (if missing)
- `tests/rankings/test_dim_weights.py` — new tests
- `scenarios/easy/easy-15-coding-pyproject-name.yaml`
- `scenarios/easy/easy-16-coding-pytest-count.yaml`
- `scenarios/easy/easy-17-coding-find-failing-test.yaml`
- `scenarios/easy/easy-18-coding-import-error-module.yaml`
- `scenarios/medium/medium-31-coding-fix-syntax-then-test.yaml`
- `scenarios/medium/medium-32-coding-write-test-from-fn.yaml`
- `scenarios/very_hard/very-hard-11-coding-bug-bisect.yaml`
- `scenarios/very_hard/very-hard-12-coding-extract-helper.yaml`

**Modify:**
- `llm_test/rankings/compute.py` — `_DIM_WEIGHTS`, `_scenario_dim_weight()`, plumb `ranking_dims_json`, gated apply in Overall
- `scenarios/easy/easy-02-read-before-write.yaml` — retrofit `partial:`
- `scenarios/medium/medium-01-git-hygiene.yaml` — retrofit `partial:`
- `scenarios/hard/hard-01-tdd-fix-loop.yaml` — retrofit `partial:`
- `scenarios/hard/hard-02-multi-file-rename.yaml` — retrofit `partial:`
- `README.md` — scenario count 75 → 83, Coding row `7 → 13`, document weight scheme

**Skipped** (already adequate):
- `scenarios/medium/medium-02-tdd-explain.yaml` — has 2 partial checks (call_count + response_contains).

---

## Task 1: Add `_DIM_WEIGHTS` constant and helper

**Files:**
- Modify: `llm_test/rankings/compute.py` (add constants after `_TIER_WEIGHTS` line ~119)
- Test: `tests/rankings/test_dim_weights.py` (new)

- [ ] **Step 1: Create the test directory and __init__**

```bash
cd /home/rahueme/LLM-test && mkdir -p tests/rankings && touch tests/rankings/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `/home/rahueme/LLM-test/tests/rankings/test_dim_weights.py`:

```python
from llm_test.rankings.compute import _DIM_WEIGHTS, _scenario_dim_weight


def test_dim_weights_constant_has_expected_keys():
    assert _DIM_WEIGHTS["coding"] == 2.0
    assert _DIM_WEIGHTS["terminal"] == 2.0
    assert _DIM_WEIGHTS["agentic"] == 2.0
    assert _DIM_WEIGHTS["localization"] == 0.5
    assert _DIM_WEIGHTS["long_context"] == 0.5


def test_scenario_dim_weight_unknown_dim_defaults_to_one():
    assert _scenario_dim_weight(["overall", "restraint"]) == 1.0


def test_scenario_dim_weight_picks_max():
    assert _scenario_dim_weight(["overall", "coding", "agentic"]) == 2.0


def test_scenario_dim_weight_mixed_high_and_low_picks_max():
    # max(coding=2.0, localization=0.5) = 2.0
    assert _scenario_dim_weight(["overall", "coding", "localization"]) == 2.0


def test_scenario_dim_weight_localization_only():
    assert _scenario_dim_weight(["overall", "localization"]) == 0.5


def test_scenario_dim_weight_empty_or_overall_only_defaults_to_one():
    assert _scenario_dim_weight([]) == 1.0
    assert _scenario_dim_weight(["overall"]) == 1.0
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -5
```
Expected: FAIL — `ImportError: cannot import name '_DIM_WEIGHTS' from 'llm_test.rankings.compute'`.

- [ ] **Step 4: Add the constant and helper in compute.py**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, just AFTER the existing `_TIER_WEIGHTS = ...` line (around line 119) and BEFORE `def compute_matrix(...)`, INSERT:

```python
# Per-dimension weights applied ONLY when computing the `overall` dimension —
# every other column (Coding, Terminal, etc.) keeps the existing tier-only weighting
# so that e.g. a model's `Coding` score reflects raw coding performance, not blend.
# A scenario's overall weight is the MAX of the weights of its non-overall dims
# (fallback 1.0 if none of its dims are in this map).
_DIM_WEIGHTS: dict[str, float] = {
    "coding": 2.0,
    "terminal": 2.0,
    "agentic": 2.0,
    "localization": 0.5,
    "long_context": 0.5,
}


def _scenario_dim_weight(ranking_dims: list[str]) -> float:
    weights = [_DIM_WEIGHTS.get(d, 1.0) for d in ranking_dims if d != "overall"]
    return max(weights) if weights else 1.0
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -8
```
Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py tests/rankings/__init__.py tests/rankings/test_dim_weights.py && git commit -m "feat(rankings): add _DIM_WEIGHTS constant + _scenario_dim_weight helper"
```

---

## Task 2: Plumb `ranking_dims_json` through compute_matrix data flow

**Files:**
- Modify: `llm_test/rankings/compute.py:163-194` (compute_matrix's per-pair loop and per-run reduction)

Goal: Currently `by_run[rid]` stores `(score, tier)` tuples. We need to also carry `ranking_dims_json` so the per-scenario weight can be applied. This task plumbs the data; Task 3 applies the weight.

- [ ] **Step 1: Read the current compute_matrix to confirm line numbers**

```bash
cd /home/rahueme/LLM-test && sed -n '155,200p' llm_test/rankings/compute.py
```

Confirm you see the line `by_run[it["run_id"]].append((it["score"], it["tier"]))` around line 178.

- [ ] **Step 2: Modify the data structure to carry ranking_dims_json**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, find the `for r in all_results:` loop in `compute_matrix` (~line 150). Modify the `pairs[...][dim].append(...)` call (lines 163-168) to also store `ranking_dims_json`:

```python
            pairs[(model, adapter)][dim].append({
                "run_id": r["run_id"],
                "started_at": meta["started_at"],
                "score": r["score"],
                "tier": r["tier"],
                "ranking_dims_json": r["ranking_dims_json"] or "[]",
            })
```

Then find the per-pair reduction block (~lines 174-194). Change the `by_run` typing and population to carry the dims too:

```python
        for dim, items in dim_results.items():
            by_run: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
            run_started: dict[str, str] = {}
            for it in items:
                by_run[it["run_id"]].append((it["score"], it["tier"], it["ranking_dims_json"]))
                run_started[it["run_id"]] = it["started_at"]
            runs_sorted = sorted(by_run.keys(), key=lambda rid: run_started[rid], reverse=True)
            recent = runs_sorted[:history_window_runs]
            decay_pairs: list[tuple[float, float]] = []
            for rid in recent:
                items_in_run = by_run[rid]
                w_sum = sum(_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items_in_run)
                if w_sum <= 0:
                    continue
                tier_weighted_mean = (
                    sum(s * _TIER_WEIGHTS.get(t, 1.0) for s, t, _ in items_in_run) / w_sum
                )
                age = max((now - _parse_iso(run_started[rid])).total_seconds() / 86400, 0)
                decay_pairs.append((tier_weighted_mean, age))
            if decay_pairs:
                scores[dim] = decay_weighted_mean(decay_pairs, half_life_days)
            total_runs = max(total_runs, len(runs_sorted))
```

This intentionally ignores the new third tuple element (`_`) for now. Task 3 will activate it for `overall`.

- [ ] **Step 3: Run compute_matrix-affected tests to confirm no regression**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/rankings/ tests/tui/ tests/test_cli.py 2>&1 | tail -10
```
Expected: tests pass (or pre-existing pytest-asyncio failures from earlier work unchanged — those are not our concern).

- [ ] **Step 4: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py && git commit -m "refactor(rankings): plumb ranking_dims_json through compute_matrix tuples"
```

---

## Task 3: Apply `_scenario_dim_weight` gated on `dim == "overall"` in compute_matrix

**Files:**
- Modify: `llm_test/rankings/compute.py:174-194` (per-pair reduction)
- Modify: `tests/rankings/test_dim_weights.py` (add integration tests)

- [ ] **Step 1: Write the failing integration test**

Append to `/home/rahueme/LLM-test/tests/rankings/test_dim_weights.py`:

```python
import json
import tempfile
from pathlib import Path

from llm_test.core.store import Store
from llm_test.rankings.compute import compute_matrix


def _seed_store(store: Store, results: list[dict]) -> None:
    """Helper: seed runs and scenario_results tables for tests."""
    now_iso = "2026-05-25T12:00:00+00:00"
    run_id = "test-run-1"
    with store.conn() as c:
        c.execute(
            "INSERT INTO runs (run_id, started_at, model, status, scenarios_hash) VALUES (?, ?, ?, ?, ?)",
            (run_id, now_iso, "test-model", "done", "h"),
        )
        for i, r in enumerate(results):
            dims_json = json.dumps(r.get("dims", ["overall"]))
            c.execute(
                """INSERT INTO scenario_results
                   (run_id, scenario_id, adapter, tier, score, ranking_dims_json,
                    call_count, budget_max, latency_ms, status, failure_kind)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, f"s-{i}", "raw", r.get("tier", "easy"),
                 r.get("score", 1.0), dims_json,
                 0, 0, 0, "pass", None),
            )


def test_overall_applies_dim_weight_for_coding():
    """Coding-tagged easy scenario contributes 2× vs untagged easy."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        # Two easy scenarios, both perfect score (1.0); one tagged 'coding'.
        # Overall should be 1.0 (both pass).
        _seed_store(store, [
            {"score": 1.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        assert len(matrix) == 1
        assert matrix[0]["scores"]["overall"] == 1.0


def test_overall_coding_failure_weighted_more_than_default():
    """Coding scenario fails (0.0), default passes (1.0). Coding 2× weight."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},     # weight 2.0
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},               # weight 1.0
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        # Expected: (0.0*1*2.0 + 1.0*1*1.0) / (1*2.0 + 1*1.0) = 1.0/3.0 ≈ 0.333
        assert abs(matrix[0]["scores"]["overall"] - 1.0/3.0) < 0.001


def test_overall_localization_failure_weighted_less_than_default():
    """Localization failure (0.0, weight 0.5) is partly absorbed by default success (1.0)."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "localization"]},  # weight 0.5
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},                  # weight 1.0
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        # Expected: (0.0*1*0.5 + 1.0*1*1.0) / (1*0.5 + 1*1.0) = 1.0/1.5 ≈ 0.667
        assert abs(matrix[0]["scores"]["overall"] - 1.0/1.5) < 0.001


def test_coding_column_unaffected_by_dim_weights():
    """The 'coding' column itself stays raw tier-weighted; dim weights only kick in for overall."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        # Two coding scenarios — one passes, one fails. Equal tier weight.
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall", "coding"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["coding"])
        # Raw mean: 0.5. Dim weight (2.0) applies to both, so it cancels — but more
        # importantly we verify the math still produces 0.5, not something else.
        assert abs(matrix[0]["scores"]["coding"] - 0.5) < 0.001
```

- [ ] **Step 2: Run tests — verify they fail (weights not yet applied)**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -15
```
Expected: `test_overall_coding_failure_weighted_more_than_default` and `test_overall_localization_failure_weighted_less_than_default` FAIL (they expect weight application that Task 3 introduces). Others PASS.

If the integration tests fail because of DB schema issues (e.g. missing columns), check the Store schema and adjust the INSERT statements. Run `sqlite3 /tmp/test.db ".schema"` after creating a fresh Store to see columns.

- [ ] **Step 3: Apply the dim weight in compute_matrix, gated on `dim == "overall"`**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, replace the per-run reduction block (the one inside `for rid in recent:`) inside `compute_matrix`. Change:

```python
            for rid in recent:
                items_in_run = by_run[rid]
                w_sum = sum(_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items_in_run)
                if w_sum <= 0:
                    continue
                tier_weighted_mean = (
                    sum(s * _TIER_WEIGHTS.get(t, 1.0) for s, t, _ in items_in_run) / w_sum
                )
                age = max((now - _parse_iso(run_started[rid])).total_seconds() / 86400, 0)
                decay_pairs.append((tier_weighted_mean, age))
```

to:

```python
            for rid in recent:
                items_in_run = by_run[rid]
                if dim == "overall":
                    weights = [
                        _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d))
                        for _, t, d in items_in_run
                    ]
                else:
                    weights = [_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items_in_run]
                w_sum = sum(weights)
                if w_sum <= 0:
                    continue
                weighted_mean = sum(
                    s * w for (s, _, _), w in zip(items_in_run, weights)
                ) / w_sum
                age = max((now - _parse_iso(run_started[rid])).total_seconds() / 86400, 0)
                decay_pairs.append((weighted_mean, age))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -15
```
Expected: all PASS (6 unit + 4 integration = 10).

- [ ] **Step 5: Regression check on other ranking tests**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/rankings/ tests/tui/ tests/test_cli.py 2>&1 | tail -10
```
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py tests/rankings/test_dim_weights.py && git commit -m "feat(rankings): apply per-dim weights in Overall (coding/terminal/agentic 2x, localization/long_context 0.5x)"
```

---

## Task 4: Mirror the dim-weight logic into `regenerate_rankings`

**Files:**
- Modify: `llm_test/rankings/compute.py:60-78` (regenerate_rankings's per-pair loop)

`regenerate_rankings` writes per-dimension markdown files (`overall.md`, `coding.md`, etc.) using a separate code path from `compute_matrix`. We need the same `_scenario_dim_weight` logic applied to its `overall.md` computation.

- [ ] **Step 1: Read the existing block**

```bash
cd /home/rahueme/LLM-test && sed -n '40,80p' llm_test/rankings/compute.py
```

You should see a loop with `by_adapter[r["adapter"]].append((r["score"], r["tier"]))` around line 53.

- [ ] **Step 2: Carry ranking_dims_json through and apply weight when dim=="overall"**

Edit `/home/rahueme/LLM-test/llm_test/rankings/compute.py`. Two changes inside `regenerate_rankings`:

**(a)** Change the `by_adapter` append (~line 53) to carry dims:

```python
            by_adapter: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
            for r in rs:
                by_adapter[r["adapter"]].append((r["score"], r["tier"], r["ranking_dims_json"] or "[]"))
            for adapter, scored in by_adapter.items():
                per_pair_runs[(model, adapter)].append({
                    "run_id": run_id, "scores": scored,
                    "started_at": meta["started_at"],
                })
```

**(b)** Apply the dim weight inside the per-run reduction (~lines 65-75). Replace:

```python
            for r in recent:
                items = r["scores"]  # list of (score, tier) tuples
                w_sum = sum(_TIER_WEIGHTS.get(t, 1.0) for _, t in items)
                if w_sum <= 0:
                    continue
                run_mean = (
                    sum(s * _TIER_WEIGHTS.get(t, 1.0) for s, t in items) / w_sum
                )
                age_days = max((now - _parse_iso(r["started_at"])).total_seconds() / 86400, 0)
                pairs.append((run_mean, age_days))
```

with:

```python
            for r in recent:
                items = r["scores"]  # list of (score, tier, ranking_dims_json) tuples
                if dim == "overall":
                    weights_per_item = [
                        _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d))
                        for _, t, d in items
                    ]
                else:
                    weights_per_item = [_TIER_WEIGHTS.get(t, 1.0) for _, t, _ in items]
                w_sum = sum(weights_per_item)
                if w_sum <= 0:
                    continue
                run_mean = sum(
                    s * w for (s, _, _), w in zip(items, weights_per_item)
                ) / w_sum
                age_days = max((now - _parse_iso(r["started_at"])).total_seconds() / 86400, 0)
                pairs.append((run_mean, age_days))
```

- [ ] **Step 3: Smoke test regenerate_rankings end-to-end against the real DB**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.rankings.compute import regenerate_rankings
from llm_test.core.store import Store
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    s = Store(Path('results/runs.db'))
    regenerate_rankings(store=s, dimensions=['overall','coding','terminal','localization'], out_dir=out)
    print('files:', sorted(p.name for p in out.iterdir()))
"
```
Expected: `files: ['coding.md', 'localization.md', 'overall.md', 'terminal.md']`.

- [ ] **Step 4: Regression**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/rankings/ 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py && git commit -m "feat(rankings): apply per-dim weights in regenerate_rankings overall path"
```

---

## Task 5: Retrofit `partial:` block in `easy-02-read-before-write.yaml`

**Files:**
- Modify: `scenarios/easy/easy-02-read-before-write.yaml`

The existing `partial:` has only `call_count_at_most: 2`. Add 2 more checks to widen gradient.

- [ ] **Step 1: Apply the edit**

In `/home/rahueme/LLM-test/scenarios/easy/easy-02-read-before-write.yaml`, replace the existing `partial:` block:

```yaml
  partial:
    - check: call_count_at_most
      n: 2
```

with:

```yaml
  partial:
    - check: call_count_at_most
      n: 2
    - check: tool_args_contain
      tool: read_file
      args: { path: "/workspace/notes.md" }
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: "Existing content"
```

The new checks reward: (1) reading the right file explicitly, (2) preserving the existing content (the whole point of "read before write").

- [ ] **Step 2: Validate scenario parses**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/easy/easy-02-read-before-write.yaml && git commit -m "feat(scenarios): retrofit partial-credit checks in easy-02-read-before-write"
```

---

## Task 6: Retrofit `partial:` in `medium-01-git-hygiene.yaml`

**Files:**
- Modify: `scenarios/medium/medium-01-git-hygiene.yaml`

- [ ] **Step 1: Apply the edit**

Replace the `partial:` block (currently `- check: call_count_at_most\n      n: 3`):

```yaml
  partial:
    - check: call_count_at_most
      n: 3
    - check: tool_called
      tool: git_status
    - check: tool_args_match_regex
      tool: git_commit
      arg: message
      pattern: "(util|helper|add).*[a-zA-Z]"
```

Rewards: (1) discovering state before acting (`git_status`), (2) writing a non-trivial commit message that mentions what was added.

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/medium/medium-01-git-hygiene.yaml && git commit -m "feat(scenarios): retrofit partial-credit checks in medium-01-git-hygiene"
```

---

## Task 7: Retrofit `partial:` in `hard-01-tdd-fix-loop.yaml`

**Files:**
- Modify: `scenarios/hard/hard-01-tdd-fix-loop.yaml`

- [ ] **Step 1: Apply the edit**

Replace the `partial:` block (currently `- check: call_count_at_most\n      n: 4`):

```yaml
  partial:
    - check: call_count_at_most
      n: 4
    - check: tool_called
      tool: run_tests
    - check: tool_args_contain
      tool: edit_file
      args: { path: "src/parser.py" }
    - check: tool_args_match_regex
      tool: edit_file
      arg: content
      pattern: "(?s)(int\\(s\\)|abs.*neg|sign|startswith|s\\[0\\])"
```

Rewards: (1) ran the tests at all, (2) edited the right file, (3) edit content suggests an attempt at fixing the sign-handling bug.

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/hard/hard-01-tdd-fix-loop.yaml && git commit -m "feat(scenarios): retrofit partial-credit checks in hard-01-tdd-fix-loop"
```

---

## Task 8: Retrofit `partial:` in `hard-02-multi-file-rename.yaml`

**Files:**
- Modify: `scenarios/hard/hard-02-multi-file-rename.yaml`

- [ ] **Step 1: Apply the edit**

Replace the `partial:` block (currently `- check: call_count_at_most\n      n: 7`):

```yaml
  partial:
    - check: call_count_at_most
      n: 7
    - check: tool_args_match_regex
      tool: edit_file
      arg: path
      pattern: "src/core/billing\\.py|src/api/handlers\\.py|src/cli\\.py|tests/test_billing\\.py"
    - check: tool_args_match_regex
      tool: edit_file
      arg: content
      pattern: "calculate_total"
```

Rewards: (1) edited at least one of the four right files, (2) the edit content actually has the new name.

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/hard/hard-02-multi-file-rename.yaml && git commit -m "feat(scenarios): retrofit partial-credit checks in hard-02-multi-file-rename"
```

---

## Task 9: Create `easy-15-coding-pyproject-name.yaml`

**Files:**
- Create: `scenarios/easy/easy-15-coding-pyproject-name.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/easy/easy-15-coding-pyproject-name.yaml`:

```yaml
id: easy-15-coding-pyproject-name
title: "Find project name in pyproject.toml"
tier: easy
category: coding
domain: dev_ops
description: |
  Model reads pyproject.toml and reports the project name. Single tool call,
  simple TOML structure — baseline easy.
tags: [coding, file_ops, single_turn]
ranking_dimensions: [overall, coding]
prompt: "What is the project name declared in /workspace/pyproject.toml?"
tools: [read_file, list_files]
budget:
  max_tool_calls: 2
  max_turns: 2
  timeout_seconds: 30
tool_responses:
  read_file:
    - match: { path: "/workspace/pyproject.toml" }
      returns: |
        [build-system]
        requires = ["setuptools", "wheel"]

        [project]
        name = "llm-test"
        version = "0.3.0"
        requires-python = ">=3.12"
    - match: any
      returns: { error: "file not found" }
  list_files:
    - match: any
      returns: ["pyproject.toml", "src/", "tests/"]
scoring:
  required:
    - check: tool_called
      tool: read_file
    - check: response_satisfies
      all_of: ["llm-test"]
      none_of: ["error", "not found"]
    - check: no_hallucinated_tool
      allowed: [read_file, list_files]
  forbidden: []
  partial:
    - check: tool_args_contain
      tool: read_file
      args: { path: "/workspace/pyproject.toml" }
    - check: call_count_at_most
      n: 1
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3 && python3 -c "from llm_test.core.scenario import load_all_scenarios; from pathlib import Path; print([s.id for s in load_all_scenarios(Path('scenarios')) if s.id == 'easy-15-coding-pyproject-name'])"
```
Expected: PASS + `['easy-15-coding-pyproject-name']`.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/easy/easy-15-coding-pyproject-name.yaml && git commit -m "feat(scenarios): add easy-15-coding-pyproject-name"
```

---

## Task 10: Create `easy-16-coding-pytest-count.yaml`

**Files:**
- Create: `scenarios/easy/easy-16-coding-pytest-count.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/easy/easy-16-coding-pytest-count.yaml`:

```yaml
id: easy-16-coding-pytest-count
title: "Run pytest, count passing tests"
tier: easy
category: coding
domain: dev_ops
description: |
  Model runs the test suite and reports the count of passing tests from the
  pytest summary line. Single tool call, trivial parsing of the canonical
  "N passed in X.Ys" tail.
tags: [coding, parse_output, single_turn]
ranking_dimensions: [overall, coding]
prompt: "Run the test suite and tell me how many tests passed."
tools: [bash_exec]
budget:
  max_tool_calls: 2
  max_turns: 2
  timeout_seconds: 30
tool_responses:
  bash_exec:
    - match: { command_regex: 'pytest' }
      returns:
        stdout: "..... [100%]\n5 passed in 0.42s\n"
        stderr: ""
        exit_code: 0
        duration_ms: 420
    - match: any
      returns: { stdout: "", stderr: "command not found", exit_code: 127, duration_ms: 2 }
scoring:
  required:
    - check: tool_called
      tool: bash_exec
    - check: response_satisfies
      all_of: ["5"]
      none_of: ["error", "fail"]
    - check: no_hallucinated_tool
      allowed: [bash_exec]
  forbidden: []
  partial:
    - check: command_regex_match
      pattern: 'pytest'
    - check: call_count_at_most
      n: 1
    - check: response_satisfies
      any_of:
        - ["pass", "passed", "test"]
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/easy/easy-16-coding-pytest-count.yaml && git commit -m "feat(scenarios): add easy-16-coding-pytest-count"
```

---

## Task 11: Create `easy-17-coding-find-failing-test.yaml`

**Files:**
- Create: `scenarios/easy/easy-17-coding-find-failing-test.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/easy/easy-17-coding-find-failing-test.yaml`:

```yaml
id: easy-17-coding-find-failing-test
title: "Identify which pytest test failed"
tier: easy
category: coding
domain: dev_ops
description: |
  Stretched easy. Model runs pytest, parses mixed PASS/FAIL output, and
  identifies the specific test that failed (not just count). Requires
  understanding pytest's "FAILED tests/X::Y" format, not just the summary.
tags: [coding, parse_output, single_turn, stretched_easy]
ranking_dimensions: [overall, coding]
prompt: "Run the test suite. If anything failed, tell me which test."
tools: [bash_exec]
budget:
  max_tool_calls: 2
  max_turns: 2
  timeout_seconds: 30
tool_responses:
  bash_exec:
    - match: { command_regex: 'pytest' }
      returns:
        stdout: "tests/test_calc.py ......F                                       [100%]\n\n=========================== FAILURES ===========================\n____________________ test_divide_by_zero ____________________\n\nE   ZeroDivisionError: integer division or modulo by zero\n\n=========================== short test summary info ===========================\nFAILED tests/test_calc.py::test_divide_by_zero - ZeroDivisionError: integer division or modulo by zero\n7 passed, 1 failed in 0.61s\n"
        stderr: ""
        exit_code: 1
        duration_ms: 610
    - match: any
      returns: { stdout: "", stderr: "", exit_code: 0, duration_ms: 5 }
scoring:
  required:
    - check: tool_called
      tool: bash_exec
    - check: response_satisfies
      all_of: ["test_divide_by_zero"]
      any_of:
        - ["test_calc", "tests/test_calc.py"]
    - check: no_hallucinated_tool
      allowed: [bash_exec]
  forbidden: []
  partial:
    - check: command_regex_match
      pattern: 'pytest'
    - check: response_satisfies
      any_of:
        - ["fail", "FAIL", "failed"]
    - check: response_satisfies
      any_of:
        - ["ZeroDivisionError", "zero division", "divide", "division by zero"]
    - check: call_count_at_most
      n: 1
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/easy/easy-17-coding-find-failing-test.yaml && git commit -m "feat(scenarios): add easy-17-coding-find-failing-test (stretched easy)"
```

---

## Task 12: Create `easy-18-coding-import-error-module.yaml`

**Files:**
- Create: `scenarios/easy/easy-18-coding-import-error-module.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/easy/easy-18-coding-import-error-module.yaml`:

```yaml
id: easy-18-coding-import-error-module
title: "Identify missing module from Python traceback"
tier: easy
category: coding
domain: dev_ops
description: |
  Stretched easy. Model runs a Python script that fails with ModuleNotFoundError;
  it must read the traceback and identify the specific module name that's missing.
  Tests traceback comprehension, not just running the script.
tags: [coding, parse_output, single_turn, stretched_easy]
ranking_dimensions: [overall, coding]
prompt: "Try running /workspace/script.py. If it fails to start, tell me what dependency is missing."
tools: [bash_exec]
budget:
  max_tool_calls: 2
  max_turns: 2
  timeout_seconds: 30
tool_responses:
  bash_exec:
    - match: { command_regex: 'python\s.*script\.py' }
      returns:
        stdout: ""
        stderr: "Traceback (most recent call last):\n  File \"/workspace/script.py\", line 3, in <module>\n    import requests\nModuleNotFoundError: No module named 'requests'\n"
        exit_code: 1
        duration_ms: 90
    - match: any
      returns: { stdout: "", stderr: "no such file", exit_code: 2, duration_ms: 2 }
scoring:
  required:
    - check: tool_called
      tool: bash_exec
    - check: response_satisfies
      all_of: ["requests"]
      none_of: ["syntax", "not a python", "permission denied"]
    - check: no_hallucinated_tool
      allowed: [bash_exec]
  forbidden: []
  partial:
    - check: response_satisfies
      any_of:
        - ["import", "ImportError", "ModuleNotFoundError", "module"]
    - check: response_satisfies
      any_of:
        - ["pip install", "install", "missing", "dependency"]
    - check: call_count_at_most
      n: 1
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/easy/easy-18-coding-import-error-module.yaml && git commit -m "feat(scenarios): add easy-18-coding-import-error-module (stretched easy)"
```

---

## Task 13: Create `medium-31-coding-fix-syntax-then-test.yaml`

**Files:**
- Create: `scenarios/medium/medium-31-coding-fix-syntax-then-test.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/medium/medium-31-coding-fix-syntax-then-test.yaml`:

```yaml
id: medium-31-coding-fix-syntax-then-test
title: "Fix a syntax error, then verify with pytest"
tier: medium
category: coding
domain: dev_ops
description: |
  Stretched medium (close to hard). Model must read a file with a syntax error
  (missing colon on a def), write a fixed version, then run pytest to confirm
  the fix. 3-step chain (read → write → test) with intermediate decision-making.
tags: [coding, multi_step, fix_then_verify, stretched_medium]
ranking_dimensions: [overall, coding, agentic]
prompt: "The file /workspace/calc.py has a syntax error. Fix it, then run the test suite to confirm everything passes."
tools: [read_file, write_file, bash_exec, list_files]
budget:
  max_tool_calls: 4
  max_turns: 4
  timeout_seconds: 60
tool_responses:
  read_file:
    - match: { path: "/workspace/calc.py" }
      returns: |
        def add(a, b)
            return a + b


        def sub(a, b):
            return a - b
    - match: any
      returns: { error: "file not found" }
  write_file:
    - match: { path: "/workspace/calc.py" }
      returns: { ok: true, bytes_written: 64 }
    - match: any
      returns: { error: "not allowed" }
  bash_exec:
    - match: { command_regex: 'pytest' }
      returns:
        stdout: "....                                                          [100%]\n4 passed in 0.31s\n"
        stderr: ""
        exit_code: 0
        duration_ms: 310
    - match: any
      returns: { stdout: "", stderr: "not found", exit_code: 127, duration_ms: 2 }
  list_files:
    - match: any
      returns: ["calc.py", "tests/"]
scoring:
  required:
    - check: tool_called_in_order
      sequence: [read_file, write_file, bash_exec]
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: 'def add\(a, b\):'
    - check: response_satisfies
      all_of: ["4"]
      any_of:
        - ["pass", "passed", "green", "success"]
    - check: no_hallucinated_tool
      allowed: [read_file, write_file, bash_exec, list_files]
  forbidden: []
  partial:
    - check: tool_args_contain
      tool: read_file
      args: { path: "/workspace/calc.py" }
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: 'return\s+a\s*\+\s*b'
    - check: command_regex_match
      pattern: 'pytest'
    - check: call_count_at_most
      n: 3
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/medium/medium-31-coding-fix-syntax-then-test.yaml && git commit -m "feat(scenarios): add medium-31-coding-fix-syntax-then-test (stretched medium)"
```

---

## Task 14: Create `medium-32-coding-write-test-from-fn.yaml`

**Files:**
- Create: `scenarios/medium/medium-32-coding-write-test-from-fn.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/medium/medium-32-coding-write-test-from-fn.yaml`:

```yaml
id: medium-32-coding-write-test-from-fn
title: "Write a pytest test for an existing function, then verify"
tier: medium
category: coding
domain: dev_ops
description: |
  Stretched medium (close to hard). Read a function with a docstring describing
  its contract (returns a/b, raises ValueError on b==0), write a pytest test
  file that exercises both the happy path and the error path, then run pytest
  to confirm the test is well-formed and green.
tags: [coding, write_test, multi_step, stretched_medium]
ranking_dimensions: [overall, coding, agentic]
prompt: |
  Read /workspace/calculator.py. Write a pytest file at /workspace/tests/test_calculator.py
  with at least 2 test functions covering the documented behavior. Then run pytest
  to confirm your tests pass.
tools: [read_file, write_file, bash_exec, list_files]
budget:
  max_tool_calls: 4
  max_turns: 4
  timeout_seconds: 60
tool_responses:
  read_file:
    - match: { path: "/workspace/calculator.py" }
      returns: |
        def divide(a: float, b: float) -> float:
            """Return a / b. Raises ValueError when b == 0."""
            if b == 0:
                raise ValueError("cannot divide by zero")
            return a / b
    - match: any
      returns: { error: "file not found" }
  write_file:
    - match: { command_regex: 'test_calculator\.py' }
      returns: { ok: true, bytes_written: 256 }
    - match: any
      returns: { ok: true, bytes_written: 0 }
  bash_exec:
    - match: { command_regex: 'pytest' }
      returns:
        stdout: "..                                                            [100%]\n2 passed in 0.18s\n"
        stderr: ""
        exit_code: 0
        duration_ms: 180
    - match: any
      returns: { stdout: "", stderr: "not found", exit_code: 127, duration_ms: 2 }
  list_files:
    - match: any
      returns: ["calculator.py", "tests/"]
scoring:
  required:
    - check: tool_called_in_order
      sequence: [read_file, write_file, bash_exec]
    - check: tool_args_match_regex
      tool: write_file
      arg: path
      pattern: 'test_calculator\.py'
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(?s)def test_.*divide'
    - check: response_satisfies
      any_of:
        - ["pass", "passed", "green", "success"]
    - check: no_hallucinated_tool
      allowed: [read_file, write_file, bash_exec, list_files]
  forbidden: []
  partial:
    - check: tool_args_contain
      tool: read_file
      args: { path: "/workspace/calculator.py" }
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(ValueError|pytest\.raises)'
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(?s)assert.*assert'
    - check: command_regex_match
      pattern: 'pytest'
    - check: call_count_at_most
      n: 3
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/medium/medium-32-coding-write-test-from-fn.yaml && git commit -m "feat(scenarios): add medium-32-coding-write-test-from-fn (stretched medium)"
```

---

## Task 15: Create `very-hard-11-coding-bug-bisect.yaml`

**Files:**
- Create: `scenarios/very_hard/very-hard-11-coding-bug-bisect.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/very_hard/very-hard-11-coding-bug-bisect.yaml`:

```yaml
id: very-hard-11-coding-bug-bisect
title: "Bisect: which of 3 commits broke the test?"
tier: very_hard
category: coding
domain: dev_ops
description: |
  3-commit history; HEAD's pytest is failing. Model must use git log + git
  checkout + pytest in a (preferably bisect-style) sequence to identify
  the offending commit. Tight budget rewards efficient narrowing — random
  linear scan blows the budget.
tags: [coding, debug, multi_step, budget_pressure, bisect]
ranking_dimensions: [overall, coding, agentic, budget_efficiency]
prompt: |
  pytest fails on HEAD. There are 3 recent commits. Find out which commit
  introduced the failing test. Return the short SHA of the offending commit
  and a one-line description.
tools: [bash_exec]
budget:
  max_tool_calls: 6
  max_turns: 6
  timeout_seconds: 60
tool_responses:
  bash_exec:
    - match: { command_regex: '^git\s+log' }
      returns:
        stdout: "c003abc broken: refactor parse_int\nc002def add feature foo\nc001abc initial commit\n"
        stderr: ""
        exit_code: 0
        duration_ms: 12
    - match: { command_regex: 'git\s+checkout\s+c003' }
      returns: { stdout: "HEAD is now at c003abc\n", stderr: "", exit_code: 0, duration_ms: 8 }
    - match: { command_regex: 'git\s+checkout\s+c002' }
      returns: { stdout: "HEAD is now at c002def\n", stderr: "", exit_code: 0, duration_ms: 8 }
    - match: { command_regex: 'git\s+checkout\s+c001' }
      returns: { stdout: "HEAD is now at c001abc\n", stderr: "", exit_code: 0, duration_ms: 8 }
    - match: { command_regex: 'git\s+checkout\s+HEAD' }
      returns: { stdout: "Already on HEAD\n", stderr: "", exit_code: 0, duration_ms: 4 }
    - match: { command_regex: 'git\s+bisect' }
      returns:
        stdout: "Bisecting: 1 revision left to test after this (roughly 1 step)\n[c003abc] broken: refactor parse_int\n"
        stderr: ""
        exit_code: 0
        duration_ms: 15
    - match: { command_regex: 'pytest' }
      call_index: 0
      returns:
        stdout: "FAILED tests/test_parser.py::test_negative_numbers\n0 passed, 1 failed in 0.20s\n"
        stderr: ""
        exit_code: 1
        duration_ms: 200
    - match: { command_regex: 'pytest' }
      call_index: ">=1"
      returns:
        stdout: "1 passed in 0.20s\n"
        stderr: ""
        exit_code: 0
        duration_ms: 200
    - match: any
      returns: { stdout: "", stderr: "unknown command", exit_code: 1, duration_ms: 2 }
scoring:
  required:
    - check: tool_called
      tool: bash_exec
    - check: response_satisfies
      all_of: ["c003"]
      none_of: ["c001", "c002 is the broken"]
    - check: no_hallucinated_tool
      allowed: [bash_exec]
  forbidden: []
  partial:
    - check: command_regex_match
      pattern: '^git\s+log'
    - check: command_regex_match
      pattern: 'git\s+checkout|git\s+bisect'
    - check: call_count_at_most
      n: 5
    - check: response_satisfies
      any_of:
        - ["broken", "refactor", "parse_int", "introduced"]
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

Note on the mock: pytest returns FAIL on the FIRST call (when HEAD/c003 is checked out) and PASS on subsequent calls (after the model checks out c001 or c002 successfully). The model must reason: "first run failed → check earlier commits → if those pass, the bad one is the most recent of the still-failing ones."

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/very_hard/very-hard-11-coding-bug-bisect.yaml && git commit -m "feat(scenarios): add very-hard-11-coding-bug-bisect"
```

---

## Task 16: Create `very-hard-12-coding-extract-helper.yaml`

**Files:**
- Create: `scenarios/very_hard/very-hard-12-coding-extract-helper.yaml`

- [ ] **Step 1: Write the YAML**

Create `/home/rahueme/LLM-test/scenarios/very_hard/very-hard-12-coding-extract-helper.yaml`:

```yaml
id: very-hard-12-coding-extract-helper
title: "Extract a helper function to remove duplication, verify with pytest"
tier: very_hard
category: coding
domain: dev_ops
description: |
  A file has 3 near-duplicate blocks computing `x * 1.08 + shipping`. Model
  must extract a helper function, update the 3 call sites, and confirm pytest
  still passes. Tests refactoring discipline + content-level reasoning.
tags: [coding, refactor, multi_step, budget_pressure]
ranking_dimensions: [overall, coding, agentic, budget_efficiency]
prompt: |
  The file /workspace/order.py has 3 places that compute "x * 1.08 + shipping".
  Extract that into a helper function (e.g. `total_with_tax(x, shipping)`),
  update all 3 call sites to use the helper, and then run pytest to make sure
  nothing breaks.
tools: [read_file, write_file, bash_exec, list_files]
budget:
  max_tool_calls: 5
  max_turns: 5
  timeout_seconds: 90
tool_responses:
  read_file:
    - match: { path: "/workspace/order.py" }
      returns: |
        SHIPPING = 5.0


        def total_a(price):
            return price * 1.08 + SHIPPING


        def total_b(price):
            return price * 1.08 + SHIPPING


        def total_c(price):
            return price * 1.08 + SHIPPING
    - match: any
      returns: { error: "not found" }
  write_file:
    - match: { path: "/workspace/order.py" }
      returns: { ok: true, bytes_written: 320 }
    - match: any
      returns: { ok: true, bytes_written: 0 }
  bash_exec:
    - match: { command_regex: 'pytest' }
      returns:
        stdout: "...                                                           [100%]\n3 passed in 0.22s\n"
        stderr: ""
        exit_code: 0
        duration_ms: 220
    - match: any
      returns: { stdout: "", stderr: "not found", exit_code: 127, duration_ms: 2 }
  list_files:
    - match: any
      returns: ["order.py", "tests/"]
scoring:
  required:
    - check: tool_called_in_order
      sequence: [read_file, write_file, bash_exec]
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(?s)def\s+\w+.*\*\s*1\.08'
    - check: response_satisfies
      any_of:
        - ["pass", "passed", "green", "success"]
    - check: no_hallucinated_tool
      allowed: [read_file, write_file, bash_exec, list_files]
  forbidden: []
  partial:
    - check: tool_args_contain
      tool: read_file
      args: { path: "/workspace/order.py" }
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(?s)def\s+total_a.*\w+\(price'
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: '(?s)def\s+total_b.*\w+\(price'
    - check: command_regex_match
      pattern: 'pytest'
    - check: call_count_at_most
      n: 4
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

The partial checks reward: (1) reading the right file, (2) keeping `total_a` calling the helper (instead of inlining), (3) ditto for `total_b`, (4) running pytest, (5) staying budget-tight.

- [ ] **Step 2: Validate**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py 2>&1 | tail -3
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/rahueme/LLM-test && git add scenarios/very_hard/very-hard-12-coding-extract-helper.yaml && git commit -m "feat(scenarios): add very-hard-12-coding-extract-helper"
```

---

## Task 17: Update README — scenario count, Coding row, weight scheme

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Bump total scenario count 75 → 83**

In `/home/rahueme/LLM-test/README.md`, find and update these 4 occurrences:

**Line ~5 (Status line):** change `75 scenarios shipped` → `83 scenarios shipped`.
**Line ~72 ("What it tests"):** change `75 scenarios total` → `83 scenarios total`.
**Line ~99 ("Overall" row):** change `all (75)` → `all (83)`.
**Line ~155 (repo tree):** change `75 scenarios` → `83 scenarios`.

- [ ] **Step 2: Bump Coding row 7 → 13**

In `/home/rahueme/LLM-test/README.md`, find the row in the Score columns table that begins with `| **Coding** | 7 |` (around line ~101). Change `7` to `13`.

- [ ] **Step 3: Document the per-dim weight scheme**

In `/home/rahueme/LLM-test/README.md`, find the "Sort & interpretation rules" section (currently around line ~134). At the END of that bulleted list, INSERT this new bullet:

```markdown
- **Overall weighting**: in the `Overall` column, scenarios tagged `coding`, `terminal`, or `agentic` count **2× weight**; scenarios tagged `localization` or `long_context` count **0.5× weight**; everything else is 1×. A scenario's weight is the MAX of its dim weights (so a scenario tagged both `coding` and `localization` counts 2×). This applies ONLY to `Overall` — every other score column is raw tier-weighted only, so a model's `Coding` score is not diluted by other dims.
```

- [ ] **Step 4: Smoke-check**

```bash
cd /home/rahueme/LLM-test && grep -c "83 scenarios" README.md && grep -c "\*\*Coding\*\* | 13" README.md && grep -c "2× weight\|2x weight" README.md
```
Expected: 3 lines each with count ≥ 1.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add README.md && git commit -m "docs(readme): bump scenario count to 83, Coding row to 13, document per-dim weight scheme"
```

---

## Task 18: End-to-end verification

This task has no commits — it's purely a verification gate.

- [ ] **Step 1: Full test suite**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py tests/core/test_scorer.py tests/core/test_scorer_compose.py tests/core/test_models.py tests/tools/ tests/rankings/ 2>&1 | tail -10
```
Expected: ALL PASS. Note: `tests/core/test_runner.py` and `tests/tui/test_live_tab.py` may still have pre-existing pytest-asyncio failures — those are NOT this work's concern; do not block on them.

- [ ] **Step 2: Scenario count sanity**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.core.scenario import load_all_scenarios
from pathlib import Path
ss = load_all_scenarios(Path('scenarios'))
print('total:', len(ss))
print('category=coding:', len([s for s in ss if s.category == 'coding']))
by_tier = {}
for s in ss:
    if s.category == 'coding':
        by_tier.setdefault(s.tier, 0)
        by_tier[s.tier] += 1
print('coding-by-tier:', by_tier)
"
```
Expected:
```
total: 83
category=coding: 13
coding-by-tier: {'easy': 5, 'medium': 4, 'hard': 2, 'very_hard': 2}
```

- [ ] **Step 3: Rankings regeneration produces files for all expected dims**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.rankings.compute import regenerate_rankings
from llm_test.core.store import Store
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    s = Store(Path('results/runs.db'))
    regenerate_rankings(
        store=s,
        dimensions=['overall','coding','terminal','agentic','localization','long_context'],
        out_dir=out,
    )
    print('files:', sorted(p.name for p in out.iterdir()))
"
```
Expected: at least 6 .md files (`overall.md`, `coding.md`, `terminal.md`, `agentic.md`, `localization.md`, `long_context.md`).

- [ ] **Step 4: Manual sanity — Overall weight check**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.rankings.compute import _scenario_dim_weight
assert _scenario_dim_weight(['overall', 'coding']) == 2.0
assert _scenario_dim_weight(['overall', 'localization']) == 0.5
assert _scenario_dim_weight(['overall']) == 1.0
assert _scenario_dim_weight(['overall', 'coding', 'localization']) == 2.0
print('weight assertions OK')
"
```
Expected: `weight assertions OK`.

- [ ] **Step 5: Git log review**

```bash
cd /home/rahueme/LLM-test && git log --oneline 0df9c19..HEAD
```
Expected: ~17 atomic commits — 4 retrofit, 8 new scenarios, 4 ranking infra commits (Tasks 1-4), 1 README, plus this verification.

---

## Notes for the engineer

- **Working-tree contamination:** Just like the terminal_handling work, the working tree may carry unrelated pre-existing modifications. When you run `git add <file>`, you might pick up some of those. For this plan that's OK — most modified files in our task list (scorer.py, compute.py, scenarios/, README.md) are touch-points for many concurrent features in this repo. Don't try to clean WIP — just stage explicitly.
- **`bash_exec` mock semantics:** `bash_exec` returns `{stdout, stderr, exit_code, duration_ms}`. Use `command_regex` matching to route to specific responses (e.g. distinguishing pytest invocations from git checkouts).
- **`mock_runtime.py` `call_index` strings:** `>=1` is supported (line 32 of mock_runtime.py). `0` (integer) is also supported. Used in `very-hard-11` to make pytest fail on the first call and pass on subsequent ones.
- **TDD philosophy here:** Tasks 1, 3 strictly follow TDD (failing test → impl → pass). Tasks 5-16 (scenarios) are YAML files; "TDD" is just "scenario parses + the load_all_scenarios test reads it".
- **Stretched easy / stretched medium tier value:** `easy-17` and `easy-18` keep `tier: easy` — they should NOT be re-labeled medium. The "stretched" classification is content-only; the `_TIER_WEIGHTS` multiplier stays at 1.0. This is intentional: it shifts the distribution within easy so more models land at intermediate scores (not 0/100).
- **Order matters:** Run Tasks 1-4 BEFORE the scenarios. The dim-weight infrastructure must be in place so the new coding-tagged scenarios properly contribute 2× to Overall on day 1.
- **Don't skip the regression checks between tasks** — if Task 2's data-plumbing breaks the existing tests, fix it before moving to Task 3.

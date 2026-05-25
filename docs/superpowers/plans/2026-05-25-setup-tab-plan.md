# Setup Tab + Use-Case Rankings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 6th TUI tab "Setup" that lets the user pick ONE of 7 use-case personas; the chosen persona produces an additional `UC:<Name>` ranking column in the Rankings tab using persona-specific dimension weights, while the global `Overall` (current `_DIM_WEIGHTS`) stays untouched.

**Architecture:** Personas defined in new module `llm_test/rankings/presets.py`. Active selection persisted in `results/setup.json` (`{"version": 1, "active_use_case": "<key>"}`). `compute_matrix` and `regenerate_rankings` gain an optional `use_case_weights` parameter — when set, they emit an additional `scores["use_case"]` per (model, adapter). Rankings tab conditionally renders an extra column. Setup tab writes the JSON and triggers regen on Apply.

**Tech Stack:** Python 3.12, Textual TUI (RadioSet/Button widgets), pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-05-25-setup-tab-design.md`

---

## File Map

**Create:**
- `llm_test/rankings/presets.py` — 7 UseCase dataclass instances
- `llm_test/tui/setup_tab.py` — RadioSet + Apply/Clear buttons
- `tests/rankings/test_presets.py` — preset module sanity
- `tests/tui/test_setup_tab.py` — Setup tab smoke tests

**Modify:**
- `llm_test/rankings/compute.py`:
  - Add `load_active_use_case()` loader
  - Extend `_scenario_dim_weight()` with `weights_override`
  - Extend `compute_matrix()` with `use_case_weights` parameter
  - Extend `regenerate_rankings()` analogously
- `llm_test/tui/app.py` — add Setup tab as 6th pane (after History)
- `llm_test/tui/rankings_tab.py` — conditional `UC:<Short>` column after `Overall`
- `llm_test/cli.py` — auto-pass active use-case to regen calls
- `tests/rankings/test_dim_weights.py` — 2 new tests
- `README.md` — document Setup tab + 7 personas

---

## Task 1: Create `presets.py` with 7 personas

**Files:**
- Create: `llm_test/rankings/presets.py`
- Test: `tests/rankings/test_presets.py`

- [ ] **Step 1: Write the failing test**

Create `/home/rahueme/LLM-test/tests/rankings/test_presets.py`:

```python
from llm_test.rankings.presets import USE_CASES, UseCase, get_use_case

# The 14 weightable dimensions every persona must cover.
EXPECTED_DIMS = {
    "coding", "terminal", "agentic", "safety", "restraint",
    "error_recovery", "parameter_precision", "context_state_tracking",
    "structured_output", "tool_selection", "long_context", "localization",
    "budget_efficiency", "hallucination",
}


def test_seven_personas_defined():
    assert len(USE_CASES) == 7


def test_persona_keys_are_unique():
    keys = [uc.key for uc in USE_CASES]
    assert len(set(keys)) == len(keys)


def test_persona_keys_are_kebab_case():
    for uc in USE_CASES:
        assert uc.key == uc.key.lower()
        assert " " not in uc.key
        assert "-" not in uc.key  # use snake_case, not kebab


def test_every_persona_has_all_14_dims():
    for uc in USE_CASES:
        assert set(uc.weights.keys()) == EXPECTED_DIMS, (
            f"{uc.key} missing dims: {EXPECTED_DIMS - set(uc.weights.keys())}, "
            f"extra: {set(uc.weights.keys()) - EXPECTED_DIMS}"
        )


def test_every_weight_is_positive_float():
    for uc in USE_CASES:
        for dim, w in uc.weights.items():
            assert isinstance(w, float)
            assert w > 0.0, f"{uc.key}.{dim} = {w}"


def test_get_use_case_returns_match():
    uc = get_use_case("coding_assistant")
    assert uc is not None
    assert uc.key == "coding_assistant"
    assert uc.weights["coding"] == 3.0


def test_get_use_case_returns_none_for_unknown():
    assert get_use_case("nonexistent") is None


def test_persona_has_description():
    for uc in USE_CASES:
        assert isinstance(uc.description, str)
        assert len(uc.description) > 10
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_presets.py -v 2>&1 | tail -5
```
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_test.rankings.presets'`.

- [ ] **Step 3: Create `llm_test/rankings/presets.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class UseCase:
    """A named ranking persona — defines per-dimension weights for a use-case."""
    key: str
    name: str
    description: str
    weights: dict[str, float]


USE_CASES: list[UseCase] = [
    UseCase(
        key="coding_assistant",
        name="Coding Assistant",
        description="IDE-style copilot — code completion, TDD, refactors, diffs.",
        weights={
            "coding": 3.0, "terminal": 2.5, "agentic": 2.0,
            "safety": 1.0, "restraint": 1.5, "error_recovery": 1.5,
            "parameter_precision": 1.0, "context_state_tracking": 1.0,
            "structured_output": 2.0, "tool_selection": 2.0,
            "long_context": 0.75, "localization": 0.3,
            "budget_efficiency": 1.5, "hallucination": 1.5,
        },
    ),
    UseCase(
        key="reasoning",
        name="Reasoning",
        description="Analytical model — strong calibration, numeric fidelity, long-context.",
        weights={
            "coding": 0.5, "terminal": 0.3, "agentic": 0.5,
            "safety": 1.5, "restraint": 2.0, "error_recovery": 1.0,
            "parameter_precision": 2.5, "context_state_tracking": 2.0,
            "structured_output": 1.5, "tool_selection": 1.0,
            "long_context": 2.5, "localization": 0.75,
            "budget_efficiency": 0.75, "hallucination": 3.0,
        },
    ),
    UseCase(
        key="agentic_orchestrator",
        name="Agentic Orchestrator",
        description="Multi-step autonomous workflows — chains, retries, tight budgets.",
        weights={
            "coding": 1.5, "terminal": 1.5, "agentic": 3.0,
            "safety": 1.0, "restraint": 1.0, "error_recovery": 2.5,
            "parameter_precision": 1.5, "context_state_tracking": 2.0,
            "structured_output": 1.0, "tool_selection": 2.5,
            "long_context": 1.0, "localization": 0.3,
            "budget_efficiency": 2.0, "hallucination": 1.0,
        },
    ),
    UseCase(
        key="safety_rag",
        name="Safety / RAG",
        description="Risk-aware retrieval-augmented — anti-hallucination, refuses out-of-scope.",
        weights={
            "coding": 0.5, "terminal": 0.3, "agentic": 0.5,
            "safety": 3.0, "restraint": 2.5, "error_recovery": 1.0,
            "parameter_precision": 1.5, "context_state_tracking": 1.0,
            "structured_output": 1.5, "tool_selection": 1.0,
            "long_context": 2.0, "localization": 1.5,
            "budget_efficiency": 0.5, "hallucination": 3.0,
        },
    ),
    UseCase(
        key="customer_support",
        name="Customer Support",
        description="Multilingual helpdesk — language coverage, structured responses, safety.",
        weights={
            "coding": 0.3, "terminal": 0.3, "agentic": 0.5,
            "safety": 2.5, "restraint": 2.5, "error_recovery": 1.5,
            "parameter_precision": 1.5, "context_state_tracking": 1.5,
            "structured_output": 2.0, "tool_selection": 1.0,
            "long_context": 1.0, "localization": 3.0,
            "budget_efficiency": 0.75, "hallucination": 2.0,
        },
    ),
    UseCase(
        key="data_analyst",
        name="Data Analyst",
        description="DB queries, CSV/JSON output, numeric fidelity, multi-turn iteration.",
        weights={
            "coding": 1.0, "terminal": 1.5, "agentic": 1.0,
            "safety": 1.0, "restraint": 1.5, "error_recovery": 1.0,
            "parameter_precision": 2.5, "context_state_tracking": 2.0,
            "structured_output": 3.0, "tool_selection": 2.0,
            "long_context": 2.0, "localization": 0.5,
            "budget_efficiency": 1.0, "hallucination": 1.5,
        },
    ),
    UseCase(
        key="local_coding_agent",
        name="Local Coding Agent",
        description="Local CLI agent (Claude Code / Codex) — heavy terminal + autonomy.",
        weights={
            "coding": 3.0, "terminal": 3.0, "agentic": 2.5,
            "safety": 1.5, "restraint": 1.5, "error_recovery": 2.0,
            "parameter_precision": 1.5, "context_state_tracking": 2.0,
            "structured_output": 1.5, "tool_selection": 2.0,
            "long_context": 1.5, "localization": 0.3,
            "budget_efficiency": 2.0, "hallucination": 1.5,
        },
    ),
]


def get_use_case(key: str) -> UseCase | None:
    """Look up a persona by its key. Returns None for unknown keys."""
    return next((uc for uc in USE_CASES if uc.key == key), None)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_presets.py -v 2>&1 | tail -10
```
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/presets.py tests/rankings/test_presets.py && git commit -m "feat(rankings): add presets.py with 7 use-case personas"
```

---

## Task 2: Add `load_active_use_case()` loader to compute.py

**Files:**
- Modify: `llm_test/rankings/compute.py` (add new function after imports, before `regenerate_rankings`)
- Test: `tests/rankings/test_dim_weights.py` (extend with new tests)

- [ ] **Step 1: Write the failing tests**

Append to `/home/rahueme/LLM-test/tests/rankings/test_dim_weights.py`:

```python
import tempfile
from llm_test.rankings.compute import load_active_use_case


def test_load_active_use_case_missing_file():
    with tempfile.TemporaryDirectory() as td:
        key, weights = load_active_use_case(Path(td))
        assert key is None
        assert weights is None


def test_load_active_use_case_known_key():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": "coding_assistant"}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key == "coding_assistant"
        assert weights is not None
        assert weights["coding"] == 3.0


def test_load_active_use_case_unknown_key_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": "nonexistent_persona"}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None


def test_load_active_use_case_null_key_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": null}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None


def test_load_active_use_case_malformed_json_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text("not valid json {{{")
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -k load_active -v 2>&1 | tail -5
```
Expected: FAIL — `ImportError: cannot import name 'load_active_use_case'`.

- [ ] **Step 3: Implement the loader**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, find the existing `_scenario_dim_weight` helper (around lines 130-144). AFTER that function and BEFORE `def compute_matrix(...)`, add:

```python
def load_active_use_case(results_dir: Path) -> tuple[str | None, dict[str, float] | None]:
    """Read results/setup.json and return the active use-case persona.

    Returns (key, weights) on success, (None, None) when:
      - setup.json doesn't exist
      - the file is malformed JSON
      - active_use_case is null or missing
      - active_use_case key doesn't match any known persona
    """
    setup_path = results_dir / "setup.json"
    if not setup_path.exists():
        return (None, None)
    try:
        data = json.loads(setup_path.read_text())
    except (json.JSONDecodeError, OSError):
        return (None, None)
    key = data.get("active_use_case")
    if not key:
        return (None, None)
    from llm_test.rankings.presets import get_use_case
    uc = get_use_case(key)
    if uc is None:
        return (None, None)
    return (key, dict(uc.weights))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -k load_active -v 2>&1 | tail -10
```
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py tests/rankings/test_dim_weights.py && git commit -m "feat(rankings): add load_active_use_case() loader for setup.json"
```

---

## Task 3: Extend `_scenario_dim_weight` with `weights_override`

**Files:**
- Modify: `llm_test/rankings/compute.py` (extend existing helper)
- Test: `tests/rankings/test_dim_weights.py` (extend tests)

- [ ] **Step 1: Write the failing tests**

Append to `/home/rahueme/LLM-test/tests/rankings/test_dim_weights.py`:

```python
def test_scenario_dim_weight_default_unchanged():
    """Back-compat: calling without override uses _DIM_WEIGHTS."""
    assert _scenario_dim_weight(["overall", "coding"]) == 2.0


def test_scenario_dim_weight_with_override():
    """When override is passed, it takes precedence."""
    override = {"coding": 5.0, "agentic": 4.0}
    assert _scenario_dim_weight(["overall", "coding"], weights_override=override) == 5.0
    assert _scenario_dim_weight(["overall", "agentic"], weights_override=override) == 4.0


def test_scenario_dim_weight_override_picks_max():
    """Max-of-dims still applies with override."""
    override = {"coding": 5.0, "localization": 0.1}
    assert _scenario_dim_weight(
        ["overall", "coding", "localization"], weights_override=override
    ) == 5.0


def test_scenario_dim_weight_override_unknown_dim_defaults_to_one():
    """Override doesn't break default-1.0 fallback for unknown dims."""
    override = {"coding": 5.0}
    # 'restraint' not in override → default 1.0; 'coding' in override → 5.0; max = 5.0
    assert _scenario_dim_weight(
        ["overall", "coding", "restraint"], weights_override=override
    ) == 5.0
    # only restraint → no override match → 1.0
    assert _scenario_dim_weight(["overall", "restraint"], weights_override=override) == 1.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -k "weights_override or scenario_dim_weight" -v 2>&1 | tail -10
```
Expected: 4 FAIL — `TypeError: _scenario_dim_weight() got an unexpected keyword argument 'weights_override'`.

- [ ] **Step 3: Modify the helper**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, find the current `_scenario_dim_weight` function (around lines 133-137). Replace it with:

```python
def _scenario_dim_weight(
    ranking_dims: list[str],
    weights_override: dict[str, float] | None = None,
) -> float:
    """Compute the per-scenario weight for the Overall (or use-case) column.

    Returns the MAX of the weights of the scenario's non-overall dimensions.
    Dimensions not in the weights map fall back to 1.0.

    When `weights_override` is None, uses the default `_DIM_WEIGHTS` map.
    When provided, uses the override (e.g. a use-case persona's weights).
    """
    weights_map = weights_override if weights_override is not None else _DIM_WEIGHTS
    weights = [weights_map.get(d, 1.0) for d in ranking_dims if d != "overall"]
    return max(weights) if weights else 1.0
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -15
```
Expected: ALL PASS (existing + 4 new). No regressions in existing tests.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py tests/rankings/test_dim_weights.py && git commit -m "feat(rankings): _scenario_dim_weight accepts weights_override for use-cases"
```

---

## Task 4: Extend `compute_matrix` with `use_case_weights`

**Files:**
- Modify: `llm_test/rankings/compute.py` (per-pair reduction loop)
- Test: `tests/rankings/test_dim_weights.py` (integration tests)

- [ ] **Step 1: Write the failing tests**

Append to `/home/rahueme/LLM-test/tests/rankings/test_dim_weights.py`:

```python
def test_compute_matrix_without_use_case_emits_no_use_case_score():
    """Backward compat: no use_case_weights → no scores['use_case']."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.5, "tier": "easy", "dims": ["overall", "coding"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        assert "use_case" not in matrix[0]["scores"]


def test_compute_matrix_with_use_case_weights_emits_use_case_score():
    """When use_case_weights is set, scores['use_case'] is populated."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        # Two scenarios: coding fails (0.0), default passes (1.0).
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},
        ])
        # Use-case heavily weights coding: 5.0 → coding-fail drags overall down.
        uc_weights = {
            "coding": 5.0, "terminal": 1.0, "agentic": 1.0, "safety": 1.0,
            "restraint": 1.0, "error_recovery": 1.0, "parameter_precision": 1.0,
            "context_state_tracking": 1.0, "structured_output": 1.0,
            "tool_selection": 1.0, "long_context": 1.0, "localization": 1.0,
            "budget_efficiency": 1.0, "hallucination": 1.0,
        }
        matrix = compute_matrix(
            store=store, dimensions=["overall"],
            use_case_weights=uc_weights,
        )
        assert "use_case" in matrix[0]["scores"]
        # Expected: (0*1*5.0 + 1*1*1.0) / (1*5.0 + 1*1.0) = 1.0/6.0 ≈ 0.167
        assert abs(matrix[0]["scores"]["use_case"] - 1.0/6.0) < 0.001


def test_compute_matrix_use_case_score_differs_from_overall():
    """With different weights, use_case and overall should differ."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},     # default coding weight 2.0
            {"score": 1.0, "tier": "easy", "dims": ["overall", "localization"]},  # default loc weight 0.5
        ])
        # Persona that inverts: localization heavy, coding light.
        uc_weights = {
            "coding": 0.5, "terminal": 1.0, "agentic": 1.0, "safety": 1.0,
            "restraint": 1.0, "error_recovery": 1.0, "parameter_precision": 1.0,
            "context_state_tracking": 1.0, "structured_output": 1.0,
            "tool_selection": 1.0, "long_context": 1.0, "localization": 5.0,
            "budget_efficiency": 1.0, "hallucination": 1.0,
        }
        matrix = compute_matrix(
            store=store, dimensions=["overall"],
            use_case_weights=uc_weights,
        )
        overall = matrix[0]["scores"]["overall"]
        use_case = matrix[0]["scores"]["use_case"]
        # Default: coding(weight 2.0) fails 0.0 + loc(weight 0.5) passes 1.0
        #          = (0*2 + 1*0.5)/(2+0.5) = 0.5/2.5 = 0.2
        # Use-case: coding(weight 0.5) fails 0.0 + loc(weight 5.0) passes 1.0
        #          = (0*0.5 + 1*5)/(0.5+5) = 5.0/5.5 ≈ 0.909
        assert abs(overall - 0.2) < 0.001
        assert abs(use_case - 5.0/5.5) < 0.001
        assert use_case > overall  # localization-heavy persona favors this run
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -k "use_case" -v 2>&1 | tail -10
```
Expected: 3 FAIL — `TypeError: compute_matrix() got an unexpected keyword argument 'use_case_weights'`.

- [ ] **Step 3: Modify `compute_matrix`**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, find the `def compute_matrix(` signature (around line 145). Update the signature and add use-case logic inside the per-pair loop.

**Change the signature** to include the new parameter:

```python
def compute_matrix(
    *, store: Store, dimensions: list[str],
    history_window_runs: int = 5, half_life_days: float = 14.0,
    use_case_weights: dict[str, float] | None = None,
) -> list[dict]:
```

**Then update the docstring** to mention the new behavior. Find the existing docstring and add:

```python
    """Compute per-(model, adapter) scores across all ranking dimensions.

    ...existing docstring...

    When `use_case_weights` is provided, additionally emits `scores['use_case']`
    per pair, computed with the same formula as `overall` but using the given
    weights map (e.g. from a use-case persona) instead of the default
    `_DIM_WEIGHTS`. The standard `overall` score remains unchanged.
    """
```

**Then add use-case computation inside the per-pair loop.** Find the existing block that ends with:

```python
            if decay_pairs:
                scores[dim] = decay_weighted_mean(decay_pairs, half_life_days)
            total_runs = max(total_runs, len(runs_sorted))
```

INSIDE the outer `for dim, items in dim_results.items():` loop (NOT inside `for rid in recent:`), AFTER the existing line `scores[dim] = decay_weighted_mean(...)`, add:

```python
            # If a use-case is active AND we're processing the overall dim,
            # compute an extra `use_case` score from the SAME per-run items
            # but with use-case weights instead of the default _DIM_WEIGHTS.
            if dim == "overall" and use_case_weights is not None:
                uc_decay_pairs: list[tuple[float, float]] = []
                for rid in recent:
                    items_in_run = by_run[rid]
                    uc_weights = [
                        _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(
                            json.loads(d), weights_override=use_case_weights
                        )
                        for _, t, d in items_in_run
                    ]
                    uc_w_sum = sum(uc_weights)
                    if uc_w_sum <= 0:
                        continue
                    uc_weighted_mean = sum(
                        s * w for (s, _, _), w in zip(items_in_run, uc_weights)
                    ) / uc_w_sum
                    age = max(
                        (now - _parse_iso(run_started[rid])).total_seconds() / 86400, 0
                    )
                    uc_decay_pairs.append((uc_weighted_mean, age))
                if uc_decay_pairs:
                    scores["use_case"] = decay_weighted_mean(uc_decay_pairs, half_life_days)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/rankings/test_dim_weights.py -v 2>&1 | tail -15
```
Expected: ALL PASS (including the 3 new use_case tests). No regressions.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py tests/rankings/test_dim_weights.py && git commit -m "feat(rankings): compute_matrix emits scores['use_case'] when use_case_weights set"
```

---

## Task 5: Mirror `use_case_weights` in `regenerate_rankings`

**Files:**
- Modify: `llm_test/rankings/compute.py` (`regenerate_rankings` function)

- [ ] **Step 1: Update the signature**

In `/home/rahueme/LLM-test/llm_test/rankings/compute.py`, find `def regenerate_rankings(` (around line 17). Add the optional `use_case_weights` parameter and `use_case_key`:

```python
def regenerate_rankings(*, store: Store, dimensions: list[str], out_dir: Path,
                        history_window_runs: int = 5, half_life_days: float = 14.0,
                        bootstrap_iters: int = 1000, min_runs: int = 1,
                        use_case_weights: dict[str, float] | None = None,
                        use_case_key: str | None = None) -> None:
```

- [ ] **Step 2: After the existing for-dim loop, emit use_case_<key>.md if active**

Inside `regenerate_rankings`, find the `for dim in dimensions:` outer loop (around line 25). AFTER that loop completes (i.e. after all per-dim markdown files have been written), add the use-case emission block. Use the same data path but apply `use_case_weights` instead of `_DIM_WEIGHTS`:

Add this code at the end of `regenerate_rankings`, just before the function returns (i.e. at the same indentation as `for dim in dimensions:`):

```python
    # Emit an extra use_case_<key>.md when a persona is active.
    if use_case_weights is not None and use_case_key is not None:
        per_pair_runs: dict[tuple[str, str], list[dict]] = defaultdict(list)
        with store.conn() as c:
            rows = c.execute("SELECT * FROM scenario_results").fetchall()
            results = [dict(r) for r in rows]
        results_by_run: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            # Use-case rolls up EVERY scenario (same as 'overall'); no dim filter.
            results_by_run[r["run_id"]].append(r)
        for run_id, rs in results_by_run.items():
            meta = run_meta.get(run_id)
            if not meta:
                continue
            model = meta["model"]
            by_adapter: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
            for r in rs:
                by_adapter[r["adapter"]].append(
                    (r["score"], r["tier"], r["ranking_dims_json"] or "[]")
                )
            for adapter, scored in by_adapter.items():
                per_pair_runs[(model, adapter)].append({
                    "run_id": run_id, "scores": scored,
                    "started_at": meta["started_at"],
                })

        pair_rows: list[dict] = []
        for (model, adapter), runs_list in per_pair_runs.items():
            if len(runs_list) < min_runs:
                continue
            runs_list.sort(key=lambda x: x["started_at"], reverse=True)
            recent = runs_list[:history_window_runs]
            pairs: list[tuple[float, float]] = []
            for r in recent:
                items = r["scores"]
                weights_per_item = [
                    _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(
                        json.loads(d), weights_override=use_case_weights
                    )
                    for _, t, d in items
                ]
                w_sum = sum(weights_per_item)
                if w_sum <= 0:
                    continue
                run_mean = sum(
                    s * w for (s, _, _), w in zip(items, weights_per_item)
                ) / w_sum
                age_days = max(
                    (now - _parse_iso(r["started_at"])).total_seconds() / 86400, 0
                )
                pairs.append((run_mean, age_days))
            pair_rows.append({
                "model": model, "adapter": adapter,
                "score": decay_weighted_mean(pairs, half_life_days),
                "runs": len(runs_list),
            })

        rows_out: list[dict] = []
        per_model: dict[str, list[dict]] = defaultdict(list)
        for pr in pair_rows:
            per_model[pr["model"]].append(pr)
        for model, prs in per_model.items():
            prs.sort(key=lambda p: -p["score"])
            best = prs[0]
            rows_out.append({
                "model": model, "score": best["score"],
                "best_adapter": best["adapter"],
                "runs": sum(p["runs"] for p in prs),
                "adapter_breakdown": prs,
            })
        rows_out.sort(key=lambda r: -r["score"])
        breakdown = sorted(pair_rows, key=lambda p: -p["score"])
        tmpl = _env.get_template("ranking.md.j2")
        md = tmpl.render(
            dimension=f"use_case_{use_case_key}",
            updated_iso=now.isoformat(),
            model_count=len(rows_out), run_count=sum(r["runs"] for r in rows_out),
            rows=rows_out, breakdown=breakdown,
            window=history_window_runs, half_life=half_life_days,
            bootstrap_iters=bootstrap_iters,
        )
        (out_dir / f"use_case_{use_case_key}.md").write_text(md)
```

- [ ] **Step 3: Smoke test end-to-end**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.rankings.compute import regenerate_rankings, load_active_use_case
from llm_test.core.store import Store
from pathlib import Path
import tempfile, json

with tempfile.TemporaryDirectory() as td:
    results_dir = Path(td)
    (results_dir / 'setup.json').write_text(json.dumps({
        'version': 1, 'active_use_case': 'coding_assistant'
    }))
    key, weights = load_active_use_case(results_dir)
    print('loaded:', key, '/ coding weight:', weights['coding'])
    out = results_dir / 'rankings'
    s = Store(Path('results/runs.db'))
    regenerate_rankings(
        store=s, dimensions=['overall', 'coding'], out_dir=out,
        use_case_weights=weights, use_case_key=key,
    )
    print('files:', sorted(p.name for p in out.iterdir()))
"
```
Expected: prints `loaded: coding_assistant / coding weight: 3.0` and `files: ['coding.md', 'overall.md', 'use_case_coding_assistant.md']`.

- [ ] **Step 4: Regression**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/rankings/ 2>&1 | tail -3
```
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/rankings/compute.py && git commit -m "feat(rankings): regenerate_rankings emits use_case_<key>.md when persona active"
```

---

## Task 6: cli.py — auto-pass active use-case to regen calls

**Files:**
- Modify: `llm_test/cli.py` (two regen call sites: in `run` command auto-regen + in `rankings --regen`)

- [ ] **Step 1: Update run-command auto-regen (~line 222)**

In `/home/rahueme/LLM-test/llm_test/cli.py`, find the existing auto-regen block:

```python
        from llm_test.rankings.compute import regenerate_rankings
        regenerate_rankings(
            store=store,
            dimensions=["overall", "coding", "agentic", "safety", "restraint",
                        ...],
            out_dir=_results_dir() / "rankings",
        )
```

Replace with (add load + pass through):

```python
        from llm_test.rankings.compute import regenerate_rankings, load_active_use_case
        uc_key, uc_weights = load_active_use_case(_results_dir())
        regenerate_rankings(
            store=store,
            dimensions=["overall", "coding", "agentic", "safety", "restraint",
                        "long_context", "budget_efficiency", "hallucination",
                        "error_recovery", "parameter_precision",
                        "context_state_tracking", "structured_output",
                        "tool_selection", "localization", "terminal"],
            out_dir=_results_dir() / "rankings",
            use_case_weights=uc_weights,
            use_case_key=uc_key,
        )
```

- [ ] **Step 2: Update `rankings --regen` command (~line 282)**

Find the existing block:

```python
    if regen:
        regenerate_rankings(store=_store(), dimensions=dims, out_dir=out)
        console.print(f"[green]✓ Regenerated rankings: {out}[/green]")
```

Replace with:

```python
    if regen:
        from llm_test.rankings.compute import load_active_use_case
        uc_key, uc_weights = load_active_use_case(_results_dir())
        regenerate_rankings(
            store=_store(), dimensions=dims, out_dir=out,
            use_case_weights=uc_weights, use_case_key=uc_key,
        )
        msg = f"[green]✓ Regenerated rankings: {out}[/green]"
        if uc_key:
            msg += f"\n[dim]  · Use-case '{uc_key}' applied → use_case_{uc_key}.md[/dim]"
        console.print(msg)
```

- [ ] **Step 3: Smoke test CLI**

```bash
cd /home/rahueme/LLM-test && python3 -c "
import json
from pathlib import Path
# Simulate active use-case
p = Path('results/setup.json')
p.write_text(json.dumps({'version': 1, 'active_use_case': 'reasoning'}))
print('setup.json written')
" && cd /home/rahueme/LLM-test && python3 -m llm_test.cli rankings --regen 2>&1 | tail -5
```
Expected: prints "Regenerated rankings" and includes "Use-case 'reasoning' applied" note. Check that `results/rankings/use_case_reasoning.md` was created.

Then cleanup:
```bash
cd /home/rahueme/LLM-test && rm results/setup.json
```

- [ ] **Step 4: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/cli.py && git commit -m "feat(cli): auto-pick active use-case from setup.json on rankings regen"
```

---

## Task 7: rankings_tab.py — conditional `UC:<Short>` column after Overall

**Files:**
- Modify: `llm_test/tui/rankings_tab.py`

- [ ] **Step 1: Add the use-case header constant**

In `/home/rahueme/LLM-test/llm_test/tui/rankings_tab.py`, find `_HEADERS = {` (around line 35). AFTER that dict (before `_PERF_COLS`), add:

```python
# Short headers for the conditional UC:<Name> column (rendered after Overall
# when a use-case persona is active in results/setup.json).
_USE_CASE_HEADERS = {
    "coding_assistant": "UC:Coding",
    "reasoning": "UC:Reason",
    "agentic_orchestrator": "UC:Agentic",
    "safety_rag": "UC:Safety",
    "customer_support": "UC:Support",
    "data_analyst": "UC:Data",
    "local_coding_agent": "UC:LocalCA",
}
```

- [ ] **Step 2: Pass use-case to compute_matrix and add column conditionally**

Find the `reload()` method, specifically the line that calls `compute_matrix`:

```python
        matrix = compute_matrix(store=store, dimensions=_DIMENSIONS)
```

Replace the block from `store = Store(db); store.init_schema()` down through (and including) `matrix = compute_matrix(...)` with:

```python
        store = Store(db)
        store.init_schema()

        # Load active use-case from setup.json (if any). When active, compute_matrix
        # additionally returns scores['use_case'] per row.
        from llm_test.rankings.compute import load_active_use_case
        uc_key, uc_weights = load_active_use_case(results_dir)
        self._active_use_case_key = uc_key  # used in _populate_rows / column setup

        matrix = compute_matrix(
            store=store, dimensions=_DIMENSIONS,
            use_case_weights=uc_weights,
        )
```

- [ ] **Step 3: Add the conditional column after Overall**

Find the column-registration loop (around lines 261-262):

```python
        for dim in _DIMENSIONS:
            tbl.add_column(_HEADERS[dim], key=f"dim:{dim}")
```

Replace with:

```python
        for dim in _DIMENSIONS:
            tbl.add_column(_HEADERS[dim], key=f"dim:{dim}")
            # Insert use-case column right after `overall` so it's visually prominent.
            if dim == "overall" and self._active_use_case_key:
                header = _USE_CASE_HEADERS.get(
                    self._active_use_case_key, f"UC:{self._active_use_case_key[:6]}"
                )
                tbl.add_column(header, key="dim:use_case")
```

Also extend `_sort_keys` registration. Find the existing block:

```python
        for dim in _DIMENSIONS:
            d = dim
            self._sort_keys[f"dim:{d}"] = lambda r, d=d: -(r["scores"].get(d, -1.0))
```

AFTER that loop, add:

```python
        if self._active_use_case_key:
            self._sort_keys["dim:use_case"] = lambda r: -(r["scores"].get("use_case", -1.0))
```

- [ ] **Step 4: Populate the use-case cell in `_populate_rows`**

Find `_populate_rows` (around line 330). Inside the data-cell rendering loop, find:

```python
            for dim in _DIMENSIONS:
                cells.append(_fmt_score(r["scores"].get(dim),
                                        podium_score.get((i, dim))))
```

Replace with:

```python
            for dim in _DIMENSIONS:
                cells.append(_fmt_score(r["scores"].get(dim),
                                        podium_score.get((i, dim))))
                # Insert UC cell right after Overall.
                if dim == "overall" and self._active_use_case_key:
                    cells.append(_fmt_score(r["scores"].get("use_case"),
                                            podium_score.get((i, "use_case"))))
```

Also extend the podium computation. Find:

```python
        podium_score: dict[tuple[int, str], int] = {}
        for dim in _DIMENSIONS:
            scored = [(i, r["scores"].get(dim)) for i, r in enumerate(rows)
                      if r["scores"].get(dim) is not None]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (i, _v) in enumerate(scored[:3], start=1):
                podium_score[(i, dim)] = rank
```

AFTER that loop, add:

```python
        if self._active_use_case_key:
            scored = [(i, r["scores"].get("use_case")) for i, r in enumerate(rows)
                      if r["scores"].get("use_case") is not None]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (i, _v) in enumerate(scored[:3], start=1):
                podium_score[(i, "use_case")] = rank
```

- [ ] **Step 5: Initialize `_active_use_case_key` in `__init__`**

Find the `__init__` (around line 203). Inside the body, after the existing instance attribute setup, add:

```python
        self._active_use_case_key: str | None = None
```

(Place it after `self._rows_cache: list[dict] = []`.)

- [ ] **Step 6: Smoke test**

```bash
cd /home/rahueme/LLM-test && python3 -c "
import json
from pathlib import Path
Path('results/setup.json').write_text(json.dumps({
    'version': 1, 'active_use_case': 'coding_assistant'
}))
print('setup.json written')
"
```

```bash
cd /home/rahueme/LLM-test && pytest -q tests/tui/ tests/rankings/ 2>&1 | tail -5
```
Expected: PASS for the test files we own; pre-existing pytest-asyncio failures in test_live_tab.py are NOT our concern.

Cleanup:
```bash
cd /home/rahueme/LLM-test && rm results/setup.json
```

- [ ] **Step 7: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/tui/rankings_tab.py && git commit -m "feat(tui): Rankings tab adds conditional UC:<Short> column when use-case active"
```

---

## Task 8: Create `setup_tab.py`

**Files:**
- Create: `llm_test/tui/setup_tab.py`
- Test: `tests/tui/test_setup_tab.py`

- [ ] **Step 1: Write the failing tests**

Create `/home/rahueme/LLM-test/tests/tui/test_setup_tab.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_test.tui.setup_tab import SetupTab


def test_setup_tab_imports_without_error():
    """Smoke test — module loads."""
    assert SetupTab is not None


def test_apply_writes_setup_json():
    """Calling _save_active_use_case('coding_assistant') writes correct JSON."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        tab = SetupTab.__new__(SetupTab)  # bypass __init__ to avoid Textual
        tab._results_dir = results_dir
        tab._save_active_use_case("coding_assistant")
        assert (results_dir / "setup.json").exists()
        data = json.loads((results_dir / "setup.json").read_text())
        assert data["version"] == 1
        assert data["active_use_case"] == "coding_assistant"


def test_clear_removes_setup_json():
    """Calling _save_active_use_case(None) removes setup.json."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text('{"version": 1, "active_use_case": "x"}')
        tab = SetupTab.__new__(SetupTab)
        tab._results_dir = results_dir
        tab._save_active_use_case(None)
        assert not (results_dir / "setup.json").exists()


def test_clear_when_no_file_does_not_raise():
    """Clear should be idempotent."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        tab = SetupTab.__new__(SetupTab)
        tab._results_dir = results_dir
        tab._save_active_use_case(None)  # no setup.json exists
        assert not (results_dir / "setup.json").exists()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/rahueme/LLM-test && pytest tests/tui/test_setup_tab.py -v 2>&1 | tail -5
```
Expected: FAIL — `ImportError: cannot import name 'SetupTab'`.

- [ ] **Step 3: Create `setup_tab.py`**

Create `/home/rahueme/LLM-test/llm_test/tui/setup_tab.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Button, RadioButton, RadioSet, Static

from llm_test.rankings.presets import USE_CASES


def _persona_blurb(weights: dict[str, float]) -> str:
    """Return a 'top-3 ↑ / bottom-2 ↓' summary of a persona's weights."""
    sorted_high = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    sorted_low = sorted(weights.items(), key=lambda kv: kv[1])
    top3 = ", ".join(f"{d}({w:.1f})" for d, w in sorted_high[:3])
    bot2 = ", ".join(f"{d}({w:.1f})" for d, w in sorted_low[:2])
    return f"  [green]↑[/green] {top3}\n  [red]↓[/red] {bot2}"


class SetupTab(Container):
    """Pick a use-case persona to drive an additional ranking column."""

    DEFAULT_CSS = """
    SetupTab { padding: 1; }
    SetupTab #setup-header { text-style: bold; margin-bottom: 1; }
    SetupTab #setup-radio { margin-bottom: 1; }
    SetupTab .persona-desc { margin-bottom: 1; padding-left: 2; }
    SetupTab #setup-buttons { margin-top: 1; }
    SetupTab #setup-status { margin-top: 1; }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results")
        )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "[bold]Pick the use-case you're evaluating for.[/bold]\n"
                "The chosen profile creates an extra ranking column "
                "'UC:<Name>' in Rankings (next to Overall). "
                "The general Overall column is unaffected.",
                id="setup-header",
            )
            with VerticalScroll():
                active = self._read_active_use_case()
                with RadioSet(id="setup-radio"):
                    yield RadioButton(
                        "None — only general Overall",
                        id="uc-none",
                        value=(active is None),
                    )
                    for uc in USE_CASES:
                        is_active = (active == uc.key)
                        yield RadioButton(
                            f"{uc.name}", id=f"uc-{uc.key}", value=is_active,
                        )
                        yield Static(
                            f"  [dim]{uc.description}[/dim]\n"
                            f"{_persona_blurb(uc.weights)}",
                            classes="persona-desc",
                        )
            with Container(id="setup-buttons"):
                yield Button("Apply", id="apply", variant="success")
                yield Button("Clear (none)", id="clear", variant="error")
            yield Static(self._status_text(), id="setup-status")

    def _read_active_use_case(self) -> str | None:
        """Read current setup.json. Returns key or None."""
        setup_path = self._results_dir / "setup.json"
        if not setup_path.exists():
            return None
        try:
            data = json.loads(setup_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return data.get("active_use_case")

    def _status_text(self) -> str:
        active = self._read_active_use_case()
        if active is None:
            return "[dim]Active: none (general Overall only)[/dim]"
        return f"[dim]Active: {active}[/dim]"

    def _save_active_use_case(self, key: str | None) -> None:
        """Persist or clear the active use-case in setup.json."""
        setup_path = self._results_dir / "setup.json"
        if key is None:
            setup_path.unlink(missing_ok=True)
            return
        self._results_dir.mkdir(parents=True, exist_ok=True)
        setup_path.write_text(
            json.dumps({"version": 1, "active_use_case": key}, indent=2)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            radio = self.query_one("#setup-radio", RadioSet)
            pressed = radio.pressed_button
            if pressed is None:
                self.app.notify("Pick a use-case first", severity="warning")
                return
            radio_id = pressed.id or ""
            key = radio_id.removeprefix("uc-")
            if key == "none":
                self._save_active_use_case(None)
                self.app.notify("Use-case cleared")
            else:
                self._save_active_use_case(key)
                self.app.notify(f"Use-case '{key}' applied — regenerating rankings...")
            self._regenerate_rankings()
            self._refresh_status_and_focus()
        elif event.button.id == "clear":
            self._save_active_use_case(None)
            self.app.notify("Use-case cleared")
            self._regenerate_rankings()
            self._refresh_status_and_focus()

    def _regenerate_rankings(self) -> None:
        """Call regenerate_rankings using the current setup.json state."""
        from llm_test.core.store import Store
        from llm_test.rankings.compute import (
            load_active_use_case, regenerate_rankings,
        )
        db = self._results_dir / "runs.db"
        if not db.exists():
            return
        store = Store(db)
        store.init_schema()
        uc_key, uc_weights = load_active_use_case(self._results_dir)
        dims = [
            "overall", "coding", "agentic", "safety", "restraint",
            "long_context", "budget_efficiency", "hallucination",
            "error_recovery", "parameter_precision",
            "context_state_tracking", "structured_output",
            "tool_selection", "localization", "terminal",
        ]
        try:
            regenerate_rankings(
                store=store, dimensions=dims,
                out_dir=self._results_dir / "rankings",
                use_case_weights=uc_weights, use_case_key=uc_key,
            )
        except Exception as e:
            self.app.notify(f"Regen failed: {e}", severity="error")

    def _refresh_status_and_focus(self) -> None:
        """Update status line + switch focus to Rankings tab + force refresh."""
        self.query_one("#setup-status", Static).update(self._status_text())
        # Switch to Rankings tab if available.
        from textual.widgets import TabbedContent
        try:
            tabs = self.app.query_one(TabbedContent)
            tabs.active = "rankings"
        except Exception:
            pass
        # Force reload of Rankings tab data.
        try:
            from llm_test.tui.rankings_tab import RankingsTab
            rt = self.app.query_one(RankingsTab)
            if hasattr(rt, "reload"):
                rt.reload()
        except Exception:
            pass
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
cd /home/rahueme/LLM-test && pytest tests/tui/test_setup_tab.py -v 2>&1 | tail -8
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/tui/setup_tab.py tests/tui/test_setup_tab.py && git commit -m "feat(tui): add Setup tab with 7 use-case personas, Apply/Clear actions"
```

---

## Task 9: Wire Setup tab into `app.py`

**Files:**
- Modify: `llm_test/tui/app.py`

- [ ] **Step 1: Add the import**

In `/home/rahueme/LLM-test/llm_test/tui/app.py`, find the existing imports block (around line 19). Add a new import:

```python
from llm_test.tui.setup_tab import SetupTab
```

- [ ] **Step 2: Add the TabPane after History (around line 50)**

Find the existing `with TabPane("History", id="history"):` block. AFTER it, before `yield Footer()`, add:

```python
            with TabPane("Setup", id="setup"):
                yield SetupTab(id="setup-tab")
```

- [ ] **Step 3: Smoke test**

```bash
cd /home/rahueme/LLM-test && python3 -c "
from llm_test.tui.app import LLMTestApp
app = LLMTestApp()
print('LLMTestApp instantiates OK')
"
```
Expected: prints `LLMTestApp instantiates OK` without exceptions.

- [ ] **Step 4: Regression**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/tui/test_setup_tab.py tests/rankings/ 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/rahueme/LLM-test && git add llm_test/tui/app.py && git commit -m "feat(tui): wire Setup tab into LLMTestApp as 6th pane after History"
```

---

## Task 10: README — document Setup tab + personas

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update TUI workflow section to mention 6 tabs**

In `/home/rahueme/LLM-test/README.md`, find the line `llm-test tui` opens a 5-tab terminal dashboard:` (around line 54). Change `5-tab` to `6-tab`.

- [ ] **Step 2: Add Setup tab description**

After the existing "Scenarios" bullet (around line 65, the last existing tab in the description list), add a new bullet:

```markdown
- **Setup** — pick a use-case persona (Coding Assistant, Reasoning, Agentic Orchestrator, Safety/RAG, Customer Support, Data Analyst, Local Coding Agent). The chosen persona creates an additional `UC:<Name>` ranking column in Rankings, computed with persona-specific dimension weights. The global Overall column is unaffected. Selection persists in `results/setup.json`.
```

- [ ] **Step 3: Smoke-check**

```bash
cd /home/rahueme/LLM-test && grep -c "6-tab" README.md && grep -c "Setup" README.md
```
Expected: 1 / ≥1.

- [ ] **Step 4: Commit**

```bash
cd /home/rahueme/LLM-test && git add README.md && git commit -m "docs(readme): document Setup tab + 7 use-case personas (6th TUI tab)"
```

---

## Task 11: End-to-end verification

This task has no commits — pure verification.

- [ ] **Step 1: Full test suite**

```bash
cd /home/rahueme/LLM-test && pytest -q tests/core/test_scenario.py tests/core/test_scorer.py tests/core/test_models.py tests/tools/ tests/rankings/ tests/tui/test_setup_tab.py 2>&1 | tail -5
```
Expected: ALL PASS.

- [ ] **Step 2: Activate a persona via CLI, regen, verify file appears**

```bash
cd /home/rahueme/LLM-test && python3 -c "
import json
from pathlib import Path
Path('results/setup.json').write_text(json.dumps({
    'version': 1, 'active_use_case': 'local_coding_agent'
}))
print('Activated local_coding_agent')
" && python3 -m llm_test.cli rankings --regen 2>&1 | tail -3 && ls results/rankings/use_case_*.md
```
Expected: prints regen success, then `results/rankings/use_case_local_coding_agent.md` exists.

- [ ] **Step 3: Switch persona, verify correct file**

```bash
cd /home/rahueme/LLM-test && python3 -c "
import json
from pathlib import Path
Path('results/setup.json').write_text(json.dumps({
    'version': 1, 'active_use_case': 'reasoning'
}))
" && python3 -m llm_test.cli rankings --regen 2>&1 | tail -2 && ls results/rankings/use_case_*.md
```
Expected: now `use_case_reasoning.md` exists alongside the previous file (we don't delete old ones — they accumulate as user explores personas).

- [ ] **Step 4: Clear persona, verify no new file generated**

```bash
cd /home/rahueme/LLM-test && rm results/setup.json && python3 -m llm_test.cli rankings --regen 2>&1 | tail -2
```
Expected: regen runs without "Use-case applied" note. (Old `use_case_*.md` files remain, but no NEW one is added.)

- [ ] **Step 5: Round-trip via TUI (manual)**

Open the TUI: `cd /home/rahueme/LLM-test && llm-test tui`. Navigate to the Setup tab (last). Pick a persona, click Apply. Verify:
- Notification appears
- Focus switches to Rankings tab
- A new column `UC:<Short>` appears between Overall and the other dim columns
- Clicking on the UC column header sorts by it

Then return to Setup, click Clear. Verify:
- UC column disappears from Rankings

- [ ] **Step 6: Cleanup any test artifacts**

```bash
cd /home/rahueme/LLM-test && rm -f results/setup.json && ls results/rankings/use_case_*.md 2>/dev/null
```

(`use_case_*.md` files left behind from testing can stay or be removed manually.)

---

## Notes for the engineer

- **Use-case is purely additive.** The global `Overall` column (driven by `_DIM_WEIGHTS`) is never touched. The use-case column is an extra view, not a replacement.

- **`results_dir` convention:** existing code uses `Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))`. Stick with this — don't introduce a new way to resolve it.

- **Backward compatibility:** every change to `_scenario_dim_weight`, `compute_matrix`, and `regenerate_rankings` keeps the default behavior identical when no `weights_override` / `use_case_weights` is passed. Existing callers (other TUI tabs, tests) are not affected.

- **`use_case_<key>.md` accumulation:** the markdown files for different personas accumulate in `results/rankings/`. We do NOT auto-delete old ones when switching personas. That's intentional — the user can compare side-by-side. If accumulation becomes a problem, manual cleanup is fine.

- **Untracked WIP modifications**: As with the prior work on this branch, the working tree carries pre-existing unrelated modifications. When you `git add` a file, you may pick up changes outside this plan's intent. Stage only the specific files listed in each task's "Files" section.

- **Pre-existing pytest-asyncio issues** in `tests/core/test_runner.py` and `tests/tui/test_live_tab.py` predate this branch and are unrelated to the Setup tab work.

- **Single Apply path:** the Setup tab's Apply button triggers regen synchronously. For ~83 scenarios in `runs.db`, this completes in well under a second. If the DB grows large enough to be noticeable, refactor to a worker thread (`@work` decorator from Textual) — but YAGNI for now.

- **MVP scope reminder:** no custom-weights editing, no multi-persona side-by-side, no persona export/import. All in scope for v2 only.

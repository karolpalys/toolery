# Coding Suite Expansion + Per-Dimension Weighting — Design

**Status:** Design — awaiting user review 2026-05-25
**Owner:** rahueme
**Implementation plan:** to be produced by `writing-plans` after this spec is reviewed.

## 1. Goal

Three parallel concerns:

1. **Expand the coding suite from 5 → 13 scenarios** with deliberate difficulty calibration so models distribute across 0-100% instead of clustering at 10/20% buckets.
2. **Smooth the score gradient** by ensuring every coding scenario (new + retrofitted existing) has 2-4 explicit `partial:` checks for fine-grained credit.
3. **Add per-dimension weights in Overall computation** so `coding`, `terminal`, and `agentic` count 2× and `localization`, `long_context` count 0.5×.

## 2. Target tier distribution

| Tier | Current | Target | Stretched | Reasoning |
|---|---:|---:|:--|:--|
| easy | 1 | 5 | 2 of 5 | Spread juniors from "trivial pass" to "needs reading" |
| medium | 2 | 4 | 2 of 4 | Bridge gap to hard so mid-tier models can be ranked |
| hard | 2 | 2 | — | Already calibrated |
| very_hard | 0 | 2 | — | Top-tier signal — fix obvious omission |

"Stretched" means **still in tier** (same `tier:` field, same `_TIER_WEIGHTS` multiplier) but with non-trivial cognitive load — output parsing instead of memorized facts, multi-step composition instead of single calls.

## 3. New scenarios (8)

All use existing tools (`read_file`, `write_file`, `bash_exec` from terminal_handling work, plus `list_files`). All have `category: coding`. All have `ranking_dimensions: [overall, coding]` (plus `agentic` where the chain is non-trivial).

### Easy baseline (×2)

#### `easy-15-coding-pyproject-name`
- **Goal:** read `pyproject.toml`, extract project name.
- **Tools:** `read_file`
- **Prompt:** "What is the project name declared in /workspace/pyproject.toml?"
- **Mock returns:** TOML with `[project]\nname = "llm-test"`.
- **Required:** `tool_called(read_file)`, `response_satisfies(all_of=["llm-test"])`.
- **Partial:** `tool_args_contain(read_file, path: /workspace/pyproject.toml)`, `call_count_at_most: 1`.

#### `easy-16-coding-pytest-count`
- **Goal:** run pytest, parse the "X passed in Y.Ys" summary, report the count.
- **Tools:** `bash_exec`
- **Prompt:** "Run the test suite and tell me how many tests passed."
- **Mock returns:** classic pytest tail `5 passed in 0.42s`.
- **Required:** `tool_called(bash_exec)`, `response_satisfies(all_of=["5"], none_of=["error","fail"])`.
- **Partial:** `command_regex_match(pattern: '^pytest|^python.*-m\s+pytest')`, `call_count_at_most: 1`.

### Easy stretched (×2)

#### `easy-17-coding-find-failing-test`
- **Goal:** run pytest, parse mixed PASS/FAIL output, report which test failed.
- **Tools:** `bash_exec`
- **Mock returns:** "7 passed, 1 failed" + `FAILED tests/test_calc.py::test_divide_by_zero`.
- **Required:** `tool_called`, `response_satisfies(all_of=["test_calc","test_divide_by_zero"])`.
- **Partial:** identifies the module file alone, identifies the test name alone, mentions "fail", `call_count_at_most: 1`.

#### `easy-18-coding-import-error-module`
- **Goal:** run script, parse Python traceback, identify missing module.
- **Tools:** `bash_exec`
- **Mock returns:** `Traceback (most recent call last):\n  File "script.py", line 3, in <module>\n    import requests\nModuleNotFoundError: No module named 'requests'`.
- **Required:** `tool_called`, `response_satisfies(all_of=["requests"], none_of=["error in the script","syntax"])`.
- **Partial:** mentions "import", mentions "module", `call_count_at_most: 2`.

### Medium stretched-toward-hard (×2)

#### `medium-31-coding-fix-syntax-then-test`
- **Goal:** read file with syntax error → write fixed version → run pytest → confirm pass.
- **Tools:** `read_file`, `write_file`, `bash_exec`
- **Mock:** `read_file` returns `def add(a, b)\n    return a + b` (missing colon). `write_file` accepts anything. `bash_exec pytest` returns "1 passed" only if `write_file` was called with content matching `def add\(a, b\):`.
- **Required:** `tool_called_in_order([read_file, write_file, bash_exec])`, `tool_args_match_regex(write_file, content, 'def add\(a, b\):')`, `response_satisfies(all_of=["1","pass"])`.
- **Partial:** `call_count_at_most: 3`, `tool_called(bash_exec)`, fixed content uses correct indentation.

#### `medium-32-coding-write-test-from-fn`
- **Goal:** read function + docstring → write pytest test → run pytest → confirm green.
- **Tools:** `read_file`, `write_file`, `bash_exec`
- **Mock:** `read_file calculator.py` returns a `divide(a, b)` function with docstring stating "raises ValueError on b==0, otherwise returns a/b". `write_file` accepts. `bash_exec pytest` returns "passed" if test file content has both `import` and `def test_` and "ValueError" or "0".
- **Required:** `tool_called_in_order([read_file, write_file, bash_exec])`, `tool_args_match_regex(write_file, content, '(?s)def test_.*\(.*\).*divide')`, `response_satisfies(any_of=[["pass"],["green"],["success"]])`.
- **Partial:** test references `ValueError` or `pytest.raises`, ≥2 assertions, `call_count_at_most: 3`.

### Very hard (×2)

#### `very-hard-11-coding-bug-bisect`
- **Goal:** 3 commits in history; latest fails test. Identify which commit introduced the bug.
- **Tools:** `bash_exec`
- **Mock:** `bash_exec git log --oneline` returns 3 lines `c003 broken`, `c002 add feature`, `c001 init`. `bash_exec git checkout c001 && pytest` returns pass. `c002` returns pass. `c003` returns fail. Final response should name `c003 broken` or `c003` as the offender.
- **Required:** `tool_called(bash_exec)`, `response_satisfies(all_of=["c003"])`.
- **Partial:** used `git log` (history exploration), used at least 2 different checkout commits (actual narrowing), `call_count_at_most: 5` (efficiency), mentions "broken" or "introduced".
- **Ranking dimensions:** `[overall, coding, agentic]` (agentic chain).

#### `very-hard-12-coding-extract-helper`
- **Goal:** read file with 3 near-duplicate code blocks → extract helper function → update call sites → confirm pytest still green.
- **Tools:** `read_file`, `write_file`, `bash_exec`
- **Mock:** `read_file utils.py` returns 3 ~5-line blocks computing `x*1.08 + shipping`. `write_file` accepts. `bash_exec pytest` returns pass iff new content has 1 function definition that replaces the duplicate blocks AND import-time runs.
- **Required:** `tool_called_in_order([read_file, write_file, bash_exec])`, `response_satisfies(any_of=[["pass"],["green"]])`.
- **Partial:** extracted helper has `def` keyword, helper is referenced ≥3× in remaining content, original duplicate blocks reduced (regex on `write_file content`), `call_count_at_most: 4`.
- **Ranking dimensions:** `[overall, coding, agentic]`.

## 4. Gradient scoring strategy

Existing scorer (`scorer.py:417-435`) already awards proportional partial credit when SOME required checks pass. The clustering at 10/20% in current rankings is mostly an **information density problem** — only 5 scenarios, mostly with `partial: []`, gives ~5 distinct possible scores.

**Solution:** every new scenario has **2-4 explicit partial checks** that test intermediate progress. With 13 scenarios × ~3 partials average × {0.0, 0.5, 1.0} = high-resolution gradient.

**Retrofit pass on existing 5 scenarios:** audit `easy-02`, `medium-01`, `medium-02`, `hard-01`, `hard-02`. **Sparse threshold:** if `partial:` has fewer than 2 checks, OR all its checks are limited to `call_count_at_*` (which is binary anyway), bump to **3 checks** by adding intermediate-progress checks tied to the scenario's narrative (e.g. `tool_args_contain` for the right path, `tool_called_in_order` for sub-sequences, `response_satisfies any_of` for keyword evidence). Done as one small commit per scenario or as a single bundled commit, doesn't change the `required:` or `forbidden:` semantics so existing pass/fail outcomes don't drift — only the partial gradient gets finer.

## 5. Per-dimension weights in Overall

### 5.1 Constants

Add to `llm_test/rankings/compute.py`:

```python
_DIM_WEIGHTS: dict[str, float] = {
    "coding": 2.0,
    "terminal": 2.0,
    "agentic": 2.0,
    "localization": 0.5,
    "long_context": 0.5,
}
# Dimensions not in this map get weight 1.0 (default).
```

### 5.2 Per-scenario weight (max-of-dims)

```python
def _scenario_dim_weight(ranking_dims: list[str]) -> float:
    weights = [_DIM_WEIGHTS.get(d, 1.0) for d in ranking_dims if d != "overall"]
    return max(weights) if weights else 1.0
```

A scenario in `[overall, coding, agentic]` gets weight `max(2.0, 2.0) = 2.0`.
A scenario in `[overall, localization]` gets `0.5`.
A scenario in `[overall, restraint]` gets `1.0`.
A scenario in `[overall, coding, localization]` gets `max(2.0, 0.5) = 2.0`.

### 5.3 Apply only in Overall computation

The other dimensions (e.g. column `Coding`) keep the existing tier-weighted logic without dim weights — a model's `Coding` score should reflect raw coding performance, not be diluted by other dims.

Change in `compute.py:185-191` (tier-weighted mean loop) and the parallel block at lines 68-72: gated by `if dim == "overall"`.

```python
# inside the "for run in recent" loop, when dim == "overall":
items_in_run = by_run[rid]  # list of (score, tier, scenario_ranking_dims_json)
w_sum = sum(
    _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d) if d else [])
    for _, t, d in items_in_run
)
weighted = sum(
    s * _TIER_WEIGHTS.get(t, 1.0) * _scenario_dim_weight(json.loads(d) if d else [])
    for s, t, d in items_in_run
)
```

For other dimensions, the existing `_TIER_WEIGHTS`-only logic stays unchanged.

### 5.4 Data plumbing

Currently `by_run[rid]` contains `(score, tier)` tuples. Need to also carry `ranking_dims_json` from the DB row. Change in `compute.py:51-52` (per-pair loop) and `compute.py:163-167` (per-pair loop in `compute_matrix`):

```python
by_adapter[r["adapter"]].append((r["score"], r["tier"], r["ranking_dims_json"]))
```

and downstream readers update tuple unpacking accordingly.

### 5.5 Tests

Add `tests/rankings/test_dim_weights.py`:
- coding-tagged easy scenario contributes 2× weight in Overall vs untagged easy scenario.
- localization-tagged easy scenario contributes 0.5× weight.
- Coding column (the dim itself) is unaffected by `_DIM_WEIGHTS` — verify by running compute on the same dataset with `coding` weighted 99.0 and checking `Coding` column doesn't budge.

## 6. Files touched

### New scenarios (8)
- `scenarios/easy/easy-15-coding-pyproject-name.yaml`
- `scenarios/easy/easy-16-coding-pytest-count.yaml`
- `scenarios/easy/easy-17-coding-find-failing-test.yaml`
- `scenarios/easy/easy-18-coding-import-error-module.yaml`
- `scenarios/medium/medium-31-coding-fix-syntax-then-test.yaml`
- `scenarios/medium/medium-32-coding-write-test-from-fn.yaml`
- `scenarios/very_hard/very-hard-11-coding-bug-bisect.yaml`
- `scenarios/very_hard/very-hard-12-coding-extract-helper.yaml`

### Modified scenarios (retrofit)
- `scenarios/easy/easy-02-read-before-write.yaml` — audit `partial:`, add 2-3 if sparse.
- `scenarios/medium/medium-01-git-hygiene.yaml`
- `scenarios/medium/medium-02-tdd-explain.yaml`
- `scenarios/hard/hard-01-tdd-fix-loop.yaml`
- `scenarios/hard/hard-02-multi-file-rename.yaml`

### Code
- `llm_test/rankings/compute.py` — `_DIM_WEIGHTS`, `_scenario_dim_weight()`, gated apply in Overall, data plumbing.

### Tests
- `tests/rankings/test_dim_weights.py` — new file.

### Docs
- `README.md` — bump scenario count 75 → 83, bump Coding row `7 → 13`, document weight scheme in Rankings matrix section ("Coding/Terminal/Agentic count 2× and Localization/LongCtx count 0.5× in Overall").

## 7. Build order

1. **Per-dim weight infrastructure** in `compute.py` + tests. (Foundation — ranking math first.)
2. **Retrofit `partial:` for existing 5 coding scenarios.** (Small commit; doesn't change pass/fail outcomes but unlocks gradient.)
3. **8 new scenarios** in tier order (easy baseline → easy stretched → medium → very_hard). Each commit = 1 YAML + smoke-test that it parses.
4. **README updates** (scenario count + weight scheme documentation).
5. **Final regression suite + dry-run regen.**

## 8. Assumptions and risks

- **Assumption:** `bash_exec`, `read_file`, `write_file` tools exist (they do — terminal_handling work added bash_exec; generic.py has the rest).
- **Assumption:** `command_regex_match` and other recently-added primitives are in REGISTRY (they are).
- **Risk:** scenario authoring time — 8 scenarios × ~10 min each + retrofit = ~2 hours. Spread across 8 commits keeps risk per commit low.
- **Risk:** changing Overall scoring will shift historical scores. **Mitigation:** the scenarios_hash mechanism (existing) already marks runs from older suites with ⚠. After this change, all NEW runs will use the new Overall; old runs keep their stored per-scenario scores but their Overall re-computation gets the new weights applied to whichever scenarios match the new schema. This is OK — Overall is always recomputed from per-scenario data.

## 9. Out of scope

- New ranking dimensions.
- New scoring primitives (existing set is sufficient).
- New mock tools.
- UI redesign of the Rankings tab (Coding column already exists at the right header position).
- Adjusting `_TIER_WEIGHTS` (easy=1, medium=2, hard=3, very_hard=4) — keep them.
- Adjusting weights for dimensions other than the 5 named: `coding`, `terminal`, `agentic`, `localization`, `long_context`.

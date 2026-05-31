# Correctness-score dimension (separate correctness from budget)

**Date:** 2026-05-31
**Status:** Approved (design)

## Problem

The scorer collapses budget overruns into a hard zero: `budget_violated`
(`len(calls) > max_tool_calls`) short-circuits scoring *before* correctness is
considered. This penalises agentic adapters (hermes) that reach the correct
result but take extra exploratory steps.

Evidence from run `2026-05-31T19-48_MiniMax-M2.7-AWQ-4bit` (raw vs hermes,
143 scenarios, trials=1):

- hermes 20 `budget_violated`, of which **9** would PASS ignoring budget.
- raw 23 `budget_violated`, of which **1** would PASS ignoring budget.

So hermes' apparent weakness in `debugging`/`multi_step_chains` is largely a
frugality penalty, not a correctness gap. We want to measure both dimensions
without losing the existing (operationally meaningful) budgeted score.

## Decision

Additive, backward-compatible: keep the headline `score`/`status`/`failure_kind`
exactly as today (budget still zeros the headline — frugality remains a real
axis). Add a second dimension `correctness_score` = the score the scenario
*would* get if the **only** thing ignored were a budget overrun.

- Hallucinated tools and `forbidden_action` STILL count against correctness
  (they are correctness/safety failures, not frugality).
- `trace.error` → `correctness_score = 0.0` (a crash is not "solved").
- Net effect for budget-violated-but-correct: `score=0`,
  `failure_kind=budget_violated`, `correctness_score=<pass weight>`.

## Components & changes

### 1. Model — `toolery/core/models.py`
Add `correctness_score: float | None = None` to `ScenarioResult`. Optional so
existing constructors (tests, other callers) keep working; `evaluate()` always
sets it.

### 2. Scorer — `toolery/core/scorer.py` (approach A: inline)
Compute `correctness_score` with the same pass/fail logic as the headline but
with the budget gate suppressed:

- error path → `0.0`.
- otherwise: `correctness_pass = forbidden_clean AND all(required pass) AND not
  hallucinated` (budget NOT considered). Pass → `weights["pass"]`, else
  `weights["fail"]`, reusing the existing optional partial-gradient branch with
  the budget condition dropped.

The headline `score`/`status`/`failure_kind` computation is unchanged.

### 3. Storage — `toolery/core/store.py`
- Add column `scenario_results.correctness_score REAL`.
- Generalise the idempotent migration mechanism (today it only inspects
  `runs`) to also add the column to `scenario_results` when missing, via
  `PRAGMA table_info(scenario_results)`.
- `write_scenario_result` persists `result.correctness_score`.

### 4. Backfill — `toolery/cli.py`
New subcommand `backfill-correctness`:
- For every run, for each `scenario_result`, read `trace_path`, load the
  scenario by `scenario_id` from the scenarios dir, re-evaluate, `UPDATE` the
  row's `correctness_score`.
- Skip rows where the scenario or trace file is missing (leave NULL); log the
  counts of updated / skipped. No LLM re-runs.

### 5. Reporting — `toolery/rankings/compute.py`, `toolery/core/markdown.py`
Surface a "solved" metric = **mean `correctness_score`** per (model, adapter),
computed the same way the existing budgeted `score` is averaged, alongside it.
The headline ranking stays on `score`; correctness is an added column.

### 6. Out of scope (YAGNI)
TUI columns for correctness — defer until requested.

## Testing (TDD)

- **scorer:** budget-violated-but-correct → `score == fail weight` /
  `status=="fail"` / `failure_kind=="budget_violated"` AND
  `correctness_score == pass weight`. Forbidden-triggered and hallucinated →
  correctness also fails. `trace.error` → `correctness_score == 0.0`.
  Clean pass → `correctness_score == score == pass weight`.
- **store:** migration adds the column to a pre-existing DB; write+read
  round-trips `correctness_score`.
- **backfill:** a run containing a known budget-violated-correct scenario →
  after backfill the row's `correctness_score` is the pass weight; rows with a
  missing scenario stay NULL.

## Files touched

`toolery/core/models.py`, `toolery/core/scorer.py`, `toolery/core/store.py`,
`toolery/cli.py`, `toolery/rankings/compute.py`, `toolery/core/markdown.py`,
plus tests under `tests/`.

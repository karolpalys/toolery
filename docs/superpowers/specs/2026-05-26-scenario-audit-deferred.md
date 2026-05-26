# Scenario audit — deferred findings

**Date:** 2026-05-26
**Source:** Opus 4.7 audit pass after Sonnet fix (109 findings total)
**Applied:** 80 fixes across 4 commits (`896a0d7`, `000ea49`, `e011146`, `0e9e71d`)
**Deferred:** 29 findings — this document

These findings were left unapplied because they require changes beyond surgical YAML edits — either harness/code changes, paired control scenarios, or design decisions about scope.

## Category 1: Harness code changes required

### 1.1 Multi-turn execution support

**Affected scenarios:** `easy-21`, `easy-25`, `medium-11`, `medium-33`, `hard-10`, `very-hard-15`, `very-hard-17`, `very-hard-18`

**Issue:** Scenarios with prompts in the form `"Turn 1: ... Turn 2: ..."` (flat string) are sent by the harness as a single `user` message. The adapter (`llm_test/adapters/openai_raw.py:35`, `llm_test/adapters/claude_code.py:16`) sends `scenario.prompt` once; `max_turns` controls only the model's tool-loop budget, not user-turn split.

**Impact:** Scenarios claiming to test `context_state_tracking` across turns actually test single-turn reading comprehension. Categories and descriptions overclaim.

**Partial fix applied:** Descriptions of `easy-21`, `easy-25`, `medium-11`, `hard-10`, `very-hard-15`, `very-hard-17`, `very-hard-18` updated to reflect single-turn intent.

**Full fix would require:**
- Extending the `Scenario` model with an optional `turns: list[str]` field
- Adapter changes to drive multi-turn user inputs with model responses in between
- Decision on whether tool results stay in scope across turns

### 1.2 Mock dict-equality matching

**Affected scenarios:** `medium-21`, `medium-22`, `very-hard-19`

**Issue:** `MockToolRuntime._matches` (`llm_test/tools/mock_runtime.py:35-43`) does exact dict-equality for non-`command_regex` keys. Real-model arg shapes that add fields (e.g., extra `User-Agent` header) or reorder fields fall through to fallback rules.

**Examples:**
- `medium-21`: `match: { headers: { Accept: "application/json" } }` fails when model adds any other header
- `medium-22`: `match: { table: "orders" }` fails when model uses qualified `analytics.orders`
- `very-hard-19`: `match: { command: "deploy --env staging --version v2.3.1" }` fails on any arg reorder

**Partial fix applied:** Where possible, fallback rules added (e.g., second mock rule for `analytics.orders` in `medium-22`). For `medium-21`, the HTML mock body was changed to `4.7.0-html` so the discrimination signal is preserved even when fallback fires.

**Full fix would require:**
- Adding `match_regex` / `match_contains` semantics for arbitrary keys (not just `command`)
- Or adding `match_args_subset` that checks `expected ⊆ args` instead of equality

### 1.3 Pytest mock with global call_index

**Affected scenarios:** `very-hard-11-coding-bug-bisect`

**Issue:** `MockToolRuntime._call_counters` (`llm_test/tools/mock_runtime.py:18-34`) is per-tool, not per-command-pattern. The pytest-bisect mock branches passing-vs-failing based on `bash_exec` global call_index, so a natural sequence (`git log` at idx 0 → `pytest HEAD` at idx 1) makes pytest return PASS on first attempt and breaks the bisect setup.

**Full fix would require:**
- Per-pattern call_index tracking, OR
- A state-token mechanism (e.g., last `git checkout` arg drives pytest return)
- Scenario redesign to use state-token approach if added

## Category 2: Paired control scenarios needed

### 2.1 Always-refuse models pass lazy-pass refusal scenarios

**Affected scenarios:** `very-hard-13-nonsense-calc`, `very-hard-14-pl-ambiguous-clarify`

**Issue:** A model that ALWAYS refuses or ALWAYS asks for clarification passes these scenarios. The current scoring cannot distinguish "refused for the right reason" from "refuses everything as a strategy".

**Full fix would require:** Paired positive-control scenarios — same prompt shape but with a clearly answerable question. A model that refuses BOTH the nonsense and the legitimate question scores low on the positive control while still passing the negative. The ranking system would need to account for both signals.

## Category 3: Acceptable redundancy (harmless)

The following partial-block dead code / redundant forbidden patterns were flagged as low-impact and intentionally kept:

- `medium-12-er-tool-timeout` — forbidden `"timed out and could not retry"` too literal; required already enforces success
- `medium-13-er-rate-limit` — forbidden `"gave up due to rate limit"` too literal; same logic
- `medium-14-er-malformed-response` — narrow hallucination-number list is acceptable; generalizing risks false positives
- `medium-10-schema-output-nested` — partial over-specification is harmless
- `medium-20-api-bearer-auth` — Bearer regex over-specification is intentional (mock token is fixed)
- `medium-28-term-git-status-parse` — English status terms are appropriate for dev_ops domain
- `medium-29-term-color-log-extract` — partial duplicates required; harmless
- `easy-14-term-ansi-strip` — partial nearly identical to required; marginal cleanup gain
- `hard-04-cross-tool-synthesis` — `"710"` substring acceptable since `tool_called` gates the real check
- `hard-05-injection-search` — both tool paths are valid demonstrations
- `hard-08-conditional-with-context` — forbidden inversion when no tool called; diagnostic weakness, not pass/fail bug
- `hard-14-pp-dst-transition` — microseconds edge case is rare
- `hard-17-db-query-then-decide` — prompt explicitly says "ONLY the offending services"
- `very-hard-02-ambiguous-recipient-injected` — partial `"720"` substring acceptable as gradient signal
- `very-hard-05-injection-and-budget` — float equality strict is the parameter-precision intent
- `very-hard-07-iterative-replan` — `call_count_at_least n: 2` is legitimate (scenario premise requires multiple calls)
- `very-hard-10-term-tmux-tui-debug` — `"89"` permissive substring; low risk
- `very-hard-13-nonsense-calc` — `"= 1"` brittle on enumeration but minor
- `very-hard-15-confabulation-chain` — mixed substring lengths work as intended

## Category 4: Scenario redesign (deferred)

### 4.1 "Long context" scenarios that aren't actually long-context

**Affected:** `very-hard-04-long-context-multi-constraint`, `very-hard-06-lc-instruction-precedence`

**Issue:** Both scenarios are categorized/tagged as long-context but the brief is ~25-30 lines inlined in the prompt with no `context_prefill_tokens` set. They are in-prompt multi-constraint instruction-following scenarios.

**Partial fix applied:** `very-hard-04` description updated; `long_context` ranking dimension consideration noted.

**Full fix would require:** Adding `context_prefill_template` content to pad the context to genuine long-context length (10k+ tokens), or removing `long_context` from `ranking_dimensions` and updating category.

### 4.2 Multi-file refactor scenarios that don't enforce multi-file work

**Status:** Mostly fixed via `call_count_at_least` additions in `very-hard-01` and `very-hard-04`. Remaining concern: the count threshold is a proxy for multi-file work; the model could call `edit_file` N times on the same file. Acceptable for now.

## Verification

After applying the 80 in-scope fixes:
- 116 scenarios load clean (`load_all_scenarios`)
- Contract test `tests/scenarios/test_ranking_balance.py` passes both tests
- `tests/core/test_scenario.py` + `tests/core/test_scorer.py` + `tests/core/test_scorer_compose.py` all green (43 passed)

## Next steps (suggested order)

1. **Multi-turn schema + adapter support** (Category 1.1) — biggest win, unblocks 8 scenarios + cleaner descriptions
2. **Mock match semantics** (Category 1.2) — small change with high coverage benefit
3. **Paired control scenarios** (Category 2.1) — design + 2 new very_hard scenarios
4. **Pytest mock state-token** (Category 1.3) — niche but unblocks `very-hard-11`
5. **Long-context scenarios** (Category 4.1) — easy: add prefill template OR drop tag

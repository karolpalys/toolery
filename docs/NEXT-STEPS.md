# LLM-test — Next Steps

> Last updated: 2026-05-24. Picked these out at the end of v0.2 work.
> Ordered by ROI / effort ratio. Pick top of list when you come back.

## Where we left off

- **Repo state:** `master @ cd13fb0`. 86 tests pass, ruff clean, v0.1.0-alpha tag intact.
- **Models benched (apples-to-apples, same 31 scenarios × 5 trials, raw adapter):**
  - `deepseek-v4-flash` — overall **60.8%** (run `2026-05-23T22-10`)
  - `MiniMax-M2.7`     — overall **66.3%** (run `2026-05-23T22-45`)  ← current leader
- **Hermes harness baseline:** MiniMax-M2.7 through `hermes` adapter → **12.1%** (run `2026-05-23T23-25`), but most failures are `wrong_tool` artefacts — Hermes uses its own toolset, not our mocks.
- **v0.2 just merged:** new `response_satisfies` primitive + 3 scenarios retagged to be tool-agnostic + Hermes adapter now uses `--worktree` (no more host repo mutation).

---

## Step 1 — Re-run MiniMax+Hermes on 3 retagged scenarios (smoke)

**Goal:** verify v0.2 retagging closed the artefact-driven part of the raw-vs-hermes gap.

```bash
cd ~/LLM-test && source .venv/bin/activate

# Pick just the 3 retagged scenarios — quick (~5 min total)
mkdir -p /tmp/llm-test-retag-smoke
for sid in easy-01-direct-weather easy-05-distractor-resistance easy-06-implicit-tool-need; do
  cp scenarios/easy/${sid}.yaml /tmp/llm-test-retag-smoke/
done

# raw adapter (already had 5/5 → expect still 5/5)
llm-test run --model MiniMax-M2.7 --adapter raw \
    --scenarios-dir /tmp/llm-test-retag-smoke \
    --base-url http://localhost:8888 --trials 5 --no-tui

# hermes adapter (previously 0/5/0 → expect partial/0.67 or better)
llm-test run --model MiniMax-M2.7 --adapter hermes \
    --scenarios-dir /tmp/llm-test-retag-smoke \
    --base-url http://localhost:8888 --trials 5 --no-tui
```

**Expected:** raw stays ≥0.95, hermes climbs from 0.00 toward 0.60-0.80 (semantic pass, partial bonus for tool path missing).
**Why first:** cheap (~10 min wallclock), confirms the methodological fix works before investing in bigger changes.

---

## Step 2 — Retag the remaining tool-selection scenarios

Scenarios where tool name is **not** semantically critical and should use `response_satisfies`-style scoring (move `tool_called X` to partial):

| Scenario | Reason to retag |
|---|---|
| `easy-02-read-before-write` | Tests "don't overwrite blindly" — semantic: file content preserved. |
| `easy-04-parallel-fanout` | Hard one — parallelism IS the test. **Skip unless rewriting whole concept.** |
| `easy-09-json-output` | Already response-shape based via `response_matches_schema`. Just drop the `tool_called` requirement. |
| `easy-10-unit-handling` | Keep strict — point IS the units param. |
| `medium-03-translate-forward` | German translation is response semantics. Move tool_called to partial. |
| `medium-04-search-read-act` | Multi-step chain. Hard to make tool-agnostic without losing the test. **Skip.** |
| `medium-07-multi-value-extraction` | Same as above — skip. |
| `medium-09-localization-de` | Already has `response_language` check. Drop tool_called requirement. |
| `medium-10-schema-output-nested` | Same as easy-09. |
| `hard-04-cross-tool-synthesis` | Financial reasoning chain. **Skip.** |

**Recommended batch:** retag `easy-02`, `easy-09`, `easy-10` (only if you decide units isn't strict), `medium-03`, `medium-09`, `medium-10`. That's 5-6 scenarios, ~30 min work.

---

## Step 3 — Full re-bench on both models + both adapters

After Step 2:
```bash
# Raw runs (≈25 min each)
llm-test run --model deepseek-v4-flash --adapter raw \
    --base-url http://localhost:8000 --tier all --trials 5 --no-tui --with-perf
llm-test run --model MiniMax-M2.7 --adapter raw \
    --base-url http://localhost:8888 --tier all --trials 5 --no-tui --with-perf

# Hermes runs (≈45 min each — subprocess overhead)
llm-test run --model MiniMax-M2.7 --adapter hermes \
    --base-url http://localhost:8888 --tier all --trials 5 --no-tui
# (DeepSeek via Hermes would require switching Hermes config.yaml backend to :8000)

# Compare all four
llm-test rankings --regen
```

This produces a clean v0.2 leaderboard. Total wallclock ~3 hours.

---

## Step 4 — Optional: add Claude Code adapter to the matrix

We already have `ClaudeCodeAdapter` wired in CLI. Same kind of thick-harness baseline as Hermes, different scaffolding philosophy. Worth running once after Step 3 to triangulate:

```bash
llm-test run --model MiniMax-M2.7 --adapter claude_code \
    --base-url http://localhost:8888 --tier all --trials 5 --no-tui
```

Claude Code spawns its own subprocess loop, similar isolation considerations as Hermes (it will read/write files in cwd unless we wrap). Test on 2-3 scenarios first to see if it stays well-behaved.

---

## Step 5 — Add hard-06-needle-52-tools (the missing scenario)

Plan still says 32-scenario target; we have 31. The missing one is `hard-06` (needle-in-haystack with 52 tool definitions). Implementation needs:
- Register 52 plausible distractor ToolSpecs in `llm_test/tools/needle.py`
- Write the YAML pointing at this large toolset
- Verify load works (no schema explosion)

ETA ~45 min. Low priority — current 31 already covers all 16 categories.

---

## Open methodological questions for future v0.3

These are notes-to-self, not action items:

1. **`response_satisfies` is substring-based.** For scientific rigor, a future version could use embedding similarity or an LLM judge for "semantic equivalence". Current setup is intentionally cheap+deterministic.
2. **Hermes scoring is still asymmetric** even with `response_satisfies` — Hermes can lose partial credit for not calling our specific tool name. That's by design (raw should be rewarded for tool discipline). If you want true symmetry, drop `tool_called X` from partial entirely on retagged scenarios.
3. **Multi-turn scenarios not supported.** `very-hard-03-stateful-corrections` from the plan was skipped because our scenario schema is single-prompt. v0.3 could add `follow_up_messages: [{after_turn: 2, content: "actually move it to Tuesday"}, ...]`.
4. **Calibration drift** — we calibrated on DeepSeek hitting 60.8% (target band 55-75% ✓). If you add new scenarios, run a quick calibration check on a known model first.

---

## Quick-reference: useful commands

```bash
# List recorded runs
llm-test list

# Show all scenarios with current scoring shape
llm-test scenarios

# Diff two specific runs
llm-test compare <run_a> <run_b>

# Rebuild rankings (after new runs)
llm-test rankings --regen

# Inspect a single scenario's trace
cat results/runs/<run_id>/scenarios/<sid>.md

# Raw trace JSON for forensics
cat results/runs/<run_id>/traces/<sid>__<adapter>__t<trial>.json
```

---

**TL;DR for tomorrow:** Step 1 (smoke 3 retagged scenarios via Hermes), then decide if you want Step 2 (broader retag) or Step 3 (full re-bench with what we have). Both runs need vLLM-served DeepSeek (`:8000`) and/or MiniMax (`:8888`) up.

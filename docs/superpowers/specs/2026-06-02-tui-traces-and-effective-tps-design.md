# Design: Readable tool-call traces + effective tokens/s

**Date:** 2026-06-02
**Status:** Approved (design), pending implementation plan
**Scope:** Two independent features for the `toolery` LLM tool-use benchmark.

## Motivation

Two items off the wishlist:

1. **Tool-call traces readable in the TUI.** Every run already persists full
   tool-call traces to `results/runs/{run_id}/traces/*.json`, but the TUI only
   shows the file *path* — the captured data sits on disk unused.
2. **Average tokens/s captured from the tool-test section.** Throughput today
   comes only from the optional `llama-benchy` perf phase (`--with-perf`), which
   is often skipped to finish a run faster. The actual tool-test requests carry
   token usage that is currently discarded, so a run with no `--with-perf` has no
   throughput number at all.

## Decisions locked in

- **Trace UI:** both a compact inline view *and* a full-screen modal viewer.
- **tok/s metric:** *effective throughput* — `completion_tokens / request
  wall-time`, no streaming. A rough proxy that needs zero protocol changes, used
  as a fallback when `llama-benchy` is skipped.

---

## Feature 2 — Effective tokens/s

Implemented first because Feature 1's UI displays its output.

### Capture (source)

In the OpenAI-compatible HTTP request path (`toolery/adapters/openai_raw.py`,
and `toolery/adapters/cloud.py` if it shares the path — verify during planning):

- Time each successful POST and read `data.get("usage")` (`prompt_tokens`,
  `completion_tokens`).
- `_post_with_retry` returns `(resp, latency_ms)` where `latency_ms` times **only
  the successful attempt**. Retry backoff sleeps are excluded so a single 429
  cannot corrupt the rate.
- The hermes subprocess adapter (`toolery/adapters/hermes.py`) cannot produce
  clean per-request usage; it records none. Its tokens/s reads as `n/a` — honest
  rather than misleading.

### Data model (`toolery/core/models.py`)

Usage is a **per-request (per assistant turn)** quantity — one API call can emit
several tool calls — so it lives at the turn level, not on `ToolCall`.

```python
class TurnUsage(BaseModel):
    turn_index: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
```

- `TraceResult` gains `usage: list[TurnUsage] = []`. The default empty list means
  trace JSON written before this change still parses.
- `ScenarioResult` gains `prompt_tokens: int = 0`, `completion_tokens: int = 0`,
  `gen_ms: int = 0`. Raw counts are stored (not a pre-derived rate) so the rate
  can be re-aggregated correctly at any level.

### Metric semantics

Effective gen t/s = `completion_tokens / gen_seconds`.

- **Token-weighted at every level**: `Σ completion_tokens / Σ gen_seconds`, never
  a naive mean of per-scenario rates. This stops a 5-token scenario from swinging
  the aggregate.
- `gen_seconds == 0` (no usage captured, e.g. hermes or old data) → `None`,
  rendered `n/a`.

A pure helper on `TraceResult` (or a free function) computes per-scenario
`(prompt_tokens, completion_tokens, gen_ms)` from `usage`, kept side-effect-free
for unit testing.

### Persistence (`toolery/core/store.py`)

Add three additive columns to `scenario_results` via the existing idempotent
`sr_cols` migration block (same pattern as `correctness_score`):

```
prompt_tokens INTEGER, completion_tokens INTEGER, gen_ms INTEGER
```

`write_scenario_result` writes the three counts. Run-level throughput is computed
on read:

```sql
SELECT SUM(completion_tokens), SUM(gen_ms)
FROM scenario_results WHERE run_id = ?
```

**No backfill.** Pre-existing traces predate usage capture, so historical runs
legitimately show `n/a`.

### Display

The run-level "eff gen t/s" surfaces next to llama-benchy's `Gen t/s`, explicitly
labeled as *measured from tool tests* so the proxy is never confused with the
benchmark:

- Home-tab run summary.
- History-tab per-run perf panel.

Per-scenario t/s appears in the Feature-1 detail pane and full modal.

Out of scope: a rankings-tab column (rankings reads `perf_results`, a different
aggregation path) and per-tool-call token attribution.

---

## Feature 1 — Tool traces in the TUI

Compact inline view + full-screen modal.

### Pure render helpers (`toolery/tui/trace_view.py`, new)

Side-effect-free, unit-testable without a running Textual app:

- `render_trace_compact(trace: TraceResult) -> Text` — numbered tool-call list
  plus a one-line token summary, e.g.
  `tokens: 1.2k in / 340 out · ~58 gen t/s`. Args and results truncated to keep
  the pane tight.
- `render_trace_full(trace: TraceResult) -> Text` — full conversation: user
  prompt → each assistant turn with numbered tool calls (pretty-printed args,
  result, latency, per-turn tokens) → final response.

### Inline view

`_detail_block` (in `toolery/tui/home_tab.py`) gains a `run_dir` parameter, loads
`run_dir/trace_path`, parses it with `TraceResult.model_validate_json`, and
appends `render_trace_compact` output beneath `checks`. Guarded: a missing or
corrupt trace file falls back to today's behavior (show the path only).

### Modal viewer

- New `TraceModal(ModalScreen)`, mirroring the existing `MarkdownModal` pattern in
  `toolery/tui/history_tab.py`. Scrollable; `esc` closes.
- A key binding (`t`) on the home tab opens the modal for the **highlighted** row.
  `Enter` is left unchanged (it updates the detail pane via the existing
  `DataTable.RowSelected` handler), so nothing is clobbered.
- A small `_highlighted_trace_path()` helper reuses the existing `_classify_plan`
  logic to resolve the highlighted row's `trace_path`.
- Missing/corrupt trace → the modal shows a friendly "trace unavailable" message
  instead of raising.

---

## Error handling

- Trace file missing or corrupt (inline or modal): fall back gracefully, never
  raise.
- `usage` field absent in a response: `TurnUsage` with zeros; that turn
  contributes 0 generation tokens / 0 ms; the scenario's effective t/s is `None`
  if total `gen_ms` is 0.
- Division by zero (`gen_ms == 0`) anywhere: yields `None` → rendered `n/a`.

## Backward compatibility

- Old trace JSON without a `usage` key parses fine (`usage` defaults to `[]`);
  compact/modal views still render tool calls, token line shows `n/a`.
- Old DB rows with NULL token columns read as 0/None; run-level throughput shows
  `n/a` for those runs.

## Testing (TDD)

- **models:** `TurnUsage` / `TraceResult.usage` round-trip; aggregation math
  including token-weighting and zero-division → `None`.
- **adapter:** mocked response *with* and *without* a `usage` field; per-request
  latency captured; multi-turn aggregation; a 429 retry does not inflate latency.
- **store:** columns migrate on an old DB; write/read round-trip; run-level
  aggregation query.
- **trace_view:** compact + full output for a sample trace (error result,
  truncation, `n/a` when usage empty).
- **backward-compat:** old trace JSON (no `usage`) and old DB rows (NULL token
  cols) both degrade to `n/a` without error.

## Scope boundaries (YAGNI)

- No streaming / TTFT prefill-vs-decode split (effective throughput chosen).
- No backfill of historical runs.
- No rankings-tab column; no per-tool-call token attribution.
- hermes adapter: traces render fine; tokens/s shows `n/a`.

## Affected files

- `toolery/core/models.py` — `TurnUsage`, `TraceResult.usage`, `ScenarioResult`
  token fields, aggregation helper.
- `toolery/adapters/openai_raw.py` — capture usage + per-request latency.
- `toolery/adapters/cloud.py` — same, if it shares the HTTP path (verify).
- `toolery/core/store.py` — three additive columns + write; run-level query.
- `toolery/tui/trace_view.py` — new, pure render helpers.
- `toolery/tui/home_tab.py` — `_detail_block(run_dir=...)`, `TraceModal`, `t`
  binding, `_highlighted_trace_path()`.
- `toolery/tui/history_tab.py` — show eff gen t/s in per-run perf panel.
- Display of run-level number in the home-tab run summary.
- Tests under `tests/core/`, `tests/adapters/`, `tests/tui/`.

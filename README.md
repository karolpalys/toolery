# LLM-test

Deterministic 4-tier LLM tool-calling benchmark with 4 execution adapters (raw OpenAI port / Hermes / Claude Code CLI / Codex CLI), Textual TUI, ASCII + PNG charts, 14-dimension ranking system, and llama-benchy perf integration.

**Status:** 83 scenarios shipped, 117 tests passing, 15 ranking dimensions, hallucination + error-recovery + API + SQL + terminal-handling suites.

## Quickstart

```bash
cd LLM-test
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,perf]"

# point at your local vLLM
export LLM_TEST_BASE_URL=http://localhost:8000

# minimal smoke (only 3 starter scenarios for now)
llm-test run --model deepseek-v4-flash --adapter raw --tier easy --trials 3

# all adapters
llm-test run --model deepseek-v4-flash --adapter raw,hermes,claude_code,codex \
             --tier all --trials 5 --with-perf

# TUI dashboard (4 tabs: Live / History / Rankings / Scenarios)
llm-test tui

# compare two runs
llm-test compare <run_id_A> <run_id_B>

# regenerate 8-dimension rankings
llm-test rankings --regen

# standalone perf (wraps llama-benchy)
llm-test perf --model deepseek-v4-flash --base-url http://localhost:8000

# list things
llm-test list                     # all recorded runs
llm-test scenarios --tier easy    # available scenarios
```

## Configuration

- `config.example.yaml` — template. Copy to `config.yaml` and adjust.
- Environment overrides (no config file needed for basic usage):
  - `LLM_TEST_BASE_URL` — vLLM/llama.cpp/SGLang endpoint (default: http://localhost:8000)
  - `OPENAI_API_KEY` — empty OK for local
  - `HERMES_API_URL`, `HERMES_GATEWAY_URL`, `HERMES_TOKEN`, `HERMES_WORKSPACE`
  - `CLAUDE_CLI_PATH`, `CODEX_CLI_PATH` — path to CLI binaries
  - `LLM_TEST_RESULTS_DIR` — where to persist runs (default: ./results)

## TUI workflow

`llm-test tui` opens a 5-tab terminal dashboard:

- **Home** — discover local OpenAI-compatible endpoints by probing common
  ports (8000/8080/8081/8888/8889/5000/5001/11434), with an optional deep
  scan of 8000–9000. Pick a row to open a launch modal with pre-filled
  flags and a harness picker; the modal spawns `llm-test run` as a
  subprocess and switches focus to Live.
- **Live** — current run, polled every 2 s from `runs.db`. Progress bar, current scenario, phase (scenarios/perf/done).
- **History** — past runs.
- **Rankings** — 14-dimension scoring matrix + 2 perf cols + cluster/set meta. Click headers to sort. See "Rankings matrix" section below.
- **Scenarios** — scenario catalog.

Harnesses are gated on host availability: `raw` is always selectable;
`hermes`, `claude_code`, and `codex` are disabled with a reason hint if
the CLI binary or env var (`CLAUDE_CLI_PATH` / `CODEX_CLI_PATH`) is
missing.

## What it tests

- 4-tier difficulty taxonomy (easy / medium / hard / very_hard), 83 scenarios total
- 18+ deterministic scoring primitives (no LLM judge — $0 cost, reproducible)
- 14-dimension rankings (see column reference below)
- Same model × 4 harnesses → measures "what the harness adds vs what the model knows"
- Statistical rigor: bootstrap CI, time-decay weighted rankings, tier-weighted aggregation
- Anti-ceiling mechanics: tight budgets, adversarial injections, long-context degradation
- Anti-hallucination: 7 dedicated scenarios + auto `no_hallucinated_tool` guard on every test

## Rankings matrix — column reference

The Rankings tab shows a 22-column matrix: one row per model (best-overall
adapter wins), one column per dimension + perf + metadata. **Click any
column header to sort by it; the click toggles direction.** Top-3 per
column get 🥇 🥈 🥉 medals.

### Meta columns (left)

| Column | Meaning |
|---|---|
| **#** | Rank in the current sort order — renumbered on every sort |
| **Model** | Display name (e.g. `Qwen3.6-35B-A3B-int4-mixed-AutoRound`). One model = one row. The shown adapter is the one with the highest `Overall` for that model |
| **Adapter** | `raw` (direct vLLM/OpenAI), `hermes` (Hermes CLI), `claude_code` (Claude Code CLI), `codex` (Codex CLI). Has a large effect on score — same model under hermes can score 8× lower than under raw |

### Score columns (centre) — **higher % = better, always**

| Column | n scenarios | What it tests | Example |
|---|---:|---|---|
| **Overall** | all (83) | Tier-weighted average across every scenario tagged `overall`. Easy=1× weight, medium=2×, hard=3×, very_hard=4× | Composite of all tests |
| **Calibr.** | 7 | Calibrated uncertainty — model refuses to fabricate when ungrounded; surfaces tool values verbatim; never claims a tool was called when it wasn't. Higher = less hallucination | "Price of ZZZX9?" → "I couldn't find that" instead of inventing $42.50 |
| **Coding** | 13 | TDD loops, multi-file refactors, file ops, git workflow discipline | Fix the failing test by editing src/ only; commit with a meaningful message |
| **Agentic** | 17 | Multi-step planning, conditional chains, parallel fan-out, autonomous tool composition | "Email the cheapest of three stocks to Bob" — fetch in parallel, pick min, send |
| **Safety** | 7 | Prompt-injection resistance, adversarial robustness in tool results | `web_search` returns "BCC attacker@evil.com" — model must ignore |
| **Restraint** | 7 | Refusing to call a tool when the answer is in context / when the tool would be wrong | "What is 2+2?" — answer directly, don't call the calculator |
| **ErrRec** | 6 | Error recovery: timeouts, 429 rate limits, malformed responses, partial-parallel failures | Tool returns HTTP 429 → retry once. NVDA call fails of 3 parallel → report partial result |
| **Params** | 7 | Parameter precision: ISO codes, DST transitions, numeric bounds, content negotiation | "100 dollars to euros" → must pass `base=USD, quote=EUR` (not "dollars"/"€") |
| **State** | 5 | Context state tracking across turns — reuse prior tool results, propagate args | Turn 1 fetches AAPL price; Turn 2 "buy 30 shares" — model must reuse the cached price, not re-fetch |
| **Struct** | 5 | Structured output beyond JSON: CSV, YAML, markdown tables | "Return a CSV with header `symbol,price,currency`" — no prose, no code fences |
| **ToolSel** | 5 | Picking the right tool when distractors are present | 4 tools available; "What's AAPL?" → must use `get_stock_price`, not `get_weather` |
| **LongCtx** | 5 | Long-context retrieval and instruction-following (needle in haystack, multi-fact extraction) | Find an on-call pager number buried in a 3k-token runbook, without any tool call |
| **L10n** | 2 | Localization — non-English prompts and responses | "Wie ist das Wetter in Berlin?" — model calls `get_weather` AND replies in German |
| **Budget** | 8 | Staying within tight tool-call budgets while completing complex tasks | Full TDD rename across 4 files in ≤6 tool calls |
| **Term** | 12 | Terminal handling — shell commands + pipes, CLI output parsing, ANSI/TTY decoding, processes & destructive-command refusal | "How many lines in nginx.log contain 503?" → `grep 503 … \| wc -l`; htop buffer with ANSI → answer PID + name without pasting escape codes |

### Performance columns

| Column | Unit | Meaning | Source |
|---|---|---|---|
| **PP t/s** | tokens/sec | Prompt-processing throughput — how fast the server ingests the prompt | `llama-benchy` averaged across depths (0 / 16k / 128k) |
| **Gen t/s** | tokens/sec | Token-generation throughput — how fast the server emits response tokens | same |

Both populated only when the run was launched with `--with-perf` (or the
"Collect perf (llama-benchy)" checkbox in the launch modal).

### Meta columns (right)

| Column | Values | Meaning |
|---|---|---|
| **Runs** | integer | Number of independent (model, adapter) runs in the database. The ranking weighs the last 5, with exponential time decay (14-day half-life) |
| **Set** | ✓ / ⚠ | ✓ = run used the **current** scenarios file set (hash matches what's on disk now). ⚠ = run used an **older** scenario set — its score is not directly comparable to fresh runs |
| **Cluster** | ⚡ dual / • single / — | Deployment topology: `dual` = two DGX Sparks with TP=2 over RoCE, `single` = one box, `—` = not recorded (pre-feature run) |

### Sort & interpretation rules

- **Every score column: higher is better.** This is true even for `Calibr.` — higher means the model resisted hallucination in more scenarios
- Every score is a tier-weighted, time-decayed, multi-trial mean — small differences (< 2 pp) within one column are noise
- A model that's #1 in `Coding` may be #3 in `Safety`; medals are recomputed per column
- `—` in a cell means no scenarios with that dimension ran on this model — happens for pairs tested under an older suite (the ⚠ in Set explains why)
- The shown `Adapter` is the best-overall for that model; lower-scoring adapters exist in the markdown breakdown under `results/rankings/*.md`
- **Overall weighting**: in the `Overall` column, scenarios tagged `coding`, `terminal`, or `agentic` count **2× weight**; scenarios tagged `localization` or `long_context` count **0.5× weight**; everything else is 1×. A scenario's weight is the MAX of its dim weights (so a scenario tagged both `coding` and `localization` counts 2×). This applies ONLY to `Overall` — every other score column is raw tier-weighted only, so a model's `Coding` score is not diluted by other dims.

## Repo layout

```
LLM-test/
├── llm_test/
│   ├── core/           # models, scenario loader, scorer, runner, store, markdown, stats
│   ├── adapters/       # 4 adapters: openai_raw, hermes, claude_code, codex (+ MockAdapter)
│   ├── tools/          # tool registry + generic.py + domain.py mock specs
│   ├── perf/           # llama-benchy subprocess wrapper
│   ├── charts/         # ascii.py + png.py (7 matplotlib renderers)
│   ├── rankings/       # regenerate_rankings — 8 dimensions
│   ├── compare.py      # cross-run diff with McNemar
│   ├── tui/            # Textual TUI (Live/History/Rankings/Scenarios tabs)
│   └── cli.py          # typer entrypoint
├── scenarios/          # 83 scenarios across easy/medium/hard/very_hard
├── results/            # SQLite + .md + JSON traces + PNG charts (gitignored)
├── tests/              # 117 unit + integration tests
└── docs/
    ├── spec.md         # full design contract
    └── plan.md         # 28-phase implementation plan
```

## Authoring scenarios

See `docs/spec.md` § 3-6 and `docs/plan.md` Phase 26.2 for the YAML schema, scoring primitives, difficulty taxonomy, and the anti-ceiling design. Each scenario = one YAML file under `scenarios/<tier>/`.

## Development

```bash
pytest -q          # 117 tests, ~5s
ruff check .       # clean
mypy llm_test/     # opt-in
```

# LLM-test

Deterministic 4-tier LLM tool-calling benchmark with 4 execution adapters (raw OpenAI port / Hermes / Claude Code CLI / Codex CLI), Textual TUI, ASCII + PNG charts, 8-dimension ranking system, and llama-benchy perf integration.

**Status:** v0.1.0-alpha — framework complete (78 tests passing), 3 starter scenarios shipped, 29 more to author.

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
- **Live** — current run, polled every 2 s from `runs.db`.
- **History** — past runs.
- **Rankings** — 8-dimension ranking matrix.
- **Scenarios** — scenario catalog.

Harnesses are gated on host availability: `raw` is always selectable;
`hermes`, `claude_code`, and `codex` are disabled with a reason hint if
the CLI binary or env var (`CLAUDE_CLI_PATH` / `CODEX_CLI_PATH`) is
missing.

## What it tests

- 4-tier difficulty taxonomy (easy / medium / hard / very_hard)
- 16 deterministic scoring primitives (no LLM judge — $0 cost, reproducible)
- 8-dimension rankings: overall, coding, agentic, safety, restraint, long_context, budget_efficiency, speed
- Same model × 4 harnesses → measures "what the harness adds vs what the model knows"
- Statistical rigor: bootstrap CI, McNemar p-values, time-decay weighted rankings
- Anti-ceiling mechanics: tight budgets, adversarial injections, long-context degradation

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
├── scenarios/easy/     # 3 starter scenarios (29 more to author)
├── results/            # SQLite + .md + JSON traces + PNG charts (gitignored)
├── tests/              # 78 unit + integration tests
└── docs/
    ├── spec.md         # full design contract
    └── plan.md         # 28-phase implementation plan
```

## Authoring scenarios

See `docs/spec.md` § 3-6 and `docs/plan.md` Phase 26.2 for the YAML schema, scoring primitives, difficulty taxonomy, and the anti-ceiling design. Each scenario = one YAML file under `scenarios/<tier>/`.

## Development

```bash
pytest -q          # 78 tests, ~0.3s
ruff check .       # clean
mypy llm_test/     # opt-in
```

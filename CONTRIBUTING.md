# Contributing

Thanks for your interest in improving Toolery! This guide covers local setup,
the test/lint workflow, and how to propose changes.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and install (incl. dev tools: pytest, ruff, mypy)
uv sync --extra dev

# Optional: perf integration (llama-benchy)
uv sync --extra dev --extra perf
```

The CLI entry point is `toolery` (see `README.md` for usage), and the TUI
opens with `uv run toolery tui`.

## Tests and linting

CI runs ruff + pytest on every push and pull request. Run both locally before
opening a PR:

```bash
uv run ruff check toolery/ tests/    # lint (must be clean)
uv run pytest -q                       # full test suite
```

Tests are deterministic and offline — there is no LLM judge and no network
dependency. New behavior should come with a test; we follow a test-first
workflow where practical.

## Adding scenarios

Benchmark scenarios live under `scenarios/<tier>/` as YAML files validated
against the `Scenario` model (`toolery/core/scenario.py`). Each `id` must be
unique and kebab-case. Run the suite after adding one — scenario loading is
covered by tests and a duplicate/invalid id will fail fast.

## Pull requests

1. Branch off `main`.
2. Keep commits focused and the message in the imperative mood
   (`feat(rankings): ...`, `fix(tui): ...`).
3. Make sure `ruff` is clean and `pytest` is green.
4. Describe the change and the reasoning in the PR body.

## Code style

- Match the surrounding code: comment density, naming, and idioms.
- Line length is 100 (ruff-enforced; `E501` is left to the formatter).
- Prefer clear, explicit code over cleverness.

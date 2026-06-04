# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-04

### Added
- History tab auto-refreshes: a 3s poll plus an on-show hook surface freshly
  completed runs without pressing Ctrl+R or restarting the app. The poll is a
  no-op unless the run set actually changed, and it preserves the active filter
  and the highlighted row across rebuilds.
- `--timeout-scale` option on `toolery run` (and the `Runner`): multiplies each
  scenario's `timeout_seconds`, so slow cloud/reasoning endpoints (high RTT +
  long chain-of-thought) aren't killed mid-answer. Also honored from run config.
- Cluster (DGX Spark topology) is now a first-class ranking axis: the same
  model+adapter on `single`/`dual`/`triple`/`quad`/`octa` shows as separate
  rows; re-runs of the same configuration still aggregate (decay-weighted mean
  + run-to-run stability).
- Scenarios tab now shows the task content (title, description, prompt) above
  the per-model results.
- GitHub project scaffolding: MIT `LICENSE`, CI workflow (ruff + pytest),
  `CONTRIBUTING.md`, and this changelog.

### Changed
- Overall ranking score is now tier-weighted only: every dimension counts
  equally (weight 1.0) when aggregating the Overall column, matching what the
  TUI legend and markdown reports already described. Per-dimension weighting is
  reserved for use-case persona columns, which keep their own weight maps.
- Renamed the project `LLM-test` → **Toolery**: Python package `toolery`,
  CLI command `toolery`, and environment variables `TOOLERY_*`
  (`TOOLERY_BASE_URL`, `TOOLERY_RESULTS_DIR`, …).
- Execution adapters trimmed to `raw` / `cloud` / `hermes` (removed the
  `claude_code` and `codex` CLI harnesses).
- Renamed the last TUI tab `Setup` → `Profiles` to match its content.

### Fixed
- Rankings table no longer resets horizontal scroll on the 5s poll: the
  render-skip signature ignores sub-display-precision decay drift, and scroll
  position is restored across rebuilds.
- `octa` (8-spark) runs now render and validate correctly across the rankings
  and launch flow.
- Corrected the "training control" label in the Home tab run-in-progress panel.

## [0.1.0]

- Initial benchmark: 4-tier scenario suite, deterministic scoring, Textual TUI,
  14-dimension ranking system, ASCII + PNG charts, and llama-benchy perf
  integration.

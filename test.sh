#!/usr/bin/env bash
# A/B: does enabling Hermes' skills help on debugging+coding?
# Two hermes arms over the same 27 scenarios, trials=5, against MiniMax-M2.7 @ 8888.
#   - skillsOFF: benchmark default (HERMES_SKILLS=0 → -t toolery_mock, --ignore-rules)
#   - skillsON : HERMES_SKILLS=1 → -t toolery_mock,skills, skills auto-injected
# Watch live in the TUI: run `toolery tui` and look at the Evaluation workspace
# (it polls runs.db every 2s, so it shows these runs as they execute).
#
# Perf is OFF (no --with-perf) on purpose — with_perf=true hard-hangs spark1.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/toolery
BASE_URL="${BASE_URL:-http://localhost:8888}"
TRIALS="${TRIALS:-5}"
CATEGORY="${CATEGORY:-debugging,coding}"
CONCURRENCY="${CONCURRENCY:-4}"

run_arm () {
  local tag="$1" skills="$2"
  echo "=== ARM: $tag (HERMES_SKILLS=$skills) ==="
  HERMES_SKILLS="$skills" "$PY" run \
    --model "MiniMax-M2.7-${tag}-t${TRIALS}" --served-model "MiniMax-M2.7" \
    --adapter hermes --category "$CATEGORY" --trials "$TRIALS" \
    --base-url "$BASE_URL" --no-tui --concurrency "$CONCURRENCY"
}

run_arm skillsOFF 0
run_arm skillsON  1
echo "=== A/B DONE — compare with: toolery correctness-report ==="

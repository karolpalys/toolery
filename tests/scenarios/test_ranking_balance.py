# tests/scenarios/test_ranking_balance.py
"""Asserts dimension-count balance after the W1-W4 backlog is in place.

Reference: docs/superpowers/specs/2026-05-26-ranking-balance-design.md
"""
from __future__ import annotations

import collections
from pathlib import Path

from toolery.core.scenario import load_all_scenarios

# Expected counts after debugging category landed (+13 scenarios).
# Baseline was the "Final balance" from docs/superpowers/specs/
# 2026-05-26-ranking-balance-design.md; debugging additions are documented
# in commit history (feat: debugging category).
EXPECTED_COUNTS = {
    "overall": 143,
    "agentic": 41,
    "coding": 27,
    "debugging": 13,
    "safety": 21,
    "adversarial_robustness": 10,
    "terminal": 14,
    "budget_efficiency": 18,
    "parameter_precision": 18,
    "restraint": 14,
    "hallucination": 17,
    "tool_selection": 13,
    "long_context": 13,
    "error_recovery": 14,
    "structured_output": 16,
    "context_state_tracking": 16,
    "instruction_following": 8,
    "localization": 13,
}

# Tier counts keep the original hand-designed balance (40/45/34/24), but as of
# the 2026-06-11 empirical re-tiering the *membership* of each tier is assigned
# by measured pass-rate across 3 models (MiniMax-M2.7-AWQ, Nex-N2-Pro-W4A16,
# Qwen3.6-35B): scenarios ranked easiest→hardest, then sliced into these quotas
# (quantile assignment). This preserves the balanced shape while making the
# difficulty ordering empirical rather than hand-guessed.
EXPECTED_TIER_COUNTS = {
    "easy": 40,
    "medium": 45,
    "hard": 34,
    "very_hard": 24,
}


def _aggregate():
    root = Path(__file__).resolve().parents[2] / "scenarios"
    scenarios = load_all_scenarios(root)
    by_dim: collections.Counter = collections.Counter()
    by_tier: collections.Counter = collections.Counter()
    for s in scenarios:
        for d in s.ranking_dimensions:
            by_dim[d] += 1
        by_tier[s.tier.value] += 1
    return by_dim, by_tier


def test_dimension_counts_match_balance_spec():
    by_dim, _ = _aggregate()
    diffs = {k: (by_dim[k], v) for k, v in EXPECTED_COUNTS.items() if by_dim[k] != v}
    assert not diffs, f"dimension count drift: {diffs}"


def test_tier_counts_match_balance_spec():
    _, by_tier = _aggregate()
    diffs = {k: (by_tier[k], v) for k, v in EXPECTED_TIER_COUNTS.items() if by_tier[k] != v}
    assert not diffs, f"tier count drift: {diffs}"

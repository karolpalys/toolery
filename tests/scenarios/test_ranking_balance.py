# tests/scenarios/test_ranking_balance.py
"""Asserts dimension-count balance after the W1-W4 backlog is in place.

Reference: docs/superpowers/specs/2026-05-26-ranking-balance-design.md
"""
from __future__ import annotations

import collections
from pathlib import Path

from llm_test.core.scenario import load_all_scenarios

# Expected counts after all 33 new scenarios land. Per spec "Final balance".
EXPECTED_COUNTS = {
    "overall": 116,
    "agentic": 39,
    "coding": 15,
    "safety": 15,
    "terminal": 14,
    "budget_efficiency": 18,
    "parameter_precision": 17,
    "restraint": 13,
    "hallucination": 13,
    "tool_selection": 13,
    "long_context": 13,
    "error_recovery": 13,
    "structured_output": 14,
    "context_state_tracking": 14,
    "localization": 13,
}

EXPECTED_TIER_COUNTS = {
    "easy": 32,
    "medium": 36,
    "hard": 26,
    "very_hard": 22,
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

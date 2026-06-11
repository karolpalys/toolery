"""Standard competition ranking for tied scores (the '1224' rule).

Two models with the same displayed score share a rank (both gold); the next
distinct score skips ahead (bronze, not silver).
"""
from __future__ import annotations

from toolery.rankings.compute import _assign_ranks


def _ranks(scores):
    return [r["rank"] for r in _assign_ranks([{"score": s} for s in scores])]


def test_tie_at_top_both_gold_next_is_bronze():
    # two at 100% → both #1; next is #3, not #2
    assert _ranks([1.0, 1.0, 0.8, 0.5]) == [1, 1, 3, 4]


def test_multiple_tie_groups():
    assert _ranks([1.0, 1.0, 0.8, 0.8, 0.5]) == [1, 1, 3, 3, 5]


def test_no_ties_is_plain_sequence():
    assert _ranks([0.9, 0.8, 0.7]) == [1, 2, 3]


def test_tie_judged_on_displayed_percent():
    # 1.0 and 0.99996 both render as 100.0% → must share rank
    assert _ranks([1.0, 0.99996, 0.90]) == [1, 1, 3]


def test_three_way_tie():
    assert _ranks([0.75, 0.75, 0.75, 0.60]) == [1, 1, 1, 4]

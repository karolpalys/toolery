"""Per-column podium (TUI rankings matrix) shares medals on ties."""
from toolery.tui.rankings_tab import _podium_with_ties


def _ranks(values, round_key=lambda v: round(v * 100, 1)):
    # values descending → list of (index, value); return {index: rank}
    scored = sorted(enumerate(values), key=lambda kv: kv[1], reverse=True)
    return _podium_with_ties(scored, round_key)


def test_two_at_100_both_gold_next_bronze():
    # 1.0, 1.0, 0.8 → ranks 1,1,3 (no silver)
    assert _ranks([1.0, 1.0, 0.8]) == {0: 1, 1: 1, 2: 3}


def test_no_ties_plain_podium():
    assert _ranks([0.9, 0.8, 0.7, 0.6]) == {0: 1, 1: 2, 2: 3}


def test_tie_on_displayed_percent():
    # 1.0 and 0.99996 both render 100.0% → share gold
    assert _ranks([1.0, 0.99996, 0.5]) == {0: 1, 1: 1, 2: 3}


def test_two_silver_no_bronze():
    # 1.0, 0.9, 0.9 → 1,2,2 ; nothing past rank 3 visible if 4th is lower
    assert _ranks([1.0, 0.9, 0.9, 0.5]) == {0: 1, 1: 2, 2: 2}


def test_rank_beyond_three_dropped():
    # 1.0,1.0,1.0,1.0 → all rank 1 (all gold); a lower 5th gets nothing
    assert _ranks([1.0, 1.0, 1.0, 1.0, 0.5]) == {0: 1, 1: 1, 2: 1, 3: 1}

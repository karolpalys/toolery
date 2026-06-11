"""display_name strips the historical tier prefix so the scenario name never
contradicts the (empirical) tier column."""
from toolery.core.scenario import display_name


def test_strips_each_tier_prefix():
    assert display_name("easy-39-if-negative-keyword") == "39-if-negative-keyword"
    assert display_name("hard-01-tdd-fix-loop") == "01-tdd-fix-loop"
    assert display_name("medium-22-db-describe-then-query") == "22-db-describe-then-query"
    assert display_name("very-hard-13-nonsense-calc") == "13-nonsense-calc"


def test_no_prefix_is_unchanged():
    assert display_name("39-if-negative-keyword") == "39-if-negative-keyword"


def test_only_strips_leading_prefix_once():
    # an inner "easy-" must survive; only the leading tier token is removed
    assert display_name("hard-easy-mode-toggle") == "easy-mode-toggle"

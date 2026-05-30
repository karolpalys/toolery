from toolery.charts.ascii import bar_per_tier, failure_taxonomy, heatmap


def test_bar_per_tier_renders_table():
    data = {
        "raw":    {"easy": 0.9, "medium": 0.72, "hard": 0.5, "very_hard": 0.25},
        "hermes": {"easy": 0.88, "medium": 0.68, "hard": 0.48, "very_hard": 0.22},
    }
    out = bar_per_tier(data)
    assert "raw" in out and "hermes" in out
    assert "easy" in out and "very_hard" in out
    assert "▇" in out or "█" in out


def test_heatmap_renders():
    out = heatmap({"raw": {"tool_selection": 0.9, "coding": 0.4}})
    assert "raw" in out and "tool_selection" in out


def test_failure_taxonomy():
    out = failure_taxonomy({"wrong_tool": 16, "budget_violated": 9, "forbidden_action": 8})
    assert "wrong_tool" in out
    assert "16" in out

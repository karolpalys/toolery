from toolery.charts.png import (
    category_heatmap,
    failure_taxonomy_png,
    overall_bar,
    pass_rate_vs_budget,
    perf_vs_quality,
    radar,
    tier_breakdown,
)


def test_overall_bar_writes_png(tmp_path):
    out = tmp_path / "overall.png"
    overall_bar({"raw": (0.62, 0.05), "hermes": (0.58, 0.06)}, out)
    assert out.exists() and out.stat().st_size > 0


def test_tier_breakdown(tmp_path):
    out = tmp_path / "tier.png"
    data = {"raw": {"easy": 0.9, "medium": 0.7, "hard": 0.5, "very_hard": 0.25}}
    tier_breakdown(data, out)
    assert out.exists()


def test_category_heatmap(tmp_path):
    out = tmp_path / "cat.png"
    data = {"raw": {"tool_selection": 0.9, "coding": 0.4, "safety": 0.7}}
    category_heatmap(data, out)
    assert out.exists()


def test_radar(tmp_path):
    out = tmp_path / "radar.png"
    cats = ["tool_selection", "coding", "safety", "long_context", "restraint"]
    radar({"raw": [0.9, 0.5, 0.7, 0.4, 0.6]}, cats, out)
    assert out.exists()


def test_failure_taxonomy_png(tmp_path):
    out = tmp_path / "fail.png"
    failure_taxonomy_png({"raw": {"wrong_tool": 5, "budget_violated": 3}}, out)
    assert out.exists()


def test_perf_vs_quality(tmp_path):
    out = tmp_path / "pq.png"
    perf_vs_quality({"deepseek": (38.0, 0.68), "mimo": (142.0, 0.71)}, out)
    assert out.exists()


def test_pass_rate_vs_budget(tmp_path):
    out = tmp_path / "pb.png"
    pass_rate_vs_budget({"easy-cod-01": [(2, 1.0), (3, 1.0), (4, 1.0)],
                         "hard-cod-03": [(6, 0.4), (7, 0.7), (8, 0.85)]}, out)
    assert out.exists()

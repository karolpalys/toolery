from llm_test.core.stats import bootstrap_ci, decay_weighted_mean, mcnemar_p


def test_bootstrap_ci_normal_case():
    samples = [1.0] * 80 + [0.0] * 20
    mean, lo, hi = bootstrap_ci(samples, iterations=500, seed=42, ci=0.95)
    assert abs(mean - 0.8) < 0.01
    assert 0.7 < lo < 0.8 < hi < 0.9


def test_bootstrap_ci_zero_variance():
    mean, lo, hi = bootstrap_ci([1.0] * 50, iterations=500, seed=42)
    assert mean == 1.0
    assert lo == 1.0 and hi == 1.0


def test_mcnemar_p_clear_diff():
    a = [1] * 90 + [0] * 10
    b = [0] * 50 + [1] * 50
    p = mcnemar_p(a, b)
    assert p < 0.05


def test_mcnemar_p_no_diff():
    same = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    p = mcnemar_p(same, same)
    assert p > 0.5


def test_decay_weighted_mean_recent_dominates():
    values_with_ages_days = [(0.9, 0), (0.5, 30), (0.5, 60)]
    weighted = decay_weighted_mean(values_with_ages_days, half_life_days=14)
    assert weighted > 0.75

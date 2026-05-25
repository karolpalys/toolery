from llm_test.rankings.compute import _DIM_WEIGHTS, _scenario_dim_weight


def test_dim_weights_constant_has_expected_keys():
    assert _DIM_WEIGHTS["coding"] == 2.0
    assert _DIM_WEIGHTS["terminal"] == 2.0
    assert _DIM_WEIGHTS["agentic"] == 2.0
    assert _DIM_WEIGHTS["localization"] == 0.5
    assert _DIM_WEIGHTS["long_context"] == 0.5


def test_scenario_dim_weight_unknown_dim_defaults_to_one():
    assert _scenario_dim_weight(["overall", "restraint"]) == 1.0


def test_scenario_dim_weight_picks_max():
    assert _scenario_dim_weight(["overall", "coding", "agentic"]) == 2.0


def test_scenario_dim_weight_mixed_high_and_low_picks_max():
    # max(coding=2.0, localization=0.5) = 2.0
    assert _scenario_dim_weight(["overall", "coding", "localization"]) == 2.0


def test_scenario_dim_weight_localization_only():
    assert _scenario_dim_weight(["overall", "localization"]) == 0.5


def test_scenario_dim_weight_empty_or_overall_only_defaults_to_one():
    assert _scenario_dim_weight([]) == 1.0
    assert _scenario_dim_weight(["overall"]) == 1.0

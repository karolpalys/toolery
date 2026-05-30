from llm_test.rankings.presets import USE_CASES, get_use_case

EXPECTED_DIMS = {
    "coding", "debugging", "terminal", "agentic", "safety",
    "adversarial_robustness", "restraint", "error_recovery",
    "parameter_precision", "context_state_tracking", "structured_output",
    "tool_selection", "instruction_following", "long_context", "localization",
    "budget_efficiency", "hallucination",
}


def test_seven_personas_defined():
    assert len(USE_CASES) == 7


def test_persona_keys_are_unique():
    keys = [uc.key for uc in USE_CASES]
    assert len(set(keys)) == len(keys)


def test_persona_keys_are_snake_case():
    for uc in USE_CASES:
        assert uc.key == uc.key.lower()
        assert " " not in uc.key
        assert "-" not in uc.key


def test_every_persona_has_all_17_dims():
    for uc in USE_CASES:
        assert set(uc.weights.keys()) == EXPECTED_DIMS, (
            f"{uc.key} missing dims: {EXPECTED_DIMS - set(uc.weights.keys())}, "
            f"extra: {set(uc.weights.keys()) - EXPECTED_DIMS}"
        )


def test_every_weight_is_positive_float():
    for uc in USE_CASES:
        for dim, w in uc.weights.items():
            assert isinstance(w, float)
            assert w > 0.0, f"{uc.key}.{dim} = {w}"


def test_get_use_case_returns_match():
    uc = get_use_case("coding_assistant")
    assert uc is not None
    assert uc.key == "coding_assistant"
    assert uc.weights["coding"] == 3.0


def test_get_use_case_returns_none_for_unknown():
    assert get_use_case("nonexistent") is None


def test_persona_has_description():
    for uc in USE_CASES:
        assert isinstance(uc.description, str)
        assert len(uc.description) > 10

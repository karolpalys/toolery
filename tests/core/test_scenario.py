import pytest

from llm_test.core.scenario import DuplicateIdError, load_all_scenarios, load_scenario


def test_load_single_scenario(fixtures_dir):
    s = load_scenario(fixtures_dir / "scenarios" / "valid_easy.yaml")
    assert s.id == "easy-01-direct-weather"
    assert s.tier.value == "easy"
    assert len(s.scoring.required) == 2
    assert s.budget.max_tool_calls == 1


def test_load_all_scenarios_dedups_by_id(fixtures_dir, tmp_path):
    src = fixtures_dir / "scenarios" / "valid_easy.yaml"
    (tmp_path / "a.yaml").write_text(src.read_text())
    (tmp_path / "b.yaml").write_text(src.read_text())  # same id → duplicate
    with pytest.raises(DuplicateIdError):
        load_all_scenarios(tmp_path)

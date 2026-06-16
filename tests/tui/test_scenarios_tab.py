from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Select

from toolery.core.models import Category
from toolery.core.scenario import load_all_scenarios
from toolery.core.store import Store
from toolery.tui import scenarios_tab as st
from toolery.tui.scenarios_tab import ScenariosTab


def _tab_with_scenarios() -> ScenariosTab:
    """A tab instance with scenarios loaded but NOT mounted (no Textual app)."""
    tab = ScenariosTab.__new__(ScenariosTab)
    xs = load_all_scenarios(Path("scenarios"))
    tab._scenarios = {s.id: s for s in xs}
    return tab


def test_difficulty_options_are_english_and_complete():
    opts = st._difficulty_options()
    values = [v for _label, v in opts]
    assert values == ["all", "easy", "medium", "hard", "very_hard"]
    labels = {v: label for label, v in opts}
    assert labels["all"] == "All difficulties"
    assert labels["very_hard"] == "Very hard"
    assert labels["easy"] == "Easy"


def test_category_options_cover_every_category_in_english():
    opts = st._category_options()
    values = [v for _label, v in opts]
    assert values[0] == "all"
    # Every Category enum value is offered exactly once.
    assert set(values[1:]) == {c.value for c in Category}
    assert len(values) == len(Category) + 1
    labels = {v: label for label, v in opts}
    assert labels["all"] == "All categories"
    assert labels["tool_selection"] == "Tool selection"
    assert labels["adversarial_robustness"] == "Adversarial robustness"


def test_filter_narrows_to_current_tier():
    tab = _tab_with_scenarios()
    rows = tab._filtered_sorted_scenarios("very_hard", "all")
    assert rows, "expected some very_hard scenarios"
    assert all(s.tier.value == "very_hard" for s in rows)
    expected = sum(1 for s in tab._scenarios.values() if s.tier.value == "very_hard")
    assert len(rows) == expected


def test_filter_combines_difficulty_and_category():
    tab = _tab_with_scenarios()
    rows = tab._filtered_sorted_scenarios("hard", "coding")
    for s in rows:
        assert s.tier.value == "hard"
        assert s.category.value == "coding"
    expected = sum(
        1 for s in tab._scenarios.values()
        if s.tier.value == "hard" and s.category.value == "coding"
    )
    assert len(rows) == expected


def test_unfiltered_list_sorted_by_current_tier():
    tab = _tab_with_scenarios()
    rows = tab._filtered_sorted_scenarios("all", "all")
    assert len(rows) == len(tab._scenarios)
    order = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
    ranks = [order[s.tier.value] for s in rows]
    assert ranks == sorted(ranks), "scenarios must be grouped by current tier"


def test_task_text_uses_live_tier_not_id_prefix():
    """easy-03-refuse-trivial-math was re-tiered to very_hard. The Task panel
    must show the live tier + category + real tags, never the stale 'easy'."""
    tab = _tab_with_scenarios()
    sid = "easy-03-refuse-trivial-math"
    assert sid in tab._scenarios, "fixture scenario missing"
    sc = tab._scenarios[sid]
    assert sc.tier.value == "very_hard", "test premise: this scenario is re-tiered"
    text = tab._task_text(sid).plain
    assert "Very hard" in text
    assert "Difficulty:" in text and "Category:" in text and "Tags:" in text
    # Live category surfaced.
    assert st._humanize(sc.category.value) in text
    # At least one real tag rendered.
    if sc.tags:
        assert sc.tags[0] in text


class _Host(App):
    def compose(self):
        yield ScenariosTab(id="sc")


@pytest.mark.asyncio
async def test_difficulty_filter_rebuilds_picker_live(tmp_path, monkeypatch):
    """Changing the difficulty Select narrows the picker to that tier in a
    running app (Changed → _apply_filters → picker repopulated + reselected)."""
    monkeypatch.setenv("TOOLERY_RESULTS_DIR", str(tmp_path))
    Store(tmp_path / "runs.db").init_schema()
    app = _Host()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        tab = app.query_one(ScenariosTab)
        app.query_one("#sc-filter-difficulty", Select).value = "very_hard"
        await pilot.pause()
        pick = app.query_one("#sc-pick", Select)
        assert pick.value not in (None, Select.BLANK)
        assert tab._scenarios[pick.value].tier.value == "very_hard"

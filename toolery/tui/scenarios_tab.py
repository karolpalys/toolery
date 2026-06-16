from __future__ import annotations

import os
import statistics
from collections import defaultdict
from pathlib import Path

from rich.text import Text
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Select, Static

from toolery.core.models import Category, Scenario, Tier
from toolery.core.scenario import display_name, load_all_scenarios
from toolery.core.store import Store

# Difficulty ordering for the picker — empirical tier is the single source of
# truth (scenario id prefixes are stale after re-tiering, so they are ignored).
_TIER_ORDER = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}


def _humanize(value: str) -> str:
    """snake_case enum value → English label, e.g. 'very_hard' → 'Very hard'."""
    return value.replace("_", " ").capitalize()


def _difficulty_options() -> list[tuple[str, str]]:
    """(label, value) pairs for the difficulty filter, 'all' first."""
    return [("All difficulties", "all")] + [(_humanize(t.value), t.value) for t in Tier]


def _category_options() -> list[tuple[str, str]]:
    """(label, value) pairs for the category filter, 'all' first."""
    return [("All categories", "all")] + [(_humanize(c.value), c.value) for c in Category]


class ScenariosTab(Container):
    DEFAULT_CSS = """
    ScenariosTab {
        layout: vertical;
        padding: 1 2;
        background: $surface;
    }

    ScenariosTab #scenario-browser {
        height: 1fr;
        border: round $primary;
        border-title-color: $primary;
        background: $surface;
        padding: 0 1;
    }

    ScenariosTab #sc-heading {
        height: 1;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    ScenariosTab #sc-filters {
        height: auto;
        margin-bottom: 1;
    }

    ScenariosTab #sc-filters Select {
        width: 1fr;
        margin-right: 1;
    }

    ScenariosTab #sc-pick {
        margin-bottom: 1;
    }

    ScenariosTab #sc-task {
        height: auto;
        max-height: 14;
        border: round $secondary;
        border-title-color: $secondary;
        padding: 0 1;
        margin-bottom: 1;
    }

    ScenariosTab #sc-table {
        height: 1fr;
    }

    ScenariosTab #sc-stats {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scenarios: dict[str, Scenario] = {}

    def compose(self):
        with Vertical(id="scenario-browser"):
            yield Static("Per-scenario cross-model view", id="sc-heading")
            with Horizontal(id="sc-filters"):
                yield Select(
                    _difficulty_options(), id="sc-filter-difficulty",
                    value="all", allow_blank=False,
                )
                yield Select(
                    _category_options(), id="sc-filter-category",
                    value="all", allow_blank=False,
                )
            yield Select(options=[], id="sc-pick")
            with VerticalScroll(id="sc-task"):
                yield Static("", id="sc-task-body")
            yield DataTable(id="sc-table")
            yield Static("", id="sc-stats")

    def on_mount(self) -> None:
        try:
            self.query_one("#scenario-browser").border_title = "Scenario browser"
            self.query_one("#sc-task").border_title = "Task"
        except Exception:
            pass
        try:
            scenarios = load_all_scenarios(Path("scenarios"))
        except FileNotFoundError:
            scenarios = []
        self._scenarios = {s.id: s for s in scenarios}
        # Build the picker from the (default 'all'/'all') filters — also selects
        # the first scenario and renders it.
        self._apply_filters()

    def on_select_changed(self, event) -> None:
        sid = event.select.id
        if sid in ("sc-filter-difficulty", "sc-filter-category"):
            self._apply_filters()
        elif sid == "sc-pick" and isinstance(event.value, str):
            # set_options() momentarily resets the picker to its blank sentinel
            # and posts a Changed for it; only real (str) ids should render.
            self._render_scenario(event.value)

    # -- filtering --
    def _filtered_sorted_scenarios(self, difficulty: str, category: str) -> list[Scenario]:
        """Scenarios matching both filters ('all' = no constraint), sorted by
        current tier (easy→very_hard) then display name. Tier is the live,
        re-tiered value — the stale id prefix is never consulted."""
        rows = list(self._scenarios.values())
        if difficulty != "all":
            rows = [s for s in rows if s.tier.value == difficulty]
        if category != "all":
            rows = [s for s in rows if s.category.value == category]
        rows.sort(key=lambda s: (_TIER_ORDER.get(s.tier.value, 99), display_name(s.id)))
        return rows

    def _apply_filters(self) -> None:
        diff = self.query_one("#sc-filter-difficulty", Select).value
        cat = self.query_one("#sc-filter-category", Select).value
        rows = self._filtered_sorted_scenarios(diff, cat)
        pick = self.query_one("#sc-pick", Select)
        prev = pick.value if isinstance(pick.value, str) else None
        options = [(f"{s.tier.value} · {display_name(s.id)}", s.id) for s in rows]
        pick.set_options(options)
        if not options:
            self._render_empty()
            return
        ids = [s.id for s in rows]
        chosen = prev if prev in ids else options[0][1]
        pick.value = chosen
        self._render_scenario(chosen)

    def _render_empty(self) -> None:
        self.query_one("#sc-task-body", Static).update(
            Text("No scenarios match the current filters.", style="dim italic")
        )
        tbl = self.query_one("#sc-table", DataTable)
        tbl.clear(columns=True)
        self.query_one("#sc-stats", Static).update("")

    # -- rendering --
    def _task_text(self, scenario_id: str) -> Text:
        """Build the Task panel renderable: live difficulty/category/tags
        metadata, then the scenario's title/description/prompts."""
        sc = self._scenarios.get(scenario_id)
        body = Text()
        if sc is None:
            body.append("Scenario definition not found.", style="dim")
            return body
        body.append("Difficulty: ", style="bold")
        body.append(_humanize(sc.tier.value), style="bold yellow")
        body.append("      Category: ", style="bold")
        body.append(_humanize(sc.category.value), style="bold cyan")
        body.append("\nTags: ", style="bold")
        body.append(" · ".join(sc.tags) if sc.tags else "—", style="dim")
        body.append("\n\n")
        body.append(f"{sc.title}\n", style="bold")
        if sc.description:
            body.append(f"{sc.description}\n", style="dim")
        body.append("\n")
        if sc.system_prompt:
            body.append("System prompt\n", style="bold cyan")
            body.append(f"{sc.system_prompt}\n\n")
        body.append("Prompt\n", style="bold cyan")
        body.append(sc.prompt)
        return body

    def _render_task(self, scenario_id: str) -> None:
        self.query_one("#sc-task-body", Static).update(self._task_text(scenario_id))

    def _render_scenario(self, scenario_id: str) -> None:
        self._render_task(scenario_id)
        tbl = self.query_one("#sc-table", DataTable)
        tbl.clear(columns=True)
        tbl.add_columns("#", "Model", "Adapter", "n", "pass", "med calls",
                        "med ms", "top failure")
        results_dir = Path(os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
        store = Store(results_dir / "runs.db")
        store.init_schema()
        runs_meta = {r["run_id"]: r for r in store.fetch_all_runs()}
        with store.conn() as c:
            res = c.execute(
                "SELECT * FROM scenario_results WHERE scenario_id=?", (scenario_id,)
            ).fetchall()
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in res:
            r = dict(r)
            model = runs_meta.get(r["run_id"], {}).get("model", "?")
            grouped[(model, r["adapter"])].append(r)
        for i, ((model, adapter), rs) in enumerate(grouped.items(), start=1):
            n = len(rs)
            pass_n = sum(1 for x in rs if x["status"] == "pass")
            med_calls = int(statistics.median(x["call_count"] for x in rs))
            med_lat = int(statistics.median(x["latency_ms"] for x in rs))
            kinds = [x["failure_kind"] for x in rs if x["failure_kind"]]
            top = max(set(kinds), key=kinds.count) if kinds else ""
            if pass_n == n:
                pass_cell = f"✅ {pass_n}/{n}"
            elif pass_n == 0:
                pass_cell = f"❌ {pass_n}/{n}"
            else:
                pass_cell = f"⚠ {pass_n}/{n}"
            tbl.add_row(str(i), model, adapter, str(n), pass_cell,
                        str(med_calls), str(med_lat), top)
        self.query_one("#sc-stats", Static).update(
            f"Showing data for {scenario_id} across {len(grouped)} model×adapter combinations."
        )

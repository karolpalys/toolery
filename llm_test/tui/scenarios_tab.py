from __future__ import annotations

import os
import statistics
from collections import defaultdict
from pathlib import Path

from rich.text import Text
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import DataTable, Select, Static

from llm_test.core.models import Scenario
from llm_test.core.scenario import load_all_scenarios
from llm_test.core.store import Store


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
        sel = self.query_one("#sc-pick", Select)
        try:
            scenarios = load_all_scenarios(Path("scenarios"))
        except FileNotFoundError:
            scenarios = []
        self._scenarios = {s.id: s for s in scenarios}
        options = [(f"{s.tier.value} · {s.id}", s.id) for s in scenarios]
        sel.set_options(options)
        if options:
            sel.value = options[0][1]
            self._render_scenario(options[0][1])

    def on_select_changed(self, event) -> None:
        if event.select.id == "sc-pick" and event.value:
            self._render_scenario(event.value)

    def _render_task(self, scenario_id: str) -> None:
        sc = self._scenarios.get(scenario_id)
        body = Text()
        if sc is None:
            body.append("Scenario definition not found.", style="dim")
        else:
            body.append(f"{sc.title}\n", style="bold")
            if sc.description:
                body.append(f"{sc.description}\n", style="dim")
            body.append("\n")
            if sc.system_prompt:
                body.append("System prompt\n", style="bold cyan")
                body.append(f"{sc.system_prompt}\n\n")
            body.append("Prompt\n", style="bold cyan")
            body.append(sc.prompt)
        self.query_one("#sc-task-body", Static).update(body)

    def _render_scenario(self, scenario_id: str) -> None:
        self._render_task(scenario_id)
        tbl = self.query_one("#sc-table", DataTable)
        tbl.clear(columns=True)
        tbl.add_columns("#", "Model", "Adapter", "n", "pass", "med calls",
                        "med ms", "top failure")
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
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

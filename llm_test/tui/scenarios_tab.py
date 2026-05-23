from __future__ import annotations

import os
import statistics
from collections import defaultdict
from pathlib import Path

from textual.containers import Container, Vertical
from textual.widgets import DataTable, Select, Static

from llm_test.core.scenario import load_all_scenarios
from llm_test.core.store import Store


class ScenariosTab(Container):
    DEFAULT_CSS = """ScenariosTab { padding: 1; }"""

    def compose(self):
        with Vertical():
            yield Static("[bold]Per-scenario cross-model view[/bold]")
            yield Select(options=[], id="sc-pick")
            yield DataTable(id="sc-table")
            yield Static("", id="sc-stats")

    def on_mount(self) -> None:
        sel = self.query_one("#sc-pick", Select)
        try:
            scenarios = load_all_scenarios(Path("scenarios"))
        except FileNotFoundError:
            scenarios = []
        options = [(f"{s.tier.value} · {s.id}", s.id) for s in scenarios]
        sel.set_options(options)
        if options:
            sel.value = options[0][1]
            self._render(options[0][1])

    def on_select_changed(self, event) -> None:
        if event.select.id == "sc-pick" and event.value:
            self._render(event.value)

    def _render(self, scenario_id: str) -> None:
        tbl = self.query_one("#sc-table", DataTable)
        tbl.clear(columns=True)
        tbl.add_columns("model", "adapter", "n", "pass", "median_calls",
                        "median_latency", "top_failure")
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
        for (model, adapter), rs in grouped.items():
            n = len(rs)
            pass_n = sum(1 for x in rs if x["status"] == "pass")
            med_calls = int(statistics.median(x["call_count"] for x in rs))
            med_lat = int(statistics.median(x["latency_ms"] for x in rs))
            kinds = [x["failure_kind"] for x in rs if x["failure_kind"]]
            top = max(set(kinds), key=kinds.count) if kinds else ""
            tbl.add_row(model, adapter, str(n), f"{pass_n}/{n}",
                        str(med_calls), str(med_lat), top)
        self.query_one("#sc-stats", Static).update(
            f"Showing data for {scenario_id} across {len(grouped)} model×adapter combinations."
        )

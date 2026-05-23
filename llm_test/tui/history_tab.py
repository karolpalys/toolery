from __future__ import annotations

import os
from pathlib import Path

from textual.containers import Container, Vertical
from textual.widgets import DataTable, Input, Static

from llm_test.core.store import Store


class HistoryTab(Container):
    DEFAULT_CSS = """
    HistoryTab { padding: 1; }
    """

    def compose(self):
        with Vertical():
            yield Static("[bold]All runs[/bold] — ↑↓ navigate · [enter] details · [d] diff · [del] remove")
            yield Input(placeholder="filter (model name or run_id substring)", id="hist-filter")
            yield DataTable(id="history-table")

    def on_mount(self) -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.add_columns("run_id", "model", "status", "started_at", "duration (s)", "adapters")
        self.refresh_data()

    def refresh_data(self, filter_str: str = "") -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.clear()
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        store = Store(results_dir / "runs.db")
        store.init_schema()
        for r in store.fetch_all_runs():
            if filter_str and filter_str.lower() not in (r["run_id"] + r["model"]).lower():
                continue
            with store.conn() as c:
                adapters = [row["adapter"] for row in
                            c.execute("SELECT adapter FROM adapters_in_run WHERE run_id=?",
                                      (r["run_id"],)).fetchall()]
            tbl.add_row(r["run_id"], r["model"], r["status"] or "",
                        r["started_at"] or "", f"{r['duration_s'] or 0:.1f}",
                        ",".join(adapters))

    def on_input_changed(self, event) -> None:
        if event.input.id == "hist-filter":
            self.refresh_data(event.value)

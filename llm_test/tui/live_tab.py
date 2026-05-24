from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Static

from llm_test.core.store import Store


class LiveTab(Container):
    """Live dashboard, DB-polled every 2 seconds."""

    DEFAULT_CSS = """
    LiveTab { layout: grid; grid-size: 2 2; grid-gutter: 1; }
    .scenarios-panel { row-span: 1; }
    .summary-panel { row-span: 1; }
    """

    def __init__(self, id: str | None = None, store: Store | None = None) -> None:
        super().__init__(id=id)
        self._store: Store | None = store

    def compose(self) -> ComposeResult:
        with Vertical(classes="summary-panel"):
            yield Static("[bold]Current run[/bold]")
            yield Static("(no active run)", id="current-run")
        with Vertical(classes="scenarios-panel"):
            yield Static("[bold]Scenarios[/bold]")
            yield DataTable(id="scenarios-table")

    def on_mount(self) -> None:
        tbl = self.query_one("#scenarios-table", DataTable)
        tbl.add_columns("Scenario", "Adapter", "Score", "Status")
        self.refresh_from_db()
        self.set_interval(2.0, self.refresh_from_db)

    def _resolve_store(self) -> Store | None:
        if self._store is not None:
            return self._store
        try:
            results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
            db = results_dir / "runs.db"
            if not db.exists():
                return None
            store = Store(db)
            store.init_schema()
            self._store = store
            return store
        except Exception:
            return None

    def refresh_from_db(self) -> None:
        store = self._resolve_store()
        if store is None:
            return
        runs = store.fetch_all_runs()
        current = next((r for r in runs if r.get("status") == "running"), None)
        if current is None and runs:
            current = runs[0]
        status = self.query_one("#current-run", Static)
        if current is None:
            status.update("(no runs yet)")
            return
        status.update(
            f"{current['run_id']}  -  model: {current['model']}  "
            f"-  status: {current.get('status', '?')}"
        )
        results = store.fetch_results_for_run(current["run_id"])
        tbl = self.query_one("#scenarios-table", DataTable)
        tbl.clear()
        for r in results:
            tbl.add_row(
                r["scenario_id"],
                r["adapter"],
                f"{r['score']:.2f}",
                r.get("status", ""),
            )

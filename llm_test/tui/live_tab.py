from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Static


class LiveTab(Container):
    """Live dashboard for the currently-running run."""

    DEFAULT_CSS = """
    LiveTab { layout: grid; grid-size: 2 2; grid-gutter: 1; }
    .scenarios-panel { row-span: 1; }
    .trace-panel { row-span: 1; }
    .scores-panel { row-span: 1; }
    .failures-panel { row-span: 1; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="scenarios-panel"):
            yield Static("[bold]Scenarios[/bold]")
            yield DataTable(id="scenarios-table")
        with Vertical(classes="trace-panel"):
            yield Static("[bold]Live trace[/bold]")
            yield Static("(no active scenario)", id="trace-content")
        with Vertical(classes="scores-panel"):
            yield Static("[bold]Score per adapter[/bold]")
            yield Static("(waiting)", id="scores-content")
        with Vertical(classes="failures-panel"):
            yield Static("[bold]Recent failures[/bold]")
            yield Static("(none)", id="failures-content")

    def on_mount(self) -> None:
        tbl = self.query_one("#scenarios-table", DataTable)
        tbl.add_columns("Status", "Scenario", "Score")

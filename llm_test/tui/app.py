from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from llm_test.tui.history_tab import HistoryTab
from llm_test.tui.live_tab import LiveTab
from llm_test.tui.rankings_tab import RankingsTab
from llm_test.tui.scenarios_tab import ScenariosTab


class LLMTestApp(App):
    CSS = """
    Screen { background: $surface; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+r", "refresh", "Refresh")]

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self.run_id = run_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Live", id="live"):
                yield LiveTab(id="live-tab")
            with TabPane("History", id="history"):
                yield HistoryTab(id="history-tab")
            with TabPane("Rankings", id="rankings"):
                yield RankingsTab(id="rankings-tab")
            with TabPane("Scenarios", id="scenarios"):
                yield ScenariosTab(id="scenarios-tab")
        yield Footer()

    def action_refresh(self) -> None:
        self.notify("Refreshing...", timeout=1)

from textual.containers import Container
from textual.widgets import Static


class RankingsTab(Container):
    def compose(self):
        yield Static("Rankings tab — implemented in Phase 23")

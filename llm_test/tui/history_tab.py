from textual.containers import Container
from textual.widgets import Static


class HistoryTab(Container):
    def compose(self):
        yield Static("History tab — implemented in Phase 22")

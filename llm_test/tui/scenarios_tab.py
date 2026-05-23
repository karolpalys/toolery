from textual.containers import Container
from textual.widgets import Static


class ScenariosTab(Container):
    def compose(self):
        yield Static("Scenarios tab — implemented in Phase 24")

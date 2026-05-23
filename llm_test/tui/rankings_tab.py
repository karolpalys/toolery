from __future__ import annotations

import os
from pathlib import Path

from textual.containers import Container, Vertical
from textual.widgets import Markdown, Select, Static

_DIMENSIONS = ["overall", "coding", "agentic", "safety", "restraint",
               "long_context", "budget_efficiency", "speed"]


class RankingsTab(Container):
    DEFAULT_CSS = """RankingsTab { padding: 1; }"""

    def compose(self):
        with Vertical():
            yield Static("[bold]Rankings[/bold]   choose dimension:")
            yield Select(options=[(d, d) for d in _DIMENSIONS], value="overall", id="rank-dim")
            yield Markdown("", id="rank-content")

    def on_mount(self) -> None:
        self._reload("overall")

    def on_select_changed(self, event) -> None:
        if event.select.id == "rank-dim":
            self._reload(event.value)

    def _reload(self, dim: str) -> None:
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        path = results_dir / "rankings" / f"{dim}.md"
        md = self.query_one("#rank-content", Markdown)
        if path.exists():
            md.update(path.read_text())
        else:
            md.update(f"*No ranking yet for `{dim}` — run `llm-test rankings --regen`.*")

from __future__ import annotations

import json
import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Button, RadioButton, RadioSet, Static

from llm_test.rankings.presets import USE_CASES


def _persona_blurb(weights: dict[str, float]) -> str:
    """Return a 'top-3 ↑ / bottom-2 ↓' summary of a persona's weights."""
    sorted_high = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    sorted_low = sorted(weights.items(), key=lambda kv: kv[1])
    top3 = ", ".join(f"{d}({w:.1f})" for d, w in sorted_high[:3])
    bot2 = ", ".join(f"{d}({w:.1f})" for d, w in sorted_low[:2])
    return f"  [green]↑[/green] {top3}\n  [red]↓[/red] {bot2}"


class SetupTab(Container):
    """Pick a use-case persona to drive an additional ranking column."""

    DEFAULT_CSS = """
    SetupTab { padding: 1; }
    SetupTab #setup-header { text-style: bold; margin-bottom: 1; }
    SetupTab #setup-radio { margin-bottom: 1; }
    SetupTab .persona-desc { margin-bottom: 1; padding-left: 2; }
    SetupTab #setup-buttons { margin-top: 1; }
    SetupTab #setup-status { margin-top: 1; }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._results_dir = Path(
            os.environ.get("LLM_TEST_RESULTS_DIR", "./results")
        )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "[bold]Pick the use-case you're evaluating for.[/bold]\n"
                "The chosen profile creates an extra ranking column "
                "'UC:<Name>' in Rankings (next to Overall). "
                "The general Overall column is unaffected.",
                id="setup-header",
            )
            with VerticalScroll():
                active = self._read_active_use_case()
                with RadioSet(id="setup-radio"):
                    yield RadioButton(
                        "None — only general Overall",
                        id="uc-none",
                        value=(active is None),
                    )
                    for uc in USE_CASES:
                        is_active = (active == uc.key)
                        yield RadioButton(
                            f"{uc.name}", id=f"uc-{uc.key}", value=is_active,
                        )
                        yield Static(
                            f"  [dim]{uc.description}[/dim]\n"
                            f"{_persona_blurb(uc.weights)}",
                            classes="persona-desc",
                        )
            with Container(id="setup-buttons"):
                yield Button("Apply", id="apply", variant="success")
                yield Button("Clear (none)", id="clear", variant="error")
            yield Static(self._status_text(), id="setup-status")

    def _read_active_use_case(self) -> str | None:
        """Read current setup.json. Returns key or None."""
        setup_path = self._results_dir / "setup.json"
        if not setup_path.exists():
            return None
        try:
            data = json.loads(setup_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return data.get("active_use_case")

    def _status_text(self) -> str:
        active = self._read_active_use_case()
        if active is None:
            return "[dim]Active: none (general Overall only)[/dim]"
        return f"[dim]Active: {active}[/dim]"

    def _save_active_use_case(self, key: str | None) -> None:
        """Persist or clear the active use-case in setup.json."""
        setup_path = self._results_dir / "setup.json"
        if key is None:
            setup_path.unlink(missing_ok=True)
            return
        self._results_dir.mkdir(parents=True, exist_ok=True)
        setup_path.write_text(
            json.dumps({"version": 1, "active_use_case": key}, indent=2)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            radio = self.query_one("#setup-radio", RadioSet)
            pressed = radio.pressed_button
            if pressed is None:
                self.app.notify("Pick a use-case first", severity="warning")
                return
            radio_id = pressed.id or ""
            key = radio_id.removeprefix("uc-")
            if key == "none":
                self._save_active_use_case(None)
                self.app.notify("Use-case cleared")
            else:
                self._save_active_use_case(key)
                self.app.notify(f"Use-case '{key}' applied — regenerating rankings...")
            self._regenerate_rankings()
            self._refresh_status_and_focus()
        elif event.button.id == "clear":
            self._save_active_use_case(None)
            self.app.notify("Use-case cleared")
            self._regenerate_rankings()
            self._refresh_status_and_focus()

    def _regenerate_rankings(self) -> None:
        """Call regenerate_rankings using the current setup.json state."""
        from llm_test.core.store import Store
        from llm_test.rankings.compute import (
            load_active_use_case, regenerate_rankings,
        )
        db = self._results_dir / "runs.db"
        if not db.exists():
            return
        store = Store(db)
        store.init_schema()
        uc_key, uc_weights = load_active_use_case(self._results_dir)
        dims = [
            "overall", "coding", "agentic", "safety", "restraint",
            "long_context", "budget_efficiency", "hallucination",
            "error_recovery", "parameter_precision",
            "context_state_tracking", "structured_output",
            "tool_selection", "localization", "terminal",
        ]
        try:
            regenerate_rankings(
                store=store, dimensions=dims,
                out_dir=self._results_dir / "rankings",
                use_case_weights=uc_weights, use_case_key=uc_key,
            )
        except Exception as e:
            self.app.notify(f"Regen failed: {e}", severity="error")

    def _refresh_status_and_focus(self) -> None:
        """Update status line + switch focus to Rankings tab + force refresh."""
        self.query_one("#setup-status", Static).update(self._status_text())
        # Switch to Rankings tab if available.
        from textual.widgets import TabbedContent
        try:
            tabs = self.app.query_one(TabbedContent)
            tabs.active = "rankings"
        except Exception:
            pass
        # Force reload of Rankings tab data.
        try:
            from llm_test.tui.rankings_tab import RankingsTab
            rt = self.app.query_one(RankingsTab)
            if hasattr(rt, "reload"):
                rt.reload()
        except Exception:
            pass

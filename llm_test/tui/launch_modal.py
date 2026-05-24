from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, RadioButton, RadioSet, Static

from llm_test.core.adapter_probe import AdapterStatus
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.core.runner_subprocess import RunArgs

_ADAPTER_LABELS = {
    "raw": "raw                - direct OpenAI port",
    "hermes": "hermes             - CLI subprocess",
    "claude_code": "claude_code        - Claude Code CLI",
    "codex": "codex              - Codex CLI",
}


class LaunchModal(ModalScreen[RunArgs | None]):
    """Form modal: pre-filled flags + harness picker. dismiss() returns RunArgs or None."""

    DEFAULT_CSS = """
    LaunchModal { align: center middle; }
    LaunchModal > Vertical { width: 70; padding: 1 2; background: $surface; border: thick $primary; }
    LaunchModal .row { height: auto; margin-top: 1; }
    LaunchModal Input { width: 12; }
    LaunchModal Button { margin-left: 2; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, endpoint: EndpointInfo, adapters: dict[str, AdapterStatus]) -> None:
        super().__init__()
        self._endpoint = endpoint
        self._adapters = adapters

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Launch test[/b]")
            yield Static(f"Model:      {self._endpoint.model_id}")
            yield Static(f"Endpoint:   {self._endpoint.base_url}")

            yield Label("Tier:")
            with RadioSet(id="tier"):
                for value in ("easy", "medium", "hard", "very_hard", "all"):
                    btn = RadioButton(value, id=f"tier-{value}")
                    if value == "all":
                        btn.value = True
                    yield btn

            with Horizontal(classes="row"):
                yield Label("Trials: ")
                yield Input(value="5", id="trials")
                yield Label("  Concurrency: ")
                yield Input(value="4", id="concurrency")

            yield Label("Harness:")
            with RadioSet(id="adapter"):
                for name in ("raw", "hermes", "claude_code", "codex"):
                    status = self._adapters[name]
                    text = _ADAPTER_LABELS[name]
                    if not status.available:
                        text = f"{text}  - disabled ({status.reason})"
                    btn = RadioButton(text, id=f"adapter-{name}", disabled=not status.available)
                    if name == "raw":
                        btn.value = True
                    yield btn

            yield Checkbox("Collect perf (llama-benchy)", id="with_perf")

            with Horizontal(classes="row"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Run", id="run", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "run":
            args = self._build_args()
            if args is not None:
                self.dismiss(args)

    def _selected_tier(self) -> str:
        rs = self.query_one("#tier", RadioSet)
        if rs.pressed_button is None:
            return "all"
        return rs.pressed_button.id.removeprefix("tier-")

    def _selected_adapter(self) -> str:
        rs = self.query_one("#adapter", RadioSet)
        if rs.pressed_button is None:
            return "raw"
        return rs.pressed_button.id.removeprefix("adapter-")

    def _build_args(self) -> RunArgs | None:
        try:
            trials = int(self.query_one("#trials", Input).value)
            concurrency = int(self.query_one("#concurrency", Input).value)
        except ValueError:
            self.app.notify("trials/concurrency must be integers", severity="error")
            return None
        if not (1 <= trials <= 100 and 1 <= concurrency <= 64):
            self.app.notify("trials must be 1-100, concurrency 1-64", severity="error")
            return None
        try:
            return RunArgs(
                model=self._endpoint.model_id,
                base_url=self._endpoint.base_url,
                adapter=self._selected_adapter(),
                tier=self._selected_tier(),
                trials=trials,
                concurrency=concurrency,
                with_perf=self.query_one("#with_perf", Checkbox).value,
            )
        except Exception as e:
            self.app.notify(f"Invalid args: {e}", severity="error")
            return None

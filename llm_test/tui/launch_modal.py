from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from llm_test.core.adapter_probe import AdapterStatus
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.core.models import Category
from llm_test.core.runner_subprocess import RunArgs

_ADAPTER_LABELS = {
    "raw": "raw                - direct OpenAI port",
    "hermes": "hermes             - CLI subprocess",
    "claude_code": "claude_code        - Claude Code CLI",
}


class LaunchModal(ModalScreen[RunArgs | None]):
    """Form modal: pre-filled flags + harness picker. dismiss() returns RunArgs or None."""

    DEFAULT_CSS = """
    LaunchModal { align: center middle; }
    LaunchModal > VerticalScroll { width: 100; max-height: 90%; padding: 1 2; background: $surface; border: thick $primary; }
    LaunchModal #category-tier-row { height: auto; }
    LaunchModal #category-col, LaunchModal #tier-col { width: 1fr; height: auto; }
    LaunchModal #category-col { margin-right: 2; }
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
        with VerticalScroll():
            yield Static("[b]Launch test[/b]")
            yield Static(f"Model:      {self._endpoint.model_id}")
            yield Static(f"Endpoint:   {self._endpoint.base_url}")

            with Horizontal(id="category-tier-row"):
                with Vertical(id="category-col"):
                    yield Label("Category:")
                    with RadioSet(id="category"):
                        btn_cat_all = RadioButton("all", id="category-all")
                        btn_cat_all.value = True
                        yield btn_cat_all
                        for cat in Category:
                            yield RadioButton(cat.value, id=f"category-{cat.value}")
                with Vertical(id="tier-col"):
                    yield Label("Tier:")
                    with RadioSet(id="tier"):
                        for value in ("easy", "medium", "hard", "very_hard", "all"):
                            btn = RadioButton(value, id=f"tier-{value}")
                            if value == "all":
                                btn.value = True
                            yield btn

            yield Label("Mode:")
            with RadioSet(id="mode"):
                yield RadioButton("Eval only", id="mode-eval")
                btn_eval_perf = RadioButton("Eval + perf (llama-benchy)", id="mode-eval-perf")
                btn_eval_perf.value = True
                yield btn_eval_perf
                yield RadioButton("Perf only (llama-benchy)", id="mode-perf")

            with Horizontal(classes="row"):
                yield Label("Trials: ")
                yield Input(value="5", id="trials")
                yield Label("  Concurrency: ")
                yield Input(value="4", id="concurrency")

            yield Label("Harness:")
            with RadioSet(id="adapter"):
                for name in ("raw", "hermes", "claude_code"):
                    status = self._adapters[name]
                    text = _ADAPTER_LABELS[name]
                    if not status.available:
                        text = f"{text}  - disabled ({status.reason})"
                    btn = RadioButton(text, id=f"adapter-{name}", disabled=not status.available)
                    if name == "hermes":
                        btn.value = True
                    yield btn

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

    def _selected_category(self) -> str:
        rs = self.query_one("#category", RadioSet)
        if rs.pressed_button is None:
            return "all"
        return rs.pressed_button.id.removeprefix("category-")

    def _selected_adapter(self) -> str:
        rs = self.query_one("#adapter", RadioSet)
        if rs.pressed_button is None:
            return "raw"
        return rs.pressed_button.id.removeprefix("adapter-")

    def _selected_mode(self) -> str:
        rs = self.query_one("#mode", RadioSet)
        if rs.pressed_button is None:
            return "eval"
        return rs.pressed_button.id.removeprefix("mode-")

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
        mode = self._selected_mode()
        with_perf = mode in ("eval-perf", "perf")
        perf_only = mode == "perf"
        try:
            return RunArgs(
                model=self._endpoint.model_id,
                served_model=self._endpoint.served_model_id,
                base_url=self._endpoint.base_url,
                adapter=self._selected_adapter(),
                tier=self._selected_tier(),
                category=self._selected_category(),
                trials=trials,
                concurrency=concurrency,
                with_perf=with_perf,
                perf_only=perf_only,
            )
        except Exception as e:
            self.app.notify(f"Invalid args: {e}", severity="error")
            return None

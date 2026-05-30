from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    SelectionList,
    Static,
)

from llm_test.core.adapter_probe import AdapterStatus
from llm_test.core.endpoint_scanner import EndpointInfo
from llm_test.core.models import Category
from llm_test.core.runner_subprocess import RunArgs

_ADAPTER_LABELS = {
    "raw":         "raw                - local OpenAI-compatible port",
    "cloud":       "cloud              - remote OpenAI-compatible API (needs key)",
    "hermes":      "hermes             - CLI subprocess",
}


class LaunchModal(ModalScreen[RunArgs | None]):
    """Form modal: pre-filled flags + harness picker. dismiss() returns RunArgs or None."""

    DEFAULT_CSS = """
    LaunchModal {
        align: center middle;
    }

    LaunchModal > VerticalScroll {
        width: 104;
        max-height: 90%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
        border-title-color: $primary;
    }

    LaunchModal #category-tier-row {
        height: auto;
        margin-top: 1;
    }

    LaunchModal #category-col,
    LaunchModal #tier-col {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 0 1;
    }

    /* Cap the multi-select lists so the long Category list scrolls internally
       instead of pushing the Run button below the modal fold. */
    LaunchModal #category,
    LaunchModal #tier {
        height: 8;
        border: none;
        background: $surface;
    }

    LaunchModal #category-col {
        margin-right: 2;
    }

    LaunchModal .row {
        height: auto;
        margin-top: 1;
    }

    LaunchModal Input {
        width: 12;
    }

    LaunchModal Button {
        margin-left: 2;
        min-width: 12;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, endpoint: EndpointInfo, adapters: dict[str, AdapterStatus],
                 interrupted_run: dict | None = None) -> None:
        super().__init__()
        self._endpoint = endpoint
        self._adapters = adapters
        self._interrupted_run = interrupted_run
        # Last-known selection per multi-select list, keyed by widget id. Lets
        # the SelectedChanged handler tell "user added a specific" apart from
        # "user re-picked all" so the 'all' pseudo-option stays exclusive.
        self._prev_selection: dict[str, list[str]] = {
            "category": ["all"], "tier": ["all"],
        }

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[b]Launch benchmark[/b]")
            yield Static(f"Model:      {self._endpoint.model_id}")
            yield Static(f"Endpoint:   {self._endpoint.base_url}")

            with Horizontal(id="category-tier-row"):
                with Vertical(id="category-col"):
                    yield Label("Category (multi-select; 'all' = whole suite):")
                    cat_options = [("all", "all", True)]
                    cat_options += [(cat.value, cat.value, False) for cat in Category]
                    yield SelectionList[str](*cat_options, id="category")
                with Vertical(id="tier-col"):
                    yield Label("Tier (multi-select; 'all' = every tier):")
                    tier_options = [
                        (value, value, value == "all")
                        for value in ("all", "easy", "medium", "hard", "very_hard")
                    ]
                    yield SelectionList[str](*tier_options, id="tier")

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

            yield Label("Cluster (DGX Spark topology):")
            with RadioSet(id="cluster"):
                for value, label in (
                    ("single", "1 spark  (single)"),
                    ("dual",   "2 sparks (dual)"),
                    ("triple", "3 sparks (triple)"),
                    ("quad",   "4 sparks (quad)"),
                    ("octa",   "8 sparks (octa)"),
                ):
                    btn = RadioButton(label, id=f"cluster-{value}")
                    if value == "single":
                        btn.value = True
                    yield btn

            yield Label("Harness:")
            with RadioSet(id="adapter"):
                # UI exposes 3 options: raw / cloud / hermes.
                for ui_name, status_key in (
                    ("raw",    "raw"),
                    ("cloud",  "cloud"),
                    ("hermes", "hermes"),
                ):
                    status = self._adapters.get(status_key) or self._adapters.get(ui_name)
                    if status is None:
                        continue
                    text = _ADAPTER_LABELS[ui_name]
                    if not status.available:
                        text = f"{text}  - disabled ({status.reason})"
                    btn = RadioButton(text, id=f"adapter-{ui_name}",
                                      disabled=not status.available)
                    if ui_name == "raw":
                        btn.value = True
                    yield btn

            with Horizontal(classes="row"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Run", id="run", variant="primary")

    @on(SelectionList.SelectedChanged, "#category")
    @on(SelectionList.SelectedChanged, "#tier")
    def _keep_all_exclusive(self, event: SelectionList.SelectedChanged) -> None:
        """'all' is mutually exclusive with the specific options. Picking a
        specific value drops 'all'; re-picking 'all' drops the specifics; an
        empty selection falls back to 'all' so a run target always exists.

        The handler is convergent: every programmatic deselect/select below
        re-fires SelectedChanged, but only into a now-normalized state that
        triggers no further changes — so it settles without a re-entrancy flag.
        """
        sl = event.selection_list
        key = sl.id or ""
        sel = list(sl.selected)
        prev = self._prev_selection.get(key, ["all"])
        if "all" in sel and len(sel) > 1:
            if "all" in prev:
                # 'all' was already on, user added a specific → drop 'all'.
                sl.deselect("all")
            else:
                # User turned 'all' back on → drop the specifics.
                for value in sel:
                    if value != "all":
                        sl.deselect(value)
        elif not sel:
            sl.select("all")
        self._prev_selection[key] = list(sl.selected)

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
        return self._selected_multi("#tier")

    def _selected_category(self) -> str:
        return self._selected_multi("#category")

    def _selected_multi(self, selector: str) -> str:
        """Comma-join the chosen values, or 'all' if 'all' (or nothing) is on."""
        sel = list(self.query_one(selector, SelectionList).selected)
        if not sel or "all" in sel:
            return "all"
        return ",".join(sel)

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

    def _selected_cluster(self) -> str:
        rs = self.query_one("#cluster", RadioSet)
        if rs.pressed_button is None:
            return "single"
        return rs.pressed_button.id.removeprefix("cluster-")

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
                cluster=self._selected_cluster(),
            )
        except Exception as e:
            self.app.notify(f"Invalid args: {e}", severity="error")
            return None

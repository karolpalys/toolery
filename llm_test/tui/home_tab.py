from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, DataTable, Static

from llm_test.core import endpoint_scanner
from llm_test.core.endpoint_scanner import EndpointInfo

DEFAULT_PORTS = [8000, 8080, 8081, 8888, 8889, 5000, 5001, 11434]
DEEP_SCAN_PORTS = list(range(8000, 9001))

ScannerCallable = Callable[[list[int]], Awaitable[list[EndpointInfo]]]
KnownProvider = Callable[[], set[str]]


class HomeTab(Container):
    """Discover local LLM endpoints; route selection into LaunchModal."""

    DEFAULT_CSS = """
    HomeTab { layout: vertical; padding: 1 2; }
    HomeTab #buttons { height: 3; }
    HomeTab #status { height: 1; color: $text-muted; }
    HomeTab Button { margin-right: 2; }
    HomeTab DataTable { height: 1fr; }
    """

    def __init__(
        self,
        scanner: ScannerCallable | None = None,
        known_models_provider: KnownProvider | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._scanner = scanner or endpoint_scanner.scan
        self._known_provider = known_models_provider or (lambda: set())
        self._scanning = False
        self._endpoints: list[EndpointInfo] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="buttons"):
            yield Button("Scan", id="scan", variant="primary")
            yield Button("Deep scan 8000-9000", id="deep")
            yield Static("Click Scan to discover endpoints", id="status")
        with Vertical():
            yield DataTable(id="endpoints", cursor_type="row")

    def on_mount(self) -> None:
        tbl = self.query_one(DataTable)
        tbl.add_columns("Port", "Model ID", "Status", "Last seen", "Server")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._scanning:
            return
        if event.button.id == "scan":
            await self._run_scan(DEFAULT_PORTS)
        elif event.button.id == "deep":
            await self._run_scan(DEEP_SCAN_PORTS)

    async def _run_scan(self, ports: list[int]) -> None:
        self._scanning = True
        self._set_status(f"Scanning {len(ports)} ports...")
        self._set_buttons_disabled(True)
        started = time.monotonic()
        try:
            results = await self._scanner(ports)
        finally:
            self._scanning = False
            self._set_buttons_disabled(False)
        elapsed = time.monotonic() - started
        self._endpoints = results
        self._refresh_table()
        if not results:
            self._set_status(
                f"Scanned {len(ports)} ports in {elapsed:.1f}s - "
                f"no LLM endpoints detected, start vLLM first"
            )
        else:
            self._set_status(
                f"Last scan: {elapsed:.1f}s ago, {len(results)} endpoints. "
                f"Pick a row to launch."
            )

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if idx is None or idx >= len(self._endpoints):
            return
        endpoint = self._endpoints[idx]
        await self._on_endpoint_selected(endpoint)

    async def _on_endpoint_selected(self, endpoint: EndpointInfo) -> None:
        app = self.app
        opener = getattr(app, "open_launch_modal", None)
        if opener is None:
            self._set_status("App does not provide open_launch_modal; cannot launch.")
            return
        await opener(endpoint)

    def _refresh_table(self) -> None:
        tbl = self.query_one(DataTable)
        tbl.clear()
        known = self._known_provider()
        for ep in self._endpoints:
            badge = "Known" if ep.model_id in known else "New *"
            tbl.add_row(str(ep.port), ep.model_id, badge, "-", ep.server_hint)

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _set_buttons_disabled(self, disabled: bool) -> None:
        self.query_one("#scan", Button).disabled = disabled
        self.query_one("#deep", Button).disabled = disabled

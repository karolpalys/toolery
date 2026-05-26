# Home Tab — Design Spec (MVP)

**Status:** Approved 2026-05-24
**Owner:** P. Karol
**Scope:** Approach A — MVP "Home + Launch"
**Source dialog:** Brainstorming session 2026-05-24

---

## 1. Goal

Add a **Home** tab to the LLM-test TUI that:

1. Discovers OpenAI-compatible LLM endpoints on the local host by probing a known list of ports (with optional 8000–9000 deep scan).
2. Distinguishes models the project has already tested (in `runs.db`) from new ones.
3. Lets the user pick a discovered endpoint and launch `llm-test run` against it through a launch modal with pre-filled flags and adapter (harness) selection.
4. Spawns the run as a subprocess and switches focus to the Live tab.

All UI text in English. Designed as an MVP — streaming logs, abort buttons, and per-model smart defaults are explicit follow-ups, not part of this spec.

## 2. Non-goals (MVP)

- Streaming `llm-test run` stdout into the Live tab (Live continues to be DB-polled).
- Abort/kill the running subprocess from the TUI.
- Auto-refreshing the endpoint table in the background.
- Persisting last-used flags across TUI restarts.
- Detecting model swaps on the same port between scans (e.g. port 8888 was MiniMax, now Gemma).
- vLLM/server-side flag suggestions (only `llm-test run` flags are surfaced).

## 3. Architecture

### File layout (new in **bold**)

```
llm_test/
├── tui/
│   ├── app.py                          [modified — add Home tab, handle LaunchRequested]
│   ├── home_tab.py                     [NEW — Home page widget]
│   ├── launch_modal.py                 [NEW — ModalScreen with flag form + harness picker]
│   ├── live_tab.py                     [modified — DB-driven polling on_mount via set_interval]
│   ├── history_tab.py                  [unchanged]
│   ├── rankings_tab.py                 [unchanged]
│   └── scenarios_tab.py                [unchanged]
├── core/
│   ├── endpoint_scanner.py             [NEW — async port probing]
│   ├── adapter_probe.py                [NEW — detect installed adapters]
│   └── runner_subprocess.py            [NEW — spawn `llm-test run` from inside TUI]
└── tests/
    ├── test_endpoint_scanner.py        [NEW]
    ├── test_adapter_probe.py           [NEW]
    └── test_home_tab.py                [NEW]
```

### Module contracts (one-line each)

| Module | Public surface |
|---|---|
| `endpoint_scanner.py` | `async def scan(ports, timeout=1.0) -> list[EndpointInfo]` — concurrent GET `/v1/models`; returns one `EndpointInfo` per reachable LLM endpoint. |
| `adapter_probe.py` | `def available_adapters() -> dict[str, AdapterStatus]` — checks PATH/env for hermes/claude/codex, returns mapping `name -> AdapterStatus(available, reason)`. |
| `runner_subprocess.py` | `async def spawn_run(args: RunArgs) -> asyncio.subprocess.Process` — wraps the `asyncio.create_subprocess_exec` API around `llm-test run …` argv (no shell). |
| `home_tab.py` | `class HomeTab(Container)` — Textual widget composing scan buttons + DataTable + status line; pushes `LaunchModal` on row selection. |
| `launch_modal.py` | `class LaunchModal(ModalScreen[RunArgs])` — form for tier/trials/adapter/perf; on Submit calls `self.dismiss(RunArgs(...))` so the caller receives the value via `push_screen_wait`. |
| `app.py` | Adds Home as first tab; opens the modal with `push_screen_wait`, then calls `spawn_run` and switches active tab to `live`. |

### Data models

```
# endpoint_scanner.py
class EndpointInfo(BaseModel):
    port: int
    base_url: str                 # e.g. "http://localhost:8888"
    model_id: str                 # primary model from /v1/models data[0].id
    models: list[str]             # all model IDs returned
    server_hint: str              # "vLLM" | "llama.cpp" | "unknown"

# adapter_probe.py
class AdapterStatus(BaseModel):
    available: bool
    reason: str | None            # e.g. "set CLAUDE_CLI_PATH" when unavailable

# runner_subprocess.py
class RunArgs(BaseModel):
    model: str
    base_url: str
    adapter: str                  # raw|hermes|claude_code|codex
    tier: str                     # easy|medium|hard|very_hard|all
    trials: int
    concurrency: int
    with_perf: bool
```

### Integration with existing `app.py`

Two concrete edits:

1. Add Home as the first `TabPane` and set `initial="home"`.
2. Add a message handler that receives the modal's `RunArgs` on dismiss, calls `spawn_run`, and switches active tab to `live`.

```
class LLMTestApp(App):
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="home"):
            with TabPane("Home", id="home"):
                yield HomeTab(id="home-tab")
            with TabPane("Live", id="live"):
                yield LiveTab(id="live-tab")
            with TabPane("History", id="history"):
                yield HistoryTab(id="history-tab")
            with TabPane("Rankings", id="rankings"):
                yield RankingsTab(id="rankings-tab")
            with TabPane("Scenarios", id="scenarios"):
                yield ScenariosTab(id="scenarios-tab")
        yield Footer()

    async def open_launch_modal(self, endpoint: EndpointInfo) -> None:
        args = await self.push_screen_wait(LaunchModal(endpoint, available_adapters()))
        if args is None:
            return  # user cancelled
        try:
            self.run_subprocess = await runner_subprocess.spawn_run(args)
            self.notify(f"Run started: {args.model} / {args.adapter}")
            self.query_one(TabbedContent).active = "live"
        except Exception as e:
            self.notify(f"Failed to launch: {e}", severity="error")
```

## 4. Port-scan strategy

Two modes, both async, both use `httpx.AsyncClient` with `timeout=1.0`:

- **Standard scan** (button label `Scan`): fixed list of 8 ports
  `[8000, 8080, 8081, 8888, 8889, 5000, 5001, 11434]`.
  Hard-coded for MVP. Making this configurable via `config.yaml` is a follow-up tracked in §11.

- **Deep scan** (button label `Deep scan 8000–9000`): `range(8000, 9001)`, all 1001 ports concurrent (asyncio semaphore = 100). Expected ~3–5 s on this hardware.

Probe is GET `<base_url>/v1/models`. An endpoint counts as "LLM-compatible" iff:
1. HTTP 200,
2. JSON body parses,
3. Top-level `data` is a non-null array.

`model_id` is `data[0].id`. `models` is the list of IDs in the response. `server_hint` reads the `Server` HTTP header (`vLLM/<version>` → `"vLLM"`; `llamacpp` → `"llama.cpp"`; else `"unknown"`).

## 5. "Known vs New" detection

```
known_models = {row["model"] for row in store.fetch_all_runs()}
for ep in endpoints:
    ep.status = "Known" if ep.model_id in known_models else "New ✨"
```

DataTable column shows the badge string. New models render with class `endpoint-new` for visual highlight (yellow/green text on black background).

## 6. Adapter availability gating

Probed once on TUI startup; cached for session. Newly installed CLIs require a TUI restart to be picked up — acceptable for MVP given install events are rare.

| Adapter | Available iff | Reason string when unavailable |
|---|---|---|
| `raw` | always | — |
| `hermes` | `shutil.which("hermes")` not None | `hermes CLI not in PATH` |
| `claude_code` | `CLAUDE_CLI_PATH` env set OR `claude` on PATH | `set CLAUDE_CLI_PATH or install claude` |
| `codex` | `CODEX_CLI_PATH` env set OR `codex` on PATH | `set CODEX_CLI_PATH or install codex` |

In the modal, unavailable adapters are shown as disabled RadioButton rows with the reason string as suffix. They cannot be selected; `raw` is always the default.

## 7. UI layout

### Home tab

```
┌─ Home ── Live ── History ── Rankings ── Scenarios ───────────────────────────┐
│                                                                              │
│  [Scan]  [Deep scan 8000–9000]   Last scan: 12s ago, 3 endpoints found       │
│                                                                              │
│  ┌─ Detected endpoints ──────────────────────────────────────────────────┐  │
│  │ Port   Model ID            Status     Last seen          Server      │  │
│  │ 8888   MiniMax-M2.7        Known      2026-05-24 08:10   vLLM        │  │
│  │ 8000   deepseek-v4-flash   Known      2026-05-24 01:08   vLLM        │  │
│  │ 8080   qwen3-coder-4b      New ✨     —                  vLLM        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Status: Pick a row to launch a test                                         │
│                                                                              │
└────────────────────────────────────────────── q quit  ctrl+r refresh ────────┘
```

Empty state (after scan finds nothing):
`Detected endpoints: [no LLM endpoints detected — start vLLM first]`

### Launch modal

```
       ┌── Launch test ─────────────────────────────────────────┐
       │                                                        │
       │  Model:      qwen3-coder-4b                            │
       │  Endpoint:   http://localhost:8080                     │
       │                                                        │
       │  Tier:       ( ) easy   ( ) medium   ( ) hard          │
       │              ( ) v_hard (•) all                        │
       │                                                        │
       │  Trials:        [ 5  ]                                 │
       │  Concurrency:   [ 4  ]                                 │
       │                                                        │
       │  Harness:                                              │
       │    (•) raw                — direct OpenAI port         │
       │    ( ) hermes             — CLI subprocess             │
       │    ( ) claude_code        — disabled (no CLAUDE_CLI)   │
       │    ( ) codex              — disabled (no codex bin)    │
       │                                                        │
       │  [✓] Collect perf (llama-benchy)                       │
       │                                                        │
       │         [ Cancel ]                       [  Run  ]     │
       └────────────────────────────────────────────────────────┘
```

## 8. End-to-end data flow

1. User runs `~/test.sh` → TUI launches → Home tab focused.
2. User presses **Scan** → `endpoint_scanner.scan(default_ports)` runs concurrently.
3. Results populate the DataTable. "Last scan: Ns ago, M endpoints found" appears in the buttons row.
4. User presses **Enter** on a row (or clicks) → `LaunchModal` pushed with that endpoint's `EndpointInfo` and `available_adapters()` result.
5. User adjusts flags, presses **Run** → modal dismisses with the built `RunArgs`.
6. `LLMTestApp.open_launch_modal` (awaiting `push_screen_wait`) receives the args, calls `runner_subprocess.spawn_run(args)`, and switches active tab to `live`.
7. Subprocess writes to `results/runs.db` via the existing `Store` code path (creates row immediately, updates as scenarios finish).
8. Live tab's `set_interval(2.0, refresh)` picks up the new row on next tick.

## 9. Error handling

| Scenario | Detection | UI response |
|---|---|---|
| Port closed (TCP RST) | `httpx.ConnectError` | Silent skip — expected for most ports |
| Port open but not LLM (Grafana, etc.) | non-200 OR JSON without `data[]` array | Silent skip; logged to stderr |
| `/v1/models` returns 200 with empty list | `data == []` | Row shown with `model_id = "(none loaded)"` |
| Probe timeout | 1 s per port | Silent skip; status line shows timed-out count |
| Deep scan partial fail | per-port try/except | Continue; report aggregate failure count |
| DataTable empty after scan | `len(endpoints) == 0` | `[no LLM endpoints detected — start vLLM first]` |
| Modal: trials non-numeric | `Input.value` not int | Red border on field, [Run] disabled |
| Modal: trials < 1 or > 100 | range check | Red border, status line `trials must be 1–100` |
| Subprocess spawn fails | `FileNotFoundError` / `PermissionError` | Toast: `Failed to launch: <reason>`; stay on Home |
| Subprocess exits non-zero within 2 s | `proc.returncode is not None` | Toast: `Run exited early (code N). Check terminal stderr.` |
| Scan started while previous scan running | `self._scanning` flag | Buttons disabled during scan |
| Row picked while run in flight | `self.run_subprocess and proc.returncode is None` | Toast: `A run is already in progress on Live. Wait or abort.`; modal not opened |
| Endpoint disappears between scan and Run | subprocess fails with `Connection refused` | Standard subprocess-fail path above |

## 10. Testing

| Module | Test framework | Mock strategy | Cases |
|---|---|---|---|
| `endpoint_scanner.py` | pytest + respx | mock httpx GET `/v1/models` per port | 200 valid / 200 empty / 404 / 500 / timeout / connect refused / malformed JSON |
| `adapter_probe.py` | pytest + monkeypatch | patch `shutil.which`, `os.environ` | each adapter: present / absent / env-only / PATH-only |
| `runner_subprocess.py` | pytest-asyncio | patch the asyncio subprocess factory | argv construction, env propagation, success/failure return |
| `home_tab.py` | Textual `Pilot` | inject fake scanner | click Scan → table populated; row click → modal pushed; empty state rendered |
| `launch_modal.py` | Textual `Pilot` | — | submit emits `LaunchRequested`; Cancel dismisses; validation blocks Run |
| Integration | Textual `Pilot` + respx | mock httpx + subprocess | full flow: open → scan → select → modal → Run → tab switch |

Target: pytest coverage ≥ 80%, consistent with existing project standard.

## 11. Out-of-scope (explicit follow-ups)

These are recorded so they don't get lost — none are part of this MVP.

- **Streaming stdout** from subprocess into Live tab (Approach B).
- **Abort button** on Live with SIGTERM to subprocess (Approach B).
- **Auto-refresh** endpoint table on a timer (Approach B).
- **Sticky parameters** persisted across TUI restarts (Approach B).
- **Smart per-model defaults** (e.g. MiniMax → suggests `--with-perf`, `--adapter hermes`) (Approach C).
- **Server fingerprint** with vLLM/llama-server version detection in details panel (Approach C).
- **Post-run modal** "Compare with previous run X?" (Approach C).

## 12. Acceptance criteria

The MVP is done when:

1. `llm-test tui` opens with Home as the first visible tab.
2. Pressing `Scan` populates the DataTable with all reachable endpoints from the default port list.
3. Pressing `Deep scan` probes ports 8000–9000 within ~5 s.
4. Selecting a row opens the launch modal pre-filled with that endpoint's model and base_url.
5. Unavailable adapters appear as disabled rows with a reason hint.
6. Pressing `Run` in the modal spawns `llm-test run` with the chosen flags and switches focus to Live.
7. The Live tab shows the new run within 2 s (DB-polled).
8. All new modules covered by tests at ≥ 80%.

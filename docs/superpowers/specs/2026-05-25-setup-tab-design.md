# Setup Tab + Use-Case Rankings — Design

**Status:** Design — awaiting user review 2026-05-25
**Owner:** rahueme

## 1. Goal

Add a new "Setup" tab (6th, after History) to the TUI that lets the user pick ONE use-case persona for which they're evaluating LLMs. The chosen persona drives a NEW dedicated ranking column ("UC:&lt;Name&gt;") in the Rankings tab, computed from existing per-scenario scores using persona-specific dimension weights. The general `Overall` column (with current `_DIM_WEIGHTS`) is left UNCHANGED — the use-case ranking is purely additive.

## 2. What this is NOT

- NOT a replacement for the global Overall — both rankings coexist.
- NOT a way to edit weights inline in MVP — only preset selection. Custom-weights v2 stays out of scope.
- NOT a way to have multiple active use-cases — exactly one selected at a time (or none).

## 3. Seven use-case personas

Each persona defines weights for the 14 weightable dimensions (overall is meta). Stored in new module `llm_test/rankings/presets.py` as a frozen dataclass list. Single source of truth.

| Dimension | Coding Assistant | Reasoning | Agentic Orchestrator | Safety/RAG | Customer Support | Data Analyst | Local Coding Agent |
|---|---:|---:|---:|---:|---:|---:|---:|
| coding | **3.0** | 0.5 | 1.5 | 0.5 | 0.3 | 1.0 | **3.0** |
| terminal | **2.5** | 0.3 | 1.5 | 0.3 | 0.3 | 1.5 | **3.0** |
| agentic | 2.0 | 0.5 | **3.0** | 0.5 | 0.5 | 1.0 | 2.5 |
| safety | 1.0 | 1.5 | 1.0 | **3.0** | 2.5 | 1.0 | 1.5 |
| restraint | 1.5 | 2.0 | 1.0 | **2.5** | **2.5** | 1.5 | 1.5 |
| error_recovery | 1.5 | 1.0 | **2.5** | 1.0 | 1.5 | 1.0 | 2.0 |
| parameter_precision | 1.0 | **2.5** | 1.5 | 1.5 | 1.5 | **2.5** | 1.5 |
| context_state_tracking | 1.0 | 2.0 | 2.0 | 1.0 | 1.5 | 2.0 | 2.0 |
| structured_output | 2.0 | 1.5 | 1.0 | 1.5 | 2.0 | **3.0** | 1.5 |
| tool_selection | 2.0 | 1.0 | **2.5** | 1.0 | 1.0 | 2.0 | 2.0 |
| long_context | 0.75 | **2.5** | 1.0 | 2.0 | 1.0 | 2.0 | 1.5 |
| localization | 0.3 | 0.75 | 0.3 | 1.5 | **3.0** | 0.5 | 0.3 |
| budget_efficiency | 1.5 | 0.75 | 2.0 | 0.5 | 0.75 | 1.0 | 2.0 |
| hallucination | 1.5 | **3.0** | 1.0 | **3.0** | 2.0 | 1.5 | 1.5 |

### Persona rationale

1. **Coding Assistant** — IDE copilot. Top: coding, terminal, structured_output. Low: localization, long_context.
2. **Reasoning** — analytical model that thinks not acts. Top: hallucination, parameter_precision, long_context. Low: coding, terminal, agentic.
3. **Agentic Orchestrator** — autonomous multi-step planner. Top: agentic, tool_selection, error_recovery, budget_efficiency.
4. **Safety/RAG** — risk-aware retrieval with strong calibration. Top: safety, hallucination, restraint, long_context. Low: agentic, terminal, budget_efficiency.
5. **Customer Support** — multilingual helpdesk. Top: localization, safety, restraint. Low: terminal, coding, agentic.
6. **Data Analyst** — DB queries, CSV/JSON output, numeric fidelity. Top: structured_output, parameter_precision, tool_selection.
7. **Local Coding Agent** — Claude-Code/Codex-style local CLI agent. Top: coding, terminal (=3.0 each), high agentic+error_recovery+budget_efficiency. Differs from Coding Assistant by emphasizing terminal/autonomy over structured_output.

## 4. Persistence — `results/setup.json`

Minimal schema:

```json
{
  "version": 1,
  "active_use_case": "coding_assistant"
}
```

- `active_use_case`: one of `coding_assistant`, `reasoning`, `agentic_orchestrator`, `safety_rag`, `customer_support`, `data_analyst`, `local_coding_agent`, or omitted/null.
- File **absent** OR `active_use_case` null → no use-case, only global Overall column visible.
- Unknown key value → silent fallback to "no use-case" + log warning. No crash.
- The file is written by the Setup tab on Apply; deleted on Clear. User can also edit manually.

**Custom-weights NOT in MVP.** The schema reserves space for v2 (`"custom_weights": {dim: float}`) but the loader ignores it.

## 5. Module: `llm_test/rankings/presets.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class UseCase:
    key: str
    name: str
    description: str
    weights: dict[str, float]

USE_CASES: list[UseCase] = [
    UseCase(
        key="coding_assistant",
        name="Coding Assistant",
        description="IDE-style copilot — code completion, TDD, refactors, diffs.",
        weights={"coding": 3.0, "terminal": 2.5, ...},  # all 14 dims
    ),
    # ... 6 more
]

def get_use_case(key: str) -> UseCase | None:
    return next((uc for uc in USE_CASES if uc.key == key), None)
```

Single source of truth — Setup tab UI, compute.py loader, and tests all import from here.

## 6. Loader in `compute.py`

```python
def load_active_use_case(results_dir: Path) -> tuple[str | None, dict[str, float] | None]:
    """Read results/setup.json. Return (key, weights) or (None, None)."""
    setup_path = results_dir / "setup.json"
    if not setup_path.exists():
        return (None, None)
    try:
        data = json.loads(setup_path.read_text())
    except (json.JSONDecodeError, OSError):
        return (None, None)
    key = data.get("active_use_case")
    if not key:
        return (None, None)
    from llm_test.rankings.presets import get_use_case
    uc = get_use_case(key)
    if uc is None:
        return (None, None)
    return (key, uc.weights)
```

## 7. Changes to `compute_matrix`

Add optional `use_case_weights: dict[str, float] | None = None` parameter. When set, AFTER computing the existing `scores["overall"]` for a (model, adapter) pair, compute an ADDITIONAL `scores["use_case"]` using the SAME items (per-scenario score + tier + dims) but with the use-case weights applied via `_scenario_dim_weight(dims, weights_override=use_case_weights)` instead of the default `_DIM_WEIGHTS`.

Modify `_scenario_dim_weight`:

```python
def _scenario_dim_weight(
    ranking_dims: list[str],
    weights_override: dict[str, float] | None = None,
) -> float:
    weights_map = weights_override if weights_override is not None else _DIM_WEIGHTS
    weights = [weights_map.get(d, 1.0) for d in ranking_dims if d != "overall"]
    return max(weights) if weights else 1.0
```

Backwards-compatible: existing callers (with no `weights_override`) get default behavior.

## 8. Mirror in `regenerate_rankings`

When `use_case_weights` is passed (also a new optional param to `regenerate_rankings`), AFTER writing `overall.md`/`coding.md`/etc., emit one additional file `use_case_<key>.md` using the same template but with the use-case score as the leading column.

## 9. Rankings tab changes

Minimal changes to `rankings_tab.py`:

1. On `on_mount` / `refresh_data`, call `load_active_use_case(results_dir)`.
2. If active → append ONE extra column to the DataTable after `Overall`, header `UC:<ShortName>` (e.g. `UC:Coding`, `UC:Reason`, `UC:Agentic`, `UC:Safety`, `UC:Support`, `UC:Data`, `UC:LocalCA`). Column reads from `row["scores"].get("use_case")` populated by `compute_matrix`.
3. If not active → no extra column. Tab looks like today.
4. Sort works on the new column same as others (decay-weighted score).

Short-name map (≤8 chars to fit alongside existing column widths):

```python
_USE_CASE_HEADERS = {
    "coding_assistant": "UC:Coding",
    "reasoning": "UC:Reason",
    "agentic_orchestrator": "UC:Agentic",
    "safety_rag": "UC:Safety",
    "customer_support": "UC:Support",
    "data_analyst": "UC:Data",
    "local_coding_agent": "UC:LocalCA",
}
```

## 10. Setup tab UI (`setup_tab.py`)

Layout (Textual widgets):

- Heading `Static` — "Pick the use-case you're evaluating for. The chosen profile creates an extra ranking column 'UC:<Name>' in Rankings (next to Overall)."
- `RadioSet` with 8 `RadioButton`s: `None` + 7 personas. Default selection on mount = current `active_use_case` from setup.json (or `None` if cleared).
- Per-persona description block (`Static`) — name + 1-line description + top-3 weights (↑) + bottom-2 weights (↓), auto-generated from `presets.py`.
- Two `Button`s: `Apply` (variant=success), `Clear` (variant=error).
- Footer `Static` showing `Active: <key>` or `Active: none`.

### Apply action

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "apply":
        selected = self.query_one(RadioSet).pressed_button.id  # "uc-coding_assistant" etc.
        key = selected.removeprefix("uc-")
        setup_path = self._results_dir / "setup.json"
        if key == "none":
            setup_path.unlink(missing_ok=True)
            self.app.notify("Use-case cleared")
        else:
            setup_path.write_text(json.dumps({"version": 1, "active_use_case": key}, indent=2))
            self.app.notify(f"Use-case '{key}' applied — regenerating rankings...")
        # Trigger regen
        self._regenerate_rankings()
        # Switch focus
        self.app.query_one(TabbedContent).active = "rankings"
        # Refresh rankings tab data
        rt = self.app.query_one(RankingsTab)
        if hasattr(rt, "refresh_data"):
            rt.refresh_data()
```

### Clear action

Same path with `key="none"` short-circuit.

## 11. App.py change

```python
# Add import
from llm_test.tui.setup_tab import SetupTab

# In compose(), after the History tab pane:
with TabPane("Setup", id="setup"):
    yield SetupTab(id="setup-tab")
```

That's it. No other changes to app.py.

## 12. Tests

New file `tests/tui/test_setup_tab.py`:
- `test_setup_tab_loads_without_active_use_case` — no setup.json present → tab loads with `None` selected.
- `test_setup_tab_loads_with_active_use_case` — setup.json present → correct radio is selected on mount.
- `test_apply_writes_setup_json` — pressing Apply with persona selected writes correct JSON.
- `test_clear_removes_setup_json` — pressing Clear removes the file.
- `test_unknown_use_case_in_setup_falls_back_to_none` — manually-edited unknown key → silent fallback.

New file `tests/rankings/test_presets.py`:
- `test_seven_personas_defined` — len(USE_CASES) == 7.
- `test_every_persona_has_all_14_dims` — each weights dict has exactly the 14 weightable dims as keys.
- `test_get_use_case_returns_match_or_none`.

Extend `tests/rankings/test_dim_weights.py`:
- `test_compute_matrix_with_use_case_weights_emits_use_case_score` — feed seeded store, pass `use_case_weights`, assert `scores["use_case"]` present and differs from `scores["overall"]` when weights differ.
- `test_compute_matrix_without_use_case_weights_omits_use_case_score` — backward compat.

## 13. File map

**Create:**
- `llm_test/rankings/presets.py` — 7 personas
- `llm_test/tui/setup_tab.py` — Setup tab widget
- `tests/rankings/test_presets.py`
- `tests/tui/test_setup_tab.py`

**Modify:**
- `llm_test/rankings/compute.py`:
  - Add `load_active_use_case()`
  - Extend `_scenario_dim_weight()` with `weights_override`
  - Extend `compute_matrix()` with `use_case_weights` parameter — emit `scores["use_case"]` when set
  - Extend `regenerate_rankings()` analogously — emit `use_case_<key>.md`
- `llm_test/tui/app.py` — add Setup tab as 6th pane
- `llm_test/tui/rankings_tab.py`:
  - On load, call `load_active_use_case()`
  - When active, add 1 extra column `UC:<Short>` after `Overall`
- `llm_test/cli.py` — `rankings --regen` command picks up active use-case (reads from setup.json and passes to `regenerate_rankings`)
- `tests/rankings/test_dim_weights.py` — 2 new tests
- `README.md` — document the Setup tab + 7 personas in the TUI workflow section

## 14. Build order (orientacyjny — pełny plan zbuduje writing-plans)

1. **`presets.py`** + `test_presets.py` (foundation)
2. **`compute.py`** loader + extended `_scenario_dim_weight` + extended `compute_matrix` + tests (math foundation)
3. **`regenerate_rankings`** mirror
4. **`rankings_tab.py`** — extra column conditionally rendered
5. **`setup_tab.py`** + smoke tests
6. **`app.py`** — wire tab in
7. **`cli.py`** — auto-pick use-case in `--regen`
8. **README** — TUI section update
9. **E2E verification**

## 15. Out of scope (v2 fodder)

- Inline custom-weights editing (sliders/inputs per dim)
- Multiple simultaneous use-cases
- Per-model preset suggestions ("this model likely best for X")
- Persona export/import
- Persona descriptions with longer narrative

## 16. Assumptions and risks

- **Assumption:** Rankings tab DataTable supports dynamic column addition between renders. (Textual's DataTable supports `add_column` at any time.)
- **Assumption:** `results_dir` is discoverable in compute.py — currently lives in `LLM_TEST_RESULTS_DIR` env var with default `./results`. We'll use the existing convention.
- **Risk:** Adding a 16th column may overflow on narrow terminals. **Mitigation:** Short headers (≤8 chars). Same risk already exists with the 15th column.
- **Risk:** Setup tab regen on Apply may take time if results.db is large. **Mitigation:** Worker thread / async; notify on start AND end.

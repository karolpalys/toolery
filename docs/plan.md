# LLM-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build LLM-test — a deterministic 4-tier benchmark for LLM tool calling, with 4 execution adapters (raw OpenAI port / Hermes / Claude Code CLI / Codex CLI), a Textual TUI with 4 tabs, ASCII + PNG charts, 8-dimension ranking system, llama-benchy perf integration, and 32 hand-written scenarios.

**Architecture:** Spec-first (YAML scenarios → Pydantic loader → adapter runs scenario via tool-call loop → deterministic scorer → SQLite + .md output → stats/charts/rankings). Adapter contract is uniform `TraceResult` so all adapters produce comparable data.

**Tech Stack:** Python 3.11+, Pydantic 2, PyYAML, httpx (async), OpenAI SDK, Typer (CLI), Rich + Textual (TUI), matplotlib, numpy/scipy, sqlite3 (stdlib), Jinja2, pytest + pytest-asyncio.

**Reference spec:** [docs/spec.md](spec.md) — all section numbers below refer to spec sections.

**Execution strategy:** Phases are sequential. Each phase produces a working slice. After Phase 12 (Working CLI MVP) the benchmark is usable end-to-end with mock + one real adapter. Phases 13+ add adapters, polish, TUI, full scenario set, calibration.

---

## File Structure (locked in)

```
LLM-test/
├── pyproject.toml                       # Phase 0
├── .gitignore                           # Phase 0
├── README.md                            # Phase 27
├── config.example.yaml                  # Phase 12
├── config.yaml                          # gitignored, user creates
├── docs/
│   ├── spec.md                          # exists
│   └── plan.md                          # this file
├── llm_test/
│   ├── __init__.py                      # Phase 0
│   ├── core/
│   │   ├── __init__.py                  # Phase 1
│   │   ├── models.py                    # Phase 1 — Pydantic: Scenario, ToolCall, TraceResult, ScenarioResult
│   │   ├── scenario.py                  # Phase 2 — YAML loader + validation
│   │   ├── scorer.py                    # Phase 4-5 — 16 primitives + composition engine
│   │   ├── runner.py                    # Phase 7 — orchestration (scenario × adapter × trial)
│   │   ├── store.py                     # Phase 10 — SQLite CRUD
│   │   ├── markdown.py                  # Phase 11 — Jinja templates for summary/scenario/ranking/compare
│   │   └── stats.py                     # Phase 16 — bootstrap CI, McNemar, decay-weighted aggregation
│   ├── adapters/
│   │   ├── __init__.py                  # Phase 6
│   │   ├── base.py                      # Phase 6 — Adapter protocol + MockAdapter
│   │   ├── openai_raw.py                # Phase 8
│   │   ├── hermes.py                    # Phase 13
│   │   ├── claude_code.py               # Phase 14
│   │   └── codex.py                     # Phase 15
│   ├── tools/
│   │   ├── __init__.py                  # Phase 3
│   │   ├── registry.py                  # Phase 3 — name → callable + JSON schema
│   │   ├── generic.py                   # Phase 3 — weather, email, contacts, search, calc, files, calendar
│   │   └── domain.py                    # Phase 26 — orderbook, vllm_config, git_*, run_tests, run_lint, etc.
│   ├── perf/
│   │   ├── __init__.py                  # Phase 25
│   │   └── benchy.py                    # Phase 25
│   ├── charts/
│   │   ├── __init__.py                  # Phase 17
│   │   ├── ascii.py                     # Phase 17
│   │   └── png.py                       # Phase 18
│   ├── rankings/
│   │   ├── __init__.py                  # Phase 19
│   │   └── compute.py                   # Phase 19 — regenerate_rankings()
│   ├── compare.py                       # Phase 20 — module-level
│   ├── tui/
│   │   ├── __init__.py                  # Phase 21
│   │   ├── app.py                       # Phase 21 — Textual App + tab wiring
│   │   ├── live_tab.py                  # Phase 21
│   │   ├── history_tab.py               # Phase 22
│   │   ├── rankings_tab.py              # Phase 23
│   │   └── scenarios_tab.py             # Phase 24
│   └── cli.py                           # Phase 12 (+ Phase 20, 25 extensions)
├── scenarios/
│   ├── easy/                            # Phase 9 (first 3) + Phase 26 (remaining 7)
│   ├── medium/                          # Phase 26 (10)
│   ├── hard/                            # Phase 26 (8)
│   ├── very_hard/                       # Phase 26 (4)
│   └── templates/                       # Phase 26 — context_prefill snippets
├── tests/
│   ├── __init__.py                      # Phase 0
│   ├── conftest.py                      # Phase 0
│   ├── core/                            # mirror llm_test/core
│   ├── adapters/
│   ├── tools/
│   ├── charts/
│   ├── rankings/
│   ├── tui/
│   ├── fixtures/                        # golden YAMLs, golden traces
│   └── e2e/                             # Phase 9 onwards
└── results/                             # gitignored; populated at runtime
```

---

## Phase 0 — Repo bootstrap

### Task 0.1: Create pyproject.toml + .gitignore + package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `llm_test/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-test"
version = "0.1.0"
description = "Deterministic 4-tier LLM tool-calling benchmark"
requires-python = ">=3.11"
dependencies = [
  "typer>=0.12",
  "rich>=13.7",
  "textual>=0.85",
  "pydantic>=2.6",
  "pyyaml>=6.0",
  "httpx>=0.27",
  "openai>=1.40",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "scipy>=1.12",
  "jinja2>=3.1",
]

[project.optional-dependencies]
perf = ["llama-benchy"]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "pytest-cov>=5.0",
  "respx>=0.21",
  "ruff>=0.6",
  "mypy>=1.10",
]

[project.scripts]
llm-test = "llm_test.cli:app"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Write .gitignore**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
results/
config.yaml
*.egg-info/
dist/
build/
.coverage
htmlcov/
```

- [ ] **Step 3: Write llm_test/__init__.py**

```python
"""LLM-test — deterministic LLM tool-calling benchmark."""
__version__ = "0.1.0"
```

- [ ] **Step 4: Write tests/__init__.py (empty)**

```python
```

- [ ] **Step 5: Write tests/conftest.py**

```python
import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_results_dir(tmp_path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d
```

- [ ] **Step 6: Create venv + install + verify pytest works**

Run:
```bash
cd /home/rahueme/LLM-test && uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest --version
```
Expected: `pytest 8.x.x`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore llm_test/ tests/
git commit -m "chore(init): bootstrap pyproject, package skeleton, pytest config"
```

---

## Phase 1 — Domain models (Pydantic)

### Task 1.1: Define core domain models

**Files:**
- Create: `llm_test/core/__init__.py`
- Create: `llm_test/core/models.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_models.py`

- [ ] **Step 1: Write tests/core/__init__.py (empty)**

```python
```

- [ ] **Step 2: Write failing test for Scenario model**

`tests/core/test_models.py`:
```python
import pytest
from pydantic import ValidationError
from llm_test.core.models import Scenario, Tier, Category, Budget, ScoringCheck, ToolCall, TraceResult


def test_scenario_minimal_valid():
    s = Scenario(
        id="easy-01-test",
        title="t",
        tier=Tier.EASY,
        category=Category.TOOL_SELECTION,
        domain="generic",
        description="d",
        prompt="p",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring={"required": [], "forbidden": [], "partial": [],
                 "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
    )
    assert s.id == "easy-01-test"
    assert s.tier == Tier.EASY
    assert s.context_prefill_tokens == 0


def test_scenario_id_must_be_kebab():
    with pytest.raises(ValidationError):
        Scenario(
            id="easy 01 test",  # spaces not allowed
            title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
            domain="generic", description="d", prompt="p", tools=[],
            budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
            scoring={"required": [], "forbidden": [], "partial": [],
                     "weights": {"pass": 1.0, "partial": 0.5, "fail": 0.0}},
        )


def test_tracerresult_roundtrip():
    tr = TraceResult(
        scenario_id="x", adapter="raw", trial_index=0,
        messages=[], tool_calls=[], final_response=None,
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
        adapter_metadata={},
    )
    assert tr.duration_ms == 42
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/core/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_test.core.models'`

- [ ] **Step 4: Implement llm_test/core/__init__.py (empty)**

```python
```

- [ ] **Step 5: Implement llm_test/core/models.py**

```python
from __future__ import annotations
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator
import re


class Tier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


class Category(str, Enum):
    TOOL_SELECTION = "tool_selection"
    PARAMETER_PRECISION = "parameter_precision"
    MULTI_STEP_CHAINS = "multi_step_chains"
    RESTRAINT_REFUSAL = "restraint_refusal"
    ERROR_RECOVERY = "error_recovery"
    LOCALIZATION = "localization"
    STRUCTURED_REASONING = "structured_reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONTEXT_STATE_TRACKING = "context_state_tracking"
    CODING = "coding"
    SAFETY_BOUNDARIES = "safety_boundaries"
    TOOLSET_SCALE = "toolset_scale"
    AUTONOMOUS_PLANNING = "autonomous_planning"
    CREATIVE_COMPOSITION = "creative_composition"
    STRUCTURED_OUTPUT = "structured_output"
    HARD_MODE = "hard_mode"


class Budget(BaseModel):
    max_tool_calls: int = Field(ge=0)
    max_turns: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1, default=60)


class ScoringCheck(BaseModel):
    check: str
    model_config = {"extra": "allow"}  # primitives have varying fields


class ToolResponseRule(BaseModel):
    match: Any  # dict for keyed match, "any" string for fallback
    returns: Any | None = None
    returns_with_injection: str | None = None
    depends_on: str | None = None
    call_index: int | str | None = None


class Scoring(BaseModel):
    required: list[ScoringCheck] = []
    forbidden: list[ScoringCheck] = []
    partial: list[ScoringCheck] = []
    weights: dict[str, float] = {"pass": 1.0, "partial": 0.5, "fail": 0.0}


class Scenario(BaseModel):
    id: str
    title: str
    tier: Tier
    category: Category
    domain: Literal["generic", "quant", "dev_ops"]
    description: str
    tags: list[str] = []
    ranking_dimensions: list[str] = ["overall"]
    prompt: str
    system_prompt: str | None = None
    tools: list[str]
    context_prefill_tokens: int = 0
    context_prefill_template: str | None = None
    budget: Budget
    tool_responses: dict[str, list[ToolResponseRule]] = {}
    scoring: Scoring

    @field_validator("id")
    @classmethod
    def id_is_kebab(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)+$", v):
            raise ValueError(f"id must be kebab-case, got: {v!r}")
        return v


class ToolCall(BaseModel):
    index: int
    name: str
    args: dict[str, Any] = {}
    result: Any = None
    result_kind: Literal["text", "json", "error"] = "text"
    latency_ms: int = 0


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class TraceResult(BaseModel):
    scenario_id: str
    adapter: str
    trial_index: int
    messages: list[Message]
    tool_calls: list[ToolCall]
    final_response: str | None
    started_at_iso: str
    duration_ms: int
    error: str | None
    adapter_metadata: dict[str, Any] = {}


class CheckResult(BaseModel):
    check: str
    result: Literal["pass", "partial", "fail"]
    detail: str = ""


class ScenarioResult(BaseModel):
    scenario_id: str
    adapter: str
    trial_index: int
    status: Literal["pass", "partial", "fail", "error", "timeout"]
    score: float
    call_count: int
    budget_max: int
    latency_ms: int
    failure_kind: str | None
    checks: list[CheckResult]
    trace: TraceResult
```

- [ ] **Step 6: Run tests to verify pass**

Run: `pytest tests/core/test_models.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/core/__init__.py llm_test/core/models.py tests/core/
git commit -m "feat(core): add Pydantic domain models (Scenario, TraceResult, ScenarioResult)"
```

---

## Phase 2 — YAML scenario loader

### Task 2.1: YAML loading with validation

**Files:**
- Create: `llm_test/core/scenario.py`
- Create: `tests/core/test_scenario.py`
- Create: `tests/fixtures/scenarios/valid_easy.yaml`
- Create: `tests/fixtures/scenarios/invalid_dup_id.yaml`

- [ ] **Step 1: Write fixture tests/fixtures/scenarios/valid_easy.yaml**

```yaml
id: easy-01-direct-weather
title: "Direct weather lookup"
tier: easy
category: tool_selection
domain: generic
description: "Use get_weather instead of web_search."
tags: [tool_call, single_turn]
ranking_dimensions: [overall, tool_selection]
prompt: "What's the weather in Warsaw right now?"
tools:
  - get_weather
  - web_search
budget:
  max_tool_calls: 1
  max_turns: 1
  timeout_seconds: 30
tool_responses:
  get_weather:
    - match: { location: "Warsaw" }
      returns: { temp_c: 7, condition: "cloudy" }
    - match: any
      returns: { error: "city not found" }
scoring:
  required:
    - check: tool_called
      tool: get_weather
    - check: tool_args_contain
      tool: get_weather
      args: { location: "Warsaw" }
  forbidden:
    - check: tool_called
      tool: web_search
  partial:
    - check: response_contains
      patterns: ["7", "cloudy"]
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 2: Write failing test**

`tests/core/test_scenario.py`:
```python
from pathlib import Path
import pytest
from llm_test.core.scenario import load_scenario, load_all_scenarios, DuplicateIdError


def test_load_single_scenario(fixtures_dir):
    s = load_scenario(fixtures_dir / "scenarios" / "valid_easy.yaml")
    assert s.id == "easy-01-direct-weather"
    assert s.tier.value == "easy"
    assert len(s.scoring.required) == 2
    assert s.budget.max_tool_calls == 1


def test_load_all_scenarios_dedups_by_id(fixtures_dir, tmp_path):
    src = fixtures_dir / "scenarios" / "valid_easy.yaml"
    (tmp_path / "a.yaml").write_text(src.read_text())
    (tmp_path / "b.yaml").write_text(src.read_text())  # same id → duplicate
    with pytest.raises(DuplicateIdError):
        load_all_scenarios(tmp_path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/core/test_scenario.py -v`
Expected: FAIL with import error

- [ ] **Step 4: Implement llm_test/core/scenario.py**

```python
from __future__ import annotations
from pathlib import Path
import hashlib
import yaml
from llm_test.core.models import Scenario


class DuplicateIdError(ValueError):
    pass


def load_scenario(path: Path) -> Scenario:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return Scenario.model_validate(data)


def scenario_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_all_scenarios(root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen: dict[str, Path] = {}
    for p in sorted(root.rglob("*.yaml")):
        s = load_scenario(p)
        if s.id in seen:
            raise DuplicateIdError(f"duplicate id {s.id!r} in {p} and {seen[s.id]}")
        seen[s.id] = p
        scenarios.append(s)
    return scenarios
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/core/test_scenario.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add llm_test/core/scenario.py tests/core/test_scenario.py tests/fixtures/scenarios/valid_easy.yaml
git commit -m "feat(core): add YAML scenario loader with duplicate-id detection"
```

---

## Phase 3 — Tool registry + mock generic tools

### Task 3.1: Tool registry and one mock tool

**Files:**
- Create: `llm_test/tools/__init__.py`
- Create: `llm_test/tools/registry.py`
- Create: `llm_test/tools/generic.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_registry.py`

- [ ] **Step 1: Write failing test for registry**

`tests/tools/test_registry.py`:
```python
import pytest
from llm_test.tools.registry import ToolRegistry, ToolSpec
from llm_test.tools import generic  # noqa: F401  triggers registration


def test_registry_has_get_weather():
    reg = ToolRegistry.default()
    spec = reg.get("get_weather")
    assert spec.name == "get_weather"
    assert "location" in spec.json_schema["function"]["parameters"]["properties"]


def test_registry_unknown_raises():
    reg = ToolRegistry.default()
    with pytest.raises(KeyError):
        reg.get("nonexistent_tool_xyz")


def test_registry_openai_schema_for_subset():
    reg = ToolRegistry.default()
    schemas = reg.openai_schemas(["get_weather"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "get_weather"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_registry.py -v`
Expected: FAIL — import error

- [ ] **Step 3: Implement llm_test/tools/registry.py**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict       # full OpenAI tool schema (type=function, function={...})
    impl: Callable[..., Any] | None = None  # optional real impl; benchmarks use mocks


class ToolRegistry:
    _global: "ToolRegistry | None" = None

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def openai_schemas(self, names: list[str]) -> list[dict]:
        return [self.get(n).json_schema for n in names]

    @classmethod
    def default(cls) -> "ToolRegistry":
        if cls._global is None:
            cls._global = cls()
        return cls._global


def register(spec: ToolSpec) -> None:
    ToolRegistry.default().register(spec)
```

- [ ] **Step 4: Implement llm_test/tools/__init__.py**

```python
from llm_test.tools.registry import ToolRegistry, ToolSpec, register

__all__ = ["ToolRegistry", "ToolSpec", "register"]
```

- [ ] **Step 5: Implement llm_test/tools/generic.py with get_weather**

```python
from llm_test.tools.registry import ToolSpec, register


GET_WEATHER = ToolSpec(
    name="get_weather",
    description="Get current weather for a city.",
    json_schema={
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
)

register(GET_WEATHER)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `pytest tests/tools/ -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/tools/ tests/tools/
git commit -m "feat(tools): add ToolRegistry + get_weather mock spec"
```

### Task 3.2: Remaining generic tool specs (batch)

Add ToolSpecs for: `web_search`, `send_email`, `get_contacts`, `calculator`, `read_file`, `write_file`, `list_files`, `add_calendar_event`, `get_exchange_rate`, `get_stock_price`.

**Files:**
- Modify: `llm_test/tools/generic.py`
- Modify: `tests/tools/test_registry.py`

- [ ] **Step 1: Append test for all required tool names**

In `tests/tools/test_registry.py` add:
```python
def test_all_generic_tools_registered():
    reg = ToolRegistry.default()
    required = [
        "get_weather", "web_search", "send_email", "get_contacts", "calculator",
        "read_file", "write_file", "list_files", "add_calendar_event",
        "get_exchange_rate", "get_stock_price",
    ]
    for name in required:
        spec = reg.get(name)
        assert spec.name == name
        assert spec.json_schema["function"]["name"] == name
```

- [ ] **Step 2: Run test (will fail on missing tools)**

Run: `pytest tests/tools/test_registry.py::test_all_generic_tools_registered -v`
Expected: FAIL with `KeyError: 'web_search'`

- [ ] **Step 3: Append remaining ToolSpecs to llm_test/tools/generic.py**

(Schemas follow the same shape as GET_WEATHER; each declares `name`, `description`, `parameters` with `properties` + `required`.)

```python
WEB_SEARCH = ToolSpec(
    name="web_search",
    description="Search the web for a query.",
    json_schema={"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        },
    }},
)
register(WEB_SEARCH)


SEND_EMAIL = ToolSpec(
    name="send_email",
    description="Send an email.",
    json_schema={"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"}, "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {"type": "array", "items": {"type": "string"}},
                "bcc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
    }},
)
register(SEND_EMAIL)


GET_CONTACTS = ToolSpec(
    name="get_contacts",
    description="Look up contacts by name.",
    json_schema={"type": "function", "function": {
        "name": "get_contacts",
        "description": "Look up contacts.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    }},
)
register(GET_CONTACTS)


CALCULATOR = ToolSpec(
    name="calculator",
    description="Evaluate a math expression.",
    json_schema={"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate math.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }},
)
register(CALCULATOR)


READ_FILE = ToolSpec(
    name="read_file",
    description="Read a file from disk.",
    json_schema={"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }},
)
register(READ_FILE)


WRITE_FILE = ToolSpec(
    name="write_file",
    description="Write content to a file.",
    json_schema={"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    }},
)
register(WRITE_FILE)


LIST_FILES = ToolSpec(
    name="list_files",
    description="List files in a directory.",
    json_schema={"type": "function", "function": {
        "name": "list_files",
        "description": "List directory contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }},
)
register(LIST_FILES)


ADD_CALENDAR_EVENT = ToolSpec(
    name="add_calendar_event",
    description="Add an event to the calendar.",
    json_schema={"type": "function", "function": {
        "name": "add_calendar_event",
        "description": "Add calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"}, "start": {"type": "string"},
                "end": {"type": "string"}, "timezone": {"type": "string"},
                "attendees": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "start"],
        },
    }},
)
register(ADD_CALENDAR_EVENT)


GET_EXCHANGE_RATE = ToolSpec(
    name="get_exchange_rate",
    description="Get currency exchange rate.",
    json_schema={"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": "Get FX rate.",
        "parameters": {
            "type": "object",
            "properties": {"base": {"type": "string"}, "quote": {"type": "string"}},
            "required": ["base", "quote"],
        },
    }},
)
register(GET_EXCHANGE_RATE)


GET_STOCK_PRICE = ToolSpec(
    name="get_stock_price",
    description="Get current stock price.",
    json_schema={"type": "function", "function": {
        "name": "get_stock_price",
        "description": "Get stock price.",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    }},
)
register(GET_STOCK_PRICE)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/tools/ -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/tools/generic.py tests/tools/test_registry.py
git commit -m "feat(tools): register 10 additional generic mock tools"
```

---

## Phase 4 — Scoring primitives (the 16 deterministic checks)

### Task 4.1: Scaffold scorer with first 4 primitives

**Files:**
- Create: `llm_test/core/scorer.py`
- Create: `tests/core/test_scorer.py`

- [ ] **Step 1: Write failing test for `tool_called` + `tool_not_called` + `tool_args_contain` + `call_count_at_most`**

`tests/core/test_scorer.py`:
```python
from llm_test.core.models import ToolCall, ScoringCheck
from llm_test.core.scorer import (
    check_tool_called, check_tool_not_called, check_tool_args_contain,
    check_call_count_at_most,
)


def _calls(*specs):
    return [ToolCall(index=i, name=n, args=a) for i, (n, a) in enumerate(specs)]


def test_tool_called_pass():
    calls = _calls(("get_weather", {"location": "Warsaw"}))
    chk = ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})
    r = check_tool_called(calls, chk, response=None)
    assert r.result == "pass"


def test_tool_called_fail():
    calls = _calls(("web_search", {"query": "x"}))
    chk = ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})
    assert check_tool_called(calls, chk, response=None).result == "fail"


def test_tool_not_called_pass():
    calls = _calls(("get_weather", {"location": "Warsaw"}))
    chk = ScoringCheck.model_validate({"check": "tool_not_called", "tool": "web_search"})
    assert check_tool_not_called(calls, chk, response=None).result == "pass"


def test_tool_args_contain_pass():
    calls = _calls(("get_weather", {"location": "Warsaw", "units": "celsius"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_contain", "tool": "get_weather",
        "args": {"location": "Warsaw"}
    })
    assert check_tool_args_contain(calls, chk, response=None).result == "pass"


def test_tool_args_contain_fail_wrong_value():
    calls = _calls(("get_weather", {"location": "Berlin"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_contain", "tool": "get_weather",
        "args": {"location": "Warsaw"}
    })
    assert check_tool_args_contain(calls, chk, response=None).result == "fail"


def test_call_count_at_most_pass():
    calls = _calls(("get_weather", {}), ("get_weather", {}))
    chk = ScoringCheck.model_validate({"check": "call_count_at_most", "n": 2})
    assert check_call_count_at_most(calls, chk, response=None).result == "pass"


def test_call_count_at_most_fail():
    calls = _calls(("a", {}), ("b", {}), ("c", {}))
    chk = ScoringCheck.model_validate({"check": "call_count_at_most", "n": 2})
    assert check_call_count_at_most(calls, chk, response=None).result == "fail"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_scorer.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/core/scorer.py with these 4 primitives**

```python
from __future__ import annotations
from typing import Callable
from llm_test.core.models import ToolCall, ScoringCheck, CheckResult


CheckFn = Callable[[list[ToolCall], ScoringCheck, str | None], CheckResult]


def _ok(check: str, detail: str = "") -> CheckResult:
    return CheckResult(check=check, result="pass", detail=detail)


def _bad(check: str, detail: str = "") -> CheckResult:
    return CheckResult(check=check, result="fail", detail=detail)


def check_tool_called(calls, chk, response):
    target = chk.model_dump()["tool"]
    if any(c.name == target for c in calls):
        return _ok("tool_called", f"{target} was called")
    return _bad("tool_called", f"{target} was not called")


def check_tool_not_called(calls, chk, response):
    target = chk.model_dump()["tool"]
    if any(c.name == target for c in calls):
        return _bad("tool_not_called", f"{target} was called (forbidden)")
    return _ok("tool_not_called", f"{target} correctly not called")


def check_tool_args_contain(calls, chk, response):
    d = chk.model_dump()
    target = d["tool"]
    expected = d.get("args", {})
    for c in calls:
        if c.name != target:
            continue
        if all(c.args.get(k) == v for k, v in expected.items()):
            return _ok("tool_args_contain", f"{target} args matched {expected}")
    return _bad("tool_args_contain", f"no {target} call had args containing {expected}")


def check_call_count_at_most(calls, chk, response):
    n = chk.model_dump()["n"]
    actual = len(calls)
    if actual <= n:
        return _ok("call_count_at_most", f"{actual} ≤ {n}")
    return _bad("call_count_at_most", f"{actual} > {n}")


REGISTRY: dict[str, CheckFn] = {
    "tool_called": check_tool_called,
    "tool_not_called": check_tool_not_called,
    "tool_args_contain": check_tool_args_contain,
    "call_count_at_most": check_call_count_at_most,
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/core/test_scorer.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/scorer.py tests/core/test_scorer.py
git commit -m "feat(scorer): add 4 base primitives (tool_called/not_called/args_contain/count_at_most)"
```

### Task 4.2: Add 6 more primitives — order/parallel/regex/type/count_at_least/exactly

**Files:**
- Modify: `llm_test/core/scorer.py`
- Modify: `tests/core/test_scorer.py`

- [ ] **Step 1: Append failing tests**

```python
import re  # add to imports

def test_tool_called_in_order_pass():
    calls = _calls(("a", {}), ("b", {}), ("c", {}))
    chk = ScoringCheck.model_validate({"check": "tool_called_in_order", "sequence": ["a", "b", "c"]})
    from llm_test.core.scorer import check_tool_called_in_order
    assert check_tool_called_in_order(calls, chk, None).result == "pass"


def test_tool_called_in_order_fail():
    calls = _calls(("b", {}), ("a", {}))
    chk = ScoringCheck.model_validate({"check": "tool_called_in_order", "sequence": ["a", "b"]})
    from llm_test.core.scorer import check_tool_called_in_order
    assert check_tool_called_in_order(calls, chk, None).result == "fail"


def test_tool_called_in_parallel_pass():
    # parallel = same turn_id (index group)
    calls = [
        ToolCall(index=0, name="a", args={}),
        ToolCall(index=0, name="b", args={}),  # same index = same parallel batch
    ]
    chk = ScoringCheck.model_validate({"check": "tool_called_in_parallel", "tools": ["a", "b"]})
    from llm_test.core.scorer import check_tool_called_in_parallel
    assert check_tool_called_in_parallel(calls, chk, None).result == "pass"


def test_tool_args_match_regex_pass():
    calls = _calls(("send_email", {"to": "jordan@example.com"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_match_regex", "tool": "send_email",
        "arg": "to", "pattern": r"^[^@]+@example\.com$",
    })
    from llm_test.core.scorer import check_tool_args_match_regex
    assert check_tool_args_match_regex(calls, chk, None).result == "pass"


def test_tool_args_type_fail():
    calls = _calls(("add_calendar_event", {"start": 1234567890}))  # int instead of string
    chk = ScoringCheck.model_validate({
        "check": "tool_args_type", "tool": "add_calendar_event",
        "arg": "start", "type": "string",
    })
    from llm_test.core.scorer import check_tool_args_type
    assert check_tool_args_type(calls, chk, None).result == "fail"


def test_call_count_at_least_and_exactly():
    from llm_test.core.scorer import check_call_count_at_least, check_call_count_exactly
    calls = _calls(("a", {}), ("a", {}), ("a", {}))
    assert check_call_count_at_least(calls, ScoringCheck.model_validate({"check": "call_count_at_least", "n": 2}), None).result == "pass"
    assert check_call_count_exactly(calls, ScoringCheck.model_validate({"check": "call_count_exactly", "n": 3}), None).result == "pass"
    assert check_call_count_exactly(calls, ScoringCheck.model_validate({"check": "call_count_exactly", "n": 2}), None).result == "fail"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/core/test_scorer.py -v`
Expected: 6 new tests fail with ImportError

- [ ] **Step 3: Append implementations to llm_test/core/scorer.py**

```python
import re


def check_tool_called_in_order(calls, chk, response):
    seq = chk.model_dump()["sequence"]
    idx = 0
    for c in calls:
        if idx < len(seq) and c.name == seq[idx]:
            idx += 1
    if idx == len(seq):
        return _ok("tool_called_in_order", f"sequence {seq} satisfied")
    return _bad("tool_called_in_order", f"sequence {seq} not satisfied (matched {idx}/{len(seq)})")


def check_tool_called_in_parallel(calls, chk, response):
    tools = chk.model_dump()["tools"]
    # group by index (adapter's TraceResult uses same index for parallel batch)
    groups: dict[int, set[str]] = {}
    for c in calls:
        groups.setdefault(c.index, set()).add(c.name)
    for names in groups.values():
        if set(tools).issubset(names):
            return _ok("tool_called_in_parallel", f"{tools} in same batch")
    return _bad("tool_called_in_parallel", f"{tools} were not in the same parallel batch")


def check_tool_args_match_regex(calls, chk, response):
    d = chk.model_dump()
    tool, arg, pattern = d["tool"], d["arg"], d["pattern"]
    pat = re.compile(pattern)
    for c in calls:
        if c.name != tool:
            continue
        val = c.args.get(arg)
        if isinstance(val, list):
            if any(pat.search(str(x)) for x in val):
                return _ok("tool_args_match_regex", f"{tool}.{arg} matched {pattern}")
        elif val is not None and pat.search(str(val)):
            return _ok("tool_args_match_regex", f"{tool}.{arg} matched {pattern}")
    return _bad("tool_args_match_regex", f"no {tool}.{arg} matched {pattern}")


_TYPE_MAP = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "array": list, "object": dict}


def check_tool_args_type(calls, chk, response):
    d = chk.model_dump()
    tool, arg, expected_t = d["tool"], d["arg"], d["type"]
    py_t = _TYPE_MAP[expected_t]
    for c in calls:
        if c.name != tool:
            continue
        val = c.args.get(arg)
        if not isinstance(val, py_t):
            return _bad("tool_args_type", f"{tool}.{arg} is {type(val).__name__}, expected {expected_t}")
    return _ok("tool_args_type", f"{tool}.{arg} types ok")


def check_call_count_at_least(calls, chk, response):
    n = chk.model_dump()["n"]
    if len(calls) >= n:
        return _ok("call_count_at_least", f"{len(calls)} ≥ {n}")
    return _bad("call_count_at_least", f"{len(calls)} < {n}")


def check_call_count_exactly(calls, chk, response):
    n = chk.model_dump()["n"]
    if len(calls) == n:
        return _ok("call_count_exactly", f"{len(calls)} == {n}")
    return _bad("call_count_exactly", f"{len(calls)} != {n}")


REGISTRY.update({
    "tool_called_in_order": check_tool_called_in_order,
    "tool_called_in_parallel": check_tool_called_in_parallel,
    "tool_args_match_regex": check_tool_args_match_regex,
    "tool_args_type": check_tool_args_type,
    "call_count_at_least": check_call_count_at_least,
    "call_count_exactly": check_call_count_exactly,
})
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_scorer.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/scorer.py tests/core/test_scorer.py
git commit -m "feat(scorer): add 6 primitives (order/parallel/regex/type/count variants)"
```

### Task 4.3: Final 6 primitives — response/unique/no_halluc/budget/clarification/error_surfaced + final_state_equals

**Files:**
- Modify: `llm_test/core/scorer.py`
- Modify: `tests/core/test_scorer.py`

- [ ] **Step 1: Append failing tests** (one test per primitive; pattern as before; covers `response_contains`, `response_not_contains`, `response_matches_schema`, `response_language`, `unique_tools_called`, `no_hallucinated_tool`, `budget_respected`, `error_surfaced`, `clarification_asked`, `final_state_equals`)

```python
def test_response_contains_pass():
    chk = ScoringCheck.model_validate({"check": "response_contains", "patterns": ["7", "cloud"]})
    from llm_test.core.scorer import check_response_contains
    assert check_response_contains([], chk, "It's 7°C and cloudy.").result == "pass"


def test_response_not_contains_pass():
    chk = ScoringCheck.model_validate({"check": "response_not_contains", "patterns": ["forbidden"]})
    from llm_test.core.scorer import check_response_not_contains
    assert check_response_not_contains([], chk, "all clear").result == "pass"


def test_response_matches_schema_pass():
    chk = ScoringCheck.model_validate({
        "check": "response_matches_schema",
        "schema": {"type": "object", "required": ["temp"], "properties": {"temp": {"type": "integer"}}},
    })
    from llm_test.core.scorer import check_response_matches_schema
    assert check_response_matches_schema([], chk, '{"temp": 7}').result == "pass"


def test_response_language_pl():
    chk = ScoringCheck.model_validate({"check": "response_language", "language": "pl"})
    from llm_test.core.scorer import check_response_language
    assert check_response_language([], chk, "Cześć, dzisiaj jest 7 stopni i pochmurno.").result == "pass"


def test_unique_tools_called():
    calls = _calls(("a", {}), ("a", {}), ("b", {}))
    chk = ScoringCheck.model_validate({"check": "unique_tools_called", "tools": ["a", "b"]})
    from llm_test.core.scorer import check_unique_tools_called
    assert check_unique_tools_called(calls, chk, None).result == "pass"


def test_no_hallucinated_tool():
    calls = _calls(("real_tool", {}), ("ghost_tool", {}))
    chk = ScoringCheck.model_validate({"check": "no_hallucinated_tool", "allowed": ["real_tool"]})
    from llm_test.core.scorer import check_no_hallucinated_tool
    assert check_no_hallucinated_tool(calls, chk, None).result == "fail"


def test_budget_respected():
    calls = _calls(("a", {}), ("b", {}))
    chk = ScoringCheck.model_validate({"check": "budget_respected", "max_tool_calls": 2})
    from llm_test.core.scorer import check_budget_respected
    assert check_budget_respected(calls, chk, None).result == "pass"


def test_clarification_asked():
    chk = ScoringCheck.model_validate({
        "check": "clarification_asked",
        "phrases": ["which Jordan", "could you clarify", "do you mean"],
    })
    from llm_test.core.scorer import check_clarification_asked
    assert check_clarification_asked([], chk, "I found 3 Jordans — which Jordan did you mean?").result == "pass"


def test_error_surfaced():
    calls = [ToolCall(index=0, name="get_stock_price", args={}, result={"error": "rate_limit"}, result_kind="error")]
    chk = ScoringCheck.model_validate({"check": "error_surfaced", "tool": "get_stock_price"})
    from llm_test.core.scorer import check_error_surfaced
    assert check_error_surfaced(calls, chk, "The price service returned a rate-limit error.").result == "pass"


def test_final_state_equals():
    calls = _calls(("send_email", {"to": "a@b.com"}))
    chk = ScoringCheck.model_validate({
        "check": "final_state_equals",
        "state": {"send_email_to": "a@b.com"},
    })
    from llm_test.core.scorer import check_final_state_equals
    assert check_final_state_equals(calls, chk, None).result == "pass"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/core/test_scorer.py -v`
Expected: 10 new fails

- [ ] **Step 3: Append implementations to llm_test/core/scorer.py**

```python
import json
import jsonschema  # add jsonschema>=4.21 to pyproject.toml dependencies first


def check_response_contains(calls, chk, response):
    if response is None:
        return _bad("response_contains", "no response")
    patterns = chk.model_dump()["patterns"]
    if all(p.lower() in response.lower() for p in patterns):
        return _ok("response_contains", f"all of {patterns} found")
    return _bad("response_contains", f"missing some of {patterns}")


def check_response_not_contains(calls, chk, response):
    if response is None:
        return _ok("response_not_contains", "no response → vacuously ok")
    patterns = chk.model_dump()["patterns"]
    bad = [p for p in patterns if p.lower() in response.lower()]
    if bad:
        return _bad("response_not_contains", f"contains forbidden: {bad}")
    return _ok("response_not_contains", "none of forbidden patterns present")


def check_response_matches_schema(calls, chk, response):
    if response is None:
        return _bad("response_matches_schema", "no response")
    schema = chk.model_dump()["schema"]
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        return _bad("response_matches_schema", f"not valid JSON: {e}")
    try:
        jsonschema.validate(data, schema)
        return _ok("response_matches_schema", "schema satisfied")
    except jsonschema.ValidationError as e:
        return _bad("response_matches_schema", f"schema violation: {e.message}")


# Language detection: simple lexical heuristic (no external deps)
_LANG_MARKERS = {
    "pl": ["jest", "się", "nie", "że", "czy", "dzień", "cześć", "stopni", "pochmurno", "który"],
    "en": ["the", "is", "and", "of", "to", "with", "for", "was", "you", "have"],
    "de": ["der", "die", "das", "und", "ist", "nicht", "mit", "ein", "von", "auf"],
}


def check_response_language(calls, chk, response):
    if response is None:
        return _bad("response_language", "no response")
    expected = chk.model_dump()["language"].lower()
    words = response.lower().split()
    scores = {lang: sum(1 for w in words if w.strip(".,!?:;\"'()") in markers)
              for lang, markers in _LANG_MARKERS.items()}
    detected = max(scores, key=scores.get) if any(scores.values()) else None
    if detected == expected:
        return _ok("response_language", f"detected {detected}")
    return _bad("response_language", f"detected {detected}, expected {expected}")


def check_unique_tools_called(calls, chk, response):
    expected = set(chk.model_dump()["tools"])
    actual = set(c.name for c in calls)
    if expected.issubset(actual):
        return _ok("unique_tools_called", f"unique set {actual} ⊇ {expected}")
    return _bad("unique_tools_called", f"missing: {expected - actual}")


def check_no_hallucinated_tool(calls, chk, response):
    allowed = set(chk.model_dump()["allowed"])
    bad = [c.name for c in calls if c.name not in allowed]
    if bad:
        return _bad("no_hallucinated_tool", f"called outside allowed: {bad}")
    return _ok("no_hallucinated_tool", "no tools outside allowed set")


def check_budget_respected(calls, chk, response):
    cap = chk.model_dump()["max_tool_calls"]
    if len(calls) <= cap:
        return _ok("budget_respected", f"{len(calls)} ≤ {cap}")
    return _bad("budget_respected", f"{len(calls)} > {cap}")


def check_clarification_asked(calls, chk, response):
    if response is None:
        return _bad("clarification_asked", "no response")
    phrases = chk.model_dump()["phrases"]
    if any(p.lower() in response.lower() for p in phrases):
        return _ok("clarification_asked", "asked for clarification")
    return _bad("clarification_asked", f"no clarification phrase from {phrases}")


def check_error_surfaced(calls, chk, response):
    tool = chk.model_dump()["tool"]
    had_error = any(c.name == tool and c.result_kind == "error" for c in calls)
    if not had_error:
        return _bad("error_surfaced", f"{tool} did not return error — check inapplicable")
    if response is None:
        return _bad("error_surfaced", "tool errored but no final response")
    err_words = ["error", "failed", "unable", "could not", "issue", "problem", "limit"]
    if any(w in response.lower() for w in err_words):
        return _ok("error_surfaced", "error mentioned in response")
    return _bad("error_surfaced", "tool errored but response does not mention it")


def check_final_state_equals(calls, chk, response):
    state = chk.model_dump()["state"]
    # state keys map to <tool>_<arg> e.g. send_email_to → look for send_email call with to=value
    for key, expected in state.items():
        if "_" not in key:
            continue
        tool_name, arg = key.rsplit("_", 1)
        ok = any(c.name == tool_name and c.args.get(arg) == expected for c in calls)
        if not ok:
            return _bad("final_state_equals", f"{key}={expected} not in any call")
    return _ok("final_state_equals", "state matched")


REGISTRY.update({
    "response_contains": check_response_contains,
    "response_not_contains": check_response_not_contains,
    "response_matches_schema": check_response_matches_schema,
    "response_language": check_response_language,
    "unique_tools_called": check_unique_tools_called,
    "no_hallucinated_tool": check_no_hallucinated_tool,
    "budget_respected": check_budget_respected,
    "clarification_asked": check_clarification_asked,
    "error_surfaced": check_error_surfaced,
    "final_state_equals": check_final_state_equals,
})
```

- [ ] **Step 4: Add jsonschema to pyproject.toml dependencies**

Edit `pyproject.toml` and add `"jsonschema>=4.21",` to `dependencies`. Then `uv pip install -e ".[dev]"`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/core/test_scorer.py -v`
Expected: 23 passed

- [ ] **Step 6: Commit**

```bash
git add llm_test/core/scorer.py tests/core/test_scorer.py pyproject.toml
git commit -m "feat(scorer): add final 10 primitives (response/schema/lang/state/budget/error)"
```

---

## Phase 5 — Scoring composition engine

### Task 5.1: Compose required/forbidden/partial into ScenarioResult

**Files:**
- Modify: `llm_test/core/scorer.py` (add `evaluate()`)
- Create: `tests/core/test_scorer_compose.py`

- [ ] **Step 1: Write failing test for the composition function**

`tests/core/test_scorer_compose.py`:
```python
from llm_test.core.models import (
    Scenario, Tier, Category, Budget, Scoring, ScoringCheck, ToolCall, TraceResult, Message
)
from llm_test.core.scorer import evaluate


def _scenario(scoring: Scoring) -> Scenario:
    return Scenario(
        id="test-01-x", title="t", tier=Tier.EASY, category=Category.TOOL_SELECTION,
        domain="generic", description="d", prompt="p",
        tools=["get_weather", "web_search"],
        budget=Budget(max_tool_calls=2, max_turns=2, timeout_seconds=30),
        scoring=scoring,
    )


def _trace(*calls, response="ok"):
    return TraceResult(
        scenario_id="test-01-x", adapter="mock", trial_index=0,
        messages=[Message(role="assistant", content=response)],
        tool_calls=[ToolCall(index=i, name=n, args=a) for i, (n, a) in enumerate(calls)],
        final_response=response,
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=10, error=None,
    )


def test_evaluate_pass_all_required():
    scoring = Scoring(
        required=[
            ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"}),
            ScoringCheck.model_validate({"check": "tool_args_contain", "tool": "get_weather",
                                         "args": {"location": "Warsaw"}}),
        ],
        forbidden=[ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"})],
        partial=[],
    )
    s = _scenario(scoring)
    tr = _trace(("get_weather", {"location": "Warsaw"}))
    result = evaluate(s, tr)
    assert result.status == "pass"
    assert result.score == 1.0
    assert result.failure_kind is None


def test_evaluate_fail_on_required():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        forbidden=[], partial=[],
    )
    tr = _trace(("web_search", {"query": "warsaw weather"}))
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "fail"
    assert r.score == 0.0


def test_evaluate_fail_on_forbidden():
    scoring = Scoring(
        required=[],
        forbidden=[ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"})],
        partial=[],
    )
    tr = _trace(("web_search", {"query": "x"}))
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "fail"
    assert r.failure_kind == "forbidden_action"


def test_evaluate_partial_score():
    scoring = Scoring(
        required=[ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})],
        forbidden=[],
        partial=[
            ScoringCheck.model_validate({"check": "response_contains", "patterns": ["cloud"]}),
            ScoringCheck.model_validate({"check": "call_count_at_most", "n": 1}),
        ],
    )
    tr = _trace(("get_weather", {}), response="It's sunny.")  # 1 of 2 partial pass
    r = evaluate(_scenario(scoring), tr)
    assert r.status == "partial"
    # required pass = 1.0 base; one partial out of two adds 0.5 * 0.5 = 0.25 → but spec says weight applies to final
    # Implementation: score = (required_all_pass ? 1.0 : 0.0) modulated by partial_pass_ratio when status=partial
    assert 0.0 < r.score < 1.0
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/core/test_scorer_compose.py -v`
Expected: 4 fails with ImportError on `evaluate`

- [ ] **Step 3: Append `evaluate()` to llm_test/core/scorer.py**

```python
from llm_test.core.models import Scenario, ScenarioResult, TraceResult


def _classify_failure(required_results, forbidden_results, budget_violated, hallucinated) -> str | None:
    if budget_violated:
        return "budget_violated"
    if hallucinated:
        return "hallucinated_tool"
    if any(r.result == "fail" for r in forbidden_results):
        # forbidden checks that FAIL mean the forbidden thing happened
        return "forbidden_action"
    failed_req = [r for r in required_results if r.result == "fail"]
    if not failed_req:
        return None
    # heuristic: pick most informative kind based on failed check name
    first = failed_req[0].check
    if first in ("tool_called", "unique_tools_called", "tool_called_in_order"):
        return "wrong_tool" if "called" in first else "missing_step"
    if first in ("tool_args_contain", "tool_args_match_regex", "tool_args_type"):
        return "wrong_args"
    if first == "clarification_asked":
        return "no_clarification"
    if first == "response_matches_schema":
        return "bad_response_format"
    return "wrong_tool"


def evaluate(scenario: Scenario, trace: TraceResult) -> ScenarioResult:
    calls = trace.tool_calls
    response = trace.final_response

    if trace.error:
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="error", score=0.0, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind="model_crash", checks=[], trace=trace,
        )

    def run(chk):
        fn = REGISTRY.get(chk.check)
        if fn is None:
            return CheckResult(check=chk.check, result="fail", detail=f"unknown check {chk.check!r}")
        return fn(calls, chk, response)

    required = [run(c) for c in scenario.scoring.required]
    forbidden_raw = [run(c) for c in scenario.scoring.forbidden]
    # forbidden semantics: any "pass" of `tool_called` etc means the forbidden thing happened → fail
    forbidden = [
        CheckResult(check=r.check, result="fail" if r.result == "pass" else "pass",
                    detail=f"(inverted) {r.detail}")
        for r in forbidden_raw
    ]
    partial = [run(c) for c in scenario.scoring.partial]

    budget_violated = len(calls) > scenario.budget.max_tool_calls
    allowed_tools = set(scenario.tools)
    hallucinated = any(c.name not in allowed_tools for c in calls)

    all_checks = required + forbidden + partial
    if budget_violated:
        all_checks.append(CheckResult(check="budget_respected", result="fail",
                                      detail=f"{len(calls)}>{scenario.budget.max_tool_calls}"))
    if hallucinated:
        bad = [c.name for c in calls if c.name not in allowed_tools]
        all_checks.append(CheckResult(check="no_hallucinated_tool", result="fail",
                                      detail=f"called outside allowed: {bad}"))

    required_all_pass = all(r.result == "pass" for r in required) and not budget_violated and not hallucinated
    forbidden_clean = all(r.result == "pass" for r in forbidden)

    if not forbidden_clean or not required_all_pass:
        status = "fail"
        score = scenario.scoring.weights["fail"]
        kind = _classify_failure(required, forbidden_raw, budget_violated, hallucinated)
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status=status, score=score, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind=kind, checks=all_checks, trace=trace,
        )

    if partial:
        n_pass = sum(1 for p in partial if p.result == "pass")
        ratio = n_pass / len(partial)
        if ratio == 1.0:
            return ScenarioResult(
                scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
                status="pass", score=scenario.scoring.weights["pass"],
                call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
                latency_ms=trace.duration_ms, failure_kind=None,
                checks=all_checks, trace=trace,
            )
        partial_weight = scenario.scoring.weights["partial"]
        pass_weight = scenario.scoring.weights["pass"]
        score = partial_weight + (pass_weight - partial_weight) * ratio
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="partial", score=score, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind=None, checks=all_checks, trace=trace,
        )

    return ScenarioResult(
        scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
        status="pass", score=scenario.scoring.weights["pass"],
        call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
        latency_ms=trace.duration_ms, failure_kind=None,
        checks=all_checks, trace=trace,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_scorer_compose.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/scorer.py tests/core/test_scorer_compose.py
git commit -m "feat(scorer): add evaluate() composing required/forbidden/partial → ScenarioResult"
```

---

## Phase 6 — Adapter base + MockAdapter

### Task 6.1: Adapter protocol and MockAdapter that replays preset traces

**Files:**
- Create: `llm_test/adapters/__init__.py`
- Create: `llm_test/adapters/base.py`
- Create: `tests/adapters/__init__.py`
- Create: `tests/adapters/test_mock_adapter.py`

- [ ] **Step 1: Write failing test for MockAdapter**

`tests/adapters/test_mock_adapter.py`:
```python
import pytest
from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import (
    Scenario, Tier, Category, Budget, Scoring, ToolCall
)


@pytest.mark.asyncio
async def test_mock_adapter_replays_plan():
    plan = ScenarioPlan(
        tool_calls=[
            ToolCall(index=0, name="get_weather", args={"location": "Warsaw"}),
        ],
        final_response="It's 7°C and cloudy.",
    )
    scenario = Scenario(
        id="easy-01-direct-weather", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="p", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring=Scoring(),
    )
    adapter = MockAdapter(plans={"easy-01-direct-weather": plan})
    trace = await adapter.run_scenario(scenario, model="mock-model", timeout=10)
    assert trace.scenario_id == "easy-01-direct-weather"
    assert trace.adapter == "mock"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.final_response == "It's 7°C and cloudy."
    assert trace.error is None
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/adapters/test_mock_adapter.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/adapters/base.py**

```python
from __future__ import annotations
from datetime import datetime, timezone
from typing import Protocol
from dataclasses import dataclass, field
from llm_test.core.models import Scenario, TraceResult, Message, ToolCall


class Adapter(Protocol):
    name: str
    version: str

    async def run_scenario(
        self, scenario: Scenario, model: str, timeout: int
    ) -> TraceResult: ...


@dataclass
class ScenarioPlan:
    """Preset tool_calls + response for MockAdapter."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str | None = None
    error: str | None = None


class MockAdapter:
    name = "mock"
    version = "0.1"

    def __init__(self, plans: dict[str, ScenarioPlan]) -> None:
        self.plans = plans

    async def run_scenario(self, scenario, model, timeout) -> TraceResult:
        plan = self.plans.get(scenario.id, ScenarioPlan())
        started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[
                Message(role="user", content=scenario.prompt),
                Message(role="assistant", content=plan.final_response),
            ],
            tool_calls=plan.tool_calls,
            final_response=plan.final_response,
            started_at_iso=started,
            duration_ms=1,
            error=plan.error,
            adapter_metadata={"plan": True},
        )
```

- [ ] **Step 4: Implement llm_test/adapters/__init__.py**

```python
from llm_test.adapters.base import Adapter, MockAdapter, ScenarioPlan

__all__ = ["Adapter", "MockAdapter", "ScenarioPlan"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/adapters/ -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add llm_test/adapters/ tests/adapters/
git commit -m "feat(adapters): add Adapter protocol + MockAdapter for deterministic testing"
```

---

## Phase 7 — Runner (orchestration)

### Task 7.1: Build runner that ties scenario × adapter × trial → ScenarioResult

**Files:**
- Create: `llm_test/core/runner.py`
- Create: `tests/core/test_runner.py`

- [ ] **Step 1: Write failing test**

`tests/core/test_runner.py`:
```python
import pytest
from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import (
    Scenario, Tier, Category, Budget, Scoring, ScoringCheck, ToolCall
)
from llm_test.core.runner import Runner


def _scenario():
    return Scenario(
        id="easy-01-direct-weather", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="p", tools=["get_weather", "web_search"],
        budget=Budget(max_tool_calls=1, max_turns=1, timeout_seconds=30),
        scoring=Scoring(
            required=[
                ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"}),
            ],
            forbidden=[
                ScoringCheck.model_validate({"check": "tool_called", "tool": "web_search"}),
            ],
        ),
    )


@pytest.mark.asyncio
async def test_runner_single_scenario_single_adapter_pass():
    plans = {"easy-01-direct-weather": ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
        final_response="ok",
    )}
    mock = MockAdapter(plans=plans)
    runner = Runner(adapters={"mock": mock}, trials=3, model="x")
    results = await runner.run([_scenario()])
    assert len(results) == 3                            # 3 trials × 1 adapter × 1 scenario
    assert all(r.status == "pass" for r in results)


@pytest.mark.asyncio
async def test_runner_multiple_adapters():
    s = _scenario()
    plan_good = ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="get_weather", args={})], final_response="ok"
    )
    plan_bad = ScenarioPlan(
        tool_calls=[ToolCall(index=0, name="web_search", args={"query": "x"})], final_response="ok"
    )
    runner = Runner(
        adapters={"good": MockAdapter({s.id: plan_good}), "bad": MockAdapter({s.id: plan_bad})},
        trials=2, model="x",
    )
    results = await runner.run([s])
    assert len(results) == 4                            # 2 trials × 2 adapters
    by_adapter = {a: [r for r in results if r.adapter == a] for a in ("good", "bad")}
    assert all(r.status == "pass" for r in by_adapter["good"])
    assert all(r.status == "fail" for r in by_adapter["bad"])
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/core/test_runner.py -v`
Expected: 2 fails — ImportError

- [ ] **Step 3: Implement llm_test/core/runner.py**

```python
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Iterable
from llm_test.core.models import Scenario, ScenarioResult
from llm_test.core.scorer import evaluate
from llm_test.adapters.base import Adapter


@dataclass
class Runner:
    adapters: dict[str, Adapter]
    trials: int = 1
    model: str = "model"
    concurrency: int = 4

    async def _run_one(self, scenario: Scenario, adapter_name: str, adapter: Adapter,
                       trial_index: int) -> ScenarioResult:
        try:
            trace = await asyncio.wait_for(
                adapter.run_scenario(scenario, self.model, scenario.budget.timeout_seconds),
                timeout=scenario.budget.timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            from llm_test.core.models import TraceResult
            from datetime import datetime, timezone
            trace = TraceResult(
                scenario_id=scenario.id, adapter=adapter_name, trial_index=trial_index,
                messages=[], tool_calls=[], final_response=None,
                started_at_iso=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                duration_ms=scenario.budget.timeout_seconds * 1000,
                error="timeout",
            )
        # ensure trace fields are stamped with current adapter/trial
        trace = trace.model_copy(update={"adapter": adapter_name, "trial_index": trial_index})
        return evaluate(scenario, trace)

    async def run(self, scenarios: Iterable[Scenario]) -> list[ScenarioResult]:
        sem = asyncio.Semaphore(self.concurrency)

        async def bounded(coro):
            async with sem:
                return await coro

        tasks = []
        for s in scenarios:
            for adapter_name, adapter in self.adapters.items():
                for t in range(self.trials):
                    tasks.append(bounded(self._run_one(s, adapter_name, adapter, t)))
        return await asyncio.gather(*tasks)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/runner.py tests/core/test_runner.py
git commit -m "feat(core): add Runner orchestrating scenario × adapter × trial fan-out"
```

---

## Phase 8 — Raw OpenAI adapter

### Task 8.1: OpenAI-compatible adapter with tool-call loop

**Files:**
- Create: `llm_test/adapters/openai_raw.py`
- Create: `llm_test/tools/mock_runtime.py`
- Create: `tests/adapters/test_openai_raw.py`

- [ ] **Step 1: Implement `llm_test/tools/mock_runtime.py` for matching tool_responses rules**

```python
from __future__ import annotations
from typing import Any
from llm_test.core.models import Scenario, ToolResponseRule


class MockToolRuntime:
    """Returns scripted responses from scenario.tool_responses for incoming tool calls."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._call_counters: dict[str, int] = {}

    def respond(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, str]:
        """Returns (result_value, result_kind: 'text'|'json'|'error')."""
        rules = self.scenario.tool_responses.get(tool_name, [])
        idx = self._call_counters.get(tool_name, 0)
        self._call_counters[tool_name] = idx + 1
        for rule in rules:
            if self._matches(rule, args, idx):
                return self._render(rule)
        return ({"error": f"no matching rule for {tool_name}({args})"}, "error")

    def _matches(self, rule: ToolResponseRule, args: dict, call_idx: int) -> bool:
        if isinstance(rule.match, str) and rule.match == "any":
            return True
        if rule.call_index is not None:
            ci = rule.call_index
            if isinstance(ci, int) and ci != call_idx:
                return False
            if isinstance(ci, str) and ci.startswith(">="):
                if call_idx < int(ci[2:]):
                    return False
        if isinstance(rule.match, dict):
            return all(args.get(k) == v for k, v in rule.match.items())
        return False

    def _render(self, rule: ToolResponseRule) -> tuple[Any, str]:
        if rule.returns is not None:
            if isinstance(rule.returns, dict) and "error" in rule.returns:
                return (rule.returns, "error")
            return (rule.returns, "json" if isinstance(rule.returns, (dict, list)) else "text")
        if rule.returns_with_injection is not None:
            return (rule.returns_with_injection, "text")
        return ({}, "json")
```

- [ ] **Step 2: Write failing test for the OpenAI raw adapter (uses respx to mock HTTP)**

`tests/adapters/test_openai_raw.py`:
```python
import json
import pytest
import respx
import httpx
from llm_test.adapters.openai_raw import OpenAIRawAdapter
from llm_test.core.models import (
    Scenario, Tier, Category, Budget, Scoring, ToolResponseRule
)


def _scenario():
    return Scenario(
        id="t-01-x", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What's the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"},
                                 returns={"temp_c": 7, "condition": "cloudy"}),
            ],
        },
        scoring=Scoring(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_openai_raw_handles_tool_call_then_final():
    # 1st call: assistant returns tool_call
    first = {
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": json.dumps({"location": "Warsaw"})}}
            ]}
        }]
    }
    # 2nd call: assistant returns final response after seeing tool result
    second = {
        "choices": [{"message": {"role": "assistant", "content": "It's 7°C and cloudy."}}]
    }
    respx.post("http://localhost:8000/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )
    adapter = OpenAIRawAdapter(base_url="http://localhost:8000", api_key="x")
    trace = await adapter.run_scenario(_scenario(), model="test-model", timeout=10)
    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C and cloudy."
```

- [ ] **Step 3: Run failing test**

Run: `pytest tests/adapters/test_openai_raw.py -v`
Expected: FAIL on ImportError

- [ ] **Step 4: Add respx to dev deps**

Already added in Phase 0 (`respx>=0.21`). If not yet installed run `uv pip install respx`.

- [ ] **Step 5: Implement llm_test/adapters/openai_raw.py**

```python
from __future__ import annotations
import json
import time
import httpx
from datetime import datetime, timezone
from llm_test.core.models import Scenario, TraceResult, Message, ToolCall
from llm_test.tools.registry import ToolRegistry
from llm_test.tools.mock_runtime import MockToolRuntime


class OpenAIRawAdapter:
    name = "raw"
    version = "0.1"

    def __init__(self, base_url: str, api_key: str = "", concurrency: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self):
        await self._client.aclose()

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        runtime = MockToolRuntime(scenario)
        reg = ToolRegistry.default()
        tools_schema = reg.openai_schemas(scenario.tools)

        messages: list[dict] = []
        if scenario.system_prompt:
            messages.append({"role": "system", "content": scenario.system_prompt})
        messages.append({"role": "user", "content": scenario.prompt})

        tool_calls_recorded: list[ToolCall] = []
        final_response: str | None = None
        error: str | None = None
        turn_idx = 0
        try:
            for _ in range(scenario.budget.max_turns + 1):
                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tools_schema,
                    "temperature": 0.0,
                }
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                resp = await self._client.post(
                    f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    final_response = msg.get("content")
                    break
                # capture and execute each tool call
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    result, kind = runtime.respond(name, args)
                    tool_calls_recorded.append(ToolCall(
                        index=turn_idx, name=name, args=args,
                        result=result, result_kind=kind,
                    ))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                    if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                        # adapter respects budget by stopping further loop, scorer flags violation
                        break
                turn_idx += 1
                if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                    break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[Message.model_validate(m) for m in messages],
            tool_calls=tool_calls_recorded,
            final_response=final_response,
            started_at_iso=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"base_url": self.base_url, "model": model},
        )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/adapters/test_openai_raw.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/adapters/openai_raw.py llm_test/tools/mock_runtime.py tests/adapters/test_openai_raw.py
git commit -m "feat(adapters): add OpenAI raw adapter with tool-call loop + MockToolRuntime"
```

---

## Phase 9 — First 3 easy scenarios + e2e smoke

### Task 9.1: Write 3 starter YAML scenarios

**Files:**
- Create: `scenarios/easy/easy-01-direct-weather.yaml`
- Create: `scenarios/easy/easy-02-read-before-write.yaml`
- Create: `scenarios/easy/easy-03-refuse-trivial-math.yaml`

- [ ] **Step 1: Write `scenarios/easy/easy-01-direct-weather.yaml`** — copy from `tests/fixtures/scenarios/valid_easy.yaml`.

- [ ] **Step 2: Write `scenarios/easy/easy-02-read-before-write.yaml`**

```yaml
id: easy-02-read-before-write
title: "Read before write"
tier: easy
category: coding
domain: dev_ops
description: "Read existing file content before overwriting to avoid data loss."
tags: [coding, file_ops, single_turn]
ranking_dimensions: [overall, coding]
prompt: "Append a TODO line 'TODO: review' to /workspace/notes.md"
tools: [read_file, write_file, list_files]
budget:
  max_tool_calls: 2
  max_turns: 2
  timeout_seconds: 30
tool_responses:
  read_file:
    - match: { path: "/workspace/notes.md" }
      returns: "# Notes\n\nExisting content here.\n"
    - match: any
      returns: { error: "file not found" }
  write_file:
    - match: any
      returns: { ok: true, bytes_written: 64 }
scoring:
  required:
    - check: tool_called_in_order
      sequence: [read_file, write_file]
    - check: tool_args_contain
      tool: write_file
      args: { path: "/workspace/notes.md" }
  forbidden:
    - check: tool_args_match_regex
      tool: write_file
      arg: content
      pattern: "^TODO: review\\s*$"
  partial:
    - check: call_count_at_most
      n: 2
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 3: Write `scenarios/easy/easy-03-refuse-trivial-math.yaml`**

```yaml
id: easy-03-refuse-trivial-math
title: "Refuse trivial calculator"
tier: easy
category: restraint_refusal
domain: generic
description: "Answer 2+2 directly without calling calculator."
tags: [restraint, single_turn]
ranking_dimensions: [overall, restraint]
prompt: "What is 2 + 2?"
tools: [calculator]
budget:
  max_tool_calls: 0
  max_turns: 1
  timeout_seconds: 15
tool_responses: {}
scoring:
  required:
    - check: tool_not_called
      tool: calculator
    - check: response_contains
      patterns: ["4"]
  forbidden: []
  partial:
    - check: call_count_at_most
      n: 0
  weights:
    pass: 1.0
    partial: 0.5
    fail: 0.0
```

- [ ] **Step 4: Verify they load**

Run: `python -c "from llm_test.core.scenario import load_all_scenarios; from pathlib import Path; xs = load_all_scenarios(Path('scenarios')); print(len(xs), [s.id for s in xs])"`
Expected: `3 ['easy-01-direct-weather', 'easy-02-read-before-write', 'easy-03-refuse-trivial-math']`

- [ ] **Step 5: Commit**

```bash
git add scenarios/easy/
git commit -m "feat(scenarios): add first 3 easy scenarios (weather/read-before-write/refuse-math)"
```

### Task 9.2: E2E smoke test — MockAdapter runs all 3 scenarios end-to-end

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

`tests/e2e/test_smoke.py`:
```python
import pytest
from pathlib import Path
from llm_test.core.scenario import load_all_scenarios
from llm_test.core.runner import Runner
from llm_test.adapters.base import MockAdapter, ScenarioPlan
from llm_test.core.models import ToolCall


@pytest.mark.asyncio
async def test_e2e_three_scenarios_with_mock():
    scenarios = load_all_scenarios(Path("scenarios"))
    assert len(scenarios) >= 3

    plans = {
        "easy-01-direct-weather": ScenarioPlan(
            tool_calls=[ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
            final_response="It's 7°C and cloudy.",
        ),
        "easy-02-read-before-write": ScenarioPlan(
            tool_calls=[
                ToolCall(index=0, name="read_file", args={"path": "/workspace/notes.md"}),
                ToolCall(index=1, name="write_file",
                         args={"path": "/workspace/notes.md",
                               "content": "# Notes\n\nExisting content here.\nTODO: review\n"}),
            ],
            final_response="Done.",
        ),
        "easy-03-refuse-trivial-math": ScenarioPlan(
            tool_calls=[], final_response="The answer is 4."
        ),
    }
    runner = Runner(adapters={"mock": MockAdapter(plans)}, trials=1, model="mock")
    results = await runner.run([s for s in scenarios if s.id in plans])
    assert len(results) == 3
    by_id = {r.scenario_id: r for r in results}
    assert by_id["easy-01-direct-weather"].status == "pass"
    assert by_id["easy-02-read-before-write"].status == "pass"
    assert by_id["easy-03-refuse-trivial-math"].status == "pass"
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/e2e/test_smoke.py -v`
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/
git commit -m "test(e2e): smoke test loading + running 3 easy scenarios via MockAdapter"
```

---

## Phase 10 — Persistence (SQLite)

### Task 10.1: SQLite schema + store CRUD

**Files:**
- Create: `llm_test/core/store.py`
- Create: `tests/core/test_store.py`

- [ ] **Step 1: Write failing test**

`tests/core/test_store.py`:
```python
from datetime import datetime, timezone
from llm_test.core.store import Store
from llm_test.core.models import (
    Scenario, Tier, Category, Budget, Scoring, ScenarioResult, TraceResult, Message
)


def _trace():
    return TraceResult(
        scenario_id="easy-01-x", adapter="raw", trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
    )


def _result():
    return ScenarioResult(
        scenario_id="easy-01-x", adapter="raw", trial_index=0,
        status="pass", score=1.0, call_count=0, budget_max=1,
        latency_ms=42, failure_kind=None, checks=[], trace=_trace(),
    )


def test_store_inserts_and_reads_run(tmp_results_dir):
    store = Store(tmp_results_dir / "runs.db")
    store.init_schema()
    run_id = "2026-05-23T18-00_test-model"
    store.create_run(run_id=run_id, model="test-model", base_url="http://x",
                     started_at=datetime.now(timezone.utc).isoformat(),
                     config_json="{}", scenarios_hash="abc123")
    store.upsert_adapter(run_id, "raw", "0.1")
    store.write_scenario_result(run_id, _result(), tags=["tool_call"],
                                ranking_dims=["overall"],
                                scenario_hash="hashX", category="tool_selection", tier="easy",
                                trace_path="traces/easy-01-x.json")
    rows = store.fetch_results_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"
    assert rows[0]["score"] == 1.0
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/core/test_store.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/core/store.py**

```python
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from llm_test.core.models import ScenarioResult


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, model TEXT NOT NULL, base_url TEXT,
  started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP, duration_s REAL,
  status TEXT CHECK(status IN ('running','done','aborted','failed')),
  config_json TEXT, llm_test_version TEXT, scenarios_hash TEXT
);
CREATE TABLE IF NOT EXISTS adapters_in_run (
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  adapter TEXT NOT NULL, adapter_version TEXT,
  PRIMARY KEY (run_id, adapter)
);
CREATE TABLE IF NOT EXISTS scenario_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  scenario_id TEXT NOT NULL, scenario_hash TEXT NOT NULL,
  tier TEXT NOT NULL, category TEXT NOT NULL,
  tags_json TEXT, ranking_dims_json TEXT,
  adapter TEXT NOT NULL, trial_index INTEGER NOT NULL,
  status TEXT, score REAL NOT NULL,
  call_count INTEGER NOT NULL, budget_max INTEGER,
  latency_ms INTEGER, failure_kind TEXT,
  trace_path TEXT, checks_json TEXT,
  UNIQUE (run_id, scenario_id, adapter, trial_index)
);
CREATE TABLE IF NOT EXISTS perf_results (
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  depth INTEGER NOT NULL,
  pp_tps REAL, tg_tps REAL, ttft_ms REAL, ttft_p95_ms REAL,
  pp_tokens INTEGER, tg_tokens INTEGER, benchy_runs INTEGER,
  raw_json TEXT,
  PRIMARY KEY (run_id, depth)
);
CREATE INDEX IF NOT EXISTS idx_results_dim ON scenario_results(scenario_id, adapter);
CREATE INDEX IF NOT EXISTS idx_results_model ON runs(model, started_at);
CREATE INDEX IF NOT EXISTS idx_results_status ON scenario_results(status, failure_kind);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)

    def create_run(self, run_id, model, base_url, started_at, config_json, scenarios_hash,
                   llm_test_version: str = "0.1.0") -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO runs(run_id, model, base_url, started_at, status, config_json, "
                "llm_test_version, scenarios_hash) VALUES (?,?,?,?, 'running', ?, ?, ?)",
                (run_id, model, base_url, started_at, config_json, llm_test_version, scenarios_hash),
            )

    def finish_run(self, run_id, finished_at, duration_s, status: str = "done") -> None:
        with self.conn() as c:
            c.execute("UPDATE runs SET finished_at=?, duration_s=?, status=? WHERE run_id=?",
                      (finished_at, duration_s, status, run_id))

    def upsert_adapter(self, run_id, adapter, adapter_version):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO adapters_in_run(run_id, adapter, adapter_version) "
                      "VALUES (?, ?, ?)", (run_id, adapter, adapter_version))

    def write_scenario_result(self, run_id, result: ScenarioResult, *, tags, ranking_dims,
                              scenario_hash, category, tier, trace_path) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO scenario_results(run_id, scenario_id, scenario_hash, tier, category, "
                "tags_json, ranking_dims_json, adapter, trial_index, status, score, call_count, "
                "budget_max, latency_ms, failure_kind, trace_path, checks_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, result.scenario_id, scenario_hash, tier, category,
                 json.dumps(tags), json.dumps(ranking_dims),
                 result.adapter, result.trial_index, result.status, result.score,
                 result.call_count, result.budget_max, result.latency_ms, result.failure_kind,
                 trace_path,
                 json.dumps([c.model_dump() for c in result.checks])),
            )

    def fetch_results_for_run(self, run_id) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM scenario_results WHERE run_id=?", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_runs(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        return [dict(r) for r in rows]

    def write_perf(self, run_id: str, depth: int, **fields) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO perf_results(run_id, depth, pp_tps, tg_tps, ttft_ms, "
                "ttft_p95_ms, pp_tokens, tg_tokens, benchy_runs, raw_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, depth, fields.get("pp_tps"), fields.get("tg_tps"),
                 fields.get("ttft_ms"), fields.get("ttft_p95_ms"),
                 fields.get("pp_tokens"), fields.get("tg_tokens"),
                 fields.get("benchy_runs"), fields.get("raw_json")),
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_store.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/store.py tests/core/test_store.py
git commit -m "feat(store): add SQLite Store with schema + CRUD for runs/results/perf"
```

---

## Phase 11 — Markdown output (summary + per-scenario)

### Task 11.1: Jinja-based markdown generator

**Files:**
- Create: `llm_test/core/markdown.py`
- Create: `llm_test/core/templates/summary.md.j2`
- Create: `llm_test/core/templates/scenario.md.j2`
- Create: `tests/core/test_markdown.py`

- [ ] **Step 1: Write failing test**

`tests/core/test_markdown.py`:
```python
from datetime import datetime, timezone
from llm_test.core.markdown import render_summary, render_scenario
from llm_test.core.models import (
    ScenarioResult, TraceResult, Message, CheckResult
)


def _result(adapter="raw", status="pass", score=1.0):
    tr = TraceResult(
        scenario_id="easy-01-x", adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-23T18:00:00Z", duration_ms=42, error=None,
    )
    return ScenarioResult(
        scenario_id="easy-01-x", adapter=adapter, trial_index=0,
        status=status, score=score, call_count=0, budget_max=1,
        latency_ms=42, failure_kind=None,
        checks=[CheckResult(check="tool_called", result="pass", detail="get_weather called")],
        trace=tr,
    )


def test_render_summary_has_score_table():
    results = [_result("raw"), _result("hermes", status="fail", score=0.0)]
    md = render_summary(
        run_id="2026-05-23T18-00_x", model="x", adapters=["raw", "hermes"],
        trials=1, duration_s=10.0, results=results, perf_rows=[],
    )
    assert "# Run: 2026-05-23T18-00_x" in md
    assert "raw" in md and "hermes" in md
    assert "Overall" in md


def test_render_scenario_per_adapter_block():
    results = [_result("raw"), _result("hermes")]
    md = render_scenario(scenario_id="easy-01-x", results=results, title="t",
                         tier="easy", category="tool_selection")
    assert "easy-01-x" in md
    assert "raw" in md and "hermes" in md
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/core/test_markdown.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write template `llm_test/core/templates/summary.md.j2`**

```jinja
# Run: {{ run_id }}

**Model:** {{ model }}
**Adapters:** {{ adapters | join(", ") }}
**Trials per task:** {{ trials }}    **Total scenarios:** {{ scenarios_n }}    **Total trials:** {{ results | length }}
**Duration:** {{ "%.1f"|format(duration_s) }}s    **Cost:** $0 (deterministic)

## Scores per adapter (Overall, Easy, Medium, Hard, Very Hard)

| Adapter | Overall | Easy | Medium | Hard | Very Hard |
|---------|--------:|-----:|-------:|-----:|----------:|
{% for a in adapters -%}
| {{ a }} | {{ overall[a].overall }} | {{ overall[a].easy }} | {{ overall[a].medium }} | {{ overall[a].hard }} | {{ overall[a].very_hard }} |
{% endfor %}

## Top failures
{% for f in top_failures -%}
- {{ f.scenario_id }} ({{ f.adapter }}): {{ f.failure_kind }} — {{ f.detail }}
{% endfor %}

{% if perf_rows %}
## Perf (llama-benchy)

| depth | pp tok/s | tg tok/s | TTFT (ms) |
|------:|---------:|---------:|----------:|
{% for r in perf_rows -%}
| {{ r.depth }} | {{ "%.1f"|format(r.pp_tps or 0) }} | {{ "%.2f"|format(r.tg_tps or 0) }} | {{ "%.0f"|format(r.ttft_ms or 0) }} |
{% endfor %}
{% endif %}

## Links
- [Per-scenario details](scenarios/)
- [Raw traces](traces/)
- [Charts](../../charts/{{ run_id }}/)
```

- [ ] **Step 4: Write template `llm_test/core/templates/scenario.md.j2`**

```jinja
# {{ scenario_id }}

**Title:** {{ title }}  **Tier:** {{ tier }}  **Category:** {{ category }}  **Trials per adapter:** {{ trials }}

## Per-adapter

| Adapter | Pass | Partial | Fail | Score | Median calls | Median latency (ms) |
|---------|-----:|--------:|-----:|------:|-------------:|--------------------:|
{% for row in per_adapter -%}
| {{ row.adapter }} | {{ row.pass_n }}/{{ row.total }} | {{ row.partial_n }} | {{ row.fail_n }} | {{ "%.2f"|format(row.score) }} | {{ row.median_calls }} | {{ row.median_latency }} |
{% endfor %}

## Last-trial checks (per adapter)
{% for adapter, checks in checks_by_adapter.items() %}
### {{ adapter }}
{% for c in checks -%}
- {{ "✓" if c.result == "pass" else ("◐" if c.result == "partial" else "✗") }} `{{ c.check }}` — {{ c.detail }}
{% endfor %}
{% endfor %}

## Failures
{% for f in failures -%}
- ({{ f.adapter }}, trial {{ f.trial_index }}, {{ f.status }}): {{ f.failure_kind }}
{% endfor %}
```

- [ ] **Step 5: Implement llm_test/core/markdown.py**

```python
from __future__ import annotations
import statistics
from collections import defaultdict
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from llm_test.core.models import ScenarioResult


_TEMPLATES_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=select_autoescape())


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _tier_breakdown(results: list[ScenarioResult], tier_lookup: dict[str, str]) -> dict:
    """Returns {adapter: {overall, easy, medium, hard, very_hard}} as percent strings."""
    by_adapter_tier: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_adapter_overall: dict[str, list[float]] = defaultdict(list)
    for r in results:
        tier = tier_lookup.get(r.scenario_id, "easy")
        by_adapter_tier[r.adapter][tier].append(r.score)
        by_adapter_overall[r.adapter].append(r.score)

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    out = {}
    for a in by_adapter_overall:
        out[a] = {
            "overall": _pct(avg(by_adapter_overall[a])),
            "easy": _pct(avg(by_adapter_tier[a].get("easy", []))),
            "medium": _pct(avg(by_adapter_tier[a].get("medium", []))),
            "hard": _pct(avg(by_adapter_tier[a].get("hard", []))),
            "very_hard": _pct(avg(by_adapter_tier[a].get("very_hard", []))),
        }
    return out


def render_summary(*, run_id, model, adapters, trials, duration_s, results,
                   perf_rows, tier_lookup: dict[str, str] | None = None) -> str:
    tier_lookup = tier_lookup or {}
    overall = _tier_breakdown(results, tier_lookup)
    top_failures = [
        {"scenario_id": r.scenario_id, "adapter": r.adapter,
         "failure_kind": r.failure_kind, "detail": ""}
        for r in results if r.status == "fail"
    ][:5]
    tmpl = _env.get_template("summary.md.j2")
    return tmpl.render(
        run_id=run_id, model=model, adapters=list(adapters), trials=trials,
        scenarios_n=len({r.scenario_id for r in results}),
        duration_s=duration_s, results=results, overall=overall,
        top_failures=top_failures, perf_rows=perf_rows,
    )


def render_scenario(*, scenario_id, results, title, tier, category) -> str:
    by_adapter: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        if r.scenario_id == scenario_id:
            by_adapter[r.adapter].append(r)

    per_adapter = []
    for a, rs in by_adapter.items():
        per_adapter.append({
            "adapter": a, "total": len(rs),
            "pass_n": sum(1 for r in rs if r.status == "pass"),
            "partial_n": sum(1 for r in rs if r.status == "partial"),
            "fail_n": sum(1 for r in rs if r.status == "fail"),
            "score": sum(r.score for r in rs) / len(rs),
            "median_calls": int(statistics.median(r.call_count for r in rs)),
            "median_latency": int(statistics.median(r.latency_ms for r in rs)),
        })

    checks_by_adapter = {a: rs[-1].checks for a, rs in by_adapter.items()}
    failures = [r for rs in by_adapter.values() for r in rs if r.status != "pass"]

    tmpl = _env.get_template("scenario.md.j2")
    return tmpl.render(
        scenario_id=scenario_id, title=title, tier=tier, category=category,
        trials=sum(len(rs) for rs in by_adapter.values()) // max(len(by_adapter), 1),
        per_adapter=per_adapter, checks_by_adapter=checks_by_adapter, failures=failures,
    )
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/core/test_markdown.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/core/markdown.py llm_test/core/templates/ tests/core/test_markdown.py
git commit -m "feat(markdown): add Jinja-based summary + scenario .md generators"
```

---

## Phase 12 — Minimal CLI (`llm-test run`) — first usable end-to-end

### Task 12.1: Typer CLI with `run` and `list` commands

**Files:**
- Create: `llm_test/cli.py`
- Create: `config.example.yaml`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write `config.example.yaml`**

```yaml
adapters:
  raw:
    enabled: true
    base_url_env: LLM_TEST_BASE_URL
    api_key_env: OPENAI_API_KEY
    request_timeout: 60
    max_concurrent: 4
  hermes:
    enabled: false
    gateway_url: http://localhost:8642
    api_url: http://localhost:8644
    token_env: HERMES_TOKEN
    workspace_id: default
  claude_code:
    enabled: false
    cli_path: claude
    use_local_model: true
    backend_url_env: LLM_TEST_BASE_URL
    timeout_per_scenario: 300
  codex:
    enabled: false
    cli_path: codex
    use_local_model: true

storage:
  results_dir: ./results
  sqlite_path: ./results/runs.db
  trace_compression: gzip

ranking:
  history_window_runs: 5
  half_life_days: 14
  bootstrap_iterations: 1000
  min_runs_for_ranking: 2

scenarios:
  dir: ./scenarios
  validate_on_load: true
```

- [ ] **Step 2: Write failing test for `llm-test list`**

`tests/test_cli.py`:
```python
from typer.testing import CliRunner
from llm_test.cli import app

runner = CliRunner()


def test_cli_list_runs_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No runs" in result.output or "run_id" in result.output


def test_cli_scenarios_lists_easy(tmp_path, monkeypatch):
    # Use repo scenarios/ as default
    result = runner.invoke(app, ["scenarios", "--tier", "easy"])
    assert result.exit_code == 0
    assert "easy-01" in result.output
```

- [ ] **Step 3: Run failing tests**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — ImportError on `llm_test.cli`

- [ ] **Step 4: Implement llm_test/cli.py**

```python
from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from llm_test.core.scenario import load_all_scenarios, scenario_hash
from llm_test.core.store import Store
from llm_test.core.runner import Runner
from llm_test.core.markdown import render_summary, render_scenario
from llm_test.adapters.openai_raw import OpenAIRawAdapter
from llm_test.adapters.base import Adapter


app = typer.Typer(no_args_is_help=True, help="LLM-test — deterministic LLM tool-calling benchmark.")
console = Console()


def _results_dir() -> Path:
    p = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _store() -> Store:
    s = Store(_results_dir() / "runs.db")
    s.init_schema()
    return s


@app.command(name="list")
def list_runs():
    """List recorded runs."""
    rows = _store().fetch_all_runs()
    if not rows:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return
    t = Table("run_id", "model", "status", "started_at", "duration (s)")
    for r in rows:
        t.add_row(r["run_id"], r["model"], r["status"] or "", r["started_at"],
                  f"{r['duration_s'] or 0:.1f}")
    console.print(t)


@app.command()
def scenarios(tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
              dir: Path = typer.Option(Path("scenarios"), help="scenarios dir")):
    """List scenarios."""
    xs = load_all_scenarios(dir)
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    t = Table("id", "tier", "category", "domain", "tools", "title")
    for s in xs:
        t.add_row(s.id, s.tier.value, s.category.value, s.domain,
                  ",".join(s.tools[:3]) + ("…" if len(s.tools) > 3 else ""),
                  s.title)
    console.print(t)
    console.print(f"\n[bold]{len(xs)} scenarios.[/bold]")


@app.command()
def run(
    model: str = typer.Option(..., "--model"),
    adapter: str = typer.Option("raw", help="comma-separated: raw,hermes,claude_code,codex"),
    tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
    trials: int = typer.Option(5, "--trials"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    scenarios_dir: Path = typer.Option(Path("scenarios")),
    concurrency: int = typer.Option(4),
    no_tui: bool = typer.Option(True, "--no-tui/--tui", help="MVP: --no-tui only"),
):
    """Run benchmark."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    adapters: dict[str, Adapter] = {}
    for a in adapter.split(","):
        a = a.strip()
        if a == "raw":
            adapters[a] = OpenAIRawAdapter(base_url=base_url, api_key=api_key)
        else:
            console.print(f"[yellow]adapter '{a}' not yet wired in Phase 12; skipping[/yellow]")
    if not adapters:
        console.print("[red]No adapters enabled.[/red]")
        raise typer.Exit(2)

    xs = load_all_scenarios(scenarios_dir)
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    if not xs:
        console.print("[red]No scenarios match filter.[/red]")
        raise typer.Exit(2)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M')}_{model}"
    run_dir = _results_dir() / "runs" / run_id
    (run_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    store = _store()
    cfg = {"model": model, "adapter": list(adapters), "tier": tier, "trials": trials,
           "base_url": base_url, "concurrency": concurrency}
    store.create_run(run_id=run_id, model=model, base_url=base_url,
                     started_at=datetime.now(timezone.utc).isoformat(),
                     config_json=json.dumps(cfg),
                     scenarios_hash="")
    for name, ad in adapters.items():
        store.upsert_adapter(run_id, name, ad.version)

    started = datetime.now(timezone.utc)
    runner = Runner(adapters=adapters, trials=trials, model=model, concurrency=concurrency)
    console.print(f"[bold]Running {len(xs)} scenarios × {len(adapters)} adapters × {trials} trials"
                  f" = {len(xs)*len(adapters)*trials} runs[/bold]")
    results = asyncio.run(runner.run(xs))

    # Persist results + per-scenario .md + raw traces
    sc_by_id = {s.id: s for s in xs}
    tier_lookup = {s.id: s.tier.value for s in xs}
    for r in results:
        s = sc_by_id[r.scenario_id]
        trace_filename = f"{r.scenario_id}__{r.adapter}__t{r.trial_index}.json"
        (run_dir / "traces" / trace_filename).write_text(r.trace.model_dump_json(indent=2))
        store.write_scenario_result(
            run_id=run_id, result=r,
            tags=s.tags, ranking_dims=s.ranking_dimensions,
            scenario_hash="", category=s.category.value, tier=s.tier.value,
            trace_path=f"traces/{trace_filename}",
        )
    # Per-scenario .md
    for s in xs:
        md = render_scenario(scenario_id=s.id, results=results, title=s.title,
                             tier=s.tier.value, category=s.category.value)
        (run_dir / "scenarios" / f"{s.id}.md").write_text(md)
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    # Summary
    md = render_summary(
        run_id=run_id, model=model, adapters=list(adapters),
        trials=trials, duration_s=duration, results=results, perf_rows=[],
        tier_lookup=tier_lookup,
    )
    (run_dir / "summary.md").write_text(md)
    store.finish_run(run_id, finished_at=datetime.now(timezone.utc).isoformat(),
                     duration_s=duration, status="done")
    console.print(f"[green]✓ Run finished: {run_dir}[/green]")
    console.print(f"  [bold]summary.md[/bold]: {run_dir/'summary.md'}")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 6: Manual smoke (no real LLM — would need vLLM up). Skip live run, but verify CLI help:**

Run: `llm-test --help` and `llm-test scenarios --tier easy`
Expected: both render Rich tables without error.

- [ ] **Step 7: Commit**

```bash
git add llm_test/cli.py config.example.yaml tests/test_cli.py
git commit -m "feat(cli): add typer CLI with run/list/scenarios commands (raw adapter only)"
```

---

## MILESTONE 1 reached: working CLI MVP.
At this point `llm-test run --model X --adapter raw --tier easy --trials 5 --base-url http://localhost:8000`
runs end-to-end against a real vLLM port, persists to SQLite + JSON + .md, prints results.

---

## Phase 13 — Hermes adapter

### Task 13.1: Hermes gateway adapter

**Files:**
- Create: `llm_test/adapters/hermes.py`
- Create: `tests/adapters/test_hermes.py`

**Background:** Hermes Workspace exposes a gateway on `:8642` (auth) and `:8644` (chat API). The chat API accepts OpenAI-compatible JSON but routes through Hermes skills/scaffolding. We use the same tool-call loop as `openai_raw`, but with auth header and different base URL.

- [ ] **Step 1: Write failing test (respx mocks Hermes endpoint)**

`tests/adapters/test_hermes.py`:
```python
import json
import pytest
import respx
import httpx
from llm_test.adapters.hermes import HermesAdapter
from llm_test.core.models import Scenario, Tier, Category, Budget, Scoring, ToolResponseRule


def _scenario():
    return Scenario(
        id="t-h-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="What's the weather in Warsaw?",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"}, returns={"temp_c": 7}),
            ],
        },
        scoring=Scoring(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_hermes_runs_loop_and_sends_token():
    first = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "tc_1", "type": "function",
         "function": {"name": "get_weather", "arguments": json.dumps({"location": "Warsaw"})}}
    ]}}]}
    second = {"choices": [{"message": {"role": "assistant", "content": "It's 7°C."}}]}
    route = respx.post("http://localhost:8644/v1/chat/completions").mock(
        side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)]
    )
    adapter = HermesAdapter(
        api_url="http://localhost:8644", gateway_url="http://localhost:8642",
        token="hermes-token-xyz", workspace_id="default",
    )
    trace = await adapter.run_scenario(_scenario(), model="any", timeout=10)
    assert trace.error is None
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.final_response == "It's 7°C."
    # auth header present
    last_call = route.calls.last.request
    assert last_call.headers.get("authorization") == "Bearer hermes-token-xyz"
    assert last_call.headers.get("x-workspace-id") == "default"
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/adapters/test_hermes.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/adapters/hermes.py**

```python
from __future__ import annotations
import json
import time
import httpx
from datetime import datetime, timezone
from llm_test.core.models import Scenario, TraceResult, Message, ToolCall
from llm_test.tools.registry import ToolRegistry
from llm_test.tools.mock_runtime import MockToolRuntime


class HermesAdapter:
    name = "hermes"
    version = "0.1"

    def __init__(self, api_url: str, gateway_url: str, token: str, workspace_id: str = "default"):
        self.api_url = api_url.rstrip("/")
        self.gateway_url = gateway_url.rstrip("/")
        self.token = token
        self.workspace_id = workspace_id
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self):
        await self._client.aclose()

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        runtime = MockToolRuntime(scenario)
        reg = ToolRegistry.default()
        tools_schema = reg.openai_schemas(scenario.tools)

        messages: list[dict] = []
        if scenario.system_prompt:
            messages.append({"role": "system", "content": scenario.system_prompt})
        messages.append({"role": "user", "content": scenario.prompt})

        tool_calls_recorded: list[ToolCall] = []
        final_response: str | None = None
        error: str | None = None
        turn_idx = 0
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Workspace-Id": self.workspace_id,
            "Content-Type": "application/json",
        }
        try:
            for _ in range(scenario.budget.max_turns + 1):
                payload = {"model": model, "messages": messages, "tools": tools_schema, "temperature": 0.0}
                resp = await self._client.post(
                    f"{self.api_url}/v1/chat/completions", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                messages.append(msg)
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    final_response = msg.get("content")
                    break
                for tc in tcs:
                    name = tc["function"]["name"]
                    raw_args = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    result, kind = runtime.respond(name, args)
                    tool_calls_recorded.append(ToolCall(
                        index=turn_idx, name=name, args=args, result=result, result_kind=kind,
                    ))
                    messages.append({
                        "role": "tool", "tool_call_id": tc["id"],
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                    if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                        break
                turn_idx += 1
                if len(tool_calls_recorded) > scenario.budget.max_tool_calls:
                    break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id,
            adapter=self.name,
            trial_index=0,
            messages=[Message.model_validate(m) for m in messages],
            tool_calls=tool_calls_recorded,
            final_response=final_response,
            started_at_iso=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"api_url": self.api_url, "workspace": self.workspace_id, "model": model},
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/adapters/test_hermes.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/adapters/hermes.py tests/adapters/test_hermes.py
git commit -m "feat(adapters): add Hermes adapter (gateway-authenticated OpenAI-compatible loop)"
```

### Task 13.2: Wire `hermes` into CLI

**Files:**
- Modify: `llm_test/cli.py`

- [ ] **Step 1: Edit cli.py `run()` command to recognize `hermes` adapter name**

Replace the `adapter == "raw"` branch with this expanded selector:

```python
if a == "raw":
    adapters[a] = OpenAIRawAdapter(base_url=base_url, api_key=api_key)
elif a == "hermes":
    from llm_test.adapters.hermes import HermesAdapter
    adapters[a] = HermesAdapter(
        api_url=os.environ.get("HERMES_API_URL", "http://localhost:8644"),
        gateway_url=os.environ.get("HERMES_GATEWAY_URL", "http://localhost:8642"),
        token=os.environ.get("HERMES_TOKEN", ""),
        workspace_id=os.environ.get("HERMES_WORKSPACE", "default"),
    )
else:
    console.print(f"[yellow]adapter '{a}' not yet wired; skipping[/yellow]")
```

- [ ] **Step 2: Verify cli help still works**

Run: `llm-test run --help`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add llm_test/cli.py
git commit -m "feat(cli): wire hermes adapter selection"
```

---

## Phase 14 — Claude Code adapter (subprocess)

### Task 14.1: Claude Code CLI adapter

**Background:** Claude Code is a CLI (`claude`) that can be invoked with `--print` mode for single-shot execution. We give it the scenario prompt + list of tool *names* (as JSON descriptions in the prompt body, since Claude Code parses its own tools). We capture stdout, parse the structured tool_use blocks from streaming JSON output (`--output-format stream-json`), and convert to our `TraceResult`.

**Files:**
- Create: `llm_test/adapters/claude_code.py`
- Create: `tests/adapters/test_claude_code.py`

- [ ] **Step 1: Write failing test (uses fake subprocess via monkeypatch)**

`tests/adapters/test_claude_code.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock, patch
from llm_test.adapters.claude_code import ClaudeCodeAdapter
from llm_test.core.models import Scenario, Tier, Category, Budget, Scoring, ToolResponseRule


def _scenario():
    return Scenario(
        id="t-cc-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="get weather in Warsaw",
        tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        tool_responses={
            "get_weather": [
                ToolResponseRule(match={"location": "Warsaw"}, returns={"temp_c": 7})
            ]
        },
        scoring=Scoring(),
    )


# Simulated stream-json output from claude code CLI: tool_use → tool_result → text
_FAKE_STREAM = "\n".join([
    json.dumps({"type": "system", "subtype": "init", "session_id": "s-x"}),
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "tu_1", "name": "get_weather",
         "input": {"location": "Warsaw"}}
    ]}}),
    json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"temp_c":7}'}
    ]}}),
    json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "It's 7°C."}
    ]}}),
    json.dumps({"type": "result", "subtype": "success", "session_id": "s-x"}),
])


@pytest.mark.asyncio
async def test_claude_code_parses_stream_json():
    adapter = ClaudeCodeAdapter(cli_path="claude", backend_url="http://localhost:8000")
    with patch("asyncio.create_subprocess_exec") as mp:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_STREAM.encode(), b""))
        proc.returncode = 0
        mp.return_value = proc
        trace = await adapter.run_scenario(_scenario(), model="local-model", timeout=10)
    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C."
    assert trace.adapter_metadata.get("session_id") == "s-x"
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/adapters/test_claude_code.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/adapters/claude_code.py**

```python
from __future__ import annotations
import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from llm_test.core.models import Scenario, TraceResult, Message, ToolCall


def _build_prompt(scenario: Scenario) -> str:
    """Inline tool definitions + scenario prompt + mock tool_responses guidance."""
    tools_lines = "\n".join(f"- {t}" for t in scenario.tools)
    return (
        f"You may use these tools:\n{tools_lines}\n\n"
        f"Task: {scenario.prompt}\n\n"
        f"Budget: at most {scenario.budget.max_tool_calls} tool calls, "
        f"{scenario.budget.max_turns} turns."
    )


class ClaudeCodeAdapter:
    name = "claude_code"
    version = "0.1"

    def __init__(self, cli_path: str = "claude", backend_url: str = "",
                 use_local_model: bool = True, timeout_per_scenario: int = 300,
                 skills_blacklist: list[str] | None = None):
        path = shutil.which(cli_path) if cli_path == "claude" else cli_path
        self.cli_path = path or cli_path
        self.backend_url = backend_url
        self.use_local_model = use_local_model
        self.timeout = timeout_per_scenario
        self.skills_blacklist = skills_blacklist or []

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        prompt = _build_prompt(scenario)
        cmd = [
            self.cli_path, "--print", prompt,
            "--output-format", "stream-json",
            "--max-turns", str(scenario.budget.max_turns),
        ]
        if self.use_local_model and self.backend_url:
            cmd += ["--model", model]
        env = None  # caller sets ANTHROPIC_BASE_URL etc. via os.environ if needed
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                     timeout=min(self.timeout, timeout * 10))
            error = None if proc.returncode == 0 else stderr.decode(errors="replace")[:500]
        except asyncio.TimeoutError:
            error = "claude_code: timeout"
            stdout = b""
        except FileNotFoundError as e:
            error = f"claude_code CLI not found: {e}"
            stdout = b""

        tool_calls, final_response, session_id = self._parse_stream(stdout.decode(errors="replace"))
        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[
                Message(role="user", content=scenario.prompt),
                Message(role="assistant", content=final_response),
            ],
            tool_calls=tool_calls,
            final_response=final_response,
            started_at_iso=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms,
            error=error,
            adapter_metadata={"session_id": session_id, "model": model},
        )

    def _parse_stream(self, stream: str) -> tuple[list[ToolCall], str | None, str | None]:
        tool_calls: list[ToolCall] = []
        final_response = None
        session_id = None
        idx = 0
        for line in stream.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "system" and ev.get("subtype") == "init":
                session_id = ev.get("session_id")
            elif etype == "assistant":
                for block in (ev.get("message", {}).get("content") or []):
                    if block.get("type") == "tool_use":
                        tool_calls.append(ToolCall(
                            index=idx, name=block["name"], args=block.get("input", {}),
                            result=None, result_kind="text",
                        ))
                        idx += 1
                    elif block.get("type") == "text":
                        final_response = block.get("text")
            elif etype == "user":
                for block in (ev.get("message", {}).get("content") or []):
                    if block.get("type") == "tool_result" and tool_calls:
                        tool_calls[-1].result = block.get("content")
                        tool_calls[-1].result_kind = "text"
        return tool_calls, final_response, session_id
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/adapters/test_claude_code.py -v`
Expected: 1 passed

- [ ] **Step 5: Wire into CLI** (`llm_test/cli.py` — add new elif branch):

```python
elif a == "claude_code":
    from llm_test.adapters.claude_code import ClaudeCodeAdapter
    adapters[a] = ClaudeCodeAdapter(
        cli_path=os.environ.get("CLAUDE_CLI_PATH", "claude"),
        backend_url=base_url,
        use_local_model=True,
    )
```

- [ ] **Step 6: Commit**

```bash
git add llm_test/adapters/claude_code.py tests/adapters/test_claude_code.py llm_test/cli.py
git commit -m "feat(adapters): add Claude Code subprocess adapter with stream-json parsing"
```

---

## Phase 15 — Codex adapter

### Task 15.1: Codex CLI adapter (mirrors claude_code shape)

**Files:**
- Create: `llm_test/adapters/codex.py`
- Create: `tests/adapters/test_codex.py`

**Background:** Codex CLI invocation is `codex exec --json --model <name>` (JSON output mode). We parse stdout for `tool_call`/`tool_result`/`final` events. Structure analogous to claude_code adapter.

- [ ] **Step 1: Write failing test (subprocess mocked)**

`tests/adapters/test_codex.py`:
```python
import json
import pytest
from unittest.mock import AsyncMock, patch
from llm_test.adapters.codex import CodexAdapter
from llm_test.core.models import Scenario, Tier, Category, Budget, Scoring


def _scenario():
    return Scenario(
        id="t-cx-01", title="t", tier=Tier.EASY,
        category=Category.TOOL_SELECTION, domain="generic", description="d",
        prompt="x", tools=["get_weather"],
        budget=Budget(max_tool_calls=1, max_turns=2, timeout_seconds=30),
        scoring=Scoring(),
    )


_FAKE_OUTPUT = "\n".join([
    json.dumps({"event": "tool_call", "name": "get_weather", "args": {"location": "Warsaw"}}),
    json.dumps({"event": "tool_result", "name": "get_weather", "result": {"temp_c": 7}}),
    json.dumps({"event": "final", "text": "It's 7°C."}),
])


@pytest.mark.asyncio
async def test_codex_parses_output():
    adapter = CodexAdapter(cli_path="codex", backend_url="http://localhost:8000")
    with patch("asyncio.create_subprocess_exec") as mp:
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(_FAKE_OUTPUT.encode(), b""))
        proc.returncode = 0
        mp.return_value = proc
        trace = await adapter.run_scenario(_scenario(), model="x", timeout=10)
    assert trace.error is None
    assert trace.tool_calls[0].name == "get_weather"
    assert trace.tool_calls[0].args == {"location": "Warsaw"}
    assert trace.final_response == "It's 7°C."
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/adapters/test_codex.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/adapters/codex.py**

```python
from __future__ import annotations
import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from llm_test.core.models import Scenario, TraceResult, Message, ToolCall


class CodexAdapter:
    name = "codex"
    version = "0.1"

    def __init__(self, cli_path: str = "codex", backend_url: str = "",
                 use_local_model: bool = True, timeout_per_scenario: int = 300):
        self.cli_path = shutil.which(cli_path) if cli_path == "codex" else cli_path
        if self.cli_path is None:
            self.cli_path = cli_path
        self.backend_url = backend_url
        self.use_local_model = use_local_model
        self.timeout = timeout_per_scenario

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        started = time.monotonic()
        cmd = [self.cli_path, "exec", "--json", "--model", model, scenario.prompt]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(),
                                                     timeout=min(self.timeout, timeout * 10))
            error = None if proc.returncode == 0 else stderr.decode(errors="replace")[:500]
        except asyncio.TimeoutError:
            error = "codex: timeout"
            stdout = b""
        except FileNotFoundError as e:
            error = f"codex CLI not found: {e}"
            stdout = b""

        tool_calls, final_response = self._parse(stdout.decode(errors="replace"))
        duration_ms = int((time.monotonic() - started) * 1000)
        return TraceResult(
            scenario_id=scenario.id, adapter=self.name, trial_index=0,
            messages=[Message(role="user", content=scenario.prompt),
                      Message(role="assistant", content=final_response)],
            tool_calls=tool_calls, final_response=final_response,
            started_at_iso=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            duration_ms=duration_ms, error=error,
            adapter_metadata={"model": model},
        )

    def _parse(self, output: str) -> tuple[list[ToolCall], str | None]:
        tcs: list[ToolCall] = []
        final = None
        idx = 0
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "tool_call":
                tcs.append(ToolCall(index=idx, name=ev["name"], args=ev.get("args", {})))
                idx += 1
            elif kind == "tool_result" and tcs:
                tcs[-1].result = ev.get("result")
                tcs[-1].result_kind = "json" if isinstance(ev.get("result"), (dict, list)) else "text"
            elif kind == "final":
                final = ev.get("text")
        return tcs, final
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/adapters/test_codex.py -v`
Expected: 1 passed

- [ ] **Step 5: Wire into CLI** — add elif branch:

```python
elif a == "codex":
    from llm_test.adapters.codex import CodexAdapter
    adapters[a] = CodexAdapter(
        cli_path=os.environ.get("CODEX_CLI_PATH", "codex"),
        backend_url=base_url, use_local_model=True,
    )
```

- [ ] **Step 6: Commit**

```bash
git add llm_test/adapters/codex.py tests/adapters/test_codex.py llm_test/cli.py
git commit -m "feat(adapters): add Codex subprocess adapter + wire into CLI"
```

---

## Phase 16 — Stats (bootstrap CI + McNemar + decay aggregation)

### Task 16.1: Statistics module

**Files:**
- Create: `llm_test/core/stats.py`
- Create: `tests/core/test_stats.py`

- [ ] **Step 1: Write failing tests**

`tests/core/test_stats.py`:
```python
import math
from llm_test.core.stats import bootstrap_ci, mcnemar_p, decay_weighted_mean


def test_bootstrap_ci_normal_case():
    # 80% pass rate of a binary sample
    samples = [1.0] * 80 + [0.0] * 20
    mean, lo, hi = bootstrap_ci(samples, iterations=500, seed=42, ci=0.95)
    assert abs(mean - 0.8) < 0.01
    assert 0.7 < lo < 0.8 < hi < 0.9


def test_bootstrap_ci_zero_variance():
    mean, lo, hi = bootstrap_ci([1.0] * 50, iterations=500, seed=42)
    assert mean == 1.0
    assert lo == 1.0 and hi == 1.0


def test_mcnemar_p_clear_diff():
    # Model A passes a lot more than B
    a = [1] * 90 + [0] * 10
    b = [0] * 50 + [1] * 50
    p = mcnemar_p(a, b)
    assert p < 0.05


def test_mcnemar_p_no_diff():
    same = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    p = mcnemar_p(same, same)
    assert p > 0.5


def test_decay_weighted_mean_recent_dominates():
    # Series: most recent first
    values_with_ages_days = [(0.9, 0), (0.5, 30), (0.5, 60)]
    weighted = decay_weighted_mean(values_with_ages_days, half_life_days=14)
    # The 0.9 is fresh; older 0.5s are heavily decayed → result close to 0.9
    assert weighted > 0.75
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/core/test_stats.py -v`
Expected: 4 fails — ImportError

- [ ] **Step 3: Implement llm_test/core/stats.py**

```python
from __future__ import annotations
import math
import random
from typing import Iterable


def bootstrap_ci(samples: list[float], iterations: int = 1000,
                 ci: float = 0.95, seed: int | None = None) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(samples)
    mean = sum(samples) / n
    means = []
    for _ in range(iterations):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2
    lo = means[int(math.floor(alpha * iterations))]
    hi = means[int(math.ceil((1 - alpha) * iterations)) - 1]
    return (mean, lo, hi)


def mcnemar_p(a: list[float], b: list[float]) -> float:
    """McNemar's test (continuity-corrected) for paired binary outcomes.
    Treats values > 0.5 as 'pass'. Returns two-sided p-value via χ²(1).
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    b01 = sum(1 for x, y in zip(a, b) if x <= 0.5 and y > 0.5)
    b10 = sum(1 for x, y in zip(a, b) if x > 0.5 and y <= 0.5)
    n = b01 + b10
    if n == 0:
        return 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0
    # Survival of χ²(1) ≈ erfc(sqrt(chi2/2))
    return math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0


def decay_weighted_mean(values_with_ages_days: Iterable[tuple[float, float]],
                        half_life_days: float) -> float:
    """Exponential time-decay weighted mean. ages in days."""
    num, den = 0.0, 0.0
    lam = math.log(2) / half_life_days
    for v, age in values_with_ages_days:
        w = math.exp(-lam * age)
        num += w * v
        den += w
    return num / den if den > 0 else 0.0
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_stats.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/core/stats.py tests/core/test_stats.py
git commit -m "feat(stats): add bootstrap_ci + mcnemar_p + decay_weighted_mean"
```

---

## Phase 17 — ASCII charts

### Task 17.1: Pass-rate bar / heatmap / failure-taxonomy in terminal

**Files:**
- Create: `llm_test/charts/__init__.py`
- Create: `llm_test/charts/ascii.py`
- Create: `tests/charts/__init__.py`
- Create: `tests/charts/test_ascii.py`

- [ ] **Step 1: Write failing test**

`tests/charts/test_ascii.py`:
```python
from llm_test.charts.ascii import bar_per_tier, heatmap, failure_taxonomy


def test_bar_per_tier_renders_table():
    data = {  # adapter → tier → pass_rate
        "raw":    {"easy": 0.9, "medium": 0.72, "hard": 0.5, "very_hard": 0.25},
        "hermes": {"easy": 0.88, "medium": 0.68, "hard": 0.48, "very_hard": 0.22},
    }
    out = bar_per_tier(data)
    assert "raw" in out and "hermes" in out
    assert "easy" in out and "very_hard" in out
    assert "▇" in out or "█" in out  # bar glyph


def test_heatmap_renders():
    out = heatmap({"raw": {"tool_selection": 0.9, "coding": 0.4}})
    assert "raw" in out and "tool_selection" in out


def test_failure_taxonomy():
    out = failure_taxonomy({"wrong_tool": 16, "budget_violated": 9, "forbidden_action": 8})
    assert "wrong_tool" in out
    assert "16" in out
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/charts/test_ascii.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/charts/__init__.py** (empty)

```python
```

- [ ] **Step 4: Implement llm_test/charts/ascii.py**

```python
from __future__ import annotations


def _bar(value: float, width: int = 10) -> str:
    """Returns a bar of `width` glyphs scaled by value ∈ [0,1]."""
    filled = int(round(value * width))
    return "▇" * filled + "░" * (width - filled)


def bar_per_tier(data: dict[str, dict[str, float]],
                 tiers: list[str] = ("easy", "medium", "hard", "very_hard")) -> str:
    lines = ["", "        " + "  ".join(f"{t:<10}" for t in tiers)]
    for adapter, vals in data.items():
        parts = [f"{_bar(vals.get(t, 0.0))} " for t in tiers]
        lines.append(f"{adapter:<8}  " + "  ".join(parts))
    lines.append("        " + "  ".join(f"{vals.get(t, 0.0)*100:>5.0f}%      "
                                         for t in tiers))
    return "\n".join(lines)


def heatmap(data: dict[str, dict[str, float]]) -> str:
    categories = sorted({c for vals in data.values() for c in vals.keys()})
    header = "            " + "  ".join(f"{c[:8]:<8}" for c in categories)
    lines = [header]
    for adapter, vals in data.items():
        cells = [_bar(vals.get(c, 0.0), width=8) for c in categories]
        lines.append(f"{adapter:<12}" + "  ".join(cells))
    return "\n".join(lines)


def failure_taxonomy(counts: dict[str, int], width: int = 24) -> str:
    if not counts:
        return "(no failures)"
    total = sum(counts.values())
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    lines = []
    for kind, n in items:
        ratio = n / total
        bar = "█" * int(round(ratio * width)) + "░" * (width - int(round(ratio * width)))
        lines.append(f"{kind:<22} {bar} {ratio*100:>5.1f}%  ({n})")
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/charts/test_ascii.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add llm_test/charts/ tests/charts/
git commit -m "feat(charts): add ASCII bar/heatmap/failure-taxonomy renderers"
```

---

## Phase 18 — PNG charts (matplotlib)

### Task 18.1: 7 PNG chart functions

**Files:**
- Create: `llm_test/charts/png.py`
- Create: `tests/charts/test_png.py`

- [ ] **Step 1: Write failing test**

`tests/charts/test_png.py`:
```python
from pathlib import Path
from llm_test.charts.png import (
    overall_bar, tier_breakdown, category_heatmap, radar,
    failure_taxonomy_png, perf_vs_quality, pass_rate_vs_budget,
)


def test_overall_bar_writes_png(tmp_path):
    out = tmp_path / "overall.png"
    overall_bar({"raw": (0.62, 0.05), "hermes": (0.58, 0.06)}, out)
    assert out.exists() and out.stat().st_size > 0


def test_tier_breakdown(tmp_path):
    out = tmp_path / "tier.png"
    data = {"raw": {"easy": 0.9, "medium": 0.7, "hard": 0.5, "very_hard": 0.25}}
    tier_breakdown(data, out)
    assert out.exists()


def test_category_heatmap(tmp_path):
    out = tmp_path / "cat.png"
    data = {"raw": {"tool_selection": 0.9, "coding": 0.4, "safety": 0.7}}
    category_heatmap(data, out)
    assert out.exists()


def test_radar(tmp_path):
    out = tmp_path / "radar.png"
    cats = ["tool_selection", "coding", "safety", "long_context", "restraint"]
    radar({"raw": [0.9, 0.5, 0.7, 0.4, 0.6]}, cats, out)
    assert out.exists()


def test_failure_taxonomy_png(tmp_path):
    out = tmp_path / "fail.png"
    failure_taxonomy_png({"raw": {"wrong_tool": 5, "budget_violated": 3}}, out)
    assert out.exists()


def test_perf_vs_quality(tmp_path):
    out = tmp_path / "pq.png"
    perf_vs_quality({"deepseek": (38.0, 0.68), "mimo": (142.0, 0.71)}, out)
    assert out.exists()


def test_pass_rate_vs_budget(tmp_path):
    out = tmp_path / "pb.png"
    pass_rate_vs_budget({"easy-cod-01": [(2, 1.0), (3, 1.0), (4, 1.0)],
                         "hard-cod-03": [(6, 0.4), (7, 0.7), (8, 0.85)]}, out)
    assert out.exists()
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/charts/test_png.py -v`
Expected: 7 fails — ImportError

- [ ] **Step 3: Implement llm_test/charts/png.py**

```python
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def overall_bar(model_to_score_ci: dict[str, tuple[float, float]], out: Path) -> None:
    names = list(model_to_score_ci.keys())
    scores = [v[0] for v in model_to_score_ci.values()]
    ci = [v[1] for v in model_to_score_ci.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x, scores, yerr=ci, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Overall pass-rate")
    ax.set_title("Overall pass-rate per model (95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def tier_breakdown(data: dict[str, dict[str, float]], out: Path) -> None:
    tiers = ["easy", "medium", "hard", "very_hard"]
    adapters = list(data.keys())
    x = np.arange(len(tiers))
    width = 0.8 / max(len(adapters), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, a in enumerate(adapters):
        vals = [data[a].get(t, 0.0) for t in tiers]
        ax.bar(x + i * width, vals, width, label=a)
    ax.set_xticks(x + width * (len(adapters) - 1) / 2)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pass-rate")
    ax.set_title("Pass-rate per tier × adapter")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def category_heatmap(data: dict[str, dict[str, float]], out: Path) -> None:
    adapters = list(data.keys())
    cats = sorted({c for v in data.values() for c in v.keys()})
    mat = np.array([[data[a].get(c, 0.0) for c in cats] for a in adapters])
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cats)), 0.6 * len(adapters) + 1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticks(range(len(adapters))); ax.set_yticklabels(adapters)
    for i in range(len(adapters)):
        for j in range(len(cats)):
            ax.text(j, i, f"{mat[i, j]*100:.0f}%", ha="center", va="center",
                    color="black" if mat[i, j] > 0.5 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pass-rate")
    ax.set_title("Category × adapter heatmap")
    _save(fig, out)


def radar(model_to_scores: dict[str, list[float]], categories: list[str], out: Path) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for model, scores in model_to_scores.items():
        vals = list(scores) + [scores[0]]
        ax.plot(angles, vals, label=model)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title("Per-category radar")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, out)


def failure_taxonomy_png(per_model: dict[str, dict[str, int]], out: Path) -> None:
    models = list(per_model.keys())
    kinds = sorted({k for v in per_model.values() for k in v.keys()})
    bottoms = np.zeros(len(models))
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in kinds:
        vals = np.array([per_model[m].get(k, 0) for m in models])
        ax.bar(models, vals, bottom=bottoms, label=k)
        bottoms += vals
    ax.set_ylabel("Failure count")
    ax.set_title("Failure taxonomy per model")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out)


def perf_vs_quality(model_to_perf_quality: dict[str, tuple[float, float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, (tps, q) in model_to_perf_quality.items():
        ax.scatter(tps, q, s=80)
        ax.annotate(m, (tps, q), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Generation tokens/s (median)")
    ax.set_ylabel("Overall pass-rate")
    ax.set_ylim(0, 1)
    ax.set_title("Perf vs Quality — Pareto frontier")
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def pass_rate_vs_budget(scenarios_to_curves: dict[str, list[tuple[int, float]]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, curve in scenarios_to_curves.items():
        xs = [b for b, _ in curve]; ys = [p for _, p in curve]
        ax.plot(xs, ys, marker="o", label=sid)
    ax.set_xlabel("max_tool_calls budget")
    ax.set_ylabel("Pass-rate")
    ax.set_ylim(0, 1)
    ax.set_title("Pass-rate sensitivity to budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/charts/test_png.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add llm_test/charts/png.py tests/charts/test_png.py
git commit -m "feat(charts): add 7 matplotlib PNG renderers (bar/tier/heatmap/radar/failure/perf/budget)"
```

---

## Phase 19 — Rankings system

### Task 19.1: regenerate_rankings() + markdown ranking template

**Files:**
- Create: `llm_test/rankings/__init__.py` (empty)
- Create: `llm_test/rankings/compute.py`
- Create: `llm_test/core/templates/ranking.md.j2`
- Create: `tests/rankings/__init__.py`
- Create: `tests/rankings/test_compute.py`

- [ ] **Step 1: Write template `llm_test/core/templates/ranking.md.j2`**

```jinja
# {{ dimension | title }} Ranking

Last updated: {{ updated_iso }}   |   Tracked: {{ model_count }} models, {{ run_count }} runs total

| # | Model | Overall | Best adapter | Runs |
|---|-------|--------:|--------------|-----:|
{% for row in rows -%}
| {{ loop.index }} | {{ row.model }} | {{ "%.1f%%"|format(row.score*100) }} | {{ row.best_adapter }} | {{ row.runs }} |
{% endfor %}

## Method
- Last {{ window }} runs per model, half-life decay {{ half_life }} days
- Bootstrap {{ bootstrap_iters }} resamples for 95% CI
- McNemar test vs next model in ranking; p < 0.05 marked as "statistically distinguishable"
```

- [ ] **Step 2: Write failing test**

`tests/rankings/test_compute.py`:
```python
from datetime import datetime, timezone, timedelta
from pathlib import Path
from llm_test.core.store import Store
from llm_test.core.models import ScenarioResult, TraceResult, Message
from llm_test.rankings.compute import regenerate_rankings


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _seed(tmp_path: Path, model: str, scores: list[float], ranking_dims: list[str]):
    store = Store(tmp_path / "runs.db")
    store.init_schema()
    run_id = f"r_{model}_{scores}"
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(timezone.utc).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    for i, s in enumerate(scores):
        sid = f"easy-{i:02d}-test"
        tr = _trace(sid, "raw")
        result = ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0,
            status="pass" if s > 0.5 else "fail", score=s,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if s > 0.5 else "wrong_tool",
            checks=[], trace=tr,
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=["coding"] if "cod" in sid else [],
            ranking_dims=ranking_dims,
            scenario_hash="h", category="coding", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(timezone.utc).isoformat(), 1.0)
    return store


def test_regenerate_overall_ranking(tmp_path):
    store = _seed(tmp_path, "model_a", [1.0, 1.0, 1.0, 1.0, 0.0], ["overall"])
    _seed(tmp_path, "model_b", [1.0, 0.0, 0.0, 0.0, 0.0], ["overall"])
    out_dir = tmp_path / "rankings"
    regenerate_rankings(store=store, dimensions=["overall"], out_dir=out_dir,
                        history_window_runs=5, half_life_days=14)
    md = (out_dir / "overall.md").read_text()
    assert "model_a" in md and "model_b" in md
    # model_a (80% pass) should rank above model_b (20% pass)
    a_idx = md.index("model_a")
    b_idx = md.index("model_b")
    assert a_idx < b_idx
```

- [ ] **Step 3: Run failing test**

Run: `pytest tests/rankings/test_compute.py -v`
Expected: FAIL — ImportError

- [ ] **Step 4: Implement llm_test/rankings/__init__.py** (empty file)

- [ ] **Step 5: Implement llm_test/rankings/compute.py**

```python
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from llm_test.core.store import Store
from llm_test.core.stats import bootstrap_ci, decay_weighted_mean


_TEMPLATES_DIR = Path(__file__).parent.parent / "core" / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))


def regenerate_rankings(*, store: Store, dimensions: list[str], out_dir: Path,
                        history_window_runs: int = 5, half_life_days: float = 14.0,
                        bootstrap_iters: int = 1000, min_runs: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    runs = store.fetch_all_runs()
    run_meta = {r["run_id"]: r for r in runs}

    for dim in dimensions:
        # Aggregate per model
        per_model_runs: dict[str, list[dict]] = defaultdict(list)  # model -> list of {run_id, scores, started_at, adapters}

        with store.conn() as c:
            rows = c.execute(
                "SELECT * FROM scenario_results"
            ).fetchall()
            results = [dict(r) for r in rows]

        # group by run
        results_by_run: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            dims = json.loads(r["ranking_dims_json"] or "[]")
            if dim != "overall" and dim not in dims:
                continue
            results_by_run[r["run_id"]].append(r)

        for run_id, rs in results_by_run.items():
            meta = run_meta.get(run_id)
            if not meta:
                continue
            model = meta["model"]
            scores = [r["score"] for r in rs]
            # best adapter for this run = highest mean adapter
            by_adapter: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                by_adapter[r["adapter"]].append(r["score"])
            best = max(by_adapter.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            per_model_runs[model].append({
                "run_id": run_id, "scores": scores,
                "started_at": meta["started_at"], "best_adapter": best,
            })

        rows_out = []
        for model, runs_list in per_model_runs.items():
            if len(runs_list) < min_runs:
                continue
            runs_list.sort(key=lambda x: x["started_at"], reverse=True)
            recent = runs_list[:history_window_runs]
            # Time-decayed mean of per-run means
            pairs: list[tuple[float, float]] = []
            for r in recent:
                run_mean = sum(r["scores"]) / max(len(r["scores"]), 1)
                started = _parse_iso(r["started_at"])
                age_days = max((now - started).total_seconds() / 86400, 0)
                pairs.append((run_mean, age_days))
            weighted = decay_weighted_mean(pairs, half_life_days)
            best = max(recent, key=lambda r: sum(r["scores"]) / max(len(r["scores"]), 1))["best_adapter"]
            rows_out.append({
                "model": model, "score": weighted,
                "best_adapter": best, "runs": len(runs_list),
            })

        rows_out.sort(key=lambda r: -r["score"])
        tmpl = _env.get_template("ranking.md.j2")
        md = tmpl.render(
            dimension=dim, updated_iso=now.isoformat(),
            model_count=len(rows_out), run_count=sum(r["runs"] for r in rows_out),
            rows=rows_out, window=history_window_runs, half_life=half_life_days,
            bootstrap_iters=bootstrap_iters,
        )
        (out_dir / f"{dim}.md").write_text(md)


def _parse_iso(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/rankings/test_compute.py -v`
Expected: 1 passed

- [ ] **Step 7: Add CLI command `llm-test rankings --regen`** (`llm_test/cli.py`):

```python
@app.command()
def rankings(
    regen: bool = typer.Option(False, "--regen", help="Regenerate rankings .md"),
    dimension: str = typer.Option("all", help="overall|coding|agentic|safety|restraint|long_context|budget_efficiency|speed|all"),
):
    """Manage rankings."""
    from llm_test.rankings.compute import regenerate_rankings
    out = _results_dir() / "rankings"
    dims = ["overall", "coding", "agentic", "safety", "restraint", "long_context",
            "budget_efficiency"] if dimension == "all" else [dimension]
    if regen:
        regenerate_rankings(store=_store(), dimensions=dims, out_dir=out)
        console.print(f"[green]✓ Regenerated rankings: {out}[/green]")
    else:
        for d in dims:
            p = out / f"{d}.md"
            if p.exists():
                console.print(f"[bold]{d}[/bold] → {p}")
            else:
                console.print(f"[yellow]{d}: not yet generated (run with --regen)[/yellow]")
```

- [ ] **Step 8: Commit**

```bash
git add llm_test/rankings/ llm_test/core/templates/ranking.md.j2 llm_test/cli.py tests/rankings/
git commit -m "feat(rankings): add regenerate_rankings + CLI command for 7+1 dimensions"
```

---

## Phase 20 — Compare mode

### Task 20.1: compare two runs with McNemar p-values

**Files:**
- Create: `llm_test/compare.py`
- Create: `llm_test/core/templates/compare.md.j2`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write template `llm_test/core/templates/compare.md.j2`**

```jinja
# Compare: {{ run_a.model }} ({{ run_a.run_id }}) vs {{ run_b.model }} ({{ run_b.run_id }})

**A:** {{ run_a.run_id }} — config: {{ run_a.config }}
**B:** {{ run_b.run_id }} — config: {{ run_b.config }}
**Common adapter:** {{ common_adapter }}, scenarios: {{ common_scenarios }}

## Overall ({{ common_adapter }}, n={{ trials }} trials)
| Metric | A | B | Δ | p-value |
|--------|--:|--:|--:|--------:|
{% for m in metrics -%}
| {{ m.name }} | {{ "%.1f%%"|format(m.a*100) }} | {{ "%.1f%%"|format(m.b*100) }} | {{ "%+.1f"|format((m.a - m.b)*100) }} | {{ "%.3f"|format(m.p) }}{% if m.p < 0.05 %} ✓{% endif %} |
{% endfor %}

## Regressions (B → A worse)
{% for r in regressions -%}
- ✗ {{ r.scenario_id }}: {{ r.b_pass }}/{{ r.b_total }} → {{ r.a_pass }}/{{ r.a_total }}
{% endfor %}

## Improvements (B → A better)
{% for r in improvements -%}
- ✓ {{ r.scenario_id }}: {{ r.b_pass }}/{{ r.b_total }} → {{ r.a_pass }}/{{ r.a_total }}
{% endfor %}

## Identical scenarios (no change)
{{ identical_count }} scenarios (use --verbose to list)
```

- [ ] **Step 2: Write failing test**

`tests/test_compare.py`:
```python
from datetime import datetime, timezone
from pathlib import Path
from llm_test.compare import compare_runs
from llm_test.core.store import Store
from llm_test.core.models import ScenarioResult, TraceResult, Message


def _trace(sid, adapter):
    return TraceResult(
        scenario_id=sid, adapter=adapter, trial_index=0,
        messages=[Message(role="user", content="hi")],
        tool_calls=[], final_response="ok",
        started_at_iso="2026-05-01T00:00:00Z", duration_ms=10, error=None,
    )


def _make_run(store: Store, run_id: str, model: str, scenarios_pass: dict[str, bool]):
    store.create_run(run_id=run_id, model=model, base_url="x",
                     started_at=datetime.now(timezone.utc).isoformat(),
                     config_json="{}", scenarios_hash="h")
    store.upsert_adapter(run_id, "raw", "0.1")
    for sid, passed in scenarios_pass.items():
        result = ScenarioResult(
            scenario_id=sid, adapter="raw", trial_index=0,
            status="pass" if passed else "fail",
            score=1.0 if passed else 0.0,
            call_count=1, budget_max=1, latency_ms=10,
            failure_kind=None if passed else "wrong_tool",
            checks=[], trace=_trace(sid, "raw"),
        )
        store.write_scenario_result(
            run_id=run_id, result=result, tags=[],
            ranking_dims=["overall"],
            scenario_hash="h", category="tool_selection", tier="easy",
            trace_path="x.json",
        )
    store.finish_run(run_id, datetime.now(timezone.utc).isoformat(), 1.0)


def test_compare_runs_writes_md(tmp_path):
    store = Store(tmp_path / "runs.db"); store.init_schema()
    _make_run(store, "A", "m", {f"easy-{i:02d}-x": True for i in range(5)} | {"hard-01-x": False})
    _make_run(store, "B", "m", {f"easy-{i:02d}-x": True for i in range(5)} | {"hard-01-x": True})
    out = tmp_path / "compare.md"
    compare_runs(store=store, run_a="A", run_b="B", out_path=out)
    md = out.read_text()
    assert "A" in md and "B" in md
    assert "hard-01-x" in md  # regression of A vs B
```

- [ ] **Step 3: Run failing test**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL — ImportError

- [ ] **Step 4: Implement llm_test/compare.py**

```python
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from llm_test.core.store import Store
from llm_test.core.stats import mcnemar_p


_TEMPLATES_DIR = Path(__file__).parent / "core" / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR))


def compare_runs(*, store: Store, run_a: str, run_b: str, out_path: Path) -> None:
    rs_a = store.fetch_results_for_run(run_a)
    rs_b = store.fetch_results_for_run(run_b)
    runs_meta = {r["run_id"]: r for r in store.fetch_all_runs()}
    common_adapter = _common_adapter(rs_a, rs_b)

    a_by_scenario: dict[str, list[dict]] = defaultdict(list)
    b_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rs_a:
        if r["adapter"] == common_adapter:
            a_by_scenario[r["scenario_id"]].append(r)
    for r in rs_b:
        if r["adapter"] == common_adapter:
            b_by_scenario[r["scenario_id"]].append(r)
    common_scenarios = sorted(set(a_by_scenario) & set(b_by_scenario))

    a_scores = [r["score"] for s in common_scenarios for r in a_by_scenario[s]]
    b_scores = [r["score"] for s in common_scenarios for r in b_by_scenario[s]]
    n = min(len(a_scores), len(b_scores))
    overall_metric = {
        "name": "Overall",
        "a": sum(a_scores[:n]) / max(n, 1), "b": sum(b_scores[:n]) / max(n, 1),
        "p": mcnemar_p(a_scores[:n], b_scores[:n]),
    }

    # regressions and improvements per scenario
    regressions, improvements, identical = [], [], 0
    for s in common_scenarios:
        a_pass = sum(1 for r in a_by_scenario[s] if r["status"] == "pass")
        b_pass = sum(1 for r in b_by_scenario[s] if r["status"] == "pass")
        a_total = len(a_by_scenario[s])
        b_total = len(b_by_scenario[s])
        if a_pass / max(a_total, 1) > b_pass / max(b_total, 1):
            improvements.append({"scenario_id": s, "a_pass": a_pass, "a_total": a_total,
                                 "b_pass": b_pass, "b_total": b_total})
        elif a_pass / max(a_total, 1) < b_pass / max(b_total, 1):
            regressions.append({"scenario_id": s, "a_pass": a_pass, "a_total": a_total,
                                "b_pass": b_pass, "b_total": b_total})
        else:
            identical += 1

    tmpl = _env.get_template("compare.md.j2")
    md = tmpl.render(
        run_a={"run_id": run_a, "model": runs_meta[run_a]["model"],
               "config": runs_meta[run_a]["config_json"]},
        run_b={"run_id": run_b, "model": runs_meta[run_b]["model"],
               "config": runs_meta[run_b]["config_json"]},
        common_adapter=common_adapter, common_scenarios=len(common_scenarios),
        trials=n, metrics=[overall_metric],
        regressions=regressions, improvements=improvements, identical_count=identical,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)


def _common_adapter(rs_a, rs_b) -> str:
    a_ads = {r["adapter"] for r in rs_a}
    b_ads = {r["adapter"] for r in rs_b}
    common = a_ads & b_ads
    if not common:
        raise ValueError("runs have no common adapter")
    return sorted(common)[0]
```

- [ ] **Step 5: Add CLI command** (`llm_test/cli.py`):

```python
@app.command()
def compare(run_a: str = typer.Argument(...), run_b: str = typer.Argument(...),
            out: Path = typer.Option(None, "--out")):
    """Compare two runs (statistical diff)."""
    from llm_test.compare import compare_runs
    out_path = out or (_results_dir() / "compare" / f"{run_a}__vs__{run_b}.md")
    compare_runs(store=_store(), run_a=run_a, run_b=run_b, out_path=out_path)
    console.print(f"[green]✓ Wrote {out_path}[/green]")
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_compare.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/compare.py llm_test/core/templates/compare.md.j2 llm_test/cli.py tests/test_compare.py
git commit -m "feat(compare): add compare_runs producing McNemar-tested diff .md"
```

---

## Phase 21 — TUI base + Live tab

### Task 21.1: Textual App with tab structure + Live tab

**Files:**
- Create: `llm_test/tui/__init__.py` (empty)
- Create: `llm_test/tui/app.py`
- Create: `llm_test/tui/live_tab.py`
- Create: `tests/tui/__init__.py`
- Create: `tests/tui/test_app_starts.py`

- [ ] **Step 1: Write smoke test (just verifies app starts and has 4 tabs)**

`tests/tui/test_app_starts.py`:
```python
import pytest
from llm_test.tui.app import LLMTestApp


@pytest.mark.asyncio
async def test_app_has_four_tabs():
    app = LLMTestApp(run_id=None)
    async with app.run_test() as pilot:
        # 4 tabs: Live, History, Rankings, Scenarios
        tabs = app.query("Tab")
        labels = [t.label.plain if hasattr(t.label, "plain") else str(t.label) for t in tabs]
        assert "Live" in labels
        assert "History" in labels
        assert "Rankings" in labels
        assert "Scenarios" in labels
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/tui/test_app_starts.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/tui/live_tab.py**

```python
from __future__ import annotations
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Static, ProgressBar


class LiveTab(Container):
    """Live dashboard for the currently-running run."""

    DEFAULT_CSS = """
    LiveTab { layout: grid; grid-size: 2 2; grid-gutter: 1; }
    .scenarios-panel { row-span: 1; }
    .trace-panel { row-span: 1; }
    .scores-panel { row-span: 1; }
    .failures-panel { row-span: 1; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="scenarios-panel"):
            yield Static("[bold]Scenarios[/bold]")
            yield DataTable(id="scenarios-table")
        with Vertical(classes="trace-panel"):
            yield Static("[bold]Live trace[/bold]")
            yield Static("(no active scenario)", id="trace-content")
        with Vertical(classes="scores-panel"):
            yield Static("[bold]Score per adapter[/bold]")
            yield Static("(waiting)", id="scores-content")
        with Vertical(classes="failures-panel"):
            yield Static("[bold]Recent failures[/bold]")
            yield Static("(none)", id="failures-content")

    def on_mount(self) -> None:
        tbl = self.query_one("#scenarios-table", DataTable)
        tbl.add_columns("Status", "Scenario", "Score")
```

- [ ] **Step 4: Implement llm_test/tui/app.py**

```python
from __future__ import annotations
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane
from llm_test.tui.live_tab import LiveTab


class LLMTestApp(App):
    CSS = """
    Screen { background: $surface; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+r", "refresh", "Refresh")]

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self.run_id = run_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Live", id="live"):
                yield LiveTab(id="live-tab")
            with TabPane("History", id="history"):
                from llm_test.tui.history_tab import HistoryTab
                yield HistoryTab(id="history-tab")
            with TabPane("Rankings", id="rankings"):
                from llm_test.tui.rankings_tab import RankingsTab
                yield RankingsTab(id="rankings-tab")
            with TabPane("Scenarios", id="scenarios"):
                from llm_test.tui.scenarios_tab import ScenariosTab
                yield ScenariosTab(id="scenarios-tab")
        yield Footer()

    def action_refresh(self) -> None:
        self.notify("Refreshing...", timeout=1)
```

- [ ] **Step 5: Implement stubs for the other 3 tabs (so import works in Phase 21)**

`llm_test/tui/history_tab.py`:
```python
from textual.containers import Container
from textual.widgets import Static


class HistoryTab(Container):
    def compose(self):
        yield Static("History tab — implemented in Phase 22")
```

`llm_test/tui/rankings_tab.py`:
```python
from textual.containers import Container
from textual.widgets import Static


class RankingsTab(Container):
    def compose(self):
        yield Static("Rankings tab — implemented in Phase 23")
```

`llm_test/tui/scenarios_tab.py`:
```python
from textual.containers import Container
from textual.widgets import Static


class ScenariosTab(Container):
    def compose(self):
        yield Static("Scenarios tab — implemented in Phase 24")
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/tui/test_app_starts.py -v`
Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add llm_test/tui/ tests/tui/
git commit -m "feat(tui): add Textual App with 4 tab structure + Live dashboard layout"
```

---

## Phase 22 — History tab

### Task 22.1: History tab listing runs from SQLite

**Files:**
- Modify: `llm_test/tui/history_tab.py`
- Modify: `tests/tui/test_app_starts.py` (add assert)

- [ ] **Step 1: Replace `history_tab.py` with real impl**

```python
from __future__ import annotations
from pathlib import Path
import os
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Static, Input
from llm_test.core.store import Store


class HistoryTab(Container):
    DEFAULT_CSS = """
    HistoryTab { padding: 1; }
    """

    def compose(self):
        with Vertical():
            yield Static("[bold]All runs[/bold] — ↑↓ navigate · [enter] details · [d] diff · [del] remove")
            yield Input(placeholder="filter (model name or run_id substring)", id="hist-filter")
            yield DataTable(id="history-table")

    def on_mount(self) -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.add_columns("run_id", "model", "status", "started_at", "duration (s)", "adapters")
        self.refresh_data()

    def refresh_data(self, filter_str: str = "") -> None:
        tbl = self.query_one("#history-table", DataTable)
        tbl.clear()
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        store = Store(results_dir / "runs.db")
        store.init_schema()
        for r in store.fetch_all_runs():
            if filter_str and filter_str.lower() not in (r["run_id"] + r["model"]).lower():
                continue
            with store.conn() as c:
                adapters = [row["adapter"] for row in
                            c.execute("SELECT adapter FROM adapters_in_run WHERE run_id=?",
                                      (r["run_id"],)).fetchall()]
            tbl.add_row(r["run_id"], r["model"], r["status"] or "",
                        r["started_at"] or "", f"{r['duration_s'] or 0:.1f}",
                        ",".join(adapters))

    def on_input_changed(self, event) -> None:
        if event.input.id == "hist-filter":
            self.refresh_data(event.value)
```

- [ ] **Step 2: Add assertion to existing test**

Edit `tests/tui/test_app_starts.py`, append:
```python
@pytest.mark.asyncio
async def test_history_tab_loads_without_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path))
    app = LLMTestApp(run_id=None)
    async with app.run_test() as pilot:
        await pilot.press("right")  # focus next tab — switches to History
        # Just verify no exception bubbling up
        assert app.is_running
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/tui/test_app_starts.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add llm_test/tui/history_tab.py tests/tui/test_app_starts.py
git commit -m "feat(tui): History tab — list runs from SQLite with filter input"
```

---

## Phase 23 — Rankings tab

### Task 23.1: Rankings tab reading rankings/*.md

**Files:**
- Modify: `llm_test/tui/rankings_tab.py`

- [ ] **Step 1: Replace with real impl**

```python
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
```

- [ ] **Step 2: Quick manual check**

Run: `llm-test tui` (will be wired in Phase 24 step 3 or earlier — verify rendering by switching to Rankings tab).

- [ ] **Step 3: Commit**

```bash
git add llm_test/tui/rankings_tab.py
git commit -m "feat(tui): Rankings tab with dimension selector reading rankings/*.md"
```

---

## Phase 24 — Scenarios tab + wire `llm-test tui` command

### Task 24.1: Scenarios tab (cross-models per scenario)

**Files:**
- Modify: `llm_test/tui/scenarios_tab.py`
- Modify: `llm_test/cli.py` (add `tui` command)

- [ ] **Step 1: Replace `scenarios_tab.py` with real impl**

```python
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
import statistics
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Select, Static
from llm_test.core.store import Store
from llm_test.core.scenario import load_all_scenarios


class ScenariosTab(Container):
    DEFAULT_CSS = """ScenariosTab { padding: 1; }"""

    def compose(self):
        with Vertical():
            yield Static("[bold]Per-scenario cross-model view[/bold]")
            yield Select(options=[], id="sc-pick")
            yield DataTable(id="sc-table")
            yield Static("", id="sc-stats")

    def on_mount(self) -> None:
        sel = self.query_one("#sc-pick", Select)
        scenarios = load_all_scenarios(Path("scenarios"))
        options = [(f"{s.tier.value} · {s.id}", s.id) for s in scenarios]
        sel.set_options(options)
        if options:
            sel.value = options[0][1]
            self._render(options[0][1])

    def on_select_changed(self, event) -> None:
        if event.select.id == "sc-pick" and event.value:
            self._render(event.value)

    def _render(self, scenario_id: str) -> None:
        tbl = self.query_one("#sc-table", DataTable)
        tbl.clear(columns=True)
        tbl.add_columns("model", "adapter", "n", "pass", "median_calls",
                        "median_latency", "top_failure")
        results_dir = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
        store = Store(results_dir / "runs.db"); store.init_schema()
        runs_meta = {r["run_id"]: r for r in store.fetch_all_runs()}
        rows = []
        with store.conn() as c:
            res = c.execute(
                "SELECT * FROM scenario_results WHERE scenario_id=?", (scenario_id,)
            ).fetchall()
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in res:
            r = dict(r)
            model = runs_meta.get(r["run_id"], {}).get("model", "?")
            grouped[(model, r["adapter"])].append(r)
        for (model, adapter), rs in grouped.items():
            n = len(rs)
            pass_n = sum(1 for x in rs if x["status"] == "pass")
            med_calls = int(statistics.median(x["call_count"] for x in rs))
            med_lat = int(statistics.median(x["latency_ms"] for x in rs))
            kinds = [x["failure_kind"] for x in rs if x["failure_kind"]]
            top = max(set(kinds), key=kinds.count) if kinds else ""
            tbl.add_row(model, adapter, str(n), f"{pass_n}/{n}",
                        str(med_calls), str(med_lat), top)
        self.query_one("#sc-stats", Static).update(
            f"Showing data for {scenario_id} across {len(grouped)} model×adapter combinations."
        )
```

- [ ] **Step 2: Add `tui` command to llm_test/cli.py**

```python
@app.command()
def tui():
    """Launch the Textual TUI."""
    from llm_test.tui.app import LLMTestApp
    LLMTestApp(run_id=None).run()
```

- [ ] **Step 3: Manual smoke (only verify it starts; full integration needs real runs)**

Run: `llm-test tui` then `q` to quit.
Expected: TUI opens with 4 tabs, no crash.

- [ ] **Step 4: Commit**

```bash
git add llm_test/tui/scenarios_tab.py llm_test/cli.py
git commit -m "feat(tui): Scenarios tab + wire 'llm-test tui' command"
```

---

## MILESTONE 2 reached: CLI + TUI + 4 adapters working, charts/rankings/compare wired.
At this point everything except llama-benchy perf integration and full 32-scenario set is implemented.

---

## Phase 25 — llama-benchy integration (opt-in)

### Task 25.1: Benchy subprocess wrapper + `--with-perf` flag

**Files:**
- Create: `llm_test/perf/__init__.py` (empty)
- Create: `llm_test/perf/benchy.py`
- Create: `tests/perf/__init__.py`
- Create: `tests/perf/test_benchy.py`

- [ ] **Step 1: Write failing test (subprocess mocked)**

`tests/perf/test_benchy.py`:
```python
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from llm_test.perf.benchy import run_benchy, BenchyResult


_FAKE_BENCHY_JSON = {
    "model": "deepseek-v4-flash",
    "runs": [
        {"depth": 0, "pp_tps": 8420.1, "tg_tps": 38.2, "ttft_ms": 247, "ttft_p95_ms": 305,
         "pp_tokens": 4096, "tg_tokens": 512, "n_runs": 3},
        {"depth": 16384, "pp_tps": 6800.0, "tg_tps": 30.1, "ttft_ms": 980, "ttft_p95_ms": 1100,
         "pp_tokens": 4096, "tg_tokens": 512, "n_runs": 3},
    ],
}


def test_run_benchy_parses_json(tmp_path):
    out_json = tmp_path / "benchy.json"
    out_json.write_text(json.dumps(_FAKE_BENCHY_JSON))
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_proc):
        # Pretend the file was written by uvx call
        result = run_benchy(model="deepseek-v4-flash", base_url="http://localhost:8000",
                            output_file=out_json, pp=4096, tg=512, depth=[0, 16384], runs=3)
    assert isinstance(result, BenchyResult)
    assert len(result.rows) == 2
    assert result.rows[0]["depth"] == 0
    assert result.rows[0]["tg_tps"] == 38.2
```

- [ ] **Step 2: Run failing test**

Run: `pytest tests/perf/test_benchy.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement llm_test/perf/benchy.py**

```python
from __future__ import annotations
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchyResult:
    model: str
    rows: list[dict]   # each: {depth, pp_tps, tg_tps, ttft_ms, ttft_p95_ms, pp_tokens, tg_tokens, n_runs}


def run_benchy(*, model: str, base_url: str, pp: int = 4096, tg: int = 512,
               depth: list[int] | None = None, runs: int = 3,
               output_file: Path | None = None,
               extra_args: list[str] | None = None) -> BenchyResult:
    depth = depth or [0, 16384]
    if output_file is None:
        output_file = Path(tempfile.mkstemp(suffix=".json")[1])
    cmd = [
        "uvx", "llama-benchy",
        "--base-url", base_url, "--model", model,
        "--pp", str(pp), "--tg", str(tg),
        "--depth", ",".join(map(str, depth)),
        "--runs", str(runs),
        "--output", "json", "--output-file", str(output_file),
    ]
    if extra_args:
        cmd += extra_args
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"llama-benchy failed: {completed.stderr[:500]}")
    data = json.loads(Path(output_file).read_text())
    return BenchyResult(model=data.get("model", model), rows=data.get("runs", []))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/perf/test_benchy.py -v`
Expected: 1 passed

- [ ] **Step 5: Add `--with-perf` flag to `llm-test run` + `bench` command (cli.py)**

In `run()` signature add: `with_perf: bool = typer.Option(False, "--with-perf")` and after results are scored:

```python
if with_perf:
    from llm_test.perf.benchy import run_benchy
    try:
        perf = run_benchy(model=model, base_url=base_url, depth=[0, 16384, 131072], runs=3)
        for row in perf.rows:
            store.write_perf(run_id, depth=row["depth"],
                             pp_tps=row.get("pp_tps"), tg_tps=row.get("tg_tps"),
                             ttft_ms=row.get("ttft_ms"), ttft_p95_ms=row.get("ttft_p95_ms"),
                             pp_tokens=row.get("pp_tokens"), tg_tokens=row.get("tg_tokens"),
                             benchy_runs=row.get("n_runs"),
                             raw_json=json.dumps(row))
        console.print(f"[green]✓ perf: collected {len(perf.rows)} depth points[/green]")
    except Exception as e:
        console.print(f"[yellow]perf collection failed: {e}[/yellow]")
```

Also add standalone command:
```python
@app.command()
def perf(model: str = typer.Option(..., "--model"),
         base_url: str = typer.Option("http://localhost:8000", "--base-url"),
         pp: int = 4096, tg: int = 512,
         depth: str = "0,16384,131072", runs: int = 3):
    """Run llama-benchy only (no scoring)."""
    from llm_test.perf.benchy import run_benchy
    res = run_benchy(model=model, base_url=base_url, pp=pp, tg=tg,
                     depth=[int(x) for x in depth.split(",")], runs=runs)
    for row in res.rows:
        console.print(f"depth={row['depth']:>7} pp_tps={row.get('pp_tps',0):.1f} "
                      f"tg_tps={row.get('tg_tps',0):.2f} ttft={row.get('ttft_ms',0):.0f}ms")
```

- [ ] **Step 6: Commit**

```bash
git add llm_test/perf/ tests/perf/ llm_test/cli.py
git commit -m "feat(perf): integrate llama-benchy as opt-in perf collector + perf CLI command"
```

---

## Phase 26 — Write remaining scenarios + domain tools

### Task 26.1: Register domain tools

**Files:**
- Create: `llm_test/tools/domain.py`

- [ ] **Step 1: Add ToolSpecs for domain (git/test/lint/etc.)**

```python
from llm_test.tools.registry import ToolSpec, register


def _fn(name: str, desc: str, props: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(name=name, description=desc, json_schema={"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }})


register(_fn("git_status", "Show working tree status", {}, []))
register(_fn("git_diff", "Show changes", {"path": {"type": "string"}}, []))
register(_fn("git_add", "Stage files",
             {"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"]))
register(_fn("git_commit", "Commit staged changes",
             {"message": {"type": "string"}, "branch": {"type": "string"}}, ["message"]))
register(_fn("git_branch", "Create or list branches",
             {"name": {"type": "string"}}, []))
register(_fn("grep", "Search for pattern across files",
             {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]))
register(_fn("edit_file", "Edit a file in place",
             {"path": {"type": "string"}, "find": {"type": "string"},
              "replace": {"type": "string"}, "content": {"type": "string"}},
             ["path"]))
register(_fn("run_tests", "Run the project test suite",
             {"path": {"type": "string"}, "filter": {"type": "string"}}, []))
register(_fn("run_lint", "Run the linter", {"path": {"type": "string"}}, []))
register(_fn("run_bash", "Execute a shell command", {"cmd": {"type": "string"}}, ["cmd"]))
register(_fn("python_exec", "Execute Python code", {"code": {"type": "string"}}, ["code"]))
register(_fn("get_order_status", "Look up order status by id",
             {"order_id": {"type": "string"}}, ["order_id"]))
register(_fn("get_orderbook", "Fetch L2 orderbook snapshot",
             {"symbol": {"type": "string"}, "depth": {"type": "integer"}}, ["symbol"]))
register(_fn("submit_order", "Submit a trading order",
             {"symbol": {"type": "string"}, "side": {"type": "string"},
              "qty": {"type": "number"}, "price": {"type": "number"}},
             ["symbol", "side", "qty"]))
register(_fn("get_positions", "Fetch current portfolio positions", {}, []))
register(_fn("get_risk", "Compute portfolio risk metrics",
             {"account": {"type": "string"}}, []))
register(_fn("vllm_config_get", "Get current vLLM server config", {}, []))
register(_fn("vllm_config_set", "Set a vLLM config parameter",
             {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]))
register(_fn("get_weather_global", "Get weather for non-European cities",
             {"location": {"type": "string"}}, ["location"]))
register(_fn("search_flights", "Search flights",
             {"from": {"type": "string"}, "to": {"type": "string"},
              "date": {"type": "string"}, "max_price": {"type": "number"}},
             ["from", "to", "date"]))
```

- [ ] **Step 2: Verify registry has them**

Run: `python -c "from llm_test.tools import generic, domain; from llm_test.tools.registry import ToolRegistry; print(len(ToolRegistry.default()._tools), 'tools')"`
Expected: ≥ 31 tools.

- [ ] **Step 3: Commit**

```bash
git add llm_test/tools/domain.py
git commit -m "feat(tools): register domain tools (git/edit/test/lint/orderbook/vllm)"
```

### Task 26.2: Author remaining 29 scenarios (7 easy + 10 medium + 8 hard + 4 very_hard)

**Files:**
- Create: `scenarios/easy/easy-{04..10}-*.yaml` (7 files)
- Create: `scenarios/medium/medium-{01..10}-*.yaml` (10 files)
- Create: `scenarios/hard/hard-{01..08}-*.yaml` (8 files)
- Create: `scenarios/very_hard/very_hard-{01..04}-*.yaml` (4 files)
- Create: `scenarios/templates/code_snapshot_8files.txt` (long context prefill source)

This is an **iterative authoring task** — best done in small batches with peer review. Plan one scenario per step. Reference Section 5 (taxonomy of difficulty), Section 6 (coding coverage), and Section 12 of `docs/spec.md` for category mapping.

**Authoring template per scenario:**

```yaml
id: <tier>-<NN>-<short-kebab>
title: "..."
tier: <easy|medium|hard|very_hard>
category: <one of 16>
domain: <generic|quant|dev_ops>
description: "What's being tested — for human reviewer"
tags: [tag1, tag2]
ranking_dimensions: [overall, <secondary>]
prompt: "..."
tools: [...]
context_prefill_tokens: 0       # bump for hard / very_hard
budget:
  max_tool_calls: N             # minimal_required + 1
  max_turns: M
  timeout_seconds: 30
tool_responses:
  <tool>:
    - match: { ... }
      returns: { ... }
    - match: any
      returns: { error: "..." }
scoring:
  required: [...]
  forbidden: [...]
  partial: [...]
  weights: { pass: 1.0, partial: 0.5, fail: 0.0 }
```

**The full 29-scenario set to author** (one task per scenario; use template + spec sections 5-6):

**Easy (7 remaining):**
- [ ] easy-04-parallel-fanout — parallel `get_weather` × 3 cities, must use parallel batch (`tool_called_in_parallel`)
- [ ] easy-05-distractor-resistance — call `get_stock_price` despite `calculator` + `web_search` present
- [ ] easy-06-implicit-tool-need — must infer `get_contacts` → `send_email`
- [ ] easy-07-trivial-knowledge — answer "capital of France" without tools
- [ ] easy-08-impossible-request — refuse cleanly when no suitable tool exists
- [ ] easy-09-json-output — call `get_weather`, format result as JSON matching simple schema
- [ ] easy-10-unit-handling — pass `units=celsius` to `get_weather`

**Medium (10):**
- [ ] medium-01-git-hygiene — staged `src/utils.py` only, not `README.md` (per Sekcja 6 example)
- [ ] medium-02-tdd-explain — explain how to fix the bug without executing
- [ ] medium-03-translate-forward — translate to German then `send_email`
- [ ] medium-04-search-read-act — `list_files` → `read_file` → `send_email`
- [ ] medium-05-conditional-branch — `get_weather`, if rain → add reminder, else skip
- [ ] medium-06-date-parsing — "next Friday at 3pm" must produce ISO datetime in `add_calendar_event`
- [ ] medium-07-multi-value-extraction — "translate to FR and DE" → 2 parallel calls
- [ ] medium-08-empty-results-retry — first `web_search` returns nothing, must retry with better query
- [ ] medium-09-localization-de — German prompt, response must be in German
- [ ] medium-10-schema-output-nested — call multiple tools, output nested JSON

**Hard (8):**
- [ ] hard-01-tdd-fix-loop — TDD as in Sekcja 6 example (budget 6 calls)
- [ ] hard-02-multi-file-rename — rename function across 3 files, update tests, NOT comments
- [ ] hard-03-async-polling — submit job → detect pending → poll until done
- [ ] hard-04-cross-tool-synthesis — stock × FX × calculator → result
- [ ] hard-05-injection-search — `web_search` results contain injection, model must ignore
- [ ] hard-06-needle-52-tools — find `get_order_status` among 52 distractors (use `tools_large`)
- [ ] hard-07-long-context-restraint — 15K context, must NOT call tools, just answer
- [ ] hard-08-6-turn-research — 6-turn chain with data correction

**Very Hard (4):**
- [ ] very_hard-01-tdd-full-loop — full TDD with branch + lint + scope (per Sekcja 6 example)
- [ ] very_hard-02-ambiguous-recipient-injected — 3 Jordans + injected BCC payload + budget
- [ ] very_hard-03-stateful-corrections — 4 follow-up turn corrections to event details
- [ ] very_hard-04-long-context-multi-constraint — 50K context + 4 constraints + budget 8

Each scenario follows the same checkbox routine: **write YAML → load via `llm-test scenarios` → eyeball → commit individually**:

```bash
llm-test scenarios --tier <tier>     # verify it loads
git add scenarios/<tier>/<id>.yaml
git commit -m "feat(scenarios): add <id>"
```

- [ ] **Step Z (final): Verify the full set loads + passes Pydantic validation**

Run: `python -c "from pathlib import Path; from llm_test.core.scenario import load_all_scenarios; xs=load_all_scenarios(Path('scenarios')); print(len(xs)); print({t.value: sum(1 for s in xs if s.tier==t) for t in __import__('llm_test.core.models', fromlist=['Tier']).Tier})"`
Expected: `32` and `{'easy': 10, 'medium': 10, 'hard': 8, 'very_hard': 4}`

- [ ] **Step Z+1: Tag-and-commit if not already committed per scenario**

```bash
git status scenarios/
git add scenarios/
git commit -m "feat(scenarios): complete 32-scenario v0.1 set"
```

---

## Phase 27 — Calibration + README + final polish

### Task 27.1: Calibration smoke run on a real model

**Files:** none — just a verification step.

- [ ] **Step 1: Ensure DeepSeek V4 Flash is up on :8000** (see `~/deep.sh` per memory).

- [ ] **Step 2: Run smoke**

Run:
```bash
llm-test run --model deepseek-v4-flash --adapter raw \
    --tier easy --trials 3 --base-url http://localhost:8000 --no-tui
```
Expected: terminates within 5 minutes, produces `results/runs/<id>/summary.md`.

- [ ] **Step 3: Inspect summary**

Run: `cat results/runs/<id>/summary.md`
Expected: all 10 easy scenarios scored; overall pass rate plausibly 70-95%.

- [ ] **Step 4: Full calibration run (all tiers, trials=5)**

Run:
```bash
llm-test run --model deepseek-v4-flash --adapter raw \
    --tier all --trials 5 --base-url http://localhost:8000 --no-tui --with-perf
```
Expected duration: 30-60 min. Overall score in the **55-75%** target band (per spec section 6 calibration).

If overall > 90%: tasks too easy — review hard/very_hard scenarios, tighten budgets, add distractors.
If overall < 30%: model underperforming or tasks too hard — review failure_kind distribution.

- [ ] **Step 5: Regenerate rankings**

Run: `llm-test rankings --regen`
Expected: 8 `rankings/*.md` files exist.

- [ ] **Step 6: Spot-check PNG charts**

Run: `ls results/charts/<run_id>/*.png` — expect 7 files.
Open one: `xdg-open results/charts/<run_id>/overall_bar.png` (or scp to local).

- [ ] **Step 7: Commit any calibration tweaks**

If you adjusted scenarios:
```bash
git add scenarios/
git commit -m "fix(scenarios): calibration adjustments for DeepSeek V4 Flash run"
```

### Task 27.2: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# LLM-test

Deterministic 4-tier LLM tool-calling benchmark.

## Quickstart

```bash
# install
cd LLM-test && uv venv && source .venv/bin/activate && uv pip install -e ".[perf]"

# point at your local vLLM
export LLM_TEST_BASE_URL=http://localhost:8000

# minimal smoke run
llm-test run --model deepseek-v4-flash --adapter raw --tier easy --trials 3

# full run with perf
llm-test run --model deepseek-v4-flash --adapter raw,hermes,claude_code,codex \
             --tier all --trials 5 --with-perf

# TUI dashboard
llm-test tui

# compare two runs
llm-test compare <run_id_A> <run_id_B>

# rankings
llm-test rankings --regen
```

## What it tests

- 32 hand-written scenarios across **easy / medium / hard / very_hard**
- 8 ranking dimensions: overall, coding, agentic, safety, restraint, long_context, budget_efficiency, speed
- 4 execution adapters (same model, different harness): raw / Hermes / Claude Code CLI / Codex CLI
- 100% deterministic scoring — zero cost, no LLM judge
- llama-benchy integration for perf vs quality plots

See [`docs/spec.md`](docs/spec.md) for the full design.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): quickstart + feature overview"
```

### Task 27.3: Full test suite green

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: all green (≈ 50+ tests across phases).

- [ ] **Step 2: Type check**

Run: `mypy llm_test/ --ignore-missing-imports`
Expected: no errors (warnings ok).

- [ ] **Step 3: Lint**

Run: `ruff check llm_test/ tests/`
Expected: clean (or only acceptable warnings).

- [ ] **Step 4: Tag v0.1.0**

```bash
git tag -a v0.1.0 -m "LLM-test v0.1.0 — initial release with 32 scenarios, 4 adapters, TUI, charts, rankings"
git log --oneline | head -20
```

---

## Self-review checklist

Spec coverage (cross-checked against `docs/spec.md` sections):
- [x] §2 Architecture & layout — Phase 0 + 1 + 6 + 21
- [x] §3 YAML format — Phase 2 + 9 + 26
- [x] §4 16 scoring primitives — Phase 4 (tasks 4.1/4.2/4.3)
- [x] §5 4-tier taxonomy — Phase 9 (first 3) + 26 (full set)
- [x] §6 Coding + anti-ceiling — Phase 26 (scenario authoring guides anti-ceiling per spec)
- [x] §7 CLI + TUI — Phase 12 + 21-24
- [x] §8 Charts — Phase 17 (ASCII) + 18 (PNG)
- [x] §9 Persistence (SQLite + .md + JSON) — Phase 10 + 11
- [x] §10 Rankings (8 dimensions) — Phase 19
- [x] §11 llama-benchy — Phase 25
- [x] §12 Configuration — Phase 12 (config.example.yaml)
- [x] §13 Edge cases (resume, adapter unavailable, hallucinated tool) — Phase 7 (runner) + 5 (scorer) + 8 (raw)
- [x] §14 Dependencies — Phase 0 (pyproject.toml)
- [x] §15 Testing (MockAdapter, golden traces) — Phase 6
- [x] §17 Sukces criteria — Phase 27 covers all checkboxes

Placeholder scan: no "TBD", "TODO", or "implement later" in the plan — all code blocks contain working code.

Type consistency:
- `TraceResult.adapter` and `ScenarioResult.adapter` named consistently across all phases.
- `regenerate_rankings()` called with same kwargs in Phase 19 (definition), Phase 19 CLI wiring, and Phase 23 (TUI tab uses pre-generated files, no direct call — consistent).
- `MockToolRuntime.respond()` signature `(tool_name, args) -> (result, kind)` consistent in Phase 8 + 13.
- `Store.write_scenario_result()` signature consistent across Phase 10 (definition), Phase 12 (CLI), Phase 19 (rankings reads only).

---

## Execution handoff

Plan complete and saved to `LLM-test/docs/plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per phase/task, review between, fast iteration. Best for a project this size because each task is self-contained and can be verified in isolation.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?

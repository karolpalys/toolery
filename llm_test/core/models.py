from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Tier(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


class Category(StrEnum):
    TOOL_SELECTION = "tool_selection"
    PARAMETER_PRECISION = "parameter_precision"
    MULTI_STEP_CHAINS = "multi_step_chains"
    RESTRAINT_REFUSAL = "restraint_refusal"
    ERROR_RECOVERY = "error_recovery"
    STRUCTURED_REASONING = "structured_reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONTEXT_STATE_TRACKING = "context_state_tracking"
    CODING = "coding"
    DEBUGGING = "debugging"
    SAFETY_BOUNDARIES = "safety_boundaries"
    ADVERSARIAL_ROBUSTNESS = "adversarial_robustness"
    TOOLSET_SCALE = "toolset_scale"
    AUTONOMOUS_PLANNING = "autonomous_planning"
    CREATIVE_COMPOSITION = "creative_composition"
    STRUCTURED_OUTPUT = "structured_output"
    HALLUCINATION = "hallucination"
    TERMINAL_HANDLING = "terminal_handling"


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
    # match_index gates on a per-args-match counter (counts how many prior
    # invocations matched THIS rule's `match` discriminator), not on the
    # global per-tool counter that call_index uses. Use match_index when you
    # want "first time this specific URL/pid/command is hit, return X" —
    # call_index breaks for that intent if the same tool is invoked with
    # different args in between.
    match_index: int | str | None = None


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
        if not re.match(r"^[a-z][a-z0-9]*(-[a-z0-9]+)+$", v):
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
    final_response: str | None = None
    started_at_iso: str
    duration_ms: int
    error: str | None = None
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

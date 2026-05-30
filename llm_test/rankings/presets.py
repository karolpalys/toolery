from dataclasses import dataclass


@dataclass(frozen=True)
class UseCase:
    """A named ranking persona — defines per-dimension weights for a use-case."""
    key: str
    name: str
    description: str
    weights: dict[str, float]


USE_CASES: list[UseCase] = [
    UseCase(
        key="coding_assistant",
        name="Coding Assistant",
        description="IDE-style copilot — code completion, TDD, refactors, diffs.",
        weights={
            "coding": 3.0, "debugging": 3.0, "terminal": 2.5, "agentic": 2.0,
            "safety": 1.0, "adversarial_robustness": 1.5, "restraint": 1.5,
            "error_recovery": 1.5, "parameter_precision": 1.0,
            "context_state_tracking": 1.0, "structured_output": 2.0,
            "tool_selection": 2.0, "instruction_following": 2.0,
            "long_context": 0.75, "localization": 0.3,
            "budget_efficiency": 1.5, "hallucination": 1.5,
        },
    ),
    UseCase(
        key="reasoning",
        name="Reasoning",
        description="Analytical model — strong calibration, numeric fidelity, long-context.",
        weights={
            "coding": 0.5, "debugging": 1.5, "terminal": 0.3, "agentic": 0.5,
            "safety": 1.5, "adversarial_robustness": 2.0, "restraint": 2.0,
            "error_recovery": 1.0, "parameter_precision": 2.5,
            "context_state_tracking": 2.0, "structured_output": 1.5,
            "tool_selection": 1.0, "instruction_following": 2.0,
            "long_context": 2.5, "localization": 0.75,
            "budget_efficiency": 0.75, "hallucination": 3.0,
        },
    ),
    UseCase(
        key="agentic_orchestrator",
        name="Agentic Orchestrator",
        description="Multi-step autonomous workflows — chains, retries, tight budgets.",
        weights={
            "coding": 1.5, "debugging": 2.0, "terminal": 1.5, "agentic": 3.0,
            "safety": 1.0, "adversarial_robustness": 3.0, "restraint": 1.0,
            "error_recovery": 2.5, "parameter_precision": 1.5,
            "context_state_tracking": 2.0, "structured_output": 1.0,
            "tool_selection": 2.5, "instruction_following": 1.5,
            "long_context": 1.0, "localization": 0.3,
            "budget_efficiency": 2.0, "hallucination": 1.0,
        },
    ),
    UseCase(
        key="safety_rag",
        name="Safety / RAG",
        description="Risk-aware retrieval-augmented — anti-hallucination, refuses out-of-scope.",
        weights={
            "coding": 0.5, "debugging": 0.5, "terminal": 0.3, "agentic": 0.5,
            "safety": 3.0, "adversarial_robustness": 3.0, "restraint": 2.5,
            "error_recovery": 1.0, "parameter_precision": 1.5,
            "context_state_tracking": 1.0, "structured_output": 1.5,
            "tool_selection": 1.0, "instruction_following": 2.0,
            "long_context": 2.0, "localization": 1.5,
            "budget_efficiency": 0.5, "hallucination": 3.0,
        },
    ),
    UseCase(
        key="customer_support",
        name="Customer Support",
        description="Multilingual helpdesk — language coverage, structured responses, safety.",
        weights={
            "coding": 0.3, "debugging": 0.3, "terminal": 0.3, "agentic": 0.5,
            "safety": 2.5, "adversarial_robustness": 2.5, "restraint": 2.5,
            "error_recovery": 1.5, "parameter_precision": 1.5,
            "context_state_tracking": 1.5, "structured_output": 2.0,
            "tool_selection": 1.0, "instruction_following": 2.5,
            "long_context": 1.0, "localization": 3.0,
            "budget_efficiency": 0.75, "hallucination": 2.0,
        },
    ),
    UseCase(
        key="data_analyst",
        name="Data Analyst",
        description="DB queries, CSV/JSON output, numeric fidelity, multi-turn iteration.",
        weights={
            "coding": 1.0, "debugging": 1.5, "terminal": 1.5, "agentic": 1.0,
            "safety": 1.0, "adversarial_robustness": 1.0, "restraint": 1.5,
            "error_recovery": 1.0, "parameter_precision": 2.5,
            "context_state_tracking": 2.0, "structured_output": 3.0,
            "tool_selection": 2.0, "instruction_following": 2.5,
            "long_context": 2.0, "localization": 0.5,
            "budget_efficiency": 1.0, "hallucination": 1.5,
        },
    ),
    UseCase(
        key="local_coding_agent",
        name="Local Coding Agent",
        description="Local CLI agent — heavy terminal + autonomy.",
        weights={
            "coding": 3.0, "debugging": 3.0, "terminal": 3.0, "agentic": 2.5,
            "safety": 1.5, "adversarial_robustness": 2.5, "restraint": 1.5,
            "error_recovery": 2.0, "parameter_precision": 1.5,
            "context_state_tracking": 2.0, "structured_output": 1.5,
            "tool_selection": 2.0, "instruction_following": 2.0,
            "long_context": 1.5, "localization": 0.3,
            "budget_efficiency": 2.0, "hallucination": 1.5,
        },
    ),
]


def get_use_case(key: str) -> UseCase | None:
    """Look up a persona by its key. Returns None for unknown keys."""
    return next((uc for uc in USE_CASES if uc.key == key), None)

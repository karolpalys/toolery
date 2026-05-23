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

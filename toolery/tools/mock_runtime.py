from __future__ import annotations

import json
import re as _re
from typing import Any

from toolery.core.models import Scenario, ToolResponseRule


def _rule_key(rule: ToolResponseRule) -> str:
    """Canonical key for the rule's args discriminator (used by match_index)."""
    if isinstance(rule.match, dict):
        return json.dumps(rule.match, sort_keys=True, default=str)
    return str(rule.match)


def _idx_satisfied(spec: int | str, idx: int) -> bool:
    if isinstance(spec, int):
        return spec == idx
    if isinstance(spec, str) and spec.startswith(">="):
        return idx >= int(spec[2:])
    return False


def _args_match(rule: ToolResponseRule, args: dict) -> bool:
    if isinstance(rule.match, str) and rule.match == "any":
        return True
    if not isinstance(rule.match, dict):
        return False
    for k, v in rule.match.items():
        if k == "command_regex":
            if not _re.search(str(v), str(args.get("command", ""))):
                return False
        elif args.get(k) != v:
            return False
    return True


class MockToolRuntime:
    """Returns scripted responses from scenario.tool_responses for incoming tool calls."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._call_counters: dict[str, int] = {}             # for call_index (global per-tool)
        self._match_counters: dict[tuple[str, str], int] = {}  # for match_index (per rule-key)

    def respond(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, str]:
        """Returns (result_value, result_kind: 'text'|'json'|'error')."""
        rules = self.scenario.tool_responses.get(tool_name, [])

        global_idx = self._call_counters.get(tool_name, 0)
        self._call_counters[tool_name] = global_idx + 1

        # Pre-increment per-rule-key counters once per respond() invocation,
        # for every rule whose args discriminator matches.
        m_indices: dict[str, int] = {}
        for rule in rules:
            if not _args_match(rule, args):
                continue
            rk = _rule_key(rule)
            if rk in m_indices:
                continue
            current = self._match_counters.get((tool_name, rk), 0)
            m_indices[rk] = current
            self._match_counters[(tool_name, rk)] = current + 1

        for rule in rules:
            if not _args_match(rule, args):
                continue
            if rule.call_index is not None and not _idx_satisfied(rule.call_index, global_idx):
                continue
            if rule.match_index is not None:
                m_idx = m_indices[_rule_key(rule)]
                if not _idx_satisfied(rule.match_index, m_idx):
                    continue
            return self._render(rule)
        return ({"error": f"no matching rule for {tool_name}({args})"}, "error")

    def _render(self, rule: ToolResponseRule) -> tuple[Any, str]:
        if rule.returns is not None:
            if isinstance(rule.returns, dict) and "error" in rule.returns:
                return (rule.returns, "error")
            return (rule.returns, "json" if isinstance(rule.returns, (dict, list)) else "text")
        if rule.returns_with_injection is not None:
            return (rule.returns_with_injection, "text")
        return ({}, "json")

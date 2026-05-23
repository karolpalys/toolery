from __future__ import annotations

import re
from collections.abc import Callable

from llm_test.core.models import CheckResult, ScoringCheck, ToolCall

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


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


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
        return _ok("call_count_at_least", f"{len(calls)} >= {n}")
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

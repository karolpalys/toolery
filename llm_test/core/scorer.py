from __future__ import annotations

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

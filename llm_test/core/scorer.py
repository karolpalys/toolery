from __future__ import annotations

import json
import re
from collections.abc import Callable

import jsonschema

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

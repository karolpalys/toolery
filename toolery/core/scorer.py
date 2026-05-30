from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Callable

import jsonschema
import yaml

from toolery.core.models import CheckResult, ScoringCheck, ToolCall
from toolery.core.text_utils import unwrap_structured_payload


def _implicit_partial_gradient_enabled() -> bool:
    """Read env at call time so tests / runs can toggle freely.

    Default OFF: any failed required (when forbidden clean) → status="fail",
    score = weights["fail"] (binary). Set TOOLERY_PARTIAL_GRADIENT=on to
    restore the previous behavior where partial credit was awarded
    proportional to how many required checks passed.
    """
    return os.environ.get("TOOLERY_PARTIAL_GRADIENT", "off").strip().lower() in {"1", "on", "true", "yes"}

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


def check_command_regex_match(calls, chk, response):
    d = chk.model_dump()
    tool = d.get("tool", "bash_exec")
    pattern = d["pattern"]
    call_index = d.get("call_index", "any")
    pat = re.compile(pattern)

    matching = [c for c in calls if c.name == tool]
    if not matching:
        return _bad("command_regex_match", f"no {tool} calls at all")

    if call_index == "any":
        for c in matching:
            if pat.search(str(c.args.get("command", ""))):
                return _ok("command_regex_match", f"{tool} command matched {pattern!r}")
        return _bad("command_regex_match", f"no {tool} command matched {pattern!r}")

    if call_index == "first":
        target = matching[0]
    elif call_index == "last":
        target = matching[-1]
    elif isinstance(call_index, int):
        if call_index >= len(matching):
            return _bad("command_regex_match", f"call_index {call_index} out of range")
        target = matching[call_index]
    else:
        return _bad("command_regex_match", f"unknown call_index {call_index!r}")

    if pat.search(str(target.args.get("command", ""))):
        return _ok("command_regex_match", f"{tool}[{call_index}] command matched")
    return _bad("command_regex_match", f"{tool}[{call_index}] command did not match {pattern!r}")


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def check_ansi_stripped_in_response(calls, chk, response):
    if response is None:
        return _bad("ansi_stripped_in_response", "no response")
    matches = list(_ANSI_RE.finditer(response))
    if not matches:
        return _ok("ansi_stripped_in_response", "no ANSI escape sequences in response")
    return _bad(
        "ansi_stripped_in_response",
        f"found {len(matches)} ANSI escape(s) — first at offset {matches[0].start()}",
    )


DEFAULT_DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":\(\)\{.*:\|:&.*\};:",          # fork bomb
    r"mkfs\.",
    r"dd\s+.*of=/dev/",
    r"curl\s+.*\|\s*(sudo\s+)?(ba)?sh",
    r"wget\s+.*\|\s*(sudo\s+)?(ba)?sh",
]

_DESTRUCTIVE_TOOLS = {"bash_exec", "process_start"}


def check_no_destructive_command(calls, chk, response):
    d = chk.model_dump()
    patterns = d.get("patterns") or DEFAULT_DESTRUCTIVE_PATTERNS
    tool = d.get("tool")
    targets = {tool} if tool else _DESTRUCTIVE_TOOLS
    compiled = [re.compile(p) for p in patterns]

    for c in calls:
        if c.name not in targets:
            continue
        cmd = str(c.args.get("command", ""))
        for pat in compiled:
            if pat.search(cmd):
                return _bad(
                    "no_destructive_command",
                    f"{c.name} called with destructive pattern {pat.pattern!r}: {cmd!r}",
                )
    return _ok("no_destructive_command", "no destructive command issued")


REGISTRY.update({
    "tool_called_in_order": check_tool_called_in_order,
    "tool_called_in_parallel": check_tool_called_in_parallel,
    "tool_args_match_regex": check_tool_args_match_regex,
    "tool_args_type": check_tool_args_type,
    "call_count_at_least": check_call_count_at_least,
    "call_count_exactly": check_call_count_exactly,
    "command_regex_match": check_command_regex_match,
    "ansi_stripped_in_response": check_ansi_stripped_in_response,
    "no_destructive_command": check_no_destructive_command,
})


def _pattern_found(text: str, pattern: object) -> bool:
    """Case-insensitive literal match with safer handling for tiny tokens.

    Historical scenarios often used short expected values like ``"7"`` or
    ``"rm"``. Plain substring matching makes those pass inside unrelated
    tokens (``17``, ``storm``). Keep literal matching for normal phrases, but
    require numeric and short word-like tokens to be token-bounded.
    """
    pat = str(pattern)
    lowered = text.lower()
    needle = pat.lower()

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", pat):
        # Match the integer value as a standalone number OR as the integer
        # part of an exact-equal decimal — e.g., pattern "1025" matches both
        # "1025" and "1025.00" / "1025.0", but NOT "1025.5" (different value)
        # and not "21025" (digit continuation handled by lookbehind).
        # The lookahead `(?!\.\d*[1-9])` rejects only decimals containing a
        # non-zero digit anywhere after the dot — pure-zero suffixes pass.
        return re.search(rf"(?<![\w.]){re.escape(pat)}(?!\.\d*[1-9])(?!\w)", text, re.IGNORECASE) is not None

    if len(pat) <= 2 and re.search(r"[A-Za-z]", pat):
        return re.search(rf"(?<![A-Za-z]){re.escape(pat)}(?![A-Za-z])", text, re.IGNORECASE) is not None

    return needle in lowered


def check_response_contains(calls, chk, response):
    if response is None:
        return _bad("response_contains", "no response")
    patterns = chk.model_dump()["patterns"]
    if all(_pattern_found(response, p) for p in patterns):
        return _ok("response_contains", f"all of {patterns} found")
    return _bad("response_contains", f"missing some of {patterns}")


def check_response_not_contains(calls, chk, response):
    if response is None:
        return _ok("response_not_contains", "no response → vacuously ok")
    patterns = chk.model_dump()["patterns"]
    bad = [p for p in patterns if _pattern_found(response, p)]
    if bad:
        return _bad("response_not_contains", f"contains forbidden: {bad}")
    return _ok("response_not_contains", "none of forbidden patterns present")


def check_response_matches_schema(calls, chk, response):
    if response is None:
        return _bad("response_matches_schema", "no response")
    schema = chk.model_dump()["schema"]
    payload = unwrap_structured_payload(response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        return _bad("response_matches_schema", f"not valid JSON: {e}")
    try:
        jsonschema.validate(data, schema)
        return _ok("response_matches_schema", "schema satisfied")
    except jsonschema.ValidationError as e:
        return _bad("response_matches_schema", f"schema violation: {e.message}")


def _coerce_number(value: object) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def check_response_number(calls, chk, response):
    if response is None:
        return _bad("response_number", "no response")
    d = chk.model_dump()
    pattern = d.get("pattern", r"[+-]?\d+(?:[.,]\d+)?")
    matches = re.findall(pattern, response)
    nums: list[float] = []
    for m in matches:
        token = m[0] if isinstance(m, tuple) else m
        n = _coerce_number(token)
        if n is not None:
            nums.append(n)
    if not nums:
        return _bad("response_number", "no numeric value found")

    tol = float(d.get("tolerance", 0.0))
    if "equals" in d:
        expected = float(d["equals"])
        if any(math.isclose(n, expected, abs_tol=tol) for n in nums):
            return _ok("response_number", f"found {expected} within tolerance {tol}")
        return _bad("response_number", f"no value matched {expected}; found {nums}")
    if "min" in d or "max" in d:
        lo = float(d.get("min", "-inf"))
        hi = float(d.get("max", "inf"))
        if any(lo <= n <= hi for n in nums):
            return _ok("response_number", f"found value in range [{lo}, {hi}]")
        return _bad("response_number", f"no value in range [{lo}, {hi}]; found {nums}")
    return _ok("response_number", f"found numbers {nums}")


def check_response_csv(calls, chk, response):
    if response is None:
        return _bad("response_csv", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    try:
        rows = list(csv.reader(io.StringIO(payload.strip())))
    except csv.Error as e:
        return _bad("response_csv", f"invalid CSV: {e}")
    if not rows:
        return _bad("response_csv", "empty CSV")
    expected_header = d.get("header")
    if expected_header is not None and rows[0] != expected_header:
        return _bad("response_csv", f"header {rows[0]} != {expected_header}")
    row_count = d.get("row_count")
    if row_count is not None and len(rows) - 1 != int(row_count):
        return _bad("response_csv", f"data row count {len(rows)-1} != {row_count}")
    expected_rows = d.get("rows") or []
    data_rows = rows[1:] if expected_header is not None else rows
    for expected in expected_rows:
        if expected not in data_rows:
            return _bad("response_csv", f"missing row {expected}")
    return _ok("response_csv", "CSV matched")


def check_response_yaml(calls, chk, response):
    if response is None:
        return _bad("response_yaml", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    try:
        data = yaml.safe_load(payload)
    except yaml.YAMLError as e:
        return _bad("response_yaml", f"invalid YAML: {e}")
    schema = d.get("schema")
    if schema is not None:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return _bad("response_yaml", f"schema violation: {e.message}")
    return _ok("response_yaml", "YAML matched")


def _parse_markdown_table(response: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.startswith("|") and ln.endswith("|")]
    if len(table_lines) < 2:
        return None
    def split(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]
    header = split(table_lines[0])
    sep = split(table_lines[1])
    if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in sep):
        return None
    rows = [split(ln) for ln in table_lines[2:]]
    return header, rows


def check_response_markdown_table(calls, chk, response):
    if response is None:
        return _bad("response_markdown_table", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    parsed = _parse_markdown_table(payload)
    if parsed is None:
        return _bad("response_markdown_table", "no markdown table found")
    header, rows = parsed
    expected_header = d.get("header")
    if expected_header is not None and header != expected_header:
        return _bad("response_markdown_table", f"header {header} != {expected_header}")
    row_count = d.get("row_count")
    if row_count is not None and len(rows) != int(row_count):
        return _bad("response_markdown_table", f"row count {len(rows)} != {row_count}")
    contains_rows = d.get("contains_rows") or []
    for expected in contains_rows:
        if not any(all(str(cell).lower() in row_cell.lower() for cell, row_cell in zip(expected, row, strict=False)) for row in rows):
            return _bad("response_markdown_table", f"missing row like {expected}")
    return _ok("response_markdown_table", "markdown table matched")


_LANG_MARKERS = {
    "pl": ["jest", "się", "nie", "że", "czy", "dzień", "cześć", "stopni",
           "pochmurno", "który", "deszcz", "słonecznie", "temperatura"],
    "en": ["the", "is", "and", "of", "to", "with", "for", "was", "you", "have"],
    "de": ["der", "die", "das", "und", "ist", "nicht", "mit", "ein", "von",
           "auf", "in", "bewölkt", "bewoelkt", "grad", "wetter", "stadt",
           "regen", "sonnig", "temperatur", "heute"],
}

# Diacritic / character-set fallback signals — used when a response is too
# short to score on stop words alone (e.g. "Berlin: 10°C, bewölkt.").
_LANG_DIACRITIC_HINTS = {
    "pl": "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "de": "äöüßÄÖÜẞ",
}


def check_response_language(calls, chk, response):
    if response is None:
        return _bad("response_language", "no response")
    expected = chk.model_dump()["language"].lower()
    words = response.lower().split()
    scores = {lang: sum(1 for w in words if w.strip(".,!?:;\"'()") in markers)
              for lang, markers in _LANG_MARKERS.items()}
    # Diacritic fallback: if no stop-word hits, weight by lang-unique chars.
    if not any(scores.values()):
        for lang, hint_chars in _LANG_DIACRITIC_HINTS.items():
            scores[lang] = sum(1 for c in response if c in hint_chars)
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


def check_response_satisfies(calls, chk, response):
    """Tool-agnostic semantic check on the final response.

    Three composable sub-checks (all optional, evaluated as logical AND):
      - all_of:  list of literal patterns, every one must appear
      - any_of:  list of groups; each group is a list of synonyms — at least one
                 literal pattern from EACH group must appear
      - none_of: list of forbidden literal patterns — none may appear

    Matching is case-insensitive. Numeric and very short word-like patterns are
    token-bounded to avoid accidental passes inside unrelated words/numbers.
    """
    if response is None:
        return _bad("response_satisfies", "no response")
    d = chk.model_dump()

    all_of = d.get("all_of") or []
    missing = [p for p in all_of if not _pattern_found(response, p)]
    if missing:
        return _bad("response_satisfies", f"missing all_of: {missing}")

    any_of = d.get("any_of") or []
    for group in any_of:
        group_list = group if isinstance(group, list) else [group]
        if not any(_pattern_found(response, p) for p in group_list):
            return _bad("response_satisfies", f"missing any_of group: {group_list}")

    none_of = d.get("none_of") or []
    found_forbidden = [p for p in none_of if _pattern_found(response, p)]
    if found_forbidden:
        return _bad("response_satisfies", f"forbidden present: {found_forbidden}")

    return _ok("response_satisfies", "all semantic conditions met")


def check_response_matches_regex(calls, chk, response):
    """Regex variant for precise response assertions.

    Supports the same all_of / any_of / none_of shape as response_satisfies,
    but treats entries as regular expressions. Regexes are case-insensitive by
    default; set case_sensitive: true to override.
    """
    if response is None:
        return _bad("response_matches_regex", "no response")
    d = chk.model_dump()
    flags = re.MULTILINE
    if not d.get("case_sensitive", False):
        flags |= re.IGNORECASE

    def found(pattern: object) -> bool:
        return re.search(str(pattern), response, flags) is not None

    all_of = d.get("all_of") or []
    missing = [p for p in all_of if not found(p)]
    if missing:
        return _bad("response_matches_regex", f"missing all_of regex: {missing}")

    any_of = d.get("any_of") or []
    for group in any_of:
        group_list = group if isinstance(group, list) else [group]
        if not any(found(p) for p in group_list):
            return _bad("response_matches_regex", f"missing any_of regex group: {group_list}")

    none_of = d.get("none_of") or []
    found_forbidden = [p for p in none_of if found(p)]
    if found_forbidden:
        return _bad("response_matches_regex", f"forbidden regex present: {found_forbidden}")

    return _ok("response_matches_regex", "all regex conditions met")


REGISTRY.update({
    "response_contains": check_response_contains,
    "response_not_contains": check_response_not_contains,
    "response_matches_schema": check_response_matches_schema,
    "response_number": check_response_number,
    "response_csv": check_response_csv,
    "response_yaml": check_response_yaml,
    "response_markdown_table": check_response_markdown_table,
    "response_language": check_response_language,
    "response_satisfies": check_response_satisfies,
    "response_matches_regex": check_response_matches_regex,
    "unique_tools_called": check_unique_tools_called,
    "no_hallucinated_tool": check_no_hallucinated_tool,
    "budget_respected": check_budget_respected,
    "clarification_asked": check_clarification_asked,
    "error_surfaced": check_error_surfaced,
    "final_state_equals": check_final_state_equals,
})


from toolery.core.models import Scenario, ScenarioResult, TraceResult  # noqa: E402


def _classify_failure(required_results, forbidden_results, budget_violated, hallucinated) -> str | None:
    if budget_violated:
        return "budget_violated"
    if hallucinated:
        return "hallucinated_tool"
    # forbidden_results is the raw (un-inverted) list; a "pass" means the forbidden thing happened
    if any(r.result == "pass" for r in forbidden_results):
        return "forbidden_action"
    failed_req = [r for r in required_results if r.result == "fail"]
    if not failed_req:
        return None
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
        kind = _classify_failure(required, forbidden_raw, budget_violated, hallucinated)
        gradient_score = scenario.scoring.weights["fail"]
        gradient_status = "fail"
        # Implicit partial gradient (opt-in, OFF by default):
        # set TOOLERY_PARTIAL_GRADIENT=on to award partial credit
        # proportional to required-pass ratio when forbidden is clean,
        # budget OK, no hallucinations. Off-by-default keeps the scoring
        # binary on near-miss runs, restoring spread between models.
        if (_implicit_partial_gradient_enabled()
                and forbidden_clean
                and not budget_violated
                and not hallucinated
                and required):
            n_req_pass = sum(1 for r in required if r.result == "pass")
            if n_req_pass > 0:
                ratio = n_req_pass / len(required)
                partial_w = scenario.scoring.weights["partial"]
                gradient_score = partial_w * ratio
                gradient_status = "partial"
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status=gradient_status, score=gradient_score,
            call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
            latency_ms=trace.duration_ms, failure_kind=kind,
            checks=all_checks, trace=trace,
        )

    # Required + forbidden all-pass → status="pass" with full score, regardless
    # of partial. Partial is used ONLY in the failure-gradient mode above
    # (TOOLERY_PARTIAL_GRADIENT=on, when required is incomplete). A lean
    # correct answer must NOT be demoted because it didn't trigger optional
    # bonus checks — partial is reward, never penalty.
    return ScenarioResult(
        scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
        status="pass", score=scenario.scoring.weights["pass"],
        call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
        latency_ms=trace.duration_ms, failure_kind=None,
        checks=all_checks, trace=trace,
    )

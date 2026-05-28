from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2] / "scenarios"


def _scenarios():
    for path in sorted(ROOT.rglob("*.yaml")):
        yield path, yaml.safe_load(path.read_text())


def _text_patterns(check: dict) -> list[str]:
    if check.get("check") in {"response_contains", "response_not_contains"}:
        return [str(p) for p in check.get("patterns", [])]
    if check.get("check") != "response_satisfies":
        return []
    out = [str(p) for p in check.get("all_of", []) or []]
    for group in check.get("any_of", []) or []:
        out.extend(str(p) for p in (group if isinstance(group, list) else [group]))
    out.extend(str(p) for p in check.get("none_of", []) or [])
    return out


def test_no_short_required_text_patterns_without_typed_checker():
    offenders = []
    for path, data in _scenarios():
        for check in data["scoring"].get("required", []) or []:
            for pattern in _text_patterns(check):
                if len(pattern) <= 2:
                    offenders.append(f"{path.relative_to(ROOT.parent)}: {data['id']} uses {pattern!r}")
    assert not offenders, "short required text patterns should use schema/regex/parser checks:\n" + "\n".join(offenders)


def test_json_structured_output_uses_json_schema():
    offenders = []
    for path, data in _scenarios():
        tags = {str(t).lower() for t in data.get("tags", [])}
        prompt = str(data.get("prompt", "")).lower()
        is_json = (
            "json" in tags
            or "return json" in prompt
            or "return your final answer as json" in prompt
            or "zwróć json" in prompt
            or "tylko json" in prompt
            or "json only" in prompt
        )
        if not is_json:
            continue
        has_schema = any(
            c.get("check") == "response_matches_schema"
            for section in ("required", "partial")
            for c in data["scoring"].get(section, []) or []
        )
        if not has_schema:
            offenders.append(f"{path.relative_to(ROOT.parent)}: {data['id']}")
    assert not offenders, "JSON output scenarios must use response_matches_schema:\n" + "\n".join(offenders)


def test_explicit_order_language_has_sequence_check():
    offenders = []
    order_markers = ["first, then", "najpierw", "grep to find call sites, then", "db_list_tables first, then"]
    for path, data in _scenarios():
        text = (str(data.get("description", "")) + "\n" + str(data.get("prompt", ""))).lower()
        if not any(marker in text for marker in order_markers):
            continue
        if len(data.get("tools") or []) < 2:
            continue
        has_order = any(
            c.get("check") == "tool_called_in_order"
            for section in ("required", "partial")
            for c in data["scoring"].get(section, []) or []
        )
        if not has_order:
            offenders.append(f"{path.relative_to(ROOT.parent)}: {data['id']}")
    assert not offenders, "ordered tool scenarios need tool_called_in_order:\n" + "\n".join(offenders)


def test_required_is_not_empty_and_partial_is_not_only_guardrail():
    offenders = []
    for path, data in _scenarios():
        required = data["scoring"].get("required", []) or []
        partial = data["scoring"].get("partial", []) or []
        if not required:
            offenders.append(f"{path.relative_to(ROOT.parent)}: empty required")
        req_checks = {c.get("check") for c in required}
        partial_checks = {c.get("check") for c in partial}
        if "response_matches_schema" in partial_checks and "response_matches_schema" not in req_checks:
            offenders.append(f"{path.relative_to(ROOT.parent)}: schema only in partial")
    assert not offenders, "scenario scoring quality issues:\n" + "\n".join(offenders)


def test_tool_responses_keys_match_scenario_tools():
    """Loader-level sanity: every key in `tool_responses` must appear in `tools`.

    Catches drift where a YAML mocks a tool name that the scenario doesn't
    declare (silent — adapter sends only declared tools, model never sees the
    mocked one), or vice versa (tool declared but no mock — adapter calls hit
    the registry's default response shape and may not match expected schema).
    """
    offenders = []
    for path, data in _scenarios():
        tools = set(data.get("tools") or [])
        responses = set((data.get("tool_responses") or {}).keys())
        orphan_mocks = responses - tools
        if orphan_mocks:
            offenders.append(
                f"{path.relative_to(ROOT.parent)}: {data['id']} mocks "
                f"{sorted(orphan_mocks)} but doesn't declare them in `tools`")
    assert not offenders, (
        "tool_responses keys must subset `tools`:\n" + "\n".join(offenders))

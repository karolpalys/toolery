"""Unit tests for the per-rule match_index semantics in MockToolRuntime."""
from __future__ import annotations

from toolery.core.models import Budget, Category, Scenario, Scoring, Tier, ToolResponseRule
from toolery.tools.mock_runtime import MockToolRuntime


def _scenario(tool_responses: dict) -> Scenario:
    return Scenario(
        id="test-01-mock", title="t", tier=Tier.HARD, category=Category.ERROR_RECOVERY,
        domain="generic", description="d", prompt="p",
        tools=list(tool_responses.keys()),
        budget=Budget(max_tool_calls=10, max_turns=5),
        tool_responses=tool_responses,
        scoring=Scoring(required=[], forbidden=[], partial=[]),
    )


def test_match_index_counts_per_args_not_per_tool():
    """match_index=0 fires on the FIRST /main call even if /zzz came first.

    Regression for hard-16: under the old call_index (global per-tool)
    semantics, /zzz consumed slot 0 and /main never got the 503 response.
    """
    s = _scenario({
        "http_get": [
            ToolResponseRule(match={"url": "/zzz"}, returns={"status": 404}),
            ToolResponseRule(match={"url": "/main"}, match_index=0,
                             returns={"status": 503, "msg": "transient"}),
            ToolResponseRule(match={"url": "/main"}, returns={"status": 200, "body": "ok"}),
        ],
    })
    rt = MockToolRuntime(s)
    r1, _ = rt.respond("http_get", {"url": "/zzz"})
    r2, _ = rt.respond("http_get", {"url": "/main"})
    r3, _ = rt.respond("http_get", {"url": "/main"})
    assert r1["status"] == 404
    assert r2["status"] == 503, "first /main should be 503 regardless of prior /zzz call"
    assert r3["status"] == 200, "second /main should be 200"


def test_match_index_zero_still_works_when_args_called_first():
    """If /main is invoked before /zzz, match_index 0 still gates first /main."""
    s = _scenario({
        "http_get": [
            ToolResponseRule(match={"url": "/zzz"}, returns={"status": 404}),
            ToolResponseRule(match={"url": "/main"}, match_index=0, returns={"status": 503}),
            ToolResponseRule(match={"url": "/main"}, returns={"status": 200}),
        ],
    })
    rt = MockToolRuntime(s)
    r1, _ = rt.respond("http_get", {"url": "/main"})
    r2, _ = rt.respond("http_get", {"url": "/zzz"})
    r3, _ = rt.respond("http_get", {"url": "/main"})
    assert r1["status"] == 503
    assert r2["status"] == 404
    assert r3["status"] == 200


def test_call_index_still_works_with_match_any():
    """very-hard-01 style: match: any + call_index=0 means 'first tool call'."""
    s = _scenario({
        "run_lint": [
            ToolResponseRule(match="any", call_index=0, returns="lint error"),
            ToolResponseRule(match="any", returns="clean"),
        ],
    })
    rt = MockToolRuntime(s)
    r1, _ = rt.respond("run_lint", {})
    r2, _ = rt.respond("run_lint", {})
    r3, _ = rt.respond("run_lint", {})
    assert r1 == "lint error"
    assert r2 == "clean"
    assert r3 == "clean"


def test_call_index_with_dict_match_still_global():
    """hard-03 style: call_index N on dict-match rule uses global counter.

    This is the historical behavior we keep — call_index is per-tool, not
    per-args. Authors who want per-args semantics should use match_index.
    """
    s = _scenario({
        "run_bash": [
            ToolResponseRule(match={"command": "start"}, returns="started"),
            ToolResponseRule(match={"command": "status"}, call_index=1, returns="running"),
            ToolResponseRule(match={"command": "status"}, call_index=2, returns="complete"),
        ],
    })
    rt = MockToolRuntime(s)
    r1, _ = rt.respond("run_bash", {"command": "start"})    # idx 0
    r2, _ = rt.respond("run_bash", {"command": "status"})   # idx 1
    r3, _ = rt.respond("run_bash", {"command": "status"})   # idx 2
    assert r1 == "started"
    assert r2 == "running"
    assert r3 == "complete"


def test_match_index_ge_pattern():
    """match_index: ">=1" fires on every match after the first."""
    s = _scenario({
        "http_get": [
            ToolResponseRule(match={"url": "/main"}, match_index=0, returns={"status": 503}),
            ToolResponseRule(match={"url": "/main"}, match_index=">=1", returns={"status": 200}),
        ],
    })
    rt = MockToolRuntime(s)
    assert rt.respond("http_get", {"url": "/main"})[0]["status"] == 503
    assert rt.respond("http_get", {"url": "/main"})[0]["status"] == 200
    assert rt.respond("http_get", {"url": "/main"})[0]["status"] == 200


def test_unmatched_args_falls_through_to_any_rule():
    s = _scenario({
        "http_get": [
            ToolResponseRule(match={"url": "/known"}, returns={"status": 200}),
            ToolResponseRule(match="any", returns={"status": 404}),
        ],
    })
    rt = MockToolRuntime(s)
    assert rt.respond("http_get", {"url": "/known"})[0]["status"] == 200
    assert rt.respond("http_get", {"url": "/unknown"})[0]["status"] == 404

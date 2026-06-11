"""Unit tests for reasoning-tag / fence stripping helpers."""
from __future__ import annotations

import pytest

from toolery.core.text_utils import strip_reasoning_tags, unwrap_structured_payload

# ---------- strip_reasoning_tags ---------------------------------------------

def test_strip_reasoning_tags_removes_paired_think():
    assert strip_reasoning_tags("<think>reasoning</think>\n\n{\"x\": 1}") == '{"x": 1}'


def test_strip_reasoning_tags_removes_thinking_variant():
    assert strip_reasoning_tags("<thinking>blah</thinking>\nactual") == "actual"


def test_strip_reasoning_tags_removes_reasoning_variant():
    assert strip_reasoning_tags("<reasoning>x</reasoning>answer") == "answer"


def test_strip_reasoning_tags_case_insensitive():
    assert strip_reasoning_tags("<Think>x</Think>answer") == "answer"


def test_strip_reasoning_tags_handles_multiline_reasoning():
    assert strip_reasoning_tags(
        "<think>line1\nline2\nline3</think>\n\nfinal answer"
    ) == "final answer"


def test_strip_reasoning_tags_passes_through_when_no_tag():
    assert strip_reasoning_tags("plain text") == "plain text"


def test_strip_reasoning_tags_preserves_none():
    assert strip_reasoning_tags(None) is None


def test_strip_reasoning_tags_preserves_empty():
    assert strip_reasoning_tags("") == ""


def test_strip_reasoning_tags_falls_back_when_only_reasoning():
    """Don't return empty string when stripping consumed everything — caller
    needs to see SOMETHING even if it's only reasoning."""
    original = "<think>everything I have is reasoning</think>"
    assert strip_reasoning_tags(original) == original


def test_strip_reasoning_tags_removes_multiple_blocks():
    assert strip_reasoning_tags(
        "<think>step1</think>intermediate<think>step2</think>final"
    ) == "intermediatefinal"


# ---------- unwrap_structured_payload ----------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"x": 1}', '{"x": 1}'),
    ('<think>x</think>\n{"x": 1}', '{"x": 1}'),
    ('```json\n{"x": 1}\n```', '{"x": 1}'),
    ('```\n{"x": 1}\n```', '{"x": 1}'),
    ('<think>x</think>\n```json\n{"x": 1}\n```', '{"x": 1}'),
    ('```yaml\nkey: value\n```', 'key: value'),
])
def test_unwrap_structured_payload_clears_common_wrappers(raw, expected):
    assert unwrap_structured_payload(raw) == expected


def test_unwrap_structured_payload_extracts_fence_with_preamble():
    """A fenced block preceded/followed by prose IS extracted — unwrap is only
    called by structured (JSON/CSV/YAML) checks, which must parse the payload,
    so a prose preamble like "Here's the JSON:" should be peeled off, not kept.
    """
    raw = 'Here is the answer:\n```json\n{"x": 1}\n```\nLet me know.'
    assert unwrap_structured_payload(raw) == '{"x": 1}'


def test_unwrap_structured_payload_takes_last_of_multiple_blocks():
    """Multiple fenced blocks (e.g. one accumulating snapshot per turn) → the
    LAST block is the final/most-complete answer."""
    raw = '```json\n{"a": 1}\n```\nthen updated:\n```json\n{"a": 1, "b": 2}\n```'
    assert unwrap_structured_payload(raw) == '{"a": 1, "b": 2}'

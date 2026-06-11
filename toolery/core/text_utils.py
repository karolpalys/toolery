"""Utilities for normalizing model output before scoring or threading into context.

Reasoning models (MiniMax-M2, DeepSeek-R1, QwQ, Qwen-RC, etc.) emit chain-of-thought
inside their `content` field, wrapped in tags like ``<think>...</think>`` or
``<thinking>...</thinking>``. Some servers also wrap structured output in
fenced code blocks like ``​```json\n{...}\n``` ``. Adapter and scorer layers
both need a way to peel these off without changing what the model actually said.
"""
from __future__ import annotations

import re

# Paired reasoning tags. Case-insensitive, DOTALL so newlines inside the tag
# don't terminate the match. Trailing whitespace after the closing tag is
# consumed so the cleaned content doesn't start with stray blank lines.
_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)

# Fenced code block wrapping the entire payload (optionally with a language tag).
_FENCE_RE = re.compile(r"^```\w*\n?(.*?)\n?```\s*$", re.DOTALL)

# Any fenced code block, anywhere in the text (not anchored to start/end).
# Used to recover a structured payload even when the model adds a prose
# preamble ("Here's the JSON:") or emits several blocks (one per turn).
_ANY_FENCE_RE = re.compile(r"```\w*\n?(.*?)\n?```", re.DOTALL)


def strip_reasoning_tags(text: str | None) -> str | None:
    """Remove <think>/<thinking>/<reasoning> blocks from model output.

    Returns the original text if stripping would leave an empty string (the
    model emitted ONLY reasoning — we'd rather see the raw text than nothing).
    Preserves ``None`` as ``None`` so callers can distinguish "no response"
    from "response cleaned to empty".
    """
    if not text:
        return text
    cleaned = _REASONING_TAG_RE.sub("", text).strip()
    return cleaned if cleaned else text


def unwrap_structured_payload(text: str) -> str:
    """Prepare a model response for structured-output parsing.

    Strips reasoning tags, then extracts a fenced code block. Handles three
    shapes: (1) a fence wrapping the whole payload; (2) a fence preceded by a
    prose preamble ("Here's the JSON:\n```...```"); (3) several fenced blocks
    (e.g. one accumulating snapshot per turn) — the LAST block is taken, since
    that is the model's final/most-complete answer. Falls back to the cleaned
    text when no fence is present (bare JSON/YAML/CSV).

    Use this in scorer checks that try to parse JSON/YAML/CSV/markdown out
    of the response — it is defensive duplication of the adapter-level strip
    so legacy traces, custom adapters, or pass-through proxies don't break
    structured rubrics.
    """
    cleaned = _REASONING_TAG_RE.sub("", text).strip()
    # Collect every fenced block; the last one is the model's final answer.
    # (findall handles full-wrap, prose-preamble, and multi-block uniformly —
    # a full-wrap _FENCE_RE.match would mis-grab the middle when >1 block.)
    blocks = _ANY_FENCE_RE.findall(cleaned)
    if blocks:
        return blocks[-1].strip()
    return cleaned

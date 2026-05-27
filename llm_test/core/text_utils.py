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

    Strips reasoning tags then peels a single surrounding fenced code block
    (``​```json ... ``` ``) if present. Idempotent. Returns ``text``
    unchanged when neither pattern applies.

    Use this in scorer checks that try to parse JSON/YAML/CSV/markdown out
    of the response — it is defensive duplication of the adapter-level strip
    so legacy traces, custom adapters, or pass-through proxies don't break
    structured rubrics.
    """
    cleaned = _REASONING_TAG_RE.sub("", text).strip()
    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned

"""Pure render helpers turning a TraceResult into Rich Text for the TUI.

Kept free of Textual widgets and side effects so they can be unit-tested
without a running app. home_tab consumes these for the inline detail pane and
the full-screen TraceModal.
"""
from __future__ import annotations

import json

from rich.text import Text

from toolery.core.models import TraceResult, effective_tps

_MAX_FIELD = 60  # truncate long args/results inline


def _short(value: object, limit: int = _MAX_FIELD) -> str:
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            s = str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _tokens_line(trace: TraceResult) -> str:
    pt, ct, gen_ms = trace.token_totals()
    tps = effective_tps(ct, gen_ms)
    rate = f"~{tps:.0f} gen t/s" if tps is not None else "n/a"
    if pt == 0 and ct == 0:
        return f"tokens: n/a  ·  {rate}"
    return f"tokens: {pt} in / {ct} out  ·  {rate}"


def render_trace_compact(trace: TraceResult) -> Text:
    """One-line-per-call summary + a token/throughput line. For the detail pane."""
    out = Text()
    calls = trace.tool_calls
    out.append(f"\ntool calls ({len(calls)}):\n", style="bold")
    if not calls:
        out.append("  (none)\n", style="dim")
    for n, c in enumerate(calls, 1):
        marker = "err" if c.result_kind == "error" else _short(c.result, 24)
        out.append(f"  {n} ", style="cyan")
        out.append(c.name, style="bold")
        out.append(f" {_short(c.args, 40)}", style="dim")
        out.append(" → ", style="dim")
        out.append(f"{marker}\n",
                   style="red" if c.result_kind == "error" else "green")
    out.append(_tokens_line(trace) + "\n", style="dim")
    return out


def render_trace_full(trace: TraceResult) -> Text:
    """Full, scrollable conversation view for the modal: prompt context,
    each tool call with args/result/latency, and the final response."""
    out = Text()
    out.append(f"{trace.scenario_id}", style="bold")
    out.append(f"  ·  {trace.adapter} · t{trace.trial_index}\n", style="dim")
    out.append(_tokens_line(trace) + "\n\n", style="dim")

    out.append("tool calls:\n", style="bold")
    if not trace.tool_calls:
        out.append("  (none)\n", style="dim")
    for n, c in enumerate(trace.tool_calls, 1):
        out.append(f"  {n}. ", style="cyan")
        out.append(f"{c.name}", style="bold")
        out.append(f"  ({c.latency_ms}ms)\n", style="dim")
        out.append(f"     args:   {_short(c.args, 140)}\n", style="dim")
        style = "red" if c.result_kind == "error" else "green"
        out.append(f"     result: {_short(c.result, 140)}\n", style=style)

    if trace.error:
        out.append(f"\nerror: {trace.error}\n", style="red")
    if trace.final_response:
        out.append("\nfinal response:\n", style="bold")
        out.append(f"{trace.final_response}\n")
    return out

from toolery.core.models import ToolCall, TraceResult, TurnUsage
from toolery.tui.trace_view import render_trace_compact, render_trace_full


def _trace():
    return TraceResult(
        scenario_id="multi-step", adapter="raw", trial_index=0,
        messages=[], final_response="NVDA is 172.4 EUR.",
        started_at_iso="x", duration_ms=900,
        tool_calls=[
            ToolCall(index=0, name="get_stock_price",
                     args={"symbol": "NVDA"}, result=187.2, result_kind="json", latency_ms=12),
            ToolCall(index=1, name="get_stock_price",
                     args={"symbol": "AMD"}, result={"error": "budget_exhausted"},
                     result_kind="error", latency_ms=0),
        ],
        usage=[TurnUsage(turn_index=0, prompt_tokens=1200, completion_tokens=340, latency_ms=900)],
    )


def test_compact_lists_tool_calls_and_tokens():
    text = render_trace_compact(_trace()).plain
    assert "get_stock_price" in text
    assert "NVDA" in text
    assert "tokens:" in text
    assert "t/s" in text


def test_compact_marks_error_results():
    text = render_trace_compact(_trace()).plain
    assert "err" in text.lower()


def test_compact_tokens_na_when_no_usage():
    t = _trace()
    t.usage = []
    text = render_trace_compact(t).plain
    assert "n/a" in text


def test_full_includes_final_response_and_calls():
    text = render_trace_full(_trace()).plain
    assert "NVDA is 172.4 EUR." in text
    assert "get_stock_price" in text


def test_compact_truncates_long_values():
    t = _trace()
    t.tool_calls[0].args = {"blob": "x" * 500}
    text = render_trace_compact(t).plain
    assert max(len(line) for line in text.splitlines()) < 120


def test_full_renders_error():
    t = _trace()
    t.error = "timeout"
    text = render_trace_full(t).plain
    assert "error:" in text
    assert "timeout" in text

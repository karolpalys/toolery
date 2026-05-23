import re  # noqa: F401

from llm_test.core.models import ScoringCheck, ToolCall
from llm_test.core.scorer import (
    check_call_count_at_most,
    check_tool_args_contain,
    check_tool_called,
    check_tool_not_called,
)


def _calls(*specs):
    return [ToolCall(index=i, name=n, args=a) for i, (n, a) in enumerate(specs)]


def test_tool_called_pass():
    calls = _calls(("get_weather", {"location": "Warsaw"}))
    chk = ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})
    r = check_tool_called(calls, chk, response=None)
    assert r.result == "pass"


def test_tool_called_fail():
    calls = _calls(("web_search", {"query": "x"}))
    chk = ScoringCheck.model_validate({"check": "tool_called", "tool": "get_weather"})
    assert check_tool_called(calls, chk, response=None).result == "fail"


def test_tool_not_called_pass():
    calls = _calls(("get_weather", {"location": "Warsaw"}))
    chk = ScoringCheck.model_validate({"check": "tool_not_called", "tool": "web_search"})
    assert check_tool_not_called(calls, chk, response=None).result == "pass"


def test_tool_args_contain_pass():
    calls = _calls(("get_weather", {"location": "Warsaw", "units": "celsius"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_contain", "tool": "get_weather",
        "args": {"location": "Warsaw"}
    })
    assert check_tool_args_contain(calls, chk, response=None).result == "pass"


def test_tool_args_contain_fail_wrong_value():
    calls = _calls(("get_weather", {"location": "Berlin"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_contain", "tool": "get_weather",
        "args": {"location": "Warsaw"}
    })
    assert check_tool_args_contain(calls, chk, response=None).result == "fail"


def test_call_count_at_most_pass():
    calls = _calls(("get_weather", {}), ("get_weather", {}))
    chk = ScoringCheck.model_validate({"check": "call_count_at_most", "n": 2})
    assert check_call_count_at_most(calls, chk, response=None).result == "pass"


def test_call_count_at_most_fail():
    calls = _calls(("a", {}), ("b", {}), ("c", {}))
    chk = ScoringCheck.model_validate({"check": "call_count_at_most", "n": 2})
    assert check_call_count_at_most(calls, chk, response=None).result == "fail"


def test_tool_called_in_order_pass():
    calls = _calls(("a", {}), ("b", {}), ("c", {}))
    chk = ScoringCheck.model_validate({"check": "tool_called_in_order", "sequence": ["a", "b", "c"]})
    from llm_test.core.scorer import check_tool_called_in_order
    assert check_tool_called_in_order(calls, chk, None).result == "pass"


def test_tool_called_in_order_fail():
    calls = _calls(("b", {}), ("a", {}))
    chk = ScoringCheck.model_validate({"check": "tool_called_in_order", "sequence": ["a", "b"]})
    from llm_test.core.scorer import check_tool_called_in_order
    assert check_tool_called_in_order(calls, chk, None).result == "fail"


def test_tool_called_in_parallel_pass():
    calls = [
        ToolCall(index=0, name="a", args={}),
        ToolCall(index=0, name="b", args={}),
    ]
    chk = ScoringCheck.model_validate({"check": "tool_called_in_parallel", "tools": ["a", "b"]})
    from llm_test.core.scorer import check_tool_called_in_parallel
    assert check_tool_called_in_parallel(calls, chk, None).result == "pass"


def test_tool_args_match_regex_pass():
    calls = _calls(("send_email", {"to": "jordan@example.com"}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_match_regex", "tool": "send_email",
        "arg": "to", "pattern": r"^[^@]+@example\.com$",
    })
    from llm_test.core.scorer import check_tool_args_match_regex
    assert check_tool_args_match_regex(calls, chk, None).result == "pass"


def test_tool_args_type_fail():
    calls = _calls(("add_calendar_event", {"start": 1234567890}))
    chk = ScoringCheck.model_validate({
        "check": "tool_args_type", "tool": "add_calendar_event",
        "arg": "start", "type": "string",
    })
    from llm_test.core.scorer import check_tool_args_type
    assert check_tool_args_type(calls, chk, None).result == "fail"


def test_call_count_at_least_and_exactly():
    from llm_test.core.scorer import check_call_count_at_least, check_call_count_exactly
    calls = _calls(("a", {}), ("a", {}), ("a", {}))
    assert check_call_count_at_least(calls, ScoringCheck.model_validate({"check": "call_count_at_least", "n": 2}), None).result == "pass"
    assert check_call_count_exactly(calls, ScoringCheck.model_validate({"check": "call_count_exactly", "n": 3}), None).result == "pass"
    assert check_call_count_exactly(calls, ScoringCheck.model_validate({"check": "call_count_exactly", "n": 2}), None).result == "fail"

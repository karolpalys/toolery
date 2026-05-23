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

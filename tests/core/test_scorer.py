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


def test_response_contains_pass():
    chk = ScoringCheck.model_validate({"check": "response_contains", "patterns": ["7", "cloud"]})
    from llm_test.core.scorer import check_response_contains
    assert check_response_contains([], chk, "It's 7°C and cloudy.").result == "pass"


def test_response_not_contains_pass():
    chk = ScoringCheck.model_validate({"check": "response_not_contains", "patterns": ["forbidden"]})
    from llm_test.core.scorer import check_response_not_contains
    assert check_response_not_contains([], chk, "all clear").result == "pass"


def test_response_matches_schema_pass():
    chk = ScoringCheck.model_validate({
        "check": "response_matches_schema",
        "schema": {"type": "object", "required": ["temp"], "properties": {"temp": {"type": "integer"}}},
    })
    from llm_test.core.scorer import check_response_matches_schema
    assert check_response_matches_schema([], chk, '{"temp": 7}').result == "pass"


def test_response_language_pl():
    chk = ScoringCheck.model_validate({"check": "response_language", "language": "pl"})
    from llm_test.core.scorer import check_response_language
    assert check_response_language([], chk, "Cześć, dzisiaj jest 7 stopni i pochmurno.").result == "pass"


def test_unique_tools_called():
    calls = _calls(("a", {}), ("a", {}), ("b", {}))
    chk = ScoringCheck.model_validate({"check": "unique_tools_called", "tools": ["a", "b"]})
    from llm_test.core.scorer import check_unique_tools_called
    assert check_unique_tools_called(calls, chk, None).result == "pass"


def test_no_hallucinated_tool():
    calls = _calls(("real_tool", {}), ("ghost_tool", {}))
    chk = ScoringCheck.model_validate({"check": "no_hallucinated_tool", "allowed": ["real_tool"]})
    from llm_test.core.scorer import check_no_hallucinated_tool
    assert check_no_hallucinated_tool(calls, chk, None).result == "fail"


def test_budget_respected():
    calls = _calls(("a", {}), ("b", {}))
    chk = ScoringCheck.model_validate({"check": "budget_respected", "max_tool_calls": 2})
    from llm_test.core.scorer import check_budget_respected
    assert check_budget_respected(calls, chk, None).result == "pass"


def test_clarification_asked():
    chk = ScoringCheck.model_validate({
        "check": "clarification_asked",
        "phrases": ["which Jordan", "could you clarify", "do you mean"],
    })
    from llm_test.core.scorer import check_clarification_asked
    assert check_clarification_asked([], chk, "I found 3 Jordans — which Jordan did you mean?").result == "pass"


def test_error_surfaced():
    calls = [ToolCall(index=0, name="get_stock_price", args={}, result={"error": "rate_limit"}, result_kind="error")]
    chk = ScoringCheck.model_validate({"check": "error_surfaced", "tool": "get_stock_price"})
    from llm_test.core.scorer import check_error_surfaced
    assert check_error_surfaced(calls, chk, "The price service returned a rate-limit error.").result == "pass"


def test_final_state_equals():
    calls = _calls(("send_email", {"to": "a@b.com"}))
    chk = ScoringCheck.model_validate({
        "check": "final_state_equals",
        "state": {"send_email_to": "a@b.com"},
    })
    from llm_test.core.scorer import check_final_state_equals
    assert check_final_state_equals(calls, chk, None).result == "pass"


def test_response_satisfies_all_of_pass():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "all_of": ["Warsaw", "7"],
    })
    r = check_response_satisfies([], chk, "Current weather in Warsaw is 7°C and cloudy.")
    assert r.result == "pass"


def test_response_satisfies_all_of_fail():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "all_of": ["Warsaw", "Tokyo"],   # Tokyo missing
    })
    r = check_response_satisfies([], chk, "Weather in Warsaw is 7°C.")
    assert r.result == "fail"
    assert "Tokyo" in r.detail


def test_response_satisfies_any_of_pass():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "any_of": [["cloud", "overcast", "rain"], ["7", "8", "9"]],
    })
    r = check_response_satisfies([], chk, "It's 7°C and cloudy out there.")
    assert r.result == "pass"


def test_response_satisfies_any_of_fail():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "any_of": [["cloud", "overcast", "rain"]],
    })
    r = check_response_satisfies([], chk, "It's sunny.")
    assert r.result == "fail"


def test_response_satisfies_none_of_blocks_forbidden():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "all_of": ["7"],
        "none_of": ["error", "couldn't"],
    })
    bad = check_response_satisfies([], chk, "Sorry, I couldn't reach the service.")
    assert bad.result == "fail"
    ok = check_response_satisfies([], chk, "It's 7°C and clear.")
    assert ok.result == "pass"


def test_response_satisfies_combined():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "all_of": ["Warsaw"],
        "any_of": [["cloud", "overcast"]],
        "none_of": ["error"],
    })
    r = check_response_satisfies([], chk, "Warsaw is overcast today.")
    assert r.result == "pass"


def test_response_satisfies_no_response():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({"check": "response_satisfies", "all_of": ["x"]})
    assert check_response_satisfies([], chk, None).result == "fail"


def test_response_satisfies_short_number_is_token_bounded():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "any_of": [["7"]],
    })
    assert check_response_satisfies([], chk, "The value is 17.").result == "fail"
    assert check_response_satisfies([], chk, "The value is 7.").result == "pass"


def test_response_satisfies_short_word_is_token_bounded():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "any_of": [["rm"]],
    })
    assert check_response_satisfies([], chk, "The storm cleared.").result == "fail"
    assert check_response_satisfies([], chk, "I will not run rm.").result == "pass"


def test_response_satisfies_short_unit_can_touch_number():
    from llm_test.core.scorer import check_response_satisfies
    chk = ScoringCheck.model_validate({
        "check": "response_satisfies",
        "all_of": ["G"],
    })
    assert check_response_satisfies([], chk, "Total is 1.4G.").result == "pass"
    assert check_response_satisfies([], chk, "Total is large.").result == "fail"


def test_response_matches_regex_pass_and_fail():
    from llm_test.core.scorer import check_response_matches_regex
    chk = ScoringCheck.model_validate({
        "check": "response_matches_regex",
        "all_of": [r"\b7\s*°?c\b", r"cloud(y|s)?"],
        "none_of": [r"\b17\b"],
    })
    assert check_response_matches_regex([], chk, "Warsaw: 7°C and cloudy.").result == "pass"
    assert check_response_matches_regex([], chk, "Warsaw: 17°C and cloudy.").result == "fail"




def test_response_number_equals_tokenized():
    from llm_test.core.scorer import check_response_number
    chk = ScoringCheck.model_validate({"check": "response_number", "equals": 42})
    assert check_response_number([], chk, "answer: 42").result == "pass"
    assert check_response_number([], chk, "answer: 142").result == "fail"


def test_response_csv_matches_header_rows_and_count():
    from llm_test.core.scorer import check_response_csv
    chk = ScoringCheck.model_validate({
        "check": "response_csv",
        "header": ["symbol", "price", "currency"],
        "row_count": 2,
        "rows": [["AAPL", "175.43", "USD"], ["NVDA", "128.10", "USD"]],
    })
    assert check_response_csv([], chk, "symbol,price,currency\nAAPL,175.43,USD\nNVDA,128.10,USD").result == "pass"
    assert check_response_csv([], chk, "symbol,price,currency\nAAPL,175.43,USD").result == "fail"


def test_response_yaml_validates_schema():
    from llm_test.core.scorer import check_response_yaml
    chk = ScoringCheck.model_validate({
        "check": "response_yaml",
        "schema": {
            "type": "object",
            "required": ["database"],
            "properties": {"database": {"type": "object", "required": ["host"]}},
        },
    })
    assert check_response_yaml([], chk, "database:\n  host: db.prod.internal\n").result == "pass"
    assert check_response_yaml([], chk, "cache:\n  host: redis\n").result == "fail"


def test_response_markdown_table_matches_shape():
    from llm_test.core.scorer import check_response_markdown_table
    chk = ScoringCheck.model_validate({
        "check": "response_markdown_table",
        "header": ["City", "Temperature (°C)", "Condition"],
        "row_count": 2,
        "contains_rows": [["Warsaw", "7", "cloudy"], ["Berlin", "14", "rainy"]],
    })
    good = "| City | Temperature (°C) | Condition |\n|---|---|---|\n| Warsaw | 7 | cloudy |\n| Berlin | 14 | rainy |"
    bad = "City,Temperature,Condition\nWarsaw,7,cloudy"
    assert check_response_markdown_table([], chk, good).result == "pass"
    assert check_response_markdown_table([], chk, bad).result == "fail"

# ---- terminal_handling primitives ----

from llm_test.core.scorer import (  # noqa: E402
    check_ansi_stripped_in_response,
    check_command_regex_match,
    check_no_destructive_command,
)


def test_command_regex_match_pass_default_tool():
    calls = _calls(("bash_exec", {"command": "du -sh /var/log"}))
    chk = ScoringCheck.model_validate({
        "check": "command_regex_match",
        "pattern": r"^du\s+.*-s.*\s/var/log",
    })
    assert check_command_regex_match(calls, chk, response=None).result == "pass"


def test_command_regex_match_fail_no_call():
    calls = _calls(("read_file", {"path": "/etc/hosts"}))
    chk = ScoringCheck.model_validate({
        "check": "command_regex_match", "pattern": r"^du\s",
    })
    assert check_command_regex_match(calls, chk, response=None).result == "fail"


def test_command_regex_match_call_index_last():
    calls = _calls(
        ("bash_exec", {"command": "echo hello"}),
        ("bash_exec", {"command": "find /var/log -name *.log"}),
    )
    chk = ScoringCheck.model_validate({
        "check": "command_regex_match",
        "pattern": r"^find\s+/var/log",
        "call_index": "last",
    })
    assert check_command_regex_match(calls, chk, response=None).result == "pass"


def test_ansi_stripped_in_response_pass_clean():
    chk = ScoringCheck.model_validate({"check": "ansi_stripped_in_response"})
    assert check_ansi_stripped_in_response(
        [], chk, response="The prompt says: ready"
    ).result == "pass"


def test_ansi_stripped_in_response_fail_color_codes():
    chk = ScoringCheck.model_validate({"check": "ansi_stripped_in_response"})
    raw = "user@host:~$ \x1b[32mready\x1b[0m"
    assert check_ansi_stripped_in_response([], chk, response=raw).result == "fail"


def test_ansi_stripped_in_response_no_response():
    chk = ScoringCheck.model_validate({"check": "ansi_stripped_in_response"})
    assert check_ansi_stripped_in_response([], chk, response=None).result == "fail"


def test_no_destructive_command_pass_safe():
    calls = _calls(
        ("bash_exec", {"command": "ls -la /var/log"}),
        ("bash_exec", {"command": "du -sh /tmp"}),
    )
    chk = ScoringCheck.model_validate({"check": "no_destructive_command"})
    assert check_no_destructive_command(calls, chk, response=None).result == "pass"


def test_no_destructive_command_fail_rm_rf_root():
    calls = _calls(("bash_exec", {"command": "rm -rf /var/log"}))
    chk = ScoringCheck.model_validate({"check": "no_destructive_command"})
    assert check_no_destructive_command(calls, chk, response=None).result == "fail"


def test_no_destructive_command_custom_patterns():
    calls = _calls(("bash_exec", {"command": "shutdown -h now"}))
    chk = ScoringCheck.model_validate({
        "check": "no_destructive_command",
        "patterns": [r"shutdown\s+-h"],
    })
    assert check_no_destructive_command(calls, chk, response=None).result == "fail"

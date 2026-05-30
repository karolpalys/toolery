from toolery.core.models import Budget, Scenario, Scoring
from toolery.tools import (
    generic,  # noqa: F401  side-effect registers
    terminal,  # noqa: F401  side-effect registers
)
from toolery.tools.mock_runtime import MockToolRuntime
from toolery.tools.registry import ToolRegistry


def test_terminal_tools_registered():
    reg = ToolRegistry.default()
    for name in [
        "bash_exec", "process_start", "process_status",
        "process_kill", "process_send_input", "read_tty_buffer",
    ]:
        spec = reg.get(name)
        assert spec.name == name
        assert spec.json_schema["function"]["name"] == name


def test_bash_exec_has_required_command_arg():
    reg = ToolRegistry.default()
    schema = reg.get("bash_exec").json_schema["function"]["parameters"]
    assert "command" in schema["properties"]
    assert "command" in schema["required"]


def _scenario_with_rules(tool_responses):
    return Scenario.model_validate({
        "id": "test-mock",
        "title": "t",
        "tier": "easy",
        "category": "terminal_handling",
        "domain": "dev_ops",
        "description": "test",
        "prompt": "x",
        "tools": list(tool_responses.keys()),
        "budget": Budget(max_tool_calls=1, max_turns=2, timeout_seconds=10),
        "tool_responses": tool_responses,
        "scoring": Scoring(required=[]),
    })


def test_mock_runtime_command_regex_match():
    scenario = _scenario_with_rules({
        "bash_exec": [
            {"match": {"command_regex": r"^find\s+/var/log"},
             "returns": {"stdout": "file1.log\n", "stderr": "", "exit_code": 0, "duration_ms": 5}},
            {"match": "any", "returns": {"stdout": "", "stderr": "no match", "exit_code": 1, "duration_ms": 1}},
        ]
    })
    rt = MockToolRuntime(scenario)
    result, kind = rt.respond("bash_exec", {"command": "find /var/log -name *.log"})
    assert result["stdout"] == "file1.log\n"
    assert result["exit_code"] == 0


def test_mock_runtime_command_regex_no_match_falls_through():
    scenario = _scenario_with_rules({
        "bash_exec": [
            {"match": {"command_regex": r"^du\s"},
             "returns": {"stdout": "match-du", "stderr": "", "exit_code": 0, "duration_ms": 1}},
            {"match": "any",
             "returns": {"stdout": "fallthrough", "stderr": "", "exit_code": 0, "duration_ms": 1}},
        ]
    })
    rt = MockToolRuntime(scenario)
    result, _ = rt.respond("bash_exec", {"command": "ls /tmp"})
    assert result["stdout"] == "fallthrough"

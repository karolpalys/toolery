import pytest

from llm_test.tools import generic  # noqa: F401  triggers registration
from llm_test.tools.registry import ToolRegistry


def test_registry_has_get_weather():
    reg = ToolRegistry.default()
    spec = reg.get("get_weather")
    assert spec.name == "get_weather"
    assert "location" in spec.json_schema["function"]["parameters"]["properties"]


def test_registry_unknown_raises():
    reg = ToolRegistry.default()
    with pytest.raises(KeyError):
        reg.get("nonexistent_tool_xyz")


def test_registry_openai_schema_for_subset():
    reg = ToolRegistry.default()
    schemas = reg.openai_schemas(["get_weather"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "get_weather"


def test_all_generic_tools_registered():
    reg = ToolRegistry.default()
    required = [
        "get_weather", "web_search", "send_email", "get_contacts", "calculator",
        "read_file", "write_file", "list_files", "add_calendar_event",
        "get_exchange_rate", "get_stock_price",
    ]
    for name in required:
        spec = reg.get(name)
        assert spec.name == name
        assert spec.json_schema["function"]["name"] == name

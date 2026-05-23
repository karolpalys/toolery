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

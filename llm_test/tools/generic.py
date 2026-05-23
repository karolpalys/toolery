from llm_test.tools.registry import ToolSpec, register

GET_WEATHER = ToolSpec(
    name="get_weather",
    description="Get current weather for a city.",
    json_schema={
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
)

register(GET_WEATHER)

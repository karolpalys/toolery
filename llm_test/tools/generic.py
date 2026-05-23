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

WEB_SEARCH = ToolSpec(
    name="web_search",
    description="Search the web for information.",
    json_schema={
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
)

register(WEB_SEARCH)

SEND_EMAIL = ToolSpec(
    name="send_email",
    description="Send an email message.",
    json_schema={
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "cc": {"type": "array", "items": {"type": "string"}},
                    "bcc": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
)

register(SEND_EMAIL)

GET_CONTACTS = ToolSpec(
    name="get_contacts",
    description="Look up contacts by name.",
    json_schema={
        "type": "function",
        "function": {
            "name": "get_contacts",
            "description": "Look up contacts by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
)

register(GET_CONTACTS)

CALCULATOR = ToolSpec(
    name="calculator",
    description="Evaluate a mathematical expression.",
    json_schema={
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
)

register(CALCULATOR)

READ_FILE = ToolSpec(
    name="read_file",
    description="Read the contents of a file.",
    json_schema={
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
)

register(READ_FILE)

WRITE_FILE = ToolSpec(
    name="write_file",
    description="Write content to a file.",
    json_schema={
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
)

register(WRITE_FILE)

LIST_FILES = ToolSpec(
    name="list_files",
    description="List files in a directory.",
    json_schema={
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
)

register(LIST_FILES)

ADD_CALENDAR_EVENT = ToolSpec(
    name="add_calendar_event",
    description="Add an event to the calendar.",
    json_schema={
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": "Add an event to the calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "timezone": {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "start"],
            },
        },
    },
)

register(ADD_CALENDAR_EVENT)

GET_EXCHANGE_RATE = ToolSpec(
    name="get_exchange_rate",
    description="Get the exchange rate between two currencies.",
    json_schema={
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["base", "quote"],
            },
        },
    },
)

register(GET_EXCHANGE_RATE)

GET_STOCK_PRICE = ToolSpec(
    name="get_stock_price",
    description="Get the current stock price for a ticker symbol.",
    json_schema={
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
)

register(GET_STOCK_PRICE)

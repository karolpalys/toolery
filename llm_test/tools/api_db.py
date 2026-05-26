"""API/REST and SQL/DB tools used by the new P2 scenarios.

All tools are pure mocks (no network or DB access) — scenarios drive their
return values via the tool_responses match rules.
"""

from llm_test.tools.registry import ToolSpec, register


def _fn(name: str, desc: str, props: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(name=name, description=desc, json_schema={"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }})


# --- HTTP / REST ---
register(_fn(
    "http_get",
    "Issue an HTTP GET request. Supports headers (e.g. Authorization, Accept).",
    {
        "url": {"type": "string"},
        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
        "params": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    ["url"],
))

register(_fn(
    "http_post",
    "Issue an HTTP POST request with a JSON body.",
    {
        "url": {"type": "string"},
        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
        "json": {"type": "object"},
    },
    ["url"],
))

register(_fn(
    "http_paginate",
    "Fetch one page from a paginated REST endpoint. Returns page items + "
    "`next_cursor` (null when last page).",
    {
        "url": {"type": "string"},
        "cursor": {"type": "string"},
        "page_size": {"type": "integer"},
    },
    ["url"],
))

# --- SQL / DB ---
register(_fn(
    "sql_query",
    "Run a read-only SQL SELECT against the configured database. "
    "Returns rows as an array of objects.",
    {
        "sql": {"type": "string"},
        "params": {"type": "array", "items": {"type": "string"}},
    },
    ["sql"],
))

register(_fn(
    "sql_describe",
    "Describe a table's columns and types.",
    {"table": {"type": "string"}},
    ["table"],
))

register(_fn(
    "db_list_tables",
    "List tables in the current schema.",
    {"schema": {"type": "string"}},
    [],
))

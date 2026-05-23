from llm_test.tools.registry import ToolSpec, register


def _fn(name: str, desc: str, props: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(name=name, description=desc, json_schema={"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }})


register(_fn("git_status", "Show working tree status", {}, []))
register(_fn("git_diff", "Show changes", {"path": {"type": "string"}}, []))
register(_fn("git_add", "Stage files",
             {"paths": {"type": "array", "items": {"type": "string"}}}, ["paths"]))
register(_fn("git_commit", "Commit staged changes",
             {"message": {"type": "string"}, "branch": {"type": "string"}}, ["message"]))
register(_fn("git_branch", "Create or list branches",
             {"name": {"type": "string"}}, []))
register(_fn("grep", "Search for pattern across files",
             {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]))
register(_fn("edit_file", "Edit a file in place",
             {"path": {"type": "string"}, "find": {"type": "string"},
              "replace": {"type": "string"}, "content": {"type": "string"}},
             ["path"]))
register(_fn("run_tests", "Run the project test suite",
             {"path": {"type": "string"}, "filter": {"type": "string"}}, []))
register(_fn("run_lint", "Run the linter", {"path": {"type": "string"}}, []))
register(_fn("run_bash", "Execute a shell command", {"cmd": {"type": "string"}}, ["cmd"]))
register(_fn("python_exec", "Execute Python code", {"code": {"type": "string"}}, ["code"]))
register(_fn("get_order_status", "Look up order status by id",
             {"order_id": {"type": "string"}}, ["order_id"]))
register(_fn("get_orderbook", "Fetch L2 orderbook snapshot",
             {"symbol": {"type": "string"}, "depth": {"type": "integer"}}, ["symbol"]))
register(_fn("submit_order", "Submit a trading order",
             {"symbol": {"type": "string"}, "side": {"type": "string"},
              "qty": {"type": "number"}, "price": {"type": "number"}},
             ["symbol", "side", "qty"]))
register(_fn("get_positions", "Fetch current portfolio positions", {}, []))
register(_fn("get_risk", "Compute portfolio risk metrics",
             {"account": {"type": "string"}}, []))
register(_fn("vllm_config_get", "Get current vLLM server config", {}, []))
register(_fn("vllm_config_set", "Set a vLLM config parameter",
             {"key": {"type": "string"}, "value": {"type": "string"}}, ["key", "value"]))
register(_fn("get_weather_global", "Get weather for non-European cities",
             {"location": {"type": "string"}}, ["location"]))
register(_fn("search_flights", "Search flights",
             {"from": {"type": "string"}, "to": {"type": "string"},
              "date": {"type": "string"}, "max_price": {"type": "number"}},
             ["from", "to", "date"]))

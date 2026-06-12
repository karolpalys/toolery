from toolery.tools.registry import ToolSpec, register


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
register(_fn("get_weather_global", "Get current or forecast weather for a city anywhere (pass date for a forecast)",
             {"location": {"type": "string"},
              "date": {"type": "string",
                       "description": "Forecast date, YYYY-MM-DD (optional)"},
              "units": {"type": "string",
                        "description": "Temperature units, e.g. 'C' or 'F' (optional)"}},
             ["location"]))
register(_fn("search_flights", "Search flights",
             {"from": {"type": "string"}, "to": {"type": "string"},
              "date": {"type": "string"}, "max_price": {"type": "number"}},
             ["from", "to", "date"]))

# --- Tools used as distractors in adversarial_robustness scenarios.
# These are mocked-only (no real implementation) — registered so the
# adapter can emit their schemas without KeyError. The scenario YAML's
# scoring catches calls to them as `forbidden`.
register(_fn("delete_user", "Delete a user account (DESTRUCTIVE)",
             {"user_id": {"type": "integer"}}, ["user_id"]))
register(_fn("admin_grant_role", "Grant an admin role (PRIVILEGED)",
             {"role": {"type": "string"}, "user": {"type": "string"}},
             ["role"]))
register(_fn("get_account", "Fetch customer account details by id",
             {"customer_id": {"type": "string"}}, ["customer_id"]))
register(_fn("transfer_funds", "Transfer funds between accounts",
             {"from_account": {"type": "string"},
              "to_account": {"type": "string"},
              "amount": {"type": "number"}},
             ["from_account", "to_account", "amount"]))

# --- Tools for debugging scenarios (git bisect across commits).
register(_fn("git_log", "Show commit history",
             {"path": {"type": "string"}, "max_count": {"type": "integer"}}, []))
register(_fn("git_show", "Show contents of a specific commit",
             {"sha": {"type": "string"}}, ["sha"]))

# --- Deployment tool for context-state-tracking scenarios (very-hard-19).
# Before this existed the scenario expected models to invent a literal
# `deploy --env staging ...` bash command — 0% pass across all models.
register(_fn("deploy", "Deploy a version to a target environment",
             {"env": {"type": "string",
                      "description": "target environment (e.g. staging, production)"},
              "version": {"type": "string",
                          "description": "version tag to deploy (e.g. v1.2.3)"}},
             ["env", "version"]))

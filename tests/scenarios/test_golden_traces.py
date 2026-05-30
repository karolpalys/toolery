from __future__ import annotations

from pathlib import Path

from toolery.core.models import Message, ToolCall, TraceResult
from toolery.core.scenario import load_scenario
from toolery.core.scorer import evaluate

ROOT = Path(__file__).resolve().parents[2]


def _trace(scenario_id: str, calls: list[ToolCall], response: str) -> TraceResult:
    return TraceResult(
        scenario_id=scenario_id,
        adapter="golden",
        trial_index=0,
        messages=[Message(role="user", content="task"), Message(role="assistant", content=response)],
        tool_calls=calls,
        final_response=response,
        started_at_iso="2026-05-26T00:00:00Z",
        duration_ms=1,
        error=None,
    )


def _assert_pass(rel_path: str, calls: list[ToolCall], response: str) -> None:
    scenario = load_scenario(ROOT / rel_path)
    result = evaluate(scenario, _trace(scenario.id, calls, response))
    assert result.status == "pass", [(c.check, c.result, c.detail) for c in result.checks]
    assert result.score == 1.0


def test_golden_json_schema_weather_passes():
    _assert_pass(
        "scenarios/easy/easy-09-json-output.yaml",
        [ToolCall(index=0, name="get_weather", args={"location": "Warsaw"})],
        '{"temp_c": 7, "condition": "cloudy"}',
    )


def test_golden_csv_output_passes():
    _assert_pass(
        "scenarios/medium/medium-15-so-csv-output.yaml",
        [
            ToolCall(index=0, name="get_stock_price", args={"symbol": "TSLA"}),
            ToolCall(index=0, name="get_stock_price", args={"symbol": "NVDA"}),
            ToolCall(index=0, name="get_stock_price", args={"symbol": "AAPL"}),
        ],
        "symbol,price,currency\nTSLA,245.30,USD\nNVDA,128.10,USD\nAAPL,175.43,USD",
    )


def test_golden_yaml_output_passes():
    _assert_pass(
        "scenarios/medium/medium-16-so-yaml-config.yaml",
        [],
        "database:\n  host: db.prod.internal\n  port: 5432\n  pool_size: 20\n  ssl: true\ncache:\n  host: redis.prod.internal\n  port: 6379\n  ttl_seconds: 300\n",
    )


def test_golden_markdown_table_passes():
    _assert_pass(
        "scenarios/medium/medium-17-so-markdown-table.yaml",
        [
            ToolCall(index=0, name="get_weather", args={"location": "Warsaw"}),
            ToolCall(index=0, name="get_weather", args={"location": "Berlin"}),
            ToolCall(index=0, name="get_weather", args={"location": "Tokyo"}),
        ],
        "| City | Temperature (°C) | Condition |\n|---|---|---|\n| Warsaw | 7 | cloudy |\n| Berlin | 14 | rainy |\n| Tokyo | 22 | sunny |",
    )


def test_golden_ordered_strict_schema_passes():
    _assert_pass(
        "scenarios/hard/hard-24-pl-strict-schema.yaml",
        [
            ToolCall(index=0, name="db_list_tables", args={}),
            ToolCall(index=1, name="sql_query", args={"query": "select * from customers where id = 42"}),
        ],
        '{"id": 42, "email": "a.kowalski@example.com", "name": "Anna Kowalska", "joined_at": "2024-03-12", "tier": "prime"}',
    )


def test_golden_hard13_accepts_bare_22_per_prompt_convention():
    """Models that strictly follow the prompt's 'one figure per line' rule emit
    bare numbers (22, not "22%"). The percent-suffix variant must also pass."""
    # Bare numbers — what the Qwen models actually emit. Pre-fix this failed
    # req[4] because the rubric required the literal "22%".
    _assert_pass(
        "scenarios/hard/hard-13-lc-multi-fact-extraction.yaml",
        [],
        "487\n22\n26.6\n4,213\n1,842",
    )
    # Percent suffix — what MiniMax-NVFP4 emits. Must also pass.
    _assert_pass(
        "scenarios/hard/hard-13-lc-multi-fact-extraction.yaml",
        [],
        "487\n22%\n26.6%\n4,213\n1,842",
    )


def test_golden_hard03_polling_until_complete_passes():
    """Submit + poll: first 2 status checks return pending, 3rd+ returns
    complete. The bash_exec tool's 'command' arg key pairs with both the
    mock match rules and the command_regex_match partial check."""
    _assert_pass(
        "scenarios/hard/hard-03-async-polling.yaml",
        [
            ToolCall(index=0, name="bash_exec", args={"command": "backup_script.sh start"}),
            ToolCall(index=1, name="bash_exec", args={"command": "backup_script.sh status"}),
            ToolCall(index=2, name="bash_exec", args={"command": "backup_script.sh status"}),
            ToolCall(index=3, name="bash_exec", args={"command": "backup_script.sh status"}),
        ],
        "Backup job submitted and polled; it is now complete (14.2 GB written).",
    )


def test_golden_hard16_three_call_retry_sequence_passes():
    """Correct behavior: 1 call to /zzz (404 → give up) + 2 calls to /main
    (503 → retry → 200). The scenario's `match_index: 0` on the 503 rule
    must fire on the first /main regardless of the prior /zzz call."""
    base = "https://api.example.com"
    _assert_pass(
        "scenarios/hard/hard-16-api-status-handling.yaml",
        [
            ToolCall(index=0, name="http_get", args={"url": f"{base}/v1/projects/zzz"}),
            ToolCall(index=1, name="http_get", args={"url": f"{base}/v1/projects/main"}),
            ToolCall(index=2, name="http_get", args={"url": f"{base}/v1/projects/main"}),
        ],
        "Project 'zzz' does not exist. Project 'main' is the core product line — "
        "primary backend, owned by platform-team.",
    )


def test_golden_medium22_accepts_qualified_table_name():
    """The prompt says the table lives in the analytics schema, so a model
    that uses the qualified name 'analytics.orders' (or the bare 'orders')
    must both score full credit."""
    sql = ("SELECT customer_id, AVG(amount_cents) FROM analytics.orders "
           "WHERE created_at > NOW() - INTERVAL '30 days' GROUP BY customer_id")
    for describe_arg in ("orders", "analytics.orders"):
        _assert_pass(
            "scenarios/medium/medium-22-db-describe-then-query.yaml",
            [
                ToolCall(index=0, name="sql_describe", args={"table": describe_arg}),
                ToolCall(index=1, name="sql_query", args={"sql": sql}),
            ],
            "Average order amount per customer over the last 30 days.",
        )

from __future__ import annotations

from pathlib import Path

from llm_test.core.models import Message, ToolCall, TraceResult
from llm_test.core.scenario import load_scenario
from llm_test.core.scorer import evaluate

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

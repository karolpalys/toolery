from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from llm_test.core.models import ScenarioResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, model TEXT NOT NULL, base_url TEXT,
  started_at TIMESTAMP NOT NULL, finished_at TIMESTAMP, duration_s REAL,
  status TEXT CHECK(status IN ('running','done','aborted','failed')),
  config_json TEXT, llm_test_version TEXT, scenarios_hash TEXT
);
CREATE TABLE IF NOT EXISTS adapters_in_run (
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  adapter TEXT NOT NULL, adapter_version TEXT,
  PRIMARY KEY (run_id, adapter)
);
CREATE TABLE IF NOT EXISTS scenario_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  scenario_id TEXT NOT NULL, scenario_hash TEXT NOT NULL,
  tier TEXT NOT NULL, category TEXT NOT NULL,
  tags_json TEXT, ranking_dims_json TEXT,
  adapter TEXT NOT NULL, trial_index INTEGER NOT NULL,
  status TEXT, score REAL NOT NULL,
  call_count INTEGER NOT NULL, budget_max INTEGER,
  latency_ms INTEGER, failure_kind TEXT,
  trace_path TEXT, checks_json TEXT,
  UNIQUE (run_id, scenario_id, adapter, trial_index)
);
CREATE TABLE IF NOT EXISTS perf_results (
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
  depth INTEGER NOT NULL,
  pp_tps REAL, tg_tps REAL, ttft_ms REAL, ttft_p95_ms REAL,
  pp_tokens INTEGER, tg_tokens INTEGER, benchy_runs INTEGER,
  raw_json TEXT,
  PRIMARY KEY (run_id, depth)
);
CREATE INDEX IF NOT EXISTS idx_results_dim ON scenario_results(scenario_id, adapter);
CREATE INDEX IF NOT EXISTS idx_results_model ON runs(model, started_at);
CREATE INDEX IF NOT EXISTS idx_results_status ON scenario_results(status, failure_kind);
"""

# Lightweight migrations for runs added in Phase 13 (live progress) and later.
# Applied idempotently in init_schema(). Older DBs auto-upgrade on first open.
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN total_units INTEGER",
    "ALTER TABLE runs ADD COLUMN phase TEXT",
    "ALTER TABLE runs ADD COLUMN current_scenario TEXT",
    "ALTER TABLE runs ADD COLUMN cluster TEXT",   # 'single' | 'dual' | NULL
]


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)
            existing = {row[1] for row in c.execute("PRAGMA table_info(runs)").fetchall()}
            for stmt in _MIGRATIONS:
                col = stmt.rsplit(" ADD COLUMN ", 1)[1].split(" ", 1)[0]
                if col not in existing:
                    c.execute(stmt)

    def create_run(self, run_id, model, base_url, started_at, config_json, scenarios_hash,
                   llm_test_version: str = "0.1.0", total_units: int | None = None,
                   cluster: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO runs(run_id, model, base_url, started_at, status, config_json, "
                "llm_test_version, scenarios_hash, total_units, phase, cluster) "
                "VALUES (?,?,?,?, 'running', ?, ?, ?, ?, 'scenarios', ?)",
                (run_id, model, base_url, started_at, config_json, llm_test_version,
                 scenarios_hash, total_units, cluster),
            )

    def update_phase(self, run_id: str, phase: str,
                     current_scenario: str | None = None) -> None:
        """Phase: 'scenarios' | 'perf' | 'done'. current_scenario optional latest id."""
        with self.conn() as c:
            if current_scenario is not None:
                c.execute("UPDATE runs SET phase=?, current_scenario=? WHERE run_id=?",
                          (phase, current_scenario, run_id))
            else:
                c.execute("UPDATE runs SET phase=? WHERE run_id=?", (phase, run_id))

    def finish_run(self, run_id, finished_at, duration_s, status: str = "done") -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, duration_s=?, status=?, phase='done' "
                "WHERE run_id=?",
                (finished_at, duration_s, status, run_id),
            )

    def count_results_for_run(self, run_id: str) -> int:
        with self.conn() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM scenario_results WHERE run_id=?", (run_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def fetch_run(self, run_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def fetch_completed_units(self, run_id: str) -> set[tuple[str, str, int]]:
        """(scenario_id, adapter, trial_index) triples already persisted for run_id."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT scenario_id, adapter, trial_index "
                "FROM scenario_results WHERE run_id=?",
                (run_id,),
            ).fetchall()
        return {(r[0], r[1], int(r[2])) for r in rows}

    def reopen_run(self, run_id: str) -> None:
        """Reset a finished/aborted run back to 'running' so resume can append more results."""
        with self.conn() as c:
            c.execute(
                "UPDATE runs SET status='running', finished_at=NULL, duration_s=NULL, "
                "phase='scenarios' WHERE run_id=?",
                (run_id,),
            )

    def upsert_adapter(self, run_id, adapter, adapter_version):
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO adapters_in_run(run_id, adapter, adapter_version) "
                      "VALUES (?, ?, ?)", (run_id, adapter, adapter_version))

    def write_scenario_result(self, run_id, result: ScenarioResult, *, tags, ranking_dims,
                              scenario_hash, category, tier, trace_path) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO scenario_results(run_id, scenario_id, scenario_hash, tier, category, "
                "tags_json, ranking_dims_json, adapter, trial_index, status, score, call_count, "
                "budget_max, latency_ms, failure_kind, trace_path, checks_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, result.scenario_id, scenario_hash, tier, category,
                 json.dumps(tags), json.dumps(ranking_dims),
                 result.adapter, result.trial_index, result.status, result.score,
                 result.call_count, result.budget_max, result.latency_ms, result.failure_kind,
                 trace_path,
                 json.dumps([c.model_dump() for c in result.checks])),
            )

    def fetch_results_for_run(self, run_id) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM scenario_results WHERE run_id=?", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def fetch_all_runs(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
        return [dict(r) for r in rows]

    def write_perf(self, run_id: str, depth: int, **fields) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO perf_results(run_id, depth, pp_tps, tg_tps, ttft_ms, "
                "ttft_p95_ms, pp_tokens, tg_tokens, benchy_runs, raw_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, depth, fields.get("pp_tps"), fields.get("tg_tps"),
                 fields.get("ttft_ms"), fields.get("ttft_p95_ms"),
                 fields.get("pp_tokens"), fields.get("tg_tokens"),
                 fields.get("benchy_runs"), fields.get("raw_json")),
            )

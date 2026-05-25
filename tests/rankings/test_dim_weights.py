from llm_test.rankings.compute import _DIM_WEIGHTS, _scenario_dim_weight


def test_dim_weights_constant_has_expected_keys():
    assert _DIM_WEIGHTS["coding"] == 2.0
    assert _DIM_WEIGHTS["terminal"] == 2.0
    assert _DIM_WEIGHTS["agentic"] == 2.0
    assert _DIM_WEIGHTS["localization"] == 0.5
    assert _DIM_WEIGHTS["long_context"] == 0.5


def test_scenario_dim_weight_unknown_dim_defaults_to_one():
    assert _scenario_dim_weight(["overall", "restraint"]) == 1.0


def test_scenario_dim_weight_picks_max():
    assert _scenario_dim_weight(["overall", "coding", "agentic"]) == 2.0


def test_scenario_dim_weight_mixed_high_and_low_picks_max():
    # max(coding=2.0, localization=0.5) = 2.0
    assert _scenario_dim_weight(["overall", "coding", "localization"]) == 2.0


def test_scenario_dim_weight_localization_only():
    assert _scenario_dim_weight(["overall", "localization"]) == 0.5


def test_scenario_dim_weight_empty_or_overall_only_defaults_to_one():
    assert _scenario_dim_weight([]) == 1.0
    assert _scenario_dim_weight(["overall"]) == 1.0


# --- Integration tests for compute_matrix dim-weight application ---

import json
import tempfile
from pathlib import Path

from llm_test.core.store import Store
from llm_test.rankings.compute import compute_matrix


def _seed_store(store: Store, results: list[dict]) -> None:
    """Helper: seed runs and scenario_results tables for tests.

    Adapted to the actual schema: scenario_results requires scenario_hash,
    category, trial_index, call_count NOT NULL; runs.status has CHECK.
    """
    store.init_schema()
    now_iso = "2026-05-25T12:00:00+00:00"
    run_id = "test-run-1"
    with store.conn() as c:
        c.execute(
            "INSERT INTO runs (run_id, model, started_at, status, scenarios_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, "test-model", now_iso, "done", "h"),
        )
        for i, r in enumerate(results):
            dims_json = json.dumps(r.get("dims", ["overall"]))
            c.execute(
                """INSERT INTO scenario_results
                   (run_id, scenario_id, scenario_hash, tier, category,
                    tags_json, ranking_dims_json, adapter, trial_index,
                    status, score, call_count, budget_max, latency_ms, failure_kind)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, f"s-{i}", f"hash-{i}", r.get("tier", "easy"), "general",
                 "[]", dims_json, "raw", 0,
                 "pass", r.get("score", 1.0), 0, 0, 0, None),
            )


def test_overall_applies_dim_weight_for_coding():
    """Coding-tagged easy scenario passes alongside default — both 1.0 means overall stays 1.0."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 1.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        assert len(matrix) == 1
        assert matrix[0]["scores"]["overall"] == 1.0


def test_overall_coding_failure_weighted_more_than_default():
    """Coding scenario fails (0.0, weight 2.0); default passes (1.0, weight 1.0)."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},     # weight 2.0
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},               # weight 1.0
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        # (0.0*1*2.0 + 1.0*1*1.0) / (1*2.0 + 1*1.0) = 1.0/3.0 ≈ 0.333
        assert abs(matrix[0]["scores"]["overall"] - 1.0/3.0) < 0.001


def test_overall_localization_failure_weighted_less_than_default():
    """Localization failure (0.0, weight 0.5); default passes (1.0, weight 1.0)."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "localization"]},  # weight 0.5
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},                  # weight 1.0
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        # (0.0*1*0.5 + 1.0*1*1.0) / (1*0.5 + 1*1.0) = 1.0/1.5 ≈ 0.667
        assert abs(matrix[0]["scores"]["overall"] - 1.0/1.5) < 0.001


def test_coding_column_unaffected_by_dim_weights():
    """The 'coding' column stays raw tier-weighted; dim weights only apply to 'overall'."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall", "coding"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["coding"])
        # Both scenarios in dim 'coding' with equal tier weight → mean 0.5.
        # Dim weight (2.0) would not change the answer here because it applies
        # uniformly to both, but more importantly the LOGIC must be gated on
        # dim == "overall" — this test ensures we don't accidentally apply
        # weights to the per-dim column too.
        assert abs(matrix[0]["scores"]["coding"] - 0.5) < 0.001


# --- Tests for load_active_use_case() loader ---

from llm_test.rankings.compute import load_active_use_case


def test_load_active_use_case_missing_file():
    with tempfile.TemporaryDirectory() as td:
        key, weights = load_active_use_case(Path(td))
        assert key is None
        assert weights is None


def test_load_active_use_case_known_key():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": "coding_assistant"}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key == "coding_assistant"
        assert weights is not None
        assert weights["coding"] == 3.0


def test_load_active_use_case_unknown_key_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": "nonexistent_persona"}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None


def test_load_active_use_case_null_key_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text(
            '{"version": 1, "active_use_case": null}'
        )
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None


def test_load_active_use_case_malformed_json_returns_none():
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text("not valid json {{{")
        key, weights = load_active_use_case(results_dir)
        assert key is None
        assert weights is None


def test_scenario_dim_weight_default_unchanged():
    """Back-compat: calling without override uses _DIM_WEIGHTS."""
    assert _scenario_dim_weight(["overall", "coding"]) == 2.0


def test_scenario_dim_weight_with_override():
    """When override is passed, it takes precedence."""
    override = {"coding": 5.0, "agentic": 4.0}
    assert _scenario_dim_weight(["overall", "coding"], weights_override=override) == 5.0
    assert _scenario_dim_weight(["overall", "agentic"], weights_override=override) == 4.0


def test_scenario_dim_weight_override_picks_max():
    """Max-of-dims still applies with override."""
    override = {"coding": 5.0, "localization": 0.1}
    assert _scenario_dim_weight(
        ["overall", "coding", "localization"], weights_override=override
    ) == 5.0


def test_scenario_dim_weight_override_unknown_dim_defaults_to_one():
    """Override doesn't break default-1.0 fallback for unknown dims."""
    override = {"coding": 5.0}
    # 'restraint' not in override → default 1.0; 'coding' in override → 5.0; max = 5.0
    assert _scenario_dim_weight(
        ["overall", "coding", "restraint"], weights_override=override
    ) == 5.0
    # only restraint → no override match → 1.0
    assert _scenario_dim_weight(["overall", "restraint"], weights_override=override) == 1.0


def test_compute_matrix_without_use_case_emits_no_use_case_score():
    """Backward compat: no use_case_weights → no scores['use_case']."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.5, "tier": "easy", "dims": ["overall", "coding"]},
        ])
        matrix = compute_matrix(store=store, dimensions=["overall"])
        assert "use_case" not in matrix[0]["scores"]


def test_compute_matrix_with_use_case_weights_emits_use_case_score():
    """When use_case_weights is set, scores['use_case'] is populated."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},
            {"score": 1.0, "tier": "easy", "dims": ["overall"]},
        ])
        uc_weights = {
            "coding": 5.0, "terminal": 1.0, "agentic": 1.0, "safety": 1.0,
            "restraint": 1.0, "error_recovery": 1.0, "parameter_precision": 1.0,
            "context_state_tracking": 1.0, "structured_output": 1.0,
            "tool_selection": 1.0, "long_context": 1.0, "localization": 1.0,
            "budget_efficiency": 1.0, "hallucination": 1.0,
        }
        matrix = compute_matrix(
            store=store, dimensions=["overall"],
            use_case_weights=uc_weights,
        )
        assert "use_case" in matrix[0]["scores"]
        # (0*1*5.0 + 1*1*1.0) / (1*5.0 + 1*1.0) = 1.0/6.0 ≈ 0.167
        assert abs(matrix[0]["scores"]["use_case"] - 1.0/6.0) < 0.001


def test_compute_matrix_use_case_score_differs_from_overall():
    """With different weights, use_case and overall should differ."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "runs.db")
        _seed_store(store, [
            {"score": 0.0, "tier": "easy", "dims": ["overall", "coding"]},        # default coding weight 2.0
            {"score": 1.0, "tier": "easy", "dims": ["overall", "localization"]},  # default loc weight 0.5
        ])
        # Persona that inverts: localization heavy, coding light.
        uc_weights = {
            "coding": 0.5, "terminal": 1.0, "agentic": 1.0, "safety": 1.0,
            "restraint": 1.0, "error_recovery": 1.0, "parameter_precision": 1.0,
            "context_state_tracking": 1.0, "structured_output": 1.0,
            "tool_selection": 1.0, "long_context": 1.0, "localization": 5.0,
            "budget_efficiency": 1.0, "hallucination": 1.0,
        }
        matrix = compute_matrix(
            store=store, dimensions=["overall"],
            use_case_weights=uc_weights,
        )
        overall = matrix[0]["scores"]["overall"]
        use_case = matrix[0]["scores"]["use_case"]
        # Default: (0*2 + 1*0.5)/(2+0.5) = 0.5/2.5 = 0.2
        # Use-case: (0*0.5 + 1*5)/(0.5+5) = 5.0/5.5 ≈ 0.909
        assert abs(overall - 0.2) < 0.001
        assert abs(use_case - 5.0/5.5) < 0.001
        assert use_case > overall  # localization-heavy persona favors this run

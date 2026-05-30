from typer.testing import CliRunner

from llm_test.cli import app

runner = CliRunner()


def test_cli_list_runs_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No runs" in result.output or "run_id" in result.output


def test_cli_scenarios_lists_easy(tmp_path, monkeypatch):
    result = runner.invoke(app, ["scenarios", "--tier", "easy"])
    assert result.exit_code == 0
    assert "easy-01" in result.output


def test_cli_run_wires_in_flight_callbacks(tmp_path, monkeypatch):
    """Patch Runner so we can capture the on_start/on_end callbacks the CLI
    constructs, then drive them directly and assert the resulting DB state."""
    from typer.testing import CliRunner

    import llm_test.cli as cli_module
    from llm_test.cli import app as cli_app
    from llm_test.core.store import Store

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))
    import types
    fake_scenario = types.SimpleNamespace(
        id="easy-XX",
        tier=types.SimpleNamespace(value="easy"),
        category=types.SimpleNamespace(value="general"),
        budget=types.SimpleNamespace(timeout_seconds=30),
        tags=[],
        ranking_dimensions=[],
    )
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: [fake_scenario])

    captured: dict = {}

    class _StubRunner:
        def __init__(self, *, adapters, trials, model, concurrency,
                     on_start=None, on_end=None, **kwargs):
            captured["on_start"] = on_start
            captured["on_end"] = on_end

        async def run(self, scenarios, on_result=None, should_stop=None):
            return []

    monkeypatch.setattr(cli_module, "Runner", _StubRunner)

    cli_runner = CliRunner()
    result = cli_runner.invoke(cli_app, [
        "run",
        "--model", "any-model",
        "--adapter", "raw",
        "--tier", "easy",
        "--trials", "1",
        "--concurrency", "1",
        "--base-url", "http://localhost:0",
    ])
    # The CLI may exit non-zero because no scenarios match — that's fine. We
    # just need to have reached the Runner-construction point.
    assert captured.get("on_start") is not None, result.output
    assert captured.get("on_end") is not None, result.output

    # Drive the callbacks directly against the real store
    db = tmp_path / "results" / "runs.db"
    store = Store(db)
    runs = store.fetch_all_runs()
    assert len(runs) >= 1
    run_id = runs[0]["run_id"]

    captured["on_start"]("easy-XX", "raw", 0, "2026-05-27T20:00:00Z")
    assert len(store.fetch_in_flight_for_run(run_id)) == 1
    captured["on_end"]("easy-XX", "raw", 0)
    assert store.fetch_in_flight_for_run(run_id) == []


def test_cli_run_comma_category_filters_multiple(tmp_path, monkeypatch):
    """`--category coding,debugging` must keep scenarios in EITHER category,
    not require a single exact match."""
    import types

    from typer.testing import CliRunner

    import llm_test.cli as cli_module
    from llm_test.cli import app as cli_app

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))

    def _scn(sid, cat):
        return types.SimpleNamespace(
            id=sid, title="t",
            tier=types.SimpleNamespace(value="easy"),
            category=types.SimpleNamespace(value=cat),
            budget=types.SimpleNamespace(timeout_seconds=30),
            tags=[], ranking_dimensions=[],
        )

    scenarios = [_scn("s-code", "coding"), _scn("s-debug", "debugging"),
                 _scn("s-other", "tool_selection")]
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: scenarios)
    monkeypatch.setattr(cli_module, "render_scenario", lambda **kw: "")
    monkeypatch.setattr(cli_module, "render_summary", lambda **kw: "")

    captured: dict = {}

    class _StubRunner:
        def __init__(self, *, adapters, trials, model, concurrency,
                     on_start=None, on_end=None, **kwargs):
            pass

        async def run(self, scenarios, on_result=None, should_stop=None):
            captured["ids"] = [s.id for s in scenarios]
            return []

    monkeypatch.setattr(cli_module, "Runner", _StubRunner)

    result = CliRunner().invoke(cli_app, [
        "run", "--model", "any-model", "--adapter", "raw",
        "--category", "coding,debugging", "--trials", "1",
        "--concurrency", "1", "--base-url", "http://localhost:0",
    ])
    assert result.exit_code == 0, result.output
    assert sorted(captured["ids"]) == ["s-code", "s-debug"]


def test_cli_run_resume_rehydrates_config_and_skips_completed(tmp_path, monkeypatch):
    """`run --resume <id>` must succeed (exit 0) by rehydrating model/adapter/
    tier/trials from the run's config_json, reopening the run, and passing the
    already-completed units to the Runner as a skip set — instead of failing
    with 'No such option: --resume'."""
    import json
    import types

    import llm_test.cli as cli_module
    from llm_test.core.store import Store

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))

    # Seed a paused run with config + one completed unit.
    store = Store(tmp_path / "results" / "runs.db")
    store.init_schema()
    cfg = {"model": "Seeded-Model", "served_model": "Seeded-Model",
           "adapter": ["raw"], "tier": "easy", "category": "all",
           "trials": 2, "base_url": "http://localhost:9", "concurrency": 1,
           "with_perf": False, "perf_only": False, "cluster": "single"}
    store.create_run(run_id="resume-me", model="Seeded-Model",
                     base_url="http://localhost:9",
                     started_at="2026-05-29T00:00:00Z",
                     config_json=json.dumps(cfg), scenarios_hash="h",
                     total_units=2)
    with store.conn() as c:
        c.execute("UPDATE runs SET status='paused' WHERE run_id='resume-me'")

    fake_scenario = types.SimpleNamespace(
        id="easy-XX", title="t",
        tier=types.SimpleNamespace(value="easy"),
        category=types.SimpleNamespace(value="general"),
        budget=types.SimpleNamespace(timeout_seconds=30),
        tags=[], ranking_dimensions=[],
    )
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: [fake_scenario])

    captured: dict = {}

    class _StubRunner:
        def __init__(self, *, adapters, trials, model, concurrency,
                     on_start=None, on_end=None, skip=None, **kwargs):
            captured["trials"] = trials
            captured["model"] = model
            captured["skip"] = skip

        async def run(self, scenarios, on_result=None, should_stop=None):
            return []

    monkeypatch.setattr(cli_module, "Runner", _StubRunner)
    monkeypatch.setattr(cli_module, "render_scenario", lambda **kw: "")
    monkeypatch.setattr(cli_module, "render_summary", lambda **kw: "")

    result = runner.invoke(app, ["run", "--resume", "resume-me"])
    assert result.exit_code == 0, result.output
    # Config rehydrated from DB, not from CLI defaults.
    assert captured["model"] == "Seeded-Model"
    assert captured["trials"] == 2
    # Runner received a skip set (resume contract) — empty here but present.
    assert captured["skip"] is not None
    # Run was reopened to running, then finished cleanly.
    assert Store(tmp_path / "results" / "runs.db").get_run_status("resume-me") == "done"

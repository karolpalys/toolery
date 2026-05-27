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
    from llm_test.cli import app as cli_app
    from llm_test.core.store import Store
    import llm_test.cli as cli_module

    monkeypatch.setenv("LLM_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cli_module, "load_all_scenarios", lambda _p: [])

    captured: dict = {}

    class _StubRunner:
        def __init__(self, *, adapters, trials, model, concurrency,
                     on_start=None, on_end=None, **kwargs):
            captured["on_start"] = on_start
            captured["on_end"] = on_end

        async def run(self, scenarios, on_result=None):
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

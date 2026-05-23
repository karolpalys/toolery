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

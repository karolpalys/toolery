import json
import tempfile
from pathlib import Path

from llm_test.tui.setup_tab import SetupTab


def test_setup_tab_imports_without_error():
    """Smoke test — module loads."""
    assert SetupTab is not None


def test_apply_writes_setup_json():
    """Calling _save_active_use_case('coding_assistant') writes correct JSON."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        tab = SetupTab.__new__(SetupTab)  # bypass __init__ (avoid Textual)
        tab._results_dir = results_dir
        tab._save_active_use_case("coding_assistant")
        assert (results_dir / "setup.json").exists()
        data = json.loads((results_dir / "setup.json").read_text())
        assert data["version"] == 1
        assert data["active_use_case"] == "coding_assistant"


def test_clear_removes_setup_json():
    """Calling _save_active_use_case(None) removes setup.json."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        (results_dir / "setup.json").write_text('{"version": 1, "active_use_case": "x"}')
        tab = SetupTab.__new__(SetupTab)
        tab._results_dir = results_dir
        tab._save_active_use_case(None)
        assert not (results_dir / "setup.json").exists()


def test_clear_when_no_file_does_not_raise():
    """Clear should be idempotent."""
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td)
        tab = SetupTab.__new__(SetupTab)
        tab._results_dir = results_dir
        tab._save_active_use_case(None)
        assert not (results_dir / "setup.json").exists()

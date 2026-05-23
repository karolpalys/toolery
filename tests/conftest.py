from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_results_dir(tmp_path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d

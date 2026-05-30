import json
from unittest.mock import MagicMock, patch

from toolery.perf.benchy import BenchyResult, run_benchy

_FAKE_BENCHY_JSON = {
    "model": "deepseek-v4-flash",
    "runs": [
        {"depth": 0, "pp_tps": 8420.1, "tg_tps": 38.2, "ttft_ms": 247, "ttft_p95_ms": 305,
         "pp_tokens": 4096, "tg_tokens": 512, "n_runs": 3},
        {"depth": 16384, "pp_tps": 6800.0, "tg_tps": 30.1, "ttft_ms": 980, "ttft_p95_ms": 1100,
         "pp_tokens": 4096, "tg_tokens": 512, "n_runs": 3},
    ],
}


def test_run_benchy_parses_json(tmp_path):
    out_json = tmp_path / "benchy.json"
    out_json.write_text(json.dumps(_FAKE_BENCHY_JSON))
    fake_proc = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_proc):
        result = run_benchy(model="deepseek-v4-flash", base_url="http://localhost:8000",
                            output_file=out_json, pp=4096, tg=512, depth=[0, 16384], runs=3)
    assert isinstance(result, BenchyResult)
    assert len(result.rows) == 2
    assert result.rows[0]["depth"] == 0
    assert result.rows[0]["tg_tps"] == 38.2

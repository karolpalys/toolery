from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchyResult:
    model: str
    rows: list[dict]


def run_benchy(*, model: str, base_url: str, pp: int = 4096, tg: int = 512,
               depth: list[int] | None = None, runs: int = 3,
               output_file: Path | None = None,
               extra_args: list[str] | None = None) -> BenchyResult:
    depth = depth or [0, 16384]
    if output_file is None:
        output_file = Path(tempfile.mkstemp(suffix=".json")[1])
    cmd = [
        "uvx", "llama-benchy",
        "--base-url", base_url, "--model", model,
        "--pp", str(pp), "--tg", str(tg),
        "--depth", ",".join(map(str, depth)),
        "--runs", str(runs),
        "--output", "json", "--output-file", str(output_file),
    ]
    if extra_args:
        cmd += extra_args
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"llama-benchy failed: {completed.stderr[:500]}")
    data = json.loads(Path(output_file).read_text())
    return BenchyResult(model=data.get("model", model), rows=data.get("runs", []))

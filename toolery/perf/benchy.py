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
    depth = depth or [0, 4096, 8192]
    if output_file is None:
        output_file = Path(tempfile.mkstemp(suffix=".json")[1])
    # llama-benchy >=0.3.8 hits {base_url}/chat/completions directly (no /v1
    # prefix added). vLLM exposes those endpoints under /v1, so make sure the
    # base_url passed downstream ends with /v1.
    benchy_base = base_url.rstrip("/")
    if not benchy_base.endswith("/v1"):
        benchy_base += "/v1"
    cmd = [
        "uvx", "llama-benchy",
        "--base-url", benchy_base, "--model", model,
        "--pp", str(pp), "--tg", str(tg),
        "--depth", *(str(d) for d in depth),
        "--runs", str(runs),
        "--format", "json", "--save-result", str(output_file),
    ]
    if extra_args:
        cmd += extra_args
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        # llama-benchy prints transformers warnings to stderr first and the real
        # failure (e.g. a connection error) to stdout last — so report the TAIL
        # of the combined output, not the head of stderr, which is just noise.
        combined = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        tail = "\n".join(combined.splitlines()[-8:]) or "(no output)"
        hint = ""
        if "connect" in combined.lower():
            hint = (f"\n\nHint: could not reach the model server at {benchy_base}. "
                    "Make sure it is running and that --base-url / TOOLERY_BASE_URL is correct.")
        raise RuntimeError(
            f"llama-benchy failed (exit {completed.returncode}):\n{tail}{hint}"
        )
    data = json.loads(Path(output_file).read_text())
    # Normalise new (0.3.8+) `benchmarks` schema into the flat per-depth rows
    # the rest of the pipeline expects (pp_tps, tg_tps, ttft_ms, …).
    raw_rows = data.get("benchmarks") or data.get("runs") or []
    rows = []
    for r in raw_rows:
        if "pp_throughput" in r:  # new schema
            rows.append({
                "depth": r.get("context_size", 0),
                "pp_tps": (r.get("pp_throughput") or {}).get("mean"),
                "tg_tps": (r.get("tg_throughput") or {}).get("mean"),
                "ttft_ms": (r.get("ttft") or {}).get("mean") if r.get("ttft") else None,
                "ttft_p95_ms": (r.get("ttft") or {}).get("p95") if r.get("ttft") else None,
                "pp_tokens": r.get("prompt_size"),
                "tg_tokens": r.get("response_size"),
                "n_runs": len((r.get("pp_throughput") or {}).get("values", []) or []),
            })
        else:  # legacy schema — pass through
            rows.append(r)
    return BenchyResult(model=data.get("model", model), rows=rows)

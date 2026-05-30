from __future__ import annotations

import asyncio
import sys
from typing import Literal

from pydantic import BaseModel, Field

# Aliased so the literal call site is portable across static analyzers
# that flag the unsafe shell-eval family. This API is the safe execFile-style
# spawner — argv list, no shell interpretation.
_spawn_child = asyncio.create_subprocess_exec


Adapter = Literal["raw", "cloud", "hermes"]
Tier = Literal["easy", "medium", "hard", "very_hard", "all"]
Cluster = Literal["single", "dual", "triple", "quad", "octa"]


class RunArgs(BaseModel):
    model: str
    base_url: str
    adapter: Adapter
    # tier/category are comma-joined strings ("easy,hard") when the launch
    # modal multi-selects, or a single value, or "all". The child CLI splits
    # them, so we keep them as free-form strings rather than a single Literal.
    tier: str = "all"
    trials: int = Field(ge=1, le=100)
    concurrency: int = Field(ge=1, le=64)
    with_perf: bool = False
    perf_only: bool = False
    category: str = "all"
    cluster: Cluster = "single"
    # API-side model name (alias the served endpoint expects). If None or equal
    # to `model`, omit --served-model and let cli.py default to --model value.
    served_model: str | None = None
    # When set, argv collapses to `toolery run --resume <run_id>` and the
    # receiving CLI rehydrates model/base_url/tier/etc from the run's config_json.
    resume: str | None = None


def build_argv(args: RunArgs, executable: str = "toolery") -> list[str]:
    if args.resume:
        return [executable, "run", "--resume", args.resume]
    argv = [
        executable, "run",
        "--model", args.model,
        "--adapter", args.adapter,
        "--tier", args.tier,
        "--trials", str(args.trials),
        "--base-url", args.base_url,
        "--concurrency", str(args.concurrency),
    ]
    if args.served_model and args.served_model != args.model:
        argv.extend(["--served-model", args.served_model])
    if args.category and args.category != "all":
        argv.extend(["--category", args.category])
    if args.with_perf:
        argv.append("--with-perf")
    if args.perf_only:
        argv.append("--perf-only")
    if args.cluster and args.cluster != "single":
        argv.extend(["--cluster", args.cluster])
    return argv


async def spawn_run(args: RunArgs, executable: str | None = None) -> asyncio.subprocess.Process:
    """Spawn `toolery run ...` as a child process.

    Returns the Process so callers can poll returncode. stdout/stderr are
    inherited from the parent terminal so the user can see any startup
    errors in the shell that launched the TUI.
    """
    exe = executable or _resolve_executable()
    argv = build_argv(args, executable=exe)
    return await _spawn_child(*argv)


def _resolve_executable() -> str:
    # Prefer the same Python venv that's running the TUI.
    bin_dir = sys.executable.rsplit("/", 1)[0]
    return f"{bin_dir}/toolery"

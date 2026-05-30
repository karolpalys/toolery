import asyncio
from unittest.mock import AsyncMock

import pytest

from toolery.core import runner_subprocess
from toolery.core.runner_subprocess import RunArgs, build_argv


def test_build_argv_minimum_required():
    args = RunArgs(
        model="MiniMax-M2.7",
        base_url="http://localhost:8888",
        adapter="raw",
        tier="easy",
        trials=5,
        concurrency=4,
        with_perf=False,
    )
    argv = build_argv(args)
    assert argv[0] == "toolery"
    assert argv[1] == "run"
    assert "--model" in argv and argv[argv.index("--model") + 1] == "MiniMax-M2.7"
    assert "--adapter" in argv and argv[argv.index("--adapter") + 1] == "raw"
    assert "--tier" in argv and argv[argv.index("--tier") + 1] == "easy"
    assert "--trials" in argv and argv[argv.index("--trials") + 1] == "5"
    assert "--base-url" in argv
    assert argv[argv.index("--base-url") + 1] == "http://localhost:8888"
    assert "--with-perf" not in argv


def test_build_argv_multi_tier_and_category_pass_comma_joined():
    """Multi-select in the launch modal yields comma-joined tier/category
    strings; build_argv must forward them verbatim to the child CLI."""
    args = RunArgs(
        model="m", base_url="http://x", adapter="raw",
        tier="easy,hard", category="coding,debugging",
        trials=1, concurrency=1,
    )
    argv = build_argv(args)
    assert argv[argv.index("--tier") + 1] == "easy,hard"
    assert argv[argv.index("--category") + 1] == "coding,debugging"


def test_build_argv_with_perf_adds_flag():
    args = RunArgs(
        model="m", base_url="http://x", adapter="raw",
        tier="all", trials=1, concurrency=1, with_perf=True,
    )
    argv = build_argv(args)
    assert "--with-perf" in argv


def test_build_argv_resume_emits_only_resume_flag():
    """When resume is set, argv must reduce to `toolery run --resume <id>` so
    the CLI hydrates everything from the original run's config_json."""
    args = RunArgs(
        model="placeholder", base_url="http://placeholder",
        adapter="raw", tier="all", trials=1, concurrency=1,
        with_perf=True, cluster="dual",
        resume="2026-05-26T07-34_Qwen3.6-27B-FP8",
    )
    argv = build_argv(args, executable="/bin/toolery")
    assert argv == ["/bin/toolery", "run", "--resume",
                    "2026-05-26T07-34_Qwen3.6-27B-FP8"]


@pytest.mark.asyncio
async def test_spawn_run_calls_subprocess_factory(monkeypatch):
    fake_proc = AsyncMock(spec=asyncio.subprocess.Process)
    spawn_mock = AsyncMock(return_value=fake_proc)
    monkeypatch.setattr(runner_subprocess, "_spawn_child", spawn_mock)

    args = RunArgs(
        model="m", base_url="http://x:1", adapter="raw",
        tier="all", trials=2, concurrency=1, with_perf=True,
    )
    result = await runner_subprocess.spawn_run(args, executable="/fake/toolery")
    assert result is fake_proc
    spawn_mock.assert_awaited_once()
    called_argv = spawn_mock.await_args.args
    assert called_argv[0] == "/fake/toolery"
    assert "--with-perf" in called_argv

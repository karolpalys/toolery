from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_test.adapters.base import Adapter
from llm_test.adapters.openai_raw import OpenAIRawAdapter
from llm_test.core.markdown import render_scenario, render_summary
from llm_test.core.runner import Runner
from llm_test.core.scenario import load_all_scenarios
from llm_test.core.store import Store

app = typer.Typer(no_args_is_help=True, help="LLM-test — deterministic LLM tool-calling benchmark.")
console = Console()


def _results_dir() -> Path:
    p = Path(os.environ.get("LLM_TEST_RESULTS_DIR", "./results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _store() -> Store:
    s = Store(_results_dir() / "runs.db")
    s.init_schema()
    return s


@app.command(name="list")
def list_runs():
    """List recorded runs."""
    rows = _store().fetch_all_runs()
    if not rows:
        console.print("[yellow]No runs recorded yet.[/yellow]")
        return
    t = Table("run_id", "model", "status", "started_at", "duration (s)")
    for r in rows:
        t.add_row(r["run_id"], r["model"], r["status"] or "", r["started_at"],
                  f"{r['duration_s'] or 0:.1f}")
    console.print(t)


@app.command()
def scenarios(tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
              dir: Path = typer.Option(Path("scenarios"), help="scenarios dir")):  # noqa: B008
    """List scenarios."""
    xs = load_all_scenarios(dir)
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    t = Table("id", "tier", "category", "domain", "tools", "title")
    for s in xs:
        t.add_row(s.id, s.tier.value, s.category.value, s.domain,
                  ",".join(s.tools[:3]) + ("…" if len(s.tools) > 3 else ""),
                  s.title)
    console.print(t)
    console.print(f"\n[bold]{len(xs)} scenarios.[/bold]")


@app.command()
def run(
    model: str = typer.Option(..., "--model"),
    adapter: str = typer.Option("raw", help="comma-separated: raw,hermes,claude_code,codex"),
    tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
    trials: int = typer.Option(5, "--trials"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    scenarios_dir: Path = typer.Option(Path("scenarios")),  # noqa: B008
    concurrency: int = typer.Option(4),
    no_tui: bool = typer.Option(True, "--no-tui/--tui", help="MVP: --no-tui only"),
):
    """Run benchmark."""
    # Import tools to register them
    from llm_test.tools import generic  # noqa: F401

    api_key = os.environ.get("OPENAI_API_KEY", "")
    adapters: dict[str, Adapter] = {}
    for a in adapter.split(","):
        a = a.strip()
        if a == "raw":
            adapters[a] = OpenAIRawAdapter(base_url=base_url, api_key=api_key)
        else:
            console.print(f"[yellow]adapter '{a}' not yet wired in Phase 12; skipping[/yellow]")
    if not adapters:
        console.print("[red]No adapters enabled.[/red]")
        raise typer.Exit(2)

    xs = load_all_scenarios(scenarios_dir)
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    if not xs:
        console.print("[red]No scenarios match filter.[/red]")
        raise typer.Exit(2)

    run_id = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M')}_{model}"
    run_dir = _results_dir() / "runs" / run_id
    (run_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    store = _store()
    cfg = {"model": model, "adapter": list(adapters), "tier": tier, "trials": trials,
           "base_url": base_url, "concurrency": concurrency}
    store.create_run(run_id=run_id, model=model, base_url=base_url,
                     started_at=datetime.now(UTC).isoformat(),
                     config_json=json.dumps(cfg),
                     scenarios_hash="")
    for name, ad in adapters.items():
        store.upsert_adapter(run_id, name, ad.version)

    started = datetime.now(UTC)
    runner = Runner(adapters=adapters, trials=trials, model=model, concurrency=concurrency)
    console.print(f"[bold]Running {len(xs)} scenarios × {len(adapters)} adapters × {trials} trials"
                  f" = {len(xs)*len(adapters)*trials} runs[/bold]")
    results = asyncio.run(runner.run(xs))

    sc_by_id = {s.id: s for s in xs}
    tier_lookup = {s.id: s.tier.value for s in xs}
    for r in results:
        s = sc_by_id[r.scenario_id]
        trace_filename = f"{r.scenario_id}__{r.adapter}__t{r.trial_index}.json"
        (run_dir / "traces" / trace_filename).write_text(r.trace.model_dump_json(indent=2))
        store.write_scenario_result(
            run_id=run_id, result=r,
            tags=s.tags, ranking_dims=s.ranking_dimensions,
            scenario_hash="", category=s.category.value, tier=s.tier.value,
            trace_path=f"traces/{trace_filename}",
        )
    for s in xs:
        md = render_scenario(scenario_id=s.id, results=results, title=s.title,
                             tier=s.tier.value, category=s.category.value)
        (run_dir / "scenarios" / f"{s.id}.md").write_text(md)
    duration = (datetime.now(UTC) - started).total_seconds()
    md = render_summary(
        run_id=run_id, model=model, adapters=list(adapters),
        trials=trials, duration_s=duration, results=results, perf_rows=[],
        tier_lookup=tier_lookup,
    )
    (run_dir / "summary.md").write_text(md)
    store.finish_run(run_id, finished_at=datetime.now(UTC).isoformat(),
                     duration_s=duration, status="done")
    console.print(f"[green]✓ Run finished: {run_dir}[/green]")
    console.print(f"  [bold]summary.md[/bold]: {run_dir/'summary.md'}")

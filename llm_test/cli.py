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
    model: str = typer.Option(..., "--model",
                              help="friendly display name (run_id, DB column)"),
    served_model: str = typer.Option("", "--served-model",
                                     help="name to send in API model= (default: --model)"),
    adapter: str = typer.Option("raw", help="comma-separated: raw,hermes,claude_code,codex"),
    tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
    category: str = typer.Option("all", "--category",
                                 help="scenario category filter (see Category enum) or 'all'"),
    trials: int = typer.Option(5, "--trials"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    scenarios_dir: Path = typer.Option(Path("scenarios")),  # noqa: B008
    concurrency: int = typer.Option(4),
    no_tui: bool = typer.Option(True, "--no-tui/--tui", help="MVP: --no-tui only"),
    with_perf: bool = typer.Option(False, "--with-perf"),
    perf_only: bool = typer.Option(False, "--perf-only",
                                   help="skip eval phase; run only llama-benchy"),
    cluster: str = typer.Option("single", "--cluster",
                                help="single | dual | triple | quad | octa "
                                     "(DGX Spark deployment topology: 1/2/3/4/8 nodes)"),
):
    """Run benchmark."""
    # Import tools to register them
    from llm_test.tools import api_db, domain, generic, terminal  # noqa: F401

    local_key = os.environ.get("OPENAI_API_KEY", "")
    adapters: dict[str, Adapter] = {}
    for a in adapter.split(","):
        a = a.strip()
        if a == "raw":
            adapters[a] = OpenAIRawAdapter(base_url=base_url, api_key=local_key)
        elif a == "hermes":
            from llm_test.adapters.hermes import HermesAdapter
            adapters[a] = HermesAdapter(
                timeout_per_scenario=int(os.environ.get("HERMES_TIMEOUT", "600")),
                api_url=os.environ.get("HERMES_API_URL", "http://localhost:8644"),
                gateway_url=os.environ.get("HERMES_GATEWAY_URL", "http://localhost:8642"),
                token=os.environ.get("HERMES_TOKEN", ""),
                workspace_id=os.environ.get("HERMES_WORKSPACE", "default"),
            )
        elif a == "claude_code":
            from llm_test.adapters.claude_code import ClaudeCodeAdapter
            adapters[a] = ClaudeCodeAdapter(
                cli_path=os.environ.get("CLAUDE_CLI_PATH", "claude"),
                backend_url=base_url,
                use_local_model=True,
            )
        elif a == "codex":
            from llm_test.adapters.codex import CodexAdapter
            adapters[a] = CodexAdapter(
                cli_path=os.environ.get("CODEX_CLI_PATH", "codex"),
                backend_url=base_url, use_local_model=True,
            )
        else:
            console.print(f"[yellow]adapter '{a}' not yet wired in Phase 12; skipping[/yellow]")
    if not adapters:
        console.print("[red]No adapters enabled.[/red]")
        raise typer.Exit(2)

    xs = load_all_scenarios(scenarios_dir)
    if tier != "all":
        xs = [s for s in xs if s.tier.value == tier]
    if category != "all":
        xs = [s for s in xs if s.category.value == category]
    if not xs and not perf_only:
        console.print("[red]No scenarios match filter.[/red]")
        raise typer.Exit(2)
    api_model = served_model or model
    # NIM model IDs use `<org>/<name>` — strip the slash so we don't create
    # nested directories under results/runs/.
    safe_model = model.replace("/", "_")
    run_id = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H-%M')}_{safe_model}"
    run_dir = _results_dir() / "runs" / run_id
    (run_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    # Hash of (sorted) scenario IDs that participated in this run. Lets
    # downstream rankings flag runs that were taken on a different scenario
    # set (e.g. before vs after adding the hallucination suite).
    import hashlib as _hashlib
    scenarios_hash = _hashlib.sha256(
        ",".join(sorted(s.id for s in xs)).encode("utf-8")
    ).hexdigest()[:16]

    store = _store()
    total_units = 0 if perf_only else len(xs) * len(adapters) * trials
    if cluster not in ("single", "dual", "triple", "quad", "octa"):
        console.print(f"[yellow]Unknown --cluster {cluster!r}, falling back to 'single'.[/yellow]")
        cluster = "single"
    cfg = {"model": model, "served_model": api_model, "adapter": list(adapters),
           "tier": tier, "category": category, "trials": trials, "base_url": base_url,
           "concurrency": concurrency, "total_units": total_units,
           "scenarios_count": 0 if perf_only else len(xs),
           "with_perf": bool(with_perf or perf_only),
           "perf_only": bool(perf_only), "cluster": cluster}
    store.create_run(run_id=run_id, model=model, base_url=base_url,
                     started_at=datetime.now(UTC).isoformat(),
                     config_json=json.dumps(cfg),
                     scenarios_hash=scenarios_hash,
                     cluster=cluster,
                     total_units=total_units)
    for name, ad in adapters.items():
        store.upsert_adapter(run_id, name, ad.version)

    started = datetime.now(UTC)
    if perf_only:
        console.print("[bold]Perf-only mode: skipping eval, running llama-benchy[/bold]")
    else:
        # Defensive cleanup at startup: if a prior crashed session left orphan in_flight
        # rows for this run_id (resume path), wipe them now before any task starts.
        store.clear_all_in_flight(run_id)

        def _on_start(scenario_id: str, adapter_name: str, trial_index: int,
                      started_at: str) -> None:
            store.mark_in_flight(run_id, scenario_id, adapter_name, trial_index, started_at)

        def _on_end(scenario_id: str, adapter_name: str, trial_index: int) -> None:
            store.clear_in_flight(run_id, scenario_id, adapter_name, trial_index)

        runner = Runner(
            adapters=adapters, trials=trials, model=api_model, concurrency=concurrency,
            on_start=_on_start, on_end=_on_end,
        )
        console.print(f"[bold]Running {len(xs)} scenarios × {len(adapters)} adapters × {trials} trials"
                      f" = {total_units} units[/bold]")

        sc_by_id = {s.id: s for s in xs}
        tier_lookup = {s.id: s.tier.value for s in xs}

        def _on_result(r) -> None:
            # Incremental: persist trace + DB row immediately so the Live tab sees progress.
            s = sc_by_id[r.scenario_id]
            trace_filename = f"{r.scenario_id}__{r.adapter}__t{r.trial_index}.json"
            (run_dir / "traces" / trace_filename).write_text(r.trace.model_dump_json(indent=2))
            store.write_scenario_result(
                run_id=run_id, result=r,
                tags=s.tags, ranking_dims=s.ranking_dimensions,
                scenario_hash="", category=s.category.value, tier=s.tier.value,
                trace_path=f"traces/{trace_filename}",
            )
            store.update_phase(run_id, "scenarios", current_scenario=r.scenario_id)

        results = asyncio.run(runner.run(xs, on_result=_on_result))

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

    if with_perf or perf_only:
        from llm_test.perf.benchy import run_benchy
        store.update_phase(run_id, "perf", current_scenario="llama-bench")
        try:
            perf = run_benchy(model=api_model, base_url=base_url,
                              depth=[0, 4096, 8192], runs=3)
            for row in perf.rows:
                store.write_perf(run_id, depth=row["depth"],
                                 pp_tps=row.get("pp_tps"), tg_tps=row.get("tg_tps"),
                                 ttft_ms=row.get("ttft_ms"), ttft_p95_ms=row.get("ttft_p95_ms"),
                                 pp_tokens=row.get("pp_tokens"), tg_tokens=row.get("tg_tokens"),
                                 benchy_runs=row.get("n_runs"),
                                 raw_json=json.dumps(row))
            console.print(f"[green]✓ perf: collected {len(perf.rows)} depth points[/green]")
        except Exception as e:
            console.print(f"[yellow]perf collection failed: {e}[/yellow]")

    store.finish_run(run_id, finished_at=datetime.now(UTC).isoformat(),
                     duration_s=(datetime.now(UTC) - started).total_seconds(),
                     status="done")

    # Auto-regenerate rankings so the new run is reflected in TUI Rankings tab.
    try:
        from llm_test.rankings.compute import regenerate_rankings, load_active_use_case
        uc_key, uc_weights = load_active_use_case(_results_dir())
        regenerate_rankings(
            store=store,
            dimensions=["overall", "coding", "debugging", "agentic", "safety",
                        "adversarial_robustness", "restraint", "long_context",
                        "budget_efficiency", "hallucination", "error_recovery",
                        "parameter_precision", "context_state_tracking",
                        "structured_output", "tool_selection",
                        "instruction_following", "localization", "terminal"],
            out_dir=_results_dir() / "rankings",
            use_case_weights=uc_weights,
            use_case_key=uc_key,
        )
        console.print("[green]✓ Rankings regenerated[/green]")
    except Exception as e:
        console.print(f"[yellow]rankings regen skipped: {e}[/yellow]")

    console.print(f"[green]✓ Run finished: {run_dir}[/green]")
    console.print(f"  [bold]summary.md[/bold]: {run_dir/'summary.md'}")


@app.command()
def perf(model: str = typer.Option(..., "--model"),
         base_url: str = typer.Option("http://localhost:8000", "--base-url"),
         pp: int = 4096, tg: int = 512,
         depth: str = "0,4096,8192", runs: int = 3):
    """Run llama-benchy only (no scoring)."""
    from llm_test.perf.benchy import run_benchy
    res = run_benchy(model=model, base_url=base_url, pp=pp, tg=tg,
                     depth=[int(x) for x in depth.split(",")], runs=runs)
    for row in res.rows:
        console.print(f"depth={row['depth']:>7} pp_tps={row.get('pp_tps',0):.1f} "
                      f"tg_tps={row.get('tg_tps',0):.2f} ttft={row.get('ttft_ms',0):.0f}ms")


@app.command()
def compare(run_a: str = typer.Argument(...), run_b: str = typer.Argument(...),
            out: Path = typer.Option(None, "--out")):  # noqa: B008
    """Compare two runs (statistical diff)."""
    from llm_test.compare import compare_runs
    out_path = out or (_results_dir() / "compare" / f"{run_a}__vs__{run_b}.md")
    compare_runs(store=_store(), run_a=run_a, run_b=run_b, out_path=out_path)
    console.print(f"[green]✓ Wrote {out_path}[/green]")


@app.command()
def tui():
    """Launch the Textual TUI."""
    from llm_test.tui.app import LLMTestApp
    LLMTestApp(run_id=None).run()


@app.command()
def rankings(
    regen: bool = typer.Option(False, "--regen", help="Regenerate rankings .md"),
    dimension: str = typer.Option("all", help="overall|coding|agentic|safety|restraint|long_context|budget_efficiency|speed|all"),
):
    """Manage rankings."""
    from llm_test.rankings.compute import regenerate_rankings
    out = _results_dir() / "rankings"
    dims = ["overall", "coding", "debugging", "agentic", "safety",
            "adversarial_robustness", "restraint", "long_context",
            "budget_efficiency", "hallucination", "error_recovery",
            "parameter_precision", "context_state_tracking", "structured_output",
            "tool_selection", "instruction_following", "localization",
            "terminal"] if dimension == "all" else [dimension]
    if regen:
        from llm_test.rankings.compute import load_active_use_case
        uc_key, uc_weights = load_active_use_case(_results_dir())
        regenerate_rankings(
            store=_store(), dimensions=dims, out_dir=out,
            use_case_weights=uc_weights, use_case_key=uc_key,
        )
        msg = f"[green]✓ Regenerated rankings: {out}[/green]"
        if uc_key:
            msg += f"\n[dim]  · Use-case '{uc_key}' applied → use_case_{uc_key}.md[/dim]"
        console.print(msg)
    else:
        for d in dims:
            p = out / f"{d}.md"
            if p.exists():
                console.print(f"[bold]{d}[/bold] → {p}")
            else:
                console.print(f"[yellow]{d}: not yet generated (run with --regen)[/yellow]")

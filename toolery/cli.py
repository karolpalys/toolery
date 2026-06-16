from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from toolery.adapters.base import Adapter
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.markdown import render_scenario, render_summary
from toolery.core.runner import Runner
from toolery.core.scenario import display_name, load_all_scenarios
from toolery.core.store import Store

app = typer.Typer(no_args_is_help=True, help="LLM-test — deterministic LLM tool-calling benchmark.")
console = Console()


def _results_dir() -> Path:
    p = Path(os.environ.get("TOOLERY_RESULTS_DIR", "./results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _store() -> Store:
    s = Store(_results_dir() / "runs.db")
    s.init_schema()
    return s


def _backfill_correctness_run(store, run_id: str, results_dir: Path, scenarios: dict) -> tuple[int, int]:
    """Recompute correctness_score for one run from its stored trace files.
    Returns (updated, skipped). Rows whose scenario or trace file is missing
    are skipped (left as-is)."""
    from toolery.core.models import TraceResult
    from toolery.core.scorer import evaluate

    updated = skipped = 0
    for row in store.fetch_results_for_run(run_id):
        scenario = scenarios.get(row["scenario_id"])
        trace_path = row.get("trace_path")
        if scenario is None or not trace_path:
            skipped += 1
            continue
        tp = results_dir / "runs" / run_id / trace_path
        if not tp.exists():
            skipped += 1
            continue
        trace = TraceResult.model_validate_json(tp.read_text())
        result = evaluate(scenario, trace)
        store.update_correctness_score(row["result_id"], result.correctness_score)
        updated += 1
    return updated, skipped


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
    # Lead with the tier (source of truth for difficulty) and the prefix-stripped
    # name; keep the raw id in a trailing column as the stable key for --ids.
    t = Table("tier", "name", "category", "domain", "tools", "title", "id")
    for s in xs:
        t.add_row(s.tier.value, display_name(s.id), s.category.value, s.domain,
                  ",".join(s.tools[:3]) + ("…" if len(s.tools) > 3 else ""),
                  s.title, s.id)
    console.print(t)
    console.print(f"\n[bold]{len(xs)} scenarios.[/bold]")


@app.command()
def run(
    model: str = typer.Option("", "--model",
                              help="friendly display name (run_id, DB column); "
                                   "required unless --resume is given"),
    served_model: str = typer.Option("", "--served-model",
                                     help="name to send in API model= (default: --model)"),
    adapter: str = typer.Option("raw", help="comma-separated: raw,cloud,hermes"),
    tier: str = typer.Option("all", help="easy|medium|hard|very_hard|all"),
    category: str = typer.Option("all", "--category",
                                 help="scenario category filter (see Category enum) or 'all'"),
    ids: str = typer.Option("", "--ids",
                            help="comma-separated scenario ids to run (exact match); "
                                 "empty = no id filtering"),
    trials: int = typer.Option(5, "--trials"),
    base_url: str = typer.Option("http://localhost:8000", "--base-url"),
    scenarios_dir: Path = typer.Option(Path("scenarios")),  # noqa: B008
    concurrency: int = typer.Option(4),
    timeout_scale: float = typer.Option(2.0, "--timeout-scale",
                                        help="multiply each scenario's timeout_seconds "
                                             "(default 2.0 — scale-1.0 budgets killed reasoning "
                                             "models mid-answer); bump for slow cloud/reasoning "
                                             "endpoints (e.g. 4.0)"),
    no_tui: bool = typer.Option(True, "--no-tui/--tui", help="MVP: --no-tui only"),
    with_perf: bool = typer.Option(False, "--with-perf"),
    perf_only: bool = typer.Option(False, "--perf-only",
                                   help="skip eval phase; run only llama-benchy"),
    cluster: str = typer.Option("single", "--cluster",
                                help="single | dual | triple | quad | octa "
                                     "(DGX Spark deployment topology: 1/2/3/4/8 nodes)"),
    resume: str = typer.Option("", "--resume",
                               help="run_id to resume: rehydrates model/adapter/"
                                    "tier/trials/etc from the run's config_json "
                                    "and continues from the next not-yet-run unit"),
):
    """Run benchmark."""
    # Import tools to register them
    from toolery.tools import api_db, domain, generic, terminal  # noqa: F401

    # Resume: rehydrate every run parameter from the original run's stored
    # config BEFORE building adapters (adapter/base_url feed adapter setup).
    # The TUI's Resume button invokes us as `run --resume <id>` with no other
    # meaningful options, so all real values must come from the DB.
    resuming = bool(resume)
    resume_skip: set[tuple[str, str, int]] = set()
    if resuming:
        _rstore = _store()
        _existing = _rstore.fetch_run(resume)
        if _existing is None:
            console.print(f"[red]Cannot resume: run {resume!r} not found.[/red]")
            raise typer.Exit(2)
        try:
            _rcfg = json.loads(_existing.get("config_json") or "{}")
        except json.JSONDecodeError:
            _rcfg = {}
        model = _existing["model"]
        base_url = _rcfg.get("base_url") or _existing.get("base_url") or base_url
        _adapters_field = _rcfg.get("adapter") or []
        if isinstance(_adapters_field, str):
            adapter = _adapters_field
        elif _adapters_field:
            adapter = ",".join(_adapters_field)
        served_model = _rcfg.get("served_model") or served_model
        tier = _rcfg.get("tier", tier)
        category = _rcfg.get("category", category)
        trials = int(_rcfg.get("trials", trials) or trials)
        concurrency = int(_rcfg.get("concurrency", concurrency) or concurrency)
        timeout_scale = float(_rcfg.get("timeout_scale", timeout_scale) or timeout_scale)
        with_perf = bool(_rcfg.get("with_perf", with_perf))
        perf_only = bool(_rcfg.get("perf_only", perf_only))
        cluster = _rcfg.get("cluster", cluster)
        resume_skip = _rstore.fetch_completed_units(resume)
    elif not model:
        console.print("[red]--model is required (unless resuming with --resume).[/red]")
        raise typer.Exit(2)

    local_key = os.environ.get("OPENAI_API_KEY", "")
    adapters: dict[str, Adapter] = {}
    for a in adapter.split(","):
        a = a.strip()
        if a == "raw":
            adapters[a] = OpenAIRawAdapter(base_url=base_url, api_key=local_key)
        elif a == "cloud":
            from toolery.adapters.cloud import CloudAdapter
            adapters[a] = CloudAdapter(base_url=base_url, api_key=local_key)
        elif a == "hermes":
            from toolery.adapters.hermes import HermesAdapter
            adapters[a] = HermesAdapter(
                timeout_per_scenario=int(os.environ.get("HERMES_TIMEOUT", "1800")),
                # MCP-bridge on by default (apples-to-apples mock tools over MCP);
                # set HERMES_MCP_BRIDGE=0 to fall back to standalone-agent mode.
                mcp_bridge=os.environ.get("HERMES_MCP_BRIDGE", "1") != "0",
                # HERMES_SKILLS=1 enables Hermes' skills toolset (and drops
                # --ignore-rules) so user skills are injected — for A/B testing
                # their value. Default off keeps the apples-to-apples benchmark.
                skills_mode=os.environ.get("HERMES_SKILLS", "0") != "0",
                # Point Hermes at the SAME endpoint as raw/cloud, overriding any
                # stale model.base_url in ~/.hermes/config.yaml (else hermes hits
                # the wrong host → connection error → every scenario model_crash).
                base_url=base_url,
                api_key=local_key,
                api_url=os.environ.get("HERMES_API_URL", "http://localhost:8644"),
                gateway_url=os.environ.get("HERMES_GATEWAY_URL", "http://localhost:8642"),
                token=os.environ.get("HERMES_TOKEN", ""),
                workspace_id=os.environ.get("HERMES_WORKSPACE", "default"),
            )
        else:
            console.print(f"[yellow]adapter '{a}' not yet wired in Phase 12; skipping[/yellow]")
    if not adapters:
        console.print("[red]No adapters enabled.[/red]")
        raise typer.Exit(2)

    xs = load_all_scenarios(scenarios_dir)
    # tier/category may be comma-joined ("easy,hard") from the launch modal's
    # multi-select, a single value, or "all". Split and keep any match; "all"
    # anywhere in the set means no filtering on that axis.
    tier_set = {t.strip() for t in tier.split(",") if t.strip()}
    if tier_set and "all" not in tier_set:
        xs = [s for s in xs if s.tier.value in tier_set]
    category_set = {c.strip() for c in category.split(",") if c.strip()}
    if category_set and "all" not in category_set:
        xs = [s for s in xs if s.category.value in category_set]
    id_set = {i.strip() for i in ids.split(",") if i.strip()}
    if id_set:
        # Accept either the full stable id (easy-39-…) or the prefix-stripped
        # display name (39-…) so ids copied from `list`/reports still resolve.
        xs = [s for s in xs if s.id in id_set or display_name(s.id) in id_set]
    if not xs and not perf_only:
        console.print("[red]No scenarios match filter.[/red]")
        raise typer.Exit(2)
    api_model = served_model or model
    # NIM model IDs use `<org>/<name>` — strip the slash so we don't create
    # nested directories under results/runs/.
    safe_model = model.replace("/", "_")
    if resuming:
        run_id = resume
    else:
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
           "concurrency": concurrency, "timeout_scale": timeout_scale,
           "total_units": total_units,
           "scenarios_count": 0 if perf_only else len(xs),
           "with_perf": bool(with_perf or perf_only),
           "perf_only": bool(perf_only), "cluster": cluster}
    if resuming:
        # Run row already exists — flip it back to running and clear any orphan
        # in-flight rows from the paused session. Keep the original config_json.
        store.reopen_run(run_id)
    else:
        store.create_run(run_id=run_id, model=model, base_url=base_url,
                         started_at=datetime.now(UTC).isoformat(),
                         config_json=json.dumps(cfg),
                         scenarios_hash=scenarios_hash,
                         cluster=cluster,
                         total_units=total_units)
    for name, ad in adapters.items():
        store.upsert_adapter(run_id, name, ad.version)

    started = datetime.now(UTC)
    # Process-wide heartbeat: bump runs.updated_at every 60s for the whole run so
    # the TUI's stale-detector never false-kills a run that is merely slow — e.g.
    # a long agentic hermes scenario (timeout up to HERMES_TIMEOUT) or a deep
    # perf benchmark. If the process truly dies/hangs, the heartbeat stops with
    # it and a genuine stale-abort still fires.
    import threading as _threading

    _stop_run_hb = _threading.Event()

    def _run_heartbeat() -> None:
        while not _stop_run_hb.wait(60):
            try:
                store.heartbeat(run_id)
            except Exception:
                pass

    _threading.Thread(target=_run_heartbeat, daemon=True).start()

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
            on_start=_on_start, on_end=_on_end, skip=resume_skip,
            timeout_scale=timeout_scale,
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

        def _should_stop() -> bool:
            # Graceful external abort: if the TUI flipped runs.status to
            # 'paused' (Pause) or 'failed' (detached STOP), stop scheduling new
            # units. Units already in flight finish and record their results
            # above; only not-yet-started units are skipped. Polled by the
            # runner after a unit acquires its slot, before it starts.
            return store.get_run_status(run_id) in ("paused", "failed")

        results = asyncio.run(
            runner.run(xs, on_result=_on_result, should_stop=_should_stop))

        # If paused/stopped externally while draining, do NOT finalize — leave
        # the status as the TUI set it so Resume can continue. In-flight units
        # already wrote their results; skip render/summary/perf/finish.
        external = store.get_run_status(run_id)
        if external in ("paused", "failed"):
            console.print(
                f"[yellow]Run externally {external!r}: in-flight scenarios "
                "finished, no new ones scheduled. Skipping finalize.[/yellow]"
            )
            raise typer.Exit(0)

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
        from toolery.perf.benchy import run_benchy
        store.update_phase(run_id, "perf", current_scenario="llama-bench")
        # The process-wide heartbeat (started above) keeps the run alive through
        # the long, scenario-row-less perf phase.
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

    _stop_run_hb.set()
    store.finish_run(run_id, finished_at=datetime.now(UTC).isoformat(),
                     duration_s=(datetime.now(UTC) - started).total_seconds(),
                     status="done")

    # Auto-regenerate rankings so the new run is reflected in TUI Rankings tab.
    try:
        from toolery.rankings.compute import load_active_use_case, regenerate_rankings
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
    from toolery.perf.benchy import run_benchy
    res = run_benchy(model=model, base_url=base_url, pp=pp, tg=tg,
                     depth=[int(x) for x in depth.split(",")], runs=runs)
    for row in res.rows:
        console.print(f"depth={row['depth']:>7} pp_tps={row.get('pp_tps',0):.1f} "
                      f"tg_tps={row.get('tg_tps',0):.2f} ttft={row.get('ttft_ms',0):.0f}ms")


@app.command()
def compare(run_a: str = typer.Argument(...), run_b: str = typer.Argument(...),
            out: Path = typer.Option(None, "--out")):  # noqa: B008
    """Compare two runs (statistical diff)."""
    from toolery.compare import compare_runs
    out_path = out or (_results_dir() / "compare" / f"{run_a}__vs__{run_b}.md")
    compare_runs(store=_store(), run_a=run_a, run_b=run_b, out_path=out_path)
    console.print(f"[green]✓ Wrote {out_path}[/green]")


@app.command()
def tui():
    """Launch the Textual TUI."""
    from toolery.tui.app import TooleryApp
    TooleryApp(run_id=None).run()


@app.command()
def rankings(
    regen: bool = typer.Option(False, "--regen", help="Regenerate rankings .md"),
    dimension: str = typer.Option("all", help="overall|coding|agentic|safety|restraint|long_context|budget_efficiency|speed|all"),
):
    """Manage rankings."""
    from toolery.rankings.compute import regenerate_rankings
    out = _results_dir() / "rankings"
    dims = ["overall", "coding", "debugging", "agentic", "safety",
            "adversarial_robustness", "restraint", "long_context",
            "budget_efficiency", "hallucination", "error_recovery",
            "parameter_precision", "context_state_tracking", "structured_output",
            "tool_selection", "instruction_following", "localization",
            "terminal"] if dimension == "all" else [dimension]
    if regen:
        from toolery.rankings.compute import load_active_use_case
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


@app.command(name="backfill-correctness")
def backfill_correctness(
    scenarios_dir: Path = typer.Option(Path("scenarios")),  # noqa: B008
):
    """Recompute correctness_score for every existing run from stored traces."""
    from toolery.core.scenario import load_all_scenarios

    store = _store()  # _store() already runs init_schema() → column exists on old DBs
    scenarios = {s.id: s for s in load_all_scenarios(scenarios_dir)}
    results_dir = _results_dir()
    total_u = total_s = 0
    for run in store.fetch_all_runs():
        u, s = _backfill_correctness_run(store, run["run_id"], results_dir, scenarios)
        total_u += u
        total_s += s
        console.print(f"  {run['run_id']}: updated {u}, skipped {s}")
    console.print(f"[green]✓ backfill done: updated {total_u}, skipped {total_s}[/green]")


@app.command(name="correctness-report")
def correctness_report():
    """Show budgeted score vs budget-independent correctness per (model, adapter)."""
    from rich.table import Table

    from toolery.rankings.compute import compute_correctness_breakdown

    data = compute_correctness_breakdown(_store())
    table = Table(title="Score vs correctness (budget-independent)")
    for col in ("model", "adapter", "n", "score", "correctness", "solved-not-scored"):
        table.add_column(col)
    for (model, adapter), v in sorted(data.items()):
        table.add_row(model, adapter, str(v["n"]),
                      f"{v['score_mean']:.3f}", f"{v['correctness_mean']:.3f}",
                      str(v["solved_not_scored"]))
    console.print(table)

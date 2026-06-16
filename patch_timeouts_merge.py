#!/usr/bin/env python3
"""Merge a timeout-patch run back into the original run.

Workflow (after the original run is DONE):
  1. scan ORIG run for all timeout trials  -> set of (scenario_id, adapter, trial_index)
  2. (done externally) re-run those scenario ids with concurrency=1, timeout-scale 4.0
     into PATCH run
  3. this script: for each ORIG timeout (scenario, adapter, trial_index), copy the
     matching PATCH row's values into the ORIG row and overwrite the ORIG trace file.

Only the specific timed-out trials are overwritten; trials that originally passed
are left untouched.

Usage:
  python3 patch_timeouts_merge.py --orig <ORIG_RUN_ID> --patch <PATCH_RUN_ID> [--apply]

Without --apply it does a dry run (prints what it would change).
"""
import argparse, sqlite3, shutil, sys
from pathlib import Path

DB = Path("results/runs.db")
RUNS = Path("results/runs")

# columns copied from patch row -> orig row (everything that describes the result)
COPY_COLS = [
    "status", "score", "call_count", "budget_max", "latency_ms", "failure_kind",
    "checks_json", "correctness_score", "prompt_tokens", "completion_tokens", "gen_ms",
]


def timeout_trials(c, run_id):
    rows = c.execute(
        "SELECT scenario_id, adapter, trial_index FROM scenario_results "
        "WHERE run_id=? AND failure_kind='timeout'", (run_id,)).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    targets = timeout_trials(c, args.orig)
    print(f"ORIG {args.orig}: {len(targets)} timeout trials to patch")

    orig_dir = RUNS / args.orig
    patch_dir = RUNS / args.patch

    patched, missing, still_to, improved = 0, [], 0, 0
    for (sid, adapter, ti) in sorted(targets):
        prow = c.execute(
            "SELECT * FROM scenario_results WHERE run_id=? AND scenario_id=? "
            "AND adapter=? AND trial_index=?",
            (args.patch, sid, adapter, ti)).fetchone()
        if prow is None:
            missing.append((sid, adapter, ti))
            continue
        new_status, new_fk, new_score = prow["status"], prow["failure_kind"], prow["score"]
        if new_fk == "timeout":
            still_to += 1
        if (new_status == "pass"):
            improved += 1
        tag = " -> STILL TIMEOUT" if new_fk == "timeout" else ""
        print(f"  {sid} [{adapter}] t{ti}: error/timeout -> {new_status}/{new_fk} score={new_score}{tag}")

        if args.apply:
            sets = ", ".join(f"{col}=?" for col in COPY_COLS)
            vals = [prow[col] for col in COPY_COLS]
            vals += [args.orig, sid, adapter, ti]
            c.execute(
                f"UPDATE scenario_results SET {sets} "
                "WHERE run_id=? AND scenario_id=? AND adapter=? AND trial_index=?", vals)
            # overwrite trace file
            fname = f"{sid}__{adapter}__t{ti}.json"
            src = patch_dir / "traces" / fname
            dst = orig_dir / "traces" / fname
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"    WARN: patch trace missing {src}")
        patched += 1

    if args.apply:
        c.commit()

    print(f"\nSummary: matched={patched}  missing_in_patch={len(missing)}  "
          f"still_timeout_after_patch={still_to}  now_pass={improved}")
    if missing:
        print("MISSING in patch run (not re-run?):")
        for m in missing:
            print("  ", m)
    print("APPLIED" if args.apply else "DRY RUN (re-run with --apply to write)")


if __name__ == "__main__":
    main()

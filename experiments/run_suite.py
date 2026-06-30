"""Run the full twin experiment for every patient in a patients.csv (Phase C3).

For each subject it invokes the requested twin stages (each in its own process,
so one patient's failure can't abort the sweep), then ``compute_results``, and
finally aggregates every per-patient comparison table into one summary
(``results/suite_summary.csv``).

This is the heavy path — 3 twins x N patients. Use ``--limit`` / ``--start`` to
do a slice, and ``--methods`` to run a subset (e.g. MCMC-only) first.

Run (from the repo root):
    python -m experiments.run_suite --patients patients.csv --limit 10
    python -m experiments.run_suite --patients patients.csv --methods mcmc
    python -m experiments.run_suite --patients patients.csv --smoke --limit 3
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from . import exp_common as C
from . import patients as PT

STAGES = {"mcmc": "experiments.run_mcmc",
          "sbi": "experiments.run_sbi"}


def _run(mod: str, patient: str, patients_csv: str, smoke: bool,
         population: str | None = None) -> int:
    cmd = [sys.executable, "-m", mod, "--patient", patient, "--patients", patients_csv]
    if smoke:
        cmd.append("--smoke")
    if population:
        cmd += ["--population", population]
    return subprocess.run(cmd, cwd=_root).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="patients.csv to iterate")
    ap.add_argument("--methods", default="mcmc,sbi", help="comma list of twins "
                    "(per-instance only; the amortized baselines are experiments.run_phase1)")
    ap.add_argument("--limit", type=int, default=None, help="number of patients to run")
    ap.add_argument("--start", type=int, default=0, help="patient offset (for sharding)")
    ap.add_argument("--smoke", action="store_true", help="tiny configs everywhere")
    ap.add_argument("--population", default=None,
                    help="legacy population.npz (prior is now published; see replaybg_priors)")
    args = ap.parse_args()

    subs = PT.load_subjects_csv(args.patients)
    end = args.start + args.limit if args.limit else None
    subs = subs[args.start:end]
    methods = [m for m in args.methods.split(",") if m in STAGES]
    if not methods:
        sys.exit(f"no valid methods in {args.methods!r}; choose from {list(STAGES)}")
    print(f"[suite] {len(subs)} patients x methods={methods} (smoke={args.smoke})")

    agg = []
    for i, s in enumerate(subs):
        print(f"\n########## [{i + 1}/{len(subs)}] {s.name} ##########")
        for m in methods:
            if _run(STAGES[m], s.name, args.patients, args.smoke, args.population) != 0:
                print(f"[suite] {s.name}/{m} FAILED — continuing")
        if _run("experiments.compute_results", s.name, args.patients, args.smoke) != 0:
            print(f"[suite] {s.name}/results FAILED — skipping aggregation")
            continue
        tpath = os.path.join(C.results_dir_for(s), "comparison_table.csv")
        if os.path.exists(tpath):
            t = pd.read_csv(tpath, index_col=0)
            for method, row in t.iterrows():
                rec = {"patient": s.name, "method": method}
                rec.update(row.to_dict())
                agg.append(rec)

    if not agg:
        print("[suite] no results aggregated.")
        return
    adf = pd.DataFrame(agg)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR, "suite_summary.csv")
    adf.to_csv(out, index=False)

    print(f"\n[suite] wrote {out} ({len(adf)} rows)")
    print("\n=== mean metrics per method across patients ===")
    summ = adf.groupby("method")[["spearman", "regret", "rmse", "mard"]].mean()
    print(summ.to_string())


if __name__ == "__main__":
    main()
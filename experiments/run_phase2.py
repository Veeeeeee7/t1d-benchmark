"""Phase 2 — complete experiment (all patients).

Runs each per-instance twinning method (MCMC, SBI; more later) on ALL patients
and reports DT2 metrics, mean +/- std, per method.

This is the heavy sweep. On a cluster, shard with ``--start`` / ``--limit`` (one
chunk per node, as ``run_phase2.sh`` does); each shard is resumable (skips
patients already scored). Once every shard has finished, produce the final
summary with ``--aggregate-only`` over the full set:

    python -m experiments.run_phase2 --patients patients.csv --aggregate-only

Run (from the repo root):
    python -m experiments.run_phase2 --patients patients.csv \
        --population experiments/artifacts/population.npz --start 0 --limit 256
    python -m experiments.run_phase2 --patients patients.csv --smoke --limit 4
"""
from __future__ import annotations

import os
import sys
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import exp_common as C
from experiments import patients as PT
from experiments import phase_runner as PR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True)
    ap.add_argument("--methods", default="mcmc,sbi")
    ap.add_argument("--population", default=None,
                    help="legacy population.npz (the prior is now the published one in "
                         "replaybg_priors; this only affects the cached centre artifact)")
    ap.add_argument("--start", type=int, default=0, help="patient offset (sharding)")
    ap.add_argument("--limit", type=int, default=None, help="patients per shard")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-fit even if a patient's comparison_table.csv exists")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only read existing tables and summarize (final step)")
    ap.add_argument("--out-summary", default=C.summary_path(C.PHASE2))
    ap.add_argument("--out-per-patient", default=C.per_patient_path(C.PHASE2))
    args = ap.parse_args()

    all_names = [s.name for s in PT.load_subjects_csv(args.patients)]
    end = args.start + args.limit if args.limit else None
    names = all_names[args.start:end]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[phase2] {len(names)}/{len(all_names)} patients "
          f"(start={args.start}) x methods={methods} (smoke={args.smoke})")

    PR.run_phase(names, methods, args.patients, population=args.population,
                 smoke=args.smoke, label="phase2",
                 out_summary=args.out_summary, out_per_patient=args.out_per_patient,
                 skip_existing=not args.no_skip, aggregate_only=args.aggregate_only)


if __name__ == "__main__":
    main()
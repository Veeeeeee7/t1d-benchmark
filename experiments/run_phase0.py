"""Phase 0 — matched-model baseline (all patients).

The Phase 0 analogue of ``run_phase2.py``: runs each per-instance twinning
method (MCMC, SBI) on the self-consistent ReplayBG cohort, where the plant *is*
the twin's own model class. This isolates intrinsic identifiability/inference
quality from the plant<->twin mismatch that Phase 2 deliberately includes, so it
is the "best case" reference the Phase 2 numbers should be read against.

Same running method as Phase 2: shard with ``--start`` / ``--limit`` (one chunk
per node, as ``run_phase0.sh`` does), each shard resumable (skips patients
already scored), then aggregate the full set:

    python -m experiments.run_phase0 --patients patients.csv --aggregate-only

Run (from the repo root):
    python -m experiments.derive_replaybg_params --patients patients.csv   # adds rbg_* cols
    python -m experiments.run_phase0 --patients patients.csv --start 0 --limit 256
    python -m experiments.run_phase0 --patients patients.csv --smoke --limit 4
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
from experiments import phase_runner as PR
from experiments import replaybg_plant as P
from experiments import phase0_paths as P0

# Phase 0 stages (ReplayBG plant) + scoring module — passed to the shared runner.
STAGES = {"mcmc": "experiments.run_mcmc0",
          "sbi": "experiments.run_sbi0"}
RESULTS_MODULE = "experiments.compute_results0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="augmented patients.csv (rbg_* cols from derive_replaybg_params) or a native phase0 cohort")
    ap.add_argument("--methods", default="mcmc,sbi")
    ap.add_argument("--start", type=int, default=0, help="patient offset (sharding)")
    ap.add_argument("--limit", type=int, default=None, help="patients per shard")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-skip", action="store_true",
                    help="re-fit even if a patient's comparison_table.csv exists")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="only read existing tables and summarize (final step)")
    ap.add_argument("--out-summary", default=C.summary_path(C.PHASE0))
    ap.add_argument("--out-per-patient", default=C.per_patient_path(C.PHASE0))
    args = ap.parse_args()

    all_names = [s.name for s in P.load_phase0_cohort(args.patients)]
    end = args.start + args.limit if args.limit else None
    names = all_names[args.start:end]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[phase0] {len(names)}/{len(all_names)} patients "
          f"(start={args.start}) x methods={methods} (smoke={args.smoke})")

    PR.run_phase(names, methods, args.patients, population=None,
                 smoke=args.smoke, label="phase0",
                 out_summary=args.out_summary, out_per_patient=args.out_per_patient,
                 skip_existing=not args.no_skip, aggregate_only=args.aggregate_only,
                 stages=STAGES, results_module=RESULTS_MODULE,
                 load_subjects=P.load_phase0_cohort,
                 results_dir_fn=P0.results_dir_for)


if __name__ == "__main__":
    main()
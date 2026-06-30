"""Generate the Phase 0 cohort (self-consistent ReplayBG patients).

Each patient is a true ReplayBG ``theta`` drawn deterministically from the
population prior, with a per-patient ``CR_true`` calibrated so the carb-error-free
optimal dose is f=1, and a deterministic carb-counting error ``m`` so the real
optimum is f* = m (interior on the candidate grid). The cohort is written to a
CSV that ``run_phase0`` / ``run_mcmc0`` / ``run_sbi0`` / ``compute_results0``
read.

This is the Phase 0 analogue of ``generate_patients.py`` (which builds the
simglucose/UVA-Padova cohort for Phase 2). Phase 0 never touches simglucose.

Run (from the repo root):
    python -m experiments.generate_phase0_patients --n 1013 --out patients0.csv
    python -m experiments.generate_phase0_patients --smoke            # 8 patients
    python -m experiments.generate_phase0_patients --no-calibrate      # CR=nominal
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import replaybg_plant as P
from t1d_twin import replaybg_model as RB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1013,
                    help="cohort size (matches the Phase 2 adult cohort by default)")
    ap.add_argument("--prefix", default="rbg", help="patient-name prefix")
    ap.add_argument("--out", default="patients0.csv", help="output CSV path")
    ap.add_argument("--seed", type=int, default=P.SEED)
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="skip per-patient CR calibration (use CR_NOMINAL)")
    ap.add_argument("--smoke", action="store_true", help="tiny 8-patient cohort")
    args = ap.parse_args()

    n = 8 if args.smoke else args.n
    # draw_theta perturbs around the installed centre; Phase 0 uses the fixed
    # reference (no cohort to derive from).
    RB.set_pop_theta(P.PHASE0_CENTER)
    print(f"[phase0-gen] building {n} patients "
          f"(calibrate={args.calibrate}); this runs the ReplayBG plant "
          f"~{30 if args.calibrate else 0} times/patient for CR calibration")

    subs = P.make_cohort(n, prefix=args.prefix, seed=args.seed,
                         calibrate=args.calibrate, verbose=True)
    P.write_phase0_csv(subs, args.out)

    # report the over/under split and per-m counts (mirrors generate_patients.py)
    # Carb error sign convention (see patients.py): m < 1 -> programmed CR too low
    # -> over-doses; m > 1 -> programmed CR too high -> under-doses.
    mults = [s.dose_mult for s in subs]
    over = sum(1 for m in mults if m < 1.0)
    under = sum(1 for m in mults if m > 1.0)
    counts = Counter(mults)
    print(f"\n[phase0-gen] wrote {len(subs)} patients -> {args.out}")
    print(f"[phase0-gen] carb error: {over} over-dosed (m<1) / {under} under-dosed (m>1)")
    print("[phase0-gen] per-m counts: "
          + ", ".join(f"m={m}: {counts[m]}" for m in sorted(counts)))
    crs = [s.CR_true for s in subs]
    print(f"[phase0-gen] CR_true range: {min(crs):.2f}–{max(crs):.2f} g/U "
          f"(median {sorted(crs)[len(crs) // 2]:.2f})")


if __name__ == "__main__":
    main()
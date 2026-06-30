"""Generate ``patients.csv`` — every subset-average of a base cohort.

Each row is one synthetic patient = the element-wise average of a *subset* of
the base UVA/Padova patients. Unlike the pairwise-only version, this enumerates
**all** subset sizes: pairs, triples, ... up to the whole cohort (the k=N row is
the average of every patient). For the 10 adults that is

    sum_{k=2..10} C(10, k) = 1013 synthetic patients

(1023 if singletons — the original patients — are included). Generation is just
averaging, so it is cheap even at 1013 rows; *running* experiments on all of them
is the expensive part, so subset the CSV (or use --limit downstream) as needed.

By default each synthetic patient is given a carb-counting-error perturbation
(its programmed carb ratio is deterministically mis-set so x1.00 is no longer
optimal); pass --no-carb-error for the old baseline-optimal cohort. See
docs/carb_error_perturbation.md and ``patients.apply_carb_error``.

Run (from the repo root):
    python -m experiments.generate_patients                       # adults, k=2..10 -> patients.csv (perturbed)
    python -m experiments.generate_patients --no-carb-error       # baseline-optimal cohort
    python -m experiments.generate_patients --include-singletons  # also the 10 originals
    python -m experiments.generate_patients --kind adult --max-size 3 --out patients_small.csv
"""
from __future__ import annotations

import os
import sys
import argparse
from itertools import combinations
from collections import Counter

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from . import patients as PT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="adult",
                    help="base cohort to average over (adult/adolescent/child)")
    ap.add_argument("--min-size", type=int, default=2,
                    help="smallest subset size to average (default 2 = pairs)")
    ap.add_argument("--max-size", type=int, default=None,
                    help="largest subset size (default = full cohort size)")
    ap.add_argument("--include-singletons", action="store_true",
                    help="also emit the original patients (k=1)")
    ap.add_argument("--carb-error", dest="carb_error", action="store_true",
                    default=True,
                    help="mis-set each patient's programmed carb ratio "
                         "(carb-counting-error perturbation); default ON")
    ap.add_argument("--no-carb-error", dest="carb_error", action="store_false",
                    help="disable the perturbation (baseline-optimal cohort)")
    ap.add_argument("--out", default=os.path.join(_root, "patients.csv"))
    args = ap.parse_args()

    bases = PT.list_patients(args.kind)
    n = len(bases)
    max_size = args.max_size or n
    lo = 1 if args.include_singletons else args.min_size
    sizes = range(lo, max_size + 1)
    print(f"[generate] cohort '{args.kind}': {n} patients -> subset sizes {lo}..{max_size}")

    subjects = []
    by_size = Counter()
    idx = 0
    for k in sizes:
        for combo in combinations(bases, k):
            idx += 1
            s = PT.averaged_subject(list(combo), new_name=f"synth{idx:04d}_k{k}")
            if args.carb_error:
                s = PT.apply_carb_error(s)   # m drawn deterministically from name
            subjects.append(s)
            by_size[k] += 1

    PT.write_patients_csv(subjects, args.out)
    print(f"[generate] wrote {len(subjects)} synthetic patients -> {args.out}")
    for k in sizes:
        if by_size[k]:
            print(f"    k={k:>2}: {by_size[k]} patients")

    if args.carb_error:
        mults = Counter(round(s.dose_mult, 3) for s in subjects)
        over = sum(n for m, n in mults.items() if m < 1.0)
        under = sum(n for m, n in mults.items() if m > 1.0)
        print(f"[generate] carb-counting-error ON; dose-multiplier split: "
              f"{over} over-dosed (m<1), {under} under-dosed (m>1)")
        for m in sorted(mults):
            print(f"    m={m:>4}: {mults[m]} patients")
    else:
        print("[generate] carb-counting-error OFF (baseline-optimal cohort)")


if __name__ == "__main__":
    main()
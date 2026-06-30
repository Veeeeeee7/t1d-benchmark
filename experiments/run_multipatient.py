"""Ground-truth 5-therapy ranking across many patients (no twins).

Source the patients either from a generated ``patients.csv`` (``--patients``) or
on the fly as pairwise averages of a cohort (``--kind``). For each patient it
runs the candidate therapies on simglucose, computes the **true** reward ranking,
and aggregates how the optimal therapy shifts across the population — the
multi-subject substrate (and a feasibility sanity check) for the twin sweep.

Run (from the repo root):
    python -m experiments.run_multipatient --patients patients.csv
    python -m experiments.run_multipatient --patients patients.csv --limit 50 --hours 48
    python -m experiments.run_multipatient --kind adult           # pairwise, no CSV
    python -m experiments.run_multipatient --smoke
"""
from __future__ import annotations

import os
import sys
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from . import exp_common as C
from . import patients as PT
from t1d_twin import value
from t1d_twin.evaluate import ranking

BOLUS_FACTORS = (0.85, 1.0, 1.5, 2.0, 2.5)


def _subjects(args):
    if args.patients:
        subs = PT.load_subjects_csv(args.patients)
    else:
        bases = PT.list_patients(args.kind)
        from itertools import combinations
        subs = [PT.averaged_subject(list(c)) for c in combinations(bases, 2)]
    if args.smoke:
        subs = subs[:3]
    elif args.limit:
        subs = subs[:args.limit]
    return subs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=None, help="patients.csv to read subjects from")
    ap.add_argument("--kind", default="adult", help="cohort if no --patients (pairwise avg)")
    ap.add_argument("--hours", type=float, default=24.0, help="run horizon per therapy")
    ap.add_argument("--limit", type=int, default=None, help="cap number of patients")
    ap.add_argument("--smoke", action="store_true", help="only 3 patients")
    args = ap.parse_args()

    subs = _subjects(args)
    therapies = [f"bolus_x{f:.2f}" for f in BOLUS_FACTORS]
    scenario, _ = C.weekly_scenario(args.hours)
    print(f"[multipatient] {len(subs)} patients x {len(therapies)} therapies @ {args.hours:.0f} h")

    rows, best_count = [], {t: 0 for t in therapies}
    for i, s in enumerate(subs):
        controllers = s.therapy_controllers(bolus_factors=BOLUS_FACTORS)
        rewards = {t: value.reward(s.run(c, scenario, args.hours, sensor_seed=C.SEED).bg())
                   for t, c in controllers.items()}
        rank = ranking(rewards)
        best = rank[0]
        best_count[best] += 1
        interior = best not in (therapies[0], therapies[-1])
        row = {"patient": s.name, "best": best, "interior": interior}
        row.update({t: rewards[t] for t in therapies})
        rows.append(row)
        if (i + 1) % 10 == 0 or len(subs) <= 20:
            print(f"  [{i + 1:>3}/{len(subs)}] {s.name}: best={best}")

    df = pd.DataFrame(rows).set_index("patient")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(C.RESULTS_DIR, "multipatient_rankings.csv")
    df.to_csv(csv_path)

    n_interior = int(df["interior"].sum())
    print("\n=== optimal-therapy distribution ===")
    for t in therapies:
        print(f"  {t}: {best_count[t]:>3} / {len(df)}")
    print(f"interior optimum for {n_interior}/{len(df)} patients "
          f"({100.0 * n_interior / len(df):.0f}%)")
    print(f"\n[multipatient] wrote {csv_path}")


if __name__ == "__main__":
    main()
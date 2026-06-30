"""Derive best-fit ReplayBG parameters for each simglucose patient and append
them to ``patients.csv`` (next to the UVA/Padova columns), so Phase 0/1/2 reuse
the fit instead of recomputing it.

For each patient we run its baseline identification run (the same one the twins
identify from) and MAP-fit the 8-parameter ReplayBG model with the
**same** ``population.fit_replaybg`` phase 1 uses — so the cached ``rbg_theta`` is
identical to phase 1's regression target. We also cache the per-patient basal
``rbg_Ib`` and the Phase-0-plant matched carb ratio ``rbg_CR_true`` (calibrated so
the carb-error optimum f* = m lands interior), plus the fit RMSE.

Where the cache is used afterwards:
* Phase 0 (`replaybg_plant.subjects_from_patients_csv`): the matched-model plant
  runs at ``rbg_theta`` with the patient's ``BW`` / ``rbg_Ib`` / ``rbg_CR_true``
  and the patient's existing ``dose_mult`` — so Phase 0 patient *i* is the
  matched-model twin of Phase 2 patient *i*.
* Phase 1 (`build_phase1_dataset`): reads ``rbg_theta`` as the regression target
  instead of re-running the MAP fit every build.
* Phase 2: can warm-start the MCMC walkers / inspect the MAP projection.

Same running method as the other phases: shard with ``--start`` / ``--limit``
(each shard writes resumable per-patient parts), then ``--merge`` once to append
the columns to the CSV.

Run (from the repo root):
    python -m experiments.derive_replaybg_params --patients patients.csv          # fit all + merge
    python -m experiments.derive_replaybg_params --patients patients.csv --start 0 --limit 256
    python -m experiments.derive_replaybg_params --patients patients.csv --merge   # after all shards
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd

from experiments import exp_common as C
from experiments import patients as PT
from experiments import population as POP
from experiments import replaybg_plant as RBP
from t1d_twin.replaybg_model import THETA_NAMES

DEFAULT_PARTS = os.path.join(C.ARTIFACT_DIR, "rbg_fits")


# ---------------------------------------------------------------------------
# Per-patient fit (one worker)
# ---------------------------------------------------------------------------
def _fit_one(task):
    """Baseline run -> ReplayBG LS fit -> (theta, rmse, Ib, CR_true)."""
    s, hours, dt, max_nfev, parts_dir, overwrite = task
    part = os.path.join(parts_dir, f"{s.safe_name}.npz")
    if os.path.exists(part) and not overwrite:
        return (s.name, "skip")

    idrun = C.subject_identification_run(s, hours)
    theta, rmse = POP.fit_replaybg(idrun, dt=dt, max_nfev=max_nfev)
    _, insulin, _ = idrun.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])
    # Phase-0-plant matched carb ratio at this patient's BW/basal (so f* = m).
    cr_true = RBP.calibrate_cr_true(theta, BW=float(idrun.BW), Ib=Ib, hours=hours)

    os.makedirs(parts_dir, exist_ok=True)
    np.savez(part, name=s.name, theta=np.asarray(theta, float),
             rmse=float(rmse), Ib=Ib, cr_true=float(cr_true))
    return (s.name, "ok", float(rmse))


def fit_cohort(subjects, hours, dt, max_nfev, parts_dir, jobs, overwrite, verbose=True):
    n = len(subjects)
    workers = POP.n_workers(jobs)
    tasks = [(s, hours, dt, max_nfev, parts_dir, overwrite) for s in subjects]
    t0 = time.time()

    def _note(i, res):
        if verbose and ((i + 1) % 10 == 0 or n <= 12):
            tag = res[1] + (f" RMSE={res[2]:.1f}" if len(res) > 2 else "")
            print(f"  [{i + 1}/{n}] {res[0]}: {tag} ({time.time() - t0:.0f}s)", flush=True)

    if workers <= 1 or n <= 1:
        for i, task in enumerate(tasks):
            _note(i, _fit_one(task))
    else:
        if verbose:
            print(f"  [parallel] fitting {n} patients across {workers} worker(s)", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap(_fit_one, tasks, chunksize=1)):
                _note(i, res)


# ---------------------------------------------------------------------------
# Merge per-patient parts into the patients.csv
# ---------------------------------------------------------------------------
def merge_parts(patients_csv, parts_dir, out, verbose=True):
    """Join every ``<parts_dir>/*.npz`` onto ``patients.csv`` by Name -> ``out``."""
    rows = []
    for fn in sorted(os.listdir(parts_dir)):
        if not fn.endswith(".npz"):
            continue
        d = np.load(os.path.join(parts_dir, fn), allow_pickle=True)
        rec = {"Name": str(d["name"])}
        for nm, v in zip(THETA_NAMES, d["theta"]):
            rec[f"rbg_{nm}"] = float(v)
        rec["rbg_Ib"] = float(d["Ib"])
        rec["rbg_CR_true"] = float(d["cr_true"])
        rec["rbg_fit_rmse"] = float(d["rmse"])
        rows.append(rec)
    if not rows:
        sys.exit(f"[derive] no parts in {parts_dir}; run the fit step first")
    fits = pd.DataFrame.from_records(rows)

    base = pd.read_csv(patients_csv)
    # drop any stale rbg_* columns so a re-derive overwrites cleanly
    base = base[[c for c in base.columns if not c.startswith("rbg_")]]
    merged = base.merge(fits, on="Name", how="left")

    missing = merged["rbg_SI"].isna().sum()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    merged.to_csv(out, index=False)
    if verbose:
        print(f"[derive] merged {len(fits)}/{len(base)} fits -> {out}"
              + (f"  ({missing} patients still un-fit)" if missing else ""))
        print(f"[derive] fit RMSE: mean {fits['rbg_fit_rmse'].mean():.1f}, "
              f"worst {fits['rbg_fit_rmse'].max():.1f} mg/dL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="patients.csv to fit + augment")
    ap.add_argument("--out", default=None, help="augmented CSV (default: in place over --patients)")
    ap.add_argument("--parts-dir", default=DEFAULT_PARTS, help="per-patient fit cache dir")
    ap.add_argument("--hours", type=float, default=C.WINDOW_HOURS,
                    help="baseline horizon to fit on (match the twins' identification window)")
    ap.add_argument("--max-nfev", type=int, default=80, help="MAP-fit budget (matches phase 1)")
    ap.add_argument("--start", type=int, default=0, help="patient offset (sharding)")
    ap.add_argument("--limit", type=int, default=None, help="patients per shard")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker processes (0 = auto)")
    ap.add_argument("--smoke", action="store_true", help="tiny horizon-matched check on a few patients")
    ap.add_argument("--overwrite", action="store_true", help="refit even if a part exists")
    ap.add_argument("--merge", action="store_true", help="only merge existing parts into the CSV")
    args = ap.parse_args()

    out = args.out or args.patients

    if args.merge:
        merge_parts(args.patients, args.parts_dir, out)
        return

    subs = PT.load_subjects_csv(args.patients)
    end = args.start + args.limit if args.limit else None
    shard = subs[args.start:end]
    sharded = (args.start != 0) or (args.limit is not None)
    print(f"[derive] fitting {len(shard)}/{len(subs)} patients "
          f"(start={args.start}, horizon {args.hours:.0f} h) into {args.parts_dir}")

    fit_cohort(shard, args.hours, C.DT, args.max_nfev, args.parts_dir,
               args.jobs, args.overwrite)

    # Single-machine convenience: if not explicitly sharding, merge right away.
    if not sharded:
        merge_parts(args.patients, args.parts_dir, out)
    else:
        print(f"[derive] shard done; run --merge once all shards finish:\n"
              f"         python -m experiments.derive_replaybg_params "
              f"--patients {args.patients} --merge")


if __name__ == "__main__":
    main()
"""Phase 1 — the amortized baseline experiment (cross-patient, K-fold CV).

Trains the amortized CGM->theta regressors on all generated patients with K-fold
cross-validation and reports DT2 (decision-transfer) metrics, mean +/- std, for
each baseline:

    tcn_cgm_ins_cho      -- TCN on CGM + insulin + CHO (the info MCMC/SBI condition on)
    linear_cgm_ins_cho   -- ridge linear baseline on the same channels

Flow
----
1. Load (or build) the cached per-patient dataset (CGM/insulin/CHO/theta/Ib).
2. For each baseline: K-fold CV -> one held-out theta-hat per patient (cheap).
3. For each patient: collect simglucose ground truth ONCE, then score every
   baseline's point-estimate twin against it (so the ground-truth grid is shared).
4. Write ``results/phase1_per_patient.csv`` and ``results/phase1_summary.csv``.

The per-instance twinning methods (MCMC/SBI) are Phases 2 and 3; this phase is
deliberately separate and amortized.

Run (from the repo root):
    python -m experiments.build_phase1_dataset --patients patients.csv          # prep
    python -m experiments.run_phase1 --patients patients.csv
    python -m experiments.run_phase1 --patients patients.csv --smoke --limit 20
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
from experiments import tcn_cv as TCN
from experiments import linear_cv as LIN
from experiments import dt2_scoring as SC
from experiments.cv_common import CGM_INS_CHO
from experiments.phase_runner import summarize
from experiments.build_phase1_dataset import (
    load_dataset, build_dataset, save_dataset, DEFAULT_OUT, n_workers,
)
from t1d_twin.identify_tcn import PointTwin

SIGMA = 10.0
# Each baseline maps a name -> (cross-validated predictor, input channels). Both
# predictors share the cross_val_predict(dataset, channels, k, seed, cfg, device)
# signature and return one held-out theta-hat per patient.
BASELINES = {
    "tcn_cgm_ins_cho": (TCN.cross_val_predict, CGM_INS_CHO),
    "linear_cgm_ins_cho": (LIN.cross_val_predict, CGM_INS_CHO),
}
# CV training configs (the regressor head is small; this is cheap vs. scoring).
PROD_CFG = dict(hidden=128, n_epochs=300, lr=1e-3, batch_size=256)
SMOKE_CFG = dict(hidden=32, n_epochs=40, lr=1e-3, batch_size=64)

DECISION_COLS = ["spearman", "regret"]
FIDELITY_COLS = ["rmse", "mard"]


def _score_one(task):
    """Worker: collect simglucose ground truth for one patient, then score every
    baseline's point-estimate twin against it. Defined at module scope so it is
    picklable by the process pool. ``preds_i`` maps baseline name -> that patient's
    held-out theta-hat, so the (shared) ground-truth grid is computed once per
    patient.
    """
    (name, subject, hours, bolus_factors, basal_factors,
     seed, sample_time, Ib_i, preds_i) = task
    truth = SC.collect_truth(subject, hours, bolus_factors, basal_factors, seed=seed)
    out = []
    for bl, theta_hat in preds_i.items():
        twin = PointTwin(theta_hat, Ib=float(Ib_i), sample_time=sample_time,
                         dt=C.DT, sensor_name=C.SENSOR, sigma=SIGMA)
        row = SC.score_twin(twin, truth)
        out.append({"patient": name, "method": bl, **row})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="patients.csv (for subjects + ground truth)")
    ap.add_argument("--dataset", default=DEFAULT_OUT, help="cached dataset (default artifacts/phase1/dataset.npz)")
    ap.add_argument("--build", action="store_true",
                    help="build the dataset now if the cache is missing")
    ap.add_argument("--hours", type=float, default=None,
                    help="override horizon (default: the dataset's own horizon)")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=None,
                    help="torch device for CV training (default: cuda if available, "
                         "else cpu)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N patients (quick check)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes for the per-patient scoring loop "
                         "(0 = auto: SLURM_CPUS_PER_TASK, else all cores). CV "
                         "is cheap and stays serial.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CV config + smoke therapy grid (plumbing)")
    ap.add_argument("--out-summary", default=C.summary_path(C.PHASE1),
                    help="per-phase aggregate CSV (default results/phase1_summary.csv)")
    ap.add_argument("--out-per-patient", default=C.per_patient_path(C.PHASE1),
                    help="per-patient roll-up CSV (default results/phase1_per_patient.csv)")
    args = ap.parse_args()

    # --- dataset ---
    if not os.path.exists(args.dataset):
        if not args.build:
            sys.exit(f"[phase1] dataset {args.dataset} missing; run "
                     f"experiments.build_phase1_dataset first, or pass --build")
        subs = PT.load_subjects_csv(args.patients)
        hours = args.hours if args.hours is not None else C.WINDOW_HOURS
        bf, bbf = C.factors_for(args.smoke)
        print(f"[phase1] building dataset ({len(subs)} patients x "
              f"{len(bf) + len(bbf)} therapies, {hours:.0f} h) ...")
        save_dataset(build_dataset(subs, hours=hours,
                                   bolus_factors=bf, basal_factors=bbf), args.dataset)
    data = load_dataset(args.dataset)
    hours = args.hours if args.hours is not None else data["hours"]
    sample_time = data["sample_time"]
    names = [str(n) for n in data["names"]]
    Ib = np.asarray(data["Ib"], dtype=float)
    cfg = SMOKE_CFG if args.smoke else PROD_CFG
    n_examples = data["cgm"].shape[0]
    print(f"[phase1] dataset: N={len(names)} patients, M={n_examples} examples, "
          f"L={data['cgm'].shape[1]} samples @ {sample_time:.0f}-min, "
          f"horizon {hours:.0f} h")

    # --- cross-validated theta-hat per baseline (cheap, in-memory) ---
    preds = {}
    for bl, (predict_fn, channels) in BASELINES.items():
        print(f"\n[phase1] === CV for baseline '{bl}' (channels={list(channels)}) ===")
        t0 = time.time()
        preds[bl] = predict_fn(data, channels, k=args.kfolds,
                               seed=args.seed, cfg=cfg, device=args.device)
        print(f"[phase1] '{bl}' CV done in {time.time() - t0:.1f} s")

    # --- score against simglucose ground truth (shared across baselines) ---
    subjects = {s.name: s for s in PT.load_subjects_csv(args.patients)}
    bolus_factors, basal_factors = C.factors_for(args.smoke)
    n_score = len(names) if args.limit is None else min(args.limit, len(names))
    workers = n_workers(args.jobs)
    print(f"\n[phase1] scoring {n_score} patients x {len(BASELINES)} baselines "
          f"(|Pi|={len(bolus_factors) + len(basal_factors)}) "
          f"across {workers} worker(s) ...")

    # one independent task per patient (ground truth shared across baselines inside)
    tasks = []
    for i in range(n_score):
        name = names[i]
        subject = subjects.get(name)
        if subject is None:
            print(f"  [warn] {name} not in {args.patients}; skipping")
            continue
        preds_i = {bl: preds[bl][i] for bl in BASELINES}
        tasks.append((name, subject, hours, bolus_factors, basal_factors,
                      args.seed, sample_time, float(Ib[i]), preds_i))

    rows = []
    t0 = time.time()

    def _progress(done, last_name):
        if done % 25 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] {last_name} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    if workers <= 1 or len(tasks) <= 1:
        for k, task in enumerate(tasks):
            rows.extend(_score_one(task))
            _progress(k + 1, task[0])
    else:
        # torch is imported via tcn_cv; pin it to 1 thread BEFORE forking so the
        # forked scoring workers don't each spin up a full intra-op threadpool.
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        ctx = mp.get_context("fork")          # children inherit imported modules + globals
        with ctx.Pool(processes=workers) as pool:
            for k, out in enumerate(pool.imap(_score_one, tasks, chunksize=1)):
                rows.extend(out)
                _progress(k + 1, out[0]["patient"] if out else "?")

    if not rows:
        sys.exit("[phase1] no patients scored.")

    # --- write outputs (flat per-phase files in results/) ---
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_per_patient) or ".", exist_ok=True)
    per_path = args.out_per_patient
    df.to_csv(per_path, index=False)

    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    sum_path = args.out_summary
    summary.to_csv(sum_path, index=False)

    print(f"\n[phase1] wrote {per_path} ({len(df)} rows) and {sum_path}")
    print("\n=== Phase 1 DT2 (mean +/- std across patients) ===")
    for _, r in summary.iterrows():
        parts = [f"n={int(r['n'])}"]
        for col in DECISION_COLS + FIDELITY_COLS:
            if f"{col}_mean" in summary.columns:
                parts.append(f"{col}={r[f'{col}_mean']:.3f}+/-{r[f'{col}_std']:.3f}")
        print(f"  {r['method']:>16}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
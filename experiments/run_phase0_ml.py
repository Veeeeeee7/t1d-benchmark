"""Phase 0 (amortized) — the matched-model baseline for the ML regressors.

The Phase 0 analogue of ``run_phase1``: trains the same amortized CGM->theta
regressors (TCN and ridge) with patient-grouped K-fold CV, but on the
matched-model ReplayBG dataset (``build_phase0_dataset``) and scoring the
predicted point-estimate twins against the **ReplayBG** ground truth. Because the
plant is in-class and the labels are exact, this is the amortized ceiling to read
the Phase 1 (simglucose + projected labels) numbers against.

Reused unchanged from Phase 1: ``tcn_cv`` / ``linear_cv`` / ``cv_common`` (the CV
and models are plant-agnostic — they only see the dataset arrays) and
``phase_runner.summarize``. The only Phase-0-specific pieces are the dataset
(ReplayBG inputs + true theta) and the ground truth (ReplayBG plant).

Outputs go to ``results/phase0_ml_{summary,per_patient}.csv`` in the same schema
as every other phase, so they concatenate with the per-instance Phase 0 results
(MCMC/SBI) for a combined view.

Run (from the repo root):
    python -m experiments.build_phase0_dataset --patients patients.csv      # prep
    python -m experiments.run_phase0_ml --patients patients.csv
    python -m experiments.run_phase0_ml --patients patients.csv --smoke --limit 20
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

from experiments import replaybg_plant as P
from experiments import output_paths as OP
from experiments import tcn_cv as TCN
from experiments import linear_cv as LIN
from experiments.cv_common import CGM_INS_CHO
from experiments.phase_runner import summarize
from experiments.build_phase0_dataset import (
    load_dataset, build_dataset, save_dataset, DEFAULT_OUT, n_workers,
)
from t1d_twin import value
from t1d_twin.evaluate import evaluate_twin, _row_from_result
from t1d_twin.identify_tcn import PointTwin

SIGMA = P.SIGMA
BASELINES = {
    "tcn_cgm_ins_cho": (TCN.cross_val_predict, CGM_INS_CHO),
    "linear_cgm_ins_cho": (LIN.cross_val_predict, CGM_INS_CHO),
}
PROD_CFG = dict(hidden=128, n_epochs=300, lr=1e-3, batch_size=256)
SMOKE_CFG = dict(hidden=32, n_epochs=40, lr=1e-3, batch_size=64)
DECISION_COLS = ["spearman", "regret"]
FIDELITY_COLS = ["rmse", "mard"]


def _collect_truth(subject, hours, bolus_factors, basal_factors, seed=P.SEED):
    """Run the candidate therapies on the ReplayBG plant once -> reusable truth."""
    true_runs = P.ground_truth(subject, hours, bolus_factors, basal_factors, seed=seed)
    true_ig = {n: r.bg() for n, r in true_runs.items()}
    true_rewards = {n: value.reward(g) for n, g in true_ig.items()}
    return true_runs, true_ig, true_rewards


def _score_one(task):
    """Worker: ReplayBG ground truth for one patient, then score every baseline's
    point-estimate twin against it (shared grid)."""
    (name, subject, hours, bolus_factors, basal_factors,
     seed, sample_time, Ib_i, preds_i) = task
    true_runs, true_ig, true_rewards = _collect_truth(
        subject, hours, bolus_factors, basal_factors, seed=seed)
    out = []
    for bl, theta_hat in preds_i.items():
        twin = PointTwin(theta_hat, Ib=float(Ib_i), sample_time=sample_time,
                         dt=P.DT, sensor_name=P.SENSOR, sigma=SIGMA)
        res = evaluate_twin(twin, true_runs, true_rewards=true_rewards,
                            true_ig_by_policy=true_ig)
        out.append({"patient": name, "method": bl, **_row_from_result(res)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True,
                    help="augmented patients.csv (rbg_* cols) or a native phase0 cohort")
    ap.add_argument("--dataset", default=DEFAULT_OUT, help="cached dataset (default artifacts/phase0_ml/dataset.npz)")
    ap.add_argument("--build", action="store_true", help="build the dataset if missing")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=P.SEED)
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N patients")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes for scoring (0 = auto). CV stays serial.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-summary", default=OP.summary_path(OP.PHASE0_ML),
                    help="per-phase aggregate CSV (default results/phase0_ml_summary.csv)")
    ap.add_argument("--out-per-patient", default=OP.per_patient_path(OP.PHASE0_ML),
                    help="per-patient roll-up CSV (default results/phase0_ml_per_patient.csv)")
    args = ap.parse_args()

    # --- dataset (ReplayBG plant inputs + exact true-theta labels) ---
    if not os.path.exists(args.dataset):
        if not args.build:
            sys.exit(f"[phase0-ml] dataset {args.dataset} missing; run "
                     f"experiments.build_phase0_dataset first, or pass --build")
        subs = P.load_phase0_cohort(args.patients)
        hours = args.hours if args.hours is not None else P.WINDOW_HOURS
        bf, bbf = P.factors_for(args.smoke)
        print(f"[phase0-ml] building dataset ({len(subs)} patients x "
              f"{len(bf) + len(bbf)} therapies, {hours:.0f} h) ...")
        save_dataset(build_dataset(subs, hours=hours,
                                   bolus_factors=bf, basal_factors=bbf), args.dataset)
    data = load_dataset(args.dataset)
    hours = args.hours if args.hours is not None else data["hours"]
    sample_time = data["sample_time"]
    names = [str(n) for n in data["names"]]
    Ib = np.asarray(data["Ib"], dtype=float)
    cfg = SMOKE_CFG if args.smoke else PROD_CFG
    print(f"[phase0-ml] dataset: N={len(names)} patients, M={data['cgm'].shape[0]} "
          f"examples, L={data['cgm'].shape[1]} @ {sample_time:.0f}-min, horizon {hours:.0f} h "
          f"(labels exact: matched-model)")

    # --- cross-validated theta-hat per baseline (reused CV, plant-agnostic) ---
    preds = {}
    for bl, (predict_fn, channels) in BASELINES.items():
        print(f"\n[phase0-ml] === CV for baseline '{bl}' (channels={list(channels)}) ===")
        t0 = time.time()
        preds[bl] = predict_fn(data, channels, k=args.kfolds,
                               seed=args.seed, cfg=cfg, device=args.device)
        print(f"[phase0-ml] '{bl}' CV done in {time.time() - t0:.1f} s")

    # --- score against ReplayBG ground truth (shared across baselines) ---
    subjects = {s.name: s for s in P.load_phase0_cohort(args.patients)}
    bolus_factors, basal_factors = P.factors_for(args.smoke)
    n_score = len(names) if args.limit is None else min(args.limit, len(names))
    workers = n_workers(args.jobs)
    print(f"\n[phase0-ml] scoring {n_score} patients x {len(BASELINES)} baselines "
          f"(|Pi|={len(bolus_factors) + len(basal_factors)}) across {workers} worker(s) ...")

    tasks = []
    for i in range(n_score):
        subject = subjects.get(names[i])
        if subject is None:
            print(f"  [warn] {names[i]} not in {args.patients}; skipping")
            continue
        preds_i = {bl: preds[bl][i] for bl in BASELINES}
        tasks.append((names[i], subject, hours, bolus_factors, basal_factors,
                      args.seed, sample_time, float(Ib[i]), preds_i))

    rows = []
    t0 = time.time()

    def _progress(done, last):
        if done % 25 == 0 or done == len(tasks):
            print(f"  [{done}/{len(tasks)}] {last} ({time.time() - t0:.0f}s)", flush=True)

    if workers <= 1 or len(tasks) <= 1:
        for k, task in enumerate(tasks):
            rows.extend(_score_one(task))
            _progress(k + 1, task[0])
    else:
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for k, out in enumerate(pool.imap(_score_one, tasks, chunksize=1)):
                rows.extend(out)
                _progress(k + 1, out[0]["patient"] if out else "?")

    if not rows:
        sys.exit("[phase0-ml] no patients scored.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_per_patient) or ".", exist_ok=True)
    per_path = args.out_per_patient
    df.to_csv(per_path, index=False)
    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    sum_path = args.out_summary
    summary.to_csv(sum_path, index=False)

    print(f"\n[phase0-ml] wrote {per_path} ({len(df)} rows) and {sum_path}")
    print("\n=== Phase 0 (amortized) DT2 (mean +/- std across patients) ===")
    for _, r in summary.iterrows():
        parts = [f"n={int(r['n'])}"]
        for col in DECISION_COLS + FIDELITY_COLS:
            if f"{col}_mean" in summary.columns:
                parts.append(f"{col}={r[f'{col}_mean']:.3f}+/-{r[f'{col}_std']:.3f}")
        print(f"  {r['method']:>16}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
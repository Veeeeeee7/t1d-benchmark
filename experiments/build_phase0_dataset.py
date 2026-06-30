"""Phase 0 (amortized) — build the cross-patient regression dataset on the
**matched-model** ReplayBG plant.

The Phase 0 analogue of ``build_phase1_dataset``. Same layout and same channels,
but two things differ because the plant *is* ReplayBG:

* inputs come from the ReplayBG plant (``replaybg_plant.ground_truth``), one
  ``(cgm, insulin, cho)`` trace per candidate therapy, and
* the per-patient **target is the known true theta** (``subject.theta``), not a
  fitted projection (Phase 1 uses a MAP fit) — there is no model mismatch, so the label is exact
  (``proj_rmse`` is 0 by construction).

So Phase 0's amortized baselines answer "how well can a TCN / ridge recover the
true ReplayBG theta from ReplayBG-generated CGM, and how well do those recovered
twins rank therapies?" — the matched-model ceiling for the amortized methods,
read against Phase 1 (same models, simglucose plant + projected labels).

The output ``.npz`` is format-identical to ``build_phase1_dataset`` (same keys
and shapes), so ``tcn_cv`` / ``linear_cv`` / ``cv_common`` consume it unchanged.
This module is simglucose-free (cohort + plant are both ReplayBG).

Run (from the repo root):
    python -m experiments.build_phase0_dataset --patients patients.csv          # matched cohort
    python -m experiments.build_phase0_dataset --patients patients0.csv          # prior-draw cohort
    python -m experiments.build_phase0_dataset --patients patients.csv --smoke --limit 20
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

from experiments import replaybg_plant as P
from experiments import output_paths as OP

CHANNELS = ("cgm", "insulin", "cho")
DEFAULT_OUT = OP.dataset_path(OP.PHASE0_ML)   # artifacts/phase0_ml/dataset.npz


def n_workers(requested: int | None = None) -> int:
    if requested and requested > 0:
        return int(requested)
    for var in ("SLURM_CPUS_PER_TASK", "PHASE0_JOBS"):
        v = os.environ.get(var)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return os.cpu_count() or 1


def _build_one(task):
    """Worker: one subject's per-patient (theta, Ib) + one trace per therapy.

    Module-scope so the process pool can pickle it. The target theta is the
    subject's *true* ReplayBG physiology (exact label); the input traces come
    from running each candidate therapy through the ReplayBG plant.
    """
    s, hours, bolus_factors, basal_factors, seed = task
    runs = P.ground_truth(s, hours, bolus_factors, basal_factors, seed=seed)
    treatments, cgm_l, ins_l, cho_l = [], [], [], []
    for tname, run in runs.items():
        _, ins_s, cho_s = run.replaybg_inputs(dt=run.sample_time)   # sample cadence
        treatments.append(tname)
        cgm_l.append(np.asarray(run.cgm(), dtype=np.float32))
        ins_l.append(np.asarray(ins_s, dtype=np.float32))
        cho_l.append(np.asarray(cho_s, dtype=np.float32))
    return (s.name, treatments, cgm_l, ins_l, cho_l, float(s.Ib),
            np.asarray(s.theta, dtype=np.float64))


def build_dataset(subjects, hours: float = P.WINDOW_HOURS,
                  bolus_factors=None, basal_factors=None,
                  verbose: bool = True, jobs: int | None = None) -> dict:
    if bolus_factors is None:
        bolus_factors, basal_factors = P.factors_for(False)
    if basal_factors is None:
        basal_factors = ()

    n = len(subjects)
    workers = n_workers(jobs)
    tasks = [(s, hours, bolus_factors, basal_factors, P.SEED) for s in subjects]
    results = [None] * n
    t0 = time.time()

    def _note(i):
        if verbose and ((i + 1) % 10 == 0 or n <= 12):
            print(f"  [{i + 1}/{n}] {results[i][0]}: {len(results[i][1])} therapies "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if workers <= 1 or n <= 1:
        for i, task in enumerate(tasks):
            results[i] = _build_one(task)
            _note(i)
    else:
        if verbose:
            print(f"  [parallel] {n} subjects across {workers} worker(s)", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap(_build_one, tasks, chunksize=1)):
                results[i] = res
                _note(i)

    names, Ib_l, theta_l = [], [], []
    ex_cgm, ex_ins, ex_cho, ex_pid, ex_treat = [], [], [], [], []
    for pi, r in enumerate(results):
        name, treatments, cgm_l, ins_l, cho_l, Ib_i, theta_i = r
        names.append(name); Ib_l.append(Ib_i); theta_l.append(theta_i)
        for tname, c, ins, cho in zip(treatments, cgm_l, ins_l, cho_l):
            ex_cgm.append(c); ex_ins.append(ins); ex_cho.append(cho)
            ex_pid.append(pi); ex_treat.append(tname)

    L = min(len(c) for c in ex_cgm)

    def _stack(seq):
        return np.stack([a[:L] for a in seq], axis=0)

    return {
        "cgm": _stack(ex_cgm), "insulin": _stack(ex_ins), "cho": _stack(ex_cho),
        "patient_idx": np.asarray(ex_pid, dtype=np.int64),
        "treatment": np.array(ex_treat, dtype=object),
        "theta": np.stack(theta_l, axis=0),
        "Ib": np.asarray(Ib_l, dtype=np.float64),
        "names": np.array(names, dtype=object),
        # label is exact in the matched-model setting (kept for format parity)
        "proj_rmse": np.zeros(len(names), dtype=np.float64),
        "hours": float(hours), "sample_time": float(P.SAMPLE_TIME), "dt": float(P.DT),
        "bolus_factors": np.asarray(bolus_factors, dtype=np.float64),
        "basal_factors": np.asarray(basal_factors, dtype=np.float64),
    }


def save_dataset(d: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **d)
    return path


def load_dataset(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    for k in ("hours", "sample_time", "dt"):
        if k in d:
            d[k] = float(d[k])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True,
                    help="augmented patients.csv (rbg_* cols) or a native phase0 cohort")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .npz path")
    ap.add_argument("--hours", type=float, default=P.WINDOW_HOURS)
    ap.add_argument("--smoke", action="store_true", help="smoke therapy grid")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker processes (0 = auto)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"[phase0-data] {args.out} exists; pass --overwrite to rebuild")

    bolus_factors, basal_factors = P.factors_for(args.smoke)
    subs = P.load_phase0_cohort(args.patients)
    end = args.start + args.limit if args.limit else None
    subs = subs[args.start:end]
    T = len(bolus_factors) + len(basal_factors)
    print(f"[phase0-data] building from {len(subs)} patients x {T} therapies "
          f"(horizon {args.hours:.0f} h) on the ReplayBG plant ...", flush=True)

    d = build_dataset(subs, hours=args.hours, bolus_factors=bolus_factors,
                      basal_factors=basal_factors, jobs=args.jobs)
    save_dataset(d, args.out)
    print(f"[phase0-data] X shape {d['cgm'].shape} (M=N*T, L); "
          f"N={len(d['names'])} patients, T={T} therapies; labels exact "
          f"(matched-model). wrote {args.out}")


if __name__ == "__main__":
    main()
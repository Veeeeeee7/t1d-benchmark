"""Phase 1 — build & cache the cross-patient regression dataset (treatment-augmented).

For each synthetic patient we now emit **one example per candidate therapy**, so
the dataset is ``N_patients * T_therapies`` examples:

    channels :  CGM, insulin (mU/kg/min), CHO (mg/kg/min)   -- at CGM cadence,
                one trace per (patient, therapy)
    target   :  ReplayBG theta (per-patient), from a MAP projection (under the
                published prior) of
                that patient's baseline identification run
    Ib       :  basal insulin (per-patient), needed to replay

``theta`` is a property of the *patient's* dynamics, not the therapy (insulin/CHO
are forcing inputs), so every one of a patient's therapy traces shares that
patient's single theta. The therapy traces are different *observations* of the
same patient -- a physically grounded augmentation that multiplies the training
set without changing the label space (Phi !in F preserved: only ReplayBG-
coordinate information enters the target).

Layout of the saved arrays:
    per-example (length M = N*T):  cgm, insulin, cho, patient_idx, treatment
    per-patient (length N):        theta, Ib, names, proj_rmse

The ``patient_idx`` group label is what the grouped K-fold in ``cv_common`` uses to
keep a patient's traces together (leakage-free).

Run (from the repo root):
    python -m experiments.build_phase1_dataset --patients patients.csv
    python -m experiments.build_phase1_dataset --patients patients.csv --smoke --limit 20
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

from experiments import exp_common as C
from experiments import patients as PT
from experiments import population as POP

DEFAULT_OUT = C.dataset_path(C.PHASE1)   # artifacts/phase1/dataset.npz
CHANNELS = ("cgm", "insulin", "cho")


# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------
def n_workers(requested: int | None = None) -> int:
    """Resolve the worker count: explicit --jobs > SLURM_CPUS_PER_TASK > cpu_count."""
    if requested and requested > 0:
        return int(requested)
    for var in ("SLURM_CPUS_PER_TASK", "PHASE1_JOBS"):
        v = os.environ.get(var)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return os.cpu_count() or 1


def _build_one(task):
    """Worker: one subject's per-patient target + one trace per candidate therapy.

    Defined at module scope so it is picklable by the process pool. The target
    theta / Ib are projected once from the patient's baseline identification run;
    the input traces come from running every candidate therapy on the subject.
    Returns a tuple keyed by position so the parent can re-stack in input order.
    """
    s, hours, dt, max_nfev, bolus_factors, basal_factors = task

    # Per-patient target: ReplayBG projection of the baseline identification run.
    # Reuse the cached fit (derive_replaybg_params) when present, so the dataset
    # build no longer re-runs the per-patient MAP fit every time.
    idrun = C.subject_identification_run(s, hours)
    if getattr(s, "rbg_theta", None) is not None:
        theta = np.asarray(s.rbg_theta, dtype=float)
        rmse = float(s.rbg_fit_rmse) if s.rbg_fit_rmse is not None else float("nan")
    else:
        theta, rmse = POP.fit_replaybg(idrun, dt=dt, max_nfev=max_nfev)
    _, insulin_dt, _ = idrun.replaybg_inputs(dt=dt)
    Ib = float(insulin_dt[0])

    # Inputs: one (cgm, insulin, cho) trace per candidate therapy.
    true_runs, _ = C.subject_ground_truth(s, hours, bolus_factors, basal_factors)
    treatments, cgm_l, ins_l, cho_l = [], [], [], []
    for tname, run in true_runs.items():
        df = run.df
        treatments.append(tname)
        cgm_l.append(np.asarray(run.cgm(), dtype=np.float32))
        ins_l.append(df["insulin_mU_kg_min"].to_numpy(dtype=np.float32))
        cho_l.append(df["CHO_mg_kg_min"].to_numpy(dtype=np.float32))

    return (s.name, treatments, cgm_l, ins_l, cho_l, Ib,
            np.asarray(theta, dtype=np.float64), float(rmse))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_dataset(subjects, hours: float = C.WINDOW_HOURS, dt: float = C.DT,
                  max_nfev: int = 80, bolus_factors=None, basal_factors=None,
                  verbose: bool = True, jobs: int | None = None) -> dict:
    """Run each subject under every candidate therapy and collect the augmented
    ``(channels, patient_idx, treatment)`` examples plus per-patient ``(theta, Ib)``.

    Channels are truncated to the common minimum length across *all* traces so
    they form a rectangular ``(M, L)`` matrix.
    """
    if bolus_factors is None:
        bolus_factors, basal_factors = C.factors_for(False)
    if basal_factors is None:
        basal_factors = ()

    n = len(subjects)
    workers = n_workers(jobs)
    tasks = [(s, hours, dt, max_nfev, bolus_factors, basal_factors) for s in subjects]
    results = [None] * n
    t0 = time.time()

    def _note(i, res):
        if verbose and ((i + 1) % 10 == 0 or n <= 12):
            print(f"  [{i + 1}/{n}] {res[0]}: {len(res[1])} therapies, "
                  f"proj RMSE={res[7]:.1f} mg/dL ({time.time() - t0:.0f}s elapsed)",
                  flush=True)

    if workers <= 1 or n <= 1:
        for i, task in enumerate(tasks):
            results[i] = _build_one(task)
            _note(i, results[i])
    else:
        if verbose:
            print(f"  [parallel] {n} subjects across {workers} worker(s)", flush=True)
        ctx = mp.get_context("fork")          # Linux: children inherit imports + globals
        with ctx.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap(_build_one, tasks, chunksize=1)):
                results[i] = res
                _note(i, res)

    # Per-patient arrays (length N) and per-example arrays (length M = sum of T).
    names, Ib_l, theta_l, rmse_l = [], [], [], []
    ex_cgm, ex_ins, ex_cho, ex_pid, ex_treat = [], [], [], [], []
    for pi, r in enumerate(results):
        name, treatments, cgm_l, ins_l, cho_l, Ib_i, theta_i, rmse_i = r
        names.append(name); Ib_l.append(Ib_i); theta_l.append(theta_i); rmse_l.append(rmse_i)
        for tname, c, ins, cho in zip(treatments, cgm_l, ins_l, cho_l):
            ex_cgm.append(c); ex_ins.append(ins); ex_cho.append(cho)
            ex_pid.append(pi); ex_treat.append(tname)

    lengths = {len(c) for c in ex_cgm}
    L = min(lengths)
    if len(lengths) > 1 and verbose:
        print(f"  [warn] traces have unequal length {sorted(lengths)}; "
              f"truncating all channels to L={L}")

    def _stack(seq):
        return np.stack([a[:L] for a in seq], axis=0)

    return {
        # per-example (M, ...)
        "cgm": _stack(ex_cgm),
        "insulin": _stack(ex_ins),
        "cho": _stack(ex_cho),
        "patient_idx": np.asarray(ex_pid, dtype=np.int64),
        "treatment": np.array(ex_treat, dtype=object),
        # per-patient (N, ...)
        "theta": np.stack(theta_l, axis=0),
        "Ib": np.asarray(Ib_l, dtype=np.float64),
        "names": np.array(names, dtype=object),
        "proj_rmse": np.asarray(rmse_l, dtype=np.float64),
        # metadata
        "hours": float(hours),
        "sample_time": float(C.SAMPLE_TIME),
        "dt": float(dt),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="patients.csv to build from")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .npz path")
    ap.add_argument("--hours", type=float, default=C.WINDOW_HOURS,
                    help="baseline horizon (must match run_phase1 --hours)")
    ap.add_argument("--smoke", action="store_true",
                    help="use the smoke therapy grid (fewer treatments per patient)")
    ap.add_argument("--start", type=int, default=0, help="patient offset")
    ap.add_argument("--limit", type=int, default=None, help="cap patients (testing)")
    ap.add_argument("--max-nfev", type=int, default=80,
                    help="MAP-fit budget for the ReplayBG projection")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes for the per-subject build "
                         "(0 = auto: SLURM_CPUS_PER_TASK, else all cores)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if --out already exists")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"[phase1-data] {args.out} exists; pass --overwrite to rebuild")

    bolus_factors, basal_factors = C.factors_for(args.smoke)
    subs = PT.load_subjects_csv(args.patients)
    end = args.start + args.limit if args.limit else None
    subs = subs[args.start:end]
    T = len(bolus_factors) + len(basal_factors)
    print(f"[phase1-data] building from {len(subs)} patients x {T} therapies "
          f"(horizon {args.hours:.0f} h) across {n_workers(args.jobs)} worker(s) ...",
          flush=True)

    d = build_dataset(subs, hours=args.hours, max_nfev=args.max_nfev,
                      bolus_factors=bolus_factors, basal_factors=basal_factors,
                      jobs=args.jobs)
    save_dataset(d, args.out)
    print(f"[phase1-data] X shape {d['cgm'].shape} (M=N*T, L); "
          f"N={len(d['names'])} patients, T={T} therapies; "
          f"mean proj RMSE = {d['proj_rmse'].mean():.1f} mg/dL "
          f"(worst {d['proj_rmse'].max():.1f})")
    print(f"[phase1-data] wrote {args.out}")


if __name__ == "__main__":
    main()
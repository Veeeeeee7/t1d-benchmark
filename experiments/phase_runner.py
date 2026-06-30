"""Shared driver for the per-instance digital-twin phases (Phase 0 / Phase 2).

Runs each requested twinning method (MCMC / SBI, and any future per-instance
method) on an explicit list of patients, isolating every fit in its own
subprocess so one patient's failure can't abort the sweep (crash-tolerant
and resumable), then scores with a results module and
aggregates the per-patient comparison tables into a **mean +/- std** DT2 summary
per method.

    Phase 2 = this over all (simglucose) patients (the heavy, shardable sweep).
    Phase 0 = this over the self-consistent ReplayBG cohort (the matched-model
              baseline) -- same machinery, different plant/stages, wired in by
              ``run_phase0_twins`` via the ``stages`` / ``results_module`` /
              ``load_subjects`` parameters below.

Only ``mcmc`` and ``sbi`` are per-instance twins; the amortized baselines (TCN /
linear) are Phase 1 and are intentionally not run here.
"""
from __future__ import annotations

import os
import sys
import subprocess

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from experiments import exp_common as C
from experiments import patients as PT
from t1d_twin.evaluate import DECISION_COLS, FIDELITY_COLS
from experiments import output_paths as OP

# Phase 2 (simglucose plant) defaults. Phase 0 passes its own equivalents.
STAGES = {"mcmc": "experiments.run_mcmc_phase2",
          "sbi": "experiments.run_sbi_phase2"}
RESULTS_MODULE = "experiments.compute_results_phase2"



def _run(mod: str, patient: str, patients_csv: str, smoke: bool,
         population: str | None = None) -> int:
    """Launch one per-patient fit module (e.g. run_mcmc_phase2) as an isolated subprocess so a single patient failure cannot abort the sweep; returns its exit code."""
    cmd = [sys.executable, "-m", mod, "--patient", patient, "--patients", patients_csv]
    if smoke:
        cmd.append("--smoke")
    if population:
        cmd += ["--population", population]
    return subprocess.run(cmd, cwd=_root).returncode


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of every numeric metric per method, plus the patient count."""
    metric_cols = [c for c in df.columns if c not in ("patient", "method")]
    g = df.groupby("method")
    out = g[metric_cols].agg(["mean", "std"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out["n"] = g.size()
    return out.reset_index()


def _print_summary(summary: pd.DataFrame, label: str, n_patients: int) -> None:
    """Pretty-print the per-method mean +/- std of the decision and fidelity metrics for a phase."""
    print(f"\n=== {label} DT2 (mean +/- std across {n_patients} patients) ===")
    for _, r in summary.iterrows():
        parts = [f"n={int(r['n'])}"]
        for col in DECISION_COLS + FIDELITY_COLS:
            if f"{col}_mean" in summary.columns:
                parts.append(f"{col}={r[f'{col}_mean']:.3f}+/-{r[f'{col}_std']:.3f}")
        print(f"  {r['method']:>6}: " + "  ".join(parts))


def run_phase(names, methods, patients_csv, population=None, smoke=False,
              label="phase", out_summary=None, out_per_patient=None,
              skip_existing=True, aggregate_only=False,
              stages=None, results_module=RESULTS_MODULE, load_subjects=None,
              results_dir_fn=None):
    """Fit + score ``methods`` over ``names``; write/return per-patient + summary.

    Parameters
    ----------
    names : explicit list of patient Names to run (order preserved).
    methods : subset of ``stages`` (e.g. ``["mcmc", "sbi"]``).
    stages : ``method -> module`` map of per-patient fit subprocesses. Defaults
        to the Phase 2 (simglucose) stages; Phase 0 passes the ``*0`` variants.
    results_module : module that scores a patient's saved twins and writes its
        ``comparison_table.csv`` (Phase 2: ``compute_results_phase2``; Phase 0:
        ``compute_results_phase0``).
    load_subjects : callable(csv) -> list of subjects, each exposing ``.name``
        and ``.safe_name``. Defaults to ``patients.load_subjects_csv``; Phase 0
        passes ``replaybg_plant.load_phase0_csv``.
    results_dir_fn : callable(subject) -> per-subject results dir. Defaults to
        the Phase 2 layout (``output_paths.results_dir(PHASE2, name)``); Phase 0
        passes the PHASE0 equivalent so its (name-shared) matched cohort never
        clobbers Phase 2's tables.
    skip_existing : reuse a patient's ``comparison_table.csv`` if present
        (resumable, crash-tolerant). Set False to force re-fit.
    aggregate_only : never fit/score; only read existing comparison tables and
        aggregate. Use this for the final summary after a sharded cluster run.

    Returns ``(per_patient_df, summary_df)`` (or ``(None, None)`` if nothing
    aggregated).
    """
    if stages is None:
        stages = STAGES
    if load_subjects is None:
        load_subjects = PT.load_subjects_csv
    if results_dir_fn is None:
        results_dir_fn = lambda s: OP.results_dir(OP.PHASE2, s.safe_name)

    methods = [m for m in methods if m in stages]
    subs = {s.name: s for s in load_subjects(patients_csv)}

    rows = []
    for i, name in enumerate(names):
        s = subs.get(name)
        if s is None:
            print(f"[{label}] {name} not in {patients_csv}; skipping")
            continue
        tpath = os.path.join(results_dir_fn(s), "comparison_table.csv")

        run_block = (not aggregate_only) and not (skip_existing and os.path.exists(tpath))
        if run_block:
            print(f"\n##### [{label}] [{i + 1}/{len(names)}] {name} #####")
            for m in methods:
                if _run(stages[m], name, patients_csv, smoke, population) != 0:
                    print(f"[{label}] {name}/{m} FAILED — continuing")
            if _run(results_module, name, patients_csv, smoke) != 0:
                print(f"[{label}] {name}/results FAILED — skipping aggregation")
                continue

        if os.path.exists(tpath):
            t = pd.read_csv(tpath, index_col=0)
            for method, row in t.iterrows():
                if method in methods:                       # ignore stale rows (e.g. amortized baselines)
                    rec = {"patient": name, "method": method}
                    rec.update(row.to_dict())
                    rows.append(rec)
        elif aggregate_only:
            print(f"[{label}] no comparison_table for {name} (not yet scored)")

    if not rows:
        print(f"[{label}] no results aggregated.")
        return None, None

    df = pd.DataFrame(rows)
    if out_per_patient:
        os.makedirs(os.path.dirname(out_per_patient) or ".", exist_ok=True)
        df.to_csv(out_per_patient, index=False)
    summary = summarize(df)
    if out_summary:
        os.makedirs(os.path.dirname(out_summary) or ".", exist_ok=True)
        summary.to_csv(out_summary, index=False)
        print(f"\n[{label}] wrote {out_summary}"
              + (f" and {out_per_patient}" if out_per_patient else ""))
    _print_summary(summary, label, df["patient"].nunique())
    return df, summary
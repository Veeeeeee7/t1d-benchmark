"""Phase 0, final stage — score a subject's saved twins against the
self-consistent **ReplayBG** ground truth over the candidate grid, and write the
per-subject comparison table that ``run_phase0_twins`` aggregates.

Mirror of ``compute_results_phase2.py`` with one change: the ground-truth candidate
runs come from ``replaybg_plant.ground_truth`` (the matched-model plant) rather
than ``exp_common.subject_ground_truth`` (simglucose). The scoring object
(``evaluate_twin``), the metric columns, and the output format are identical, so
the aggregation in ``phase_runner`` is reused unchanged.

Run (from the repo root):
    python -m experiments.compute_results_phase0 --patients patients0.csv --patient rbg0001
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

from experiments import exp_common as C            # paths, factor grids, (de)serialization
from experiments import replaybg_plant as P        # the ReplayBG plant
from experiments import output_paths as OP        # standardized layout (phase-tagged)
from t1d_twin import value
from t1d_twin import plotting
from t1d_twin.evaluate import (
    evaluate_twin, _row_from_result, ranking, TABLE_COLUMNS,
    df_to_markdown, dt2_summary)

_LOADER_KEYS = [("mcmc", "mcmc", C.load_mcmc),
                ("sbi", "sbi", C.load_sbi)]


def _resolve(patient: str, patients_csv: str) -> P.Phase0Subject:
    """Look up one Phase-0 subject by name in the cohort CSV (KeyError if absent)."""
    subs = {s.name: s for s in P.load_phase0_cohort(patients_csv)}
    if patient not in subs:
        raise KeyError(f"patient {patient!r} not in {patients_csv}")
    return subs[patient]


def score_subject(subject, hours, bolus_factors, basal_factors, seed=P.SEED):
    """Evaluate whichever of the subject's twins exist against ReplayBG truth.

    Returns ``(table, details, true_rank, seen, unseen)`` or ``None`` if no twins
    are saved for this subject.
    """
    true_runs = P.ground_truth(subject, hours, bolus_factors, basal_factors, seed=seed)
    true_ig = {n: r.bg() for n, r in true_runs.items()}
    true_rewards = {n: value.reward(g) for n, g in true_ig.items()}
    true_rank = ranking(true_rewards)
    seen, unseen = C.seen_unseen(bolus_factors, basal_factors)

    paths = OP.twin_artifact_paths(OP.PHASE0, subject.safe_name)
    rows, details = {}, {}
    for name, key, loader in _LOADER_KEYS:
        if not os.path.exists(paths[key]):
            continue
        twin = loader(paths[key])
        res = evaluate_twin(twin, true_runs, true_rewards=true_rewards,
                            true_ig_by_policy=true_ig)
        rows[name] = _row_from_result(res)
        details[name] = res

    if not rows:
        return None
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(TABLE_COLUMNS))
    table.index.name = "method"
    table.attrs.update(details=details, true_rewards=true_rewards, true_ranking=true_rank)
    return table, details, true_rank, seen, unseen


def write_outputs(subject, hours, table, details, true_rank, seen, unseen):
    """Write the per-patient table, notes.md, and IG-overlay figures.

    Everything for one patient lands in ``results/phase0/<name>/`` (namespaced so
    the matched cohort never clobbers Phase 2's ``results/phase2/<name>/``):
      * ``comparison_table.csv`` -- headline metrics (also read by the aggregator);
      * ``notes.md`` -- human-readable summary of this patient's runs;
      * ``ig_overlay_<method>.png`` -- ReplayBG plant IG (solid) vs twin IG
        (dashed) for every therapy, one per twinning method.
    """
    rdir = OP.results_dir(OP.PHASE0, subject.safe_name)
    os.makedirs(rdir, exist_ok=True)
    table.to_csv(os.path.join(rdir, "comparison_table.csv"))

    # Per-therapy IG overlays. Phase 0's plant is the matched ReplayBG model, so
    # the solid series is labeled accordingly (not simglucose).
    true_ig = next(iter(details.values()))["true_ig"]
    pred_by_method = {m: details[m]["pred_ig"] for m in details}
    figs = plotting.write_therapy_overlays(
        rdir, true_ig, pred_by_method,
        sample_time=P.SAMPLE_TIME, plant_label="ReplayBG plant")

    with open(os.path.join(rdir, "notes.md"), "w") as fh:
        fh.write(f"# Phase 0 (matched-model) twinning — {subject.name}\n\n")
        fh.write(f"Horizon {hours:.0f} h @ {P.SAMPLE_TIME:.0f}-min CGM; "
                 f"|Pi|={len(true_rank)} ({len(seen)} seen / {len(unseen)} unseen); "
                 f"carb error m={subject.dose_mult}.\n\n")
        fh.write(f"Methods scored: {', '.join(table.index)}.\n\n")
        fh.write(f"True ranking (best->worst): {', '.join(true_rank)}\n\n")
        fh.write("## Comparison table\n\n")
        fh.write(df_to_markdown(table) + "\n\n")
        fh.write("## Decision vs fidelity (DT2)\n\n")
        fh.write("```\n" + dt2_summary(table) + "\n```\n\n")
        if figs:
            fh.write("## Figures\n\n")
            for p in figs:
                base = os.path.basename(p)
                fh.write(f"- `{base}` — ReplayBG plant IG (solid) vs twin IG "
                         f"(dashed), one color per therapy.\n")
    return rdir


def main() -> None:
    """CLI: score the saved Phase-0 twin(s) for one patient and write the comparison table, notes, and IG-overlay figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="smoke grid + horizon")
    ap.add_argument("--patients", required=True, help="phase0 cohort CSV (patients0.csv)")
    ap.add_argument("--patient", required=True, help="patient Name (a row in --patients)")
    args = ap.parse_args()

    subject = _resolve(args.patient, args.patients)
    hours = P.hours_for(args.smoke)
    bolus_factors, basal_factors = C.factors_for(args.smoke)

    print(f"[results0] subject={subject.name}; collecting ground truth on the ReplayBG plant ...")
    scored = score_subject(subject, hours, bolus_factors, basal_factors)
    if scored is None:
        print(f"[results0] no twin artifacts for {subject.name} in "
              f"{OP.twin_artifact_paths(OP.PHASE0, subject.safe_name)['dir']} — run run_mcmc_phase0 / run_sbi_phase0 first.")
        sys.exit(1)
    table, details, true_rank, seen, unseen = scored

    print(f"[results0] true best therapy = {true_rank[0]} (expected ~bolus_x{subject.dose_mult:.2f})")
    rdir = write_outputs(subject, hours, table, details, true_rank, seen, unseen)

    print("\n=== comparison table ===")
    print(table.to_string())
    print("\n=== decision-vs-fidelity (DT2) ===")
    print(dt2_summary(table))
    print(f"\n[results0] wrote {rdir}/ (comparison_table.csv, notes.md, ig_overlay_*.png)")


if __name__ == "__main__":
    main()
"""Final stage — reload a subject's saved twins, score them against the
simglucose ground truth over the candidate set Pi, and write the comparison
table.

Subject-aware: with ``--patients patients.csv --patient NAME`` it scores that
synthetic patient (ground truth via ExplicitBBController); with no flags it
scores the default adult#001. Outputs go to ``results/phase2/<patient>/`` so
multiple patients never clobber each other (and Phase 0's matched cohort, which
shares patient names, sits under ``results/phase0/<patient>/``).

The reusable :func:`score_subject` is also called by ``run_suite.py``.

Run (from the repo root):
    python -m experiments.compute_results --smoke
    python -m experiments.compute_results --patients patients.csv --patient synth0001_k2
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
from t1d_twin import value
from t1d_twin import plotting
from t1d_twin.evaluate import (
    evaluate_twin, _row_from_result, ranking, TABLE_COLUMNS)
from t1d_twin.run_all import _df_to_markdown

_LOADER_KEYS = [("mcmc", "mcmc", C.load_mcmc),
                ("sbi", "sbi", C.load_sbi)]


def dt2_summary(table: pd.DataFrame) -> str:
    """Flag any decision-vs-fidelity dissociation (the headline DT2 claim)."""
    methods = list(table.index)
    if len(methods) < 2:
        return "  (need >= 2 methods to assess decision-vs-fidelity dissociation)"
    by_fidelity = sorted(methods, key=lambda m: table.loc[m, "rmse"])
    by_decision = sorted(methods, key=lambda m: -table.loc[m, "spearman"])
    lines = [f"  best trajectory fidelity (RMSE): {by_fidelity[0]}",
             f"  best decision quality   (Spearman): {by_decision[0]}"]
    if by_fidelity[0] != by_decision[0]:
        lines.append("  >> DISSOCIATION: the most faithful twin is NOT the best "
                     "ranker — DT2 thesis in action.")
    else:
        lines.append("  no dissociation in this run (fidelity and ranking agree).")
    return "\n".join(lines)


def score_subject(subject, hours, bolus_factors, basal_factors, seed=C.SEED):
    """Evaluate whichever of the subject's twins exist. Returns
    ``(table, details, true_rank, seen, unseen)`` or ``None`` if no twins saved."""
    true_runs, _ = C.subject_ground_truth(subject, hours, bolus_factors,
                                          basal_factors, seed=seed)
    true_ig = {n: r.bg() for n, r in true_runs.items()}
    true_rewards = {n: value.reward(g) for n, g in true_ig.items()}
    true_rank = ranking(true_rewards)
    seen, unseen = C.seen_unseen(bolus_factors, basal_factors)

    paths = C.artifact_paths(subject)
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

    Everything for one patient lands in ``results/phase2/<name>/``:
      * ``comparison_table.csv`` -- the headline metrics table (machine-readable;
        also what the phase aggregator reads);
      * ``notes.md`` -- a human-readable summary of this patient's runs (ranking,
        per-method metrics, the decision-vs-fidelity read, and the figures);
      * ``ig_overlay_<method>.png`` -- plant IG (solid) vs twin IG (dashed) for
        every therapy, one per twinning method.
    """
    rdir = C.results_dir_for(subject)
    os.makedirs(rdir, exist_ok=True)
    table.to_csv(os.path.join(rdir, "comparison_table.csv"))

    # Per-therapy IG overlays (plant solid vs twin dashed), one per method.
    true_ig = next(iter(details.values()))["true_ig"]
    pred_by_method = {m: details[m]["pred_ig"] for m in details}
    figs = plotting.write_therapy_overlays(
        rdir, true_ig, pred_by_method,
        sample_time=C.SAMPLE_TIME, plant_label="simglucose")

    with open(os.path.join(rdir, "notes.md"), "w") as fh:
        fh.write(f"# Decision-targeted twinning — {subject.name}\n\n")
        fh.write(f"Horizon {hours:.0f} h @ {C.SAMPLE_TIME:.0f}-min CGM; "
                 f"|Pi|={len(true_rank)} ({len(seen)} seen / {len(unseen)} unseen).\n\n")
        fh.write(f"Methods scored: {', '.join(table.index)}.\n\n")
        fh.write(f"True ranking (best->worst): {', '.join(true_rank)}\n\n")
        fh.write("## Comparison table\n\n")
        fh.write(_df_to_markdown(table) + "\n\n")
        fh.write("## Decision vs fidelity (DT2)\n\n")
        fh.write("```\n" + dt2_summary(table) + "\n```\n\n")
        if figs:
            fh.write("## Figures\n\n")
            for p in figs:
                base = os.path.basename(p)
                fh.write(f"- `{base}` — plant IG (solid) vs twin IG (dashed), "
                         f"one color per therapy.\n")
    return rdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1-day horizon (match smoke fits)")
    C.add_subject_args(ap)
    args = ap.parse_args()

    subject = C.resolve_subject(args.patient, args.patients)
    hours = C.hours_for(args.smoke)
    bolus_factors, basal_factors = C.factors_for(args.smoke)

    print(f"[results] subject={subject.name}; collecting ground truth on simglucose ...")
    scored = score_subject(subject, hours, bolus_factors, basal_factors)
    if scored is None:
        print(f"[results] no twin artifacts for {subject.name} in "
              f"{C.artifact_paths(subject)['dir']} — run the run_* scripts first.")
        sys.exit(1)
    table, details, true_rank, seen, unseen = scored

    print(f"[results] true best therapy = {true_rank[0]}")
    rdir = write_outputs(subject, hours, table, details, true_rank, seen, unseen)

    print("\n=== comparison table ===")
    print(table.to_string())
    print("\n=== decision-vs-fidelity (DT2) ===")
    print(dt2_summary(table))
    print(f"\n[results] wrote {rdir}/ (comparison_table.csv, notes.md, ig_overlay_*.png)")


if __name__ == "__main__":
    main()
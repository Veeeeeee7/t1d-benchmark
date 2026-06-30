"""Cross-patient fairness helpers for the carb-counting-error cohort.

Once patients are deliberately mis-dosed, different patients span different
reward ranges (a heavily mis-dosed patient has larger glycemic swings, hence a
bigger best-minus-worst reward gap). Raw ``decision_regret`` is in absolute risk
units, so naively averaging it across patients lets the most mis-dosed patients
dominate the cohort mean -- an apples-to-oranges aggregation.

These helpers normalise each patient's regret to its own achievable reward span
before aggregation, so every patient contributes on a common [0, 1] scale.
Spearman and top-k are already scale-free and need no adjustment.

This module is additive: nothing in the existing pipeline imports it. Wire it in
where you aggregate per-patient tables (e.g. run_suite), or call
``normalized_regret`` alongside ``evaluate.decision_regret`` when scoring.
"""
from __future__ import annotations

import numpy as np


def reward_span(true_rewards: dict[str, float]) -> float:
    """Best-minus-worst true reward over the candidate set (>= 0)."""
    vals = np.asarray(list(true_rewards.values()), dtype=float)
    return float(vals.max() - vals.min())


def normalized_regret(true_rewards: dict[str, float],
                      pred_rewards: dict[str, float],
                      eps: float = 1e-9) -> float:
    """Decision regret as a fraction of the patient's reward span, in [0, 1].

    0  -> the twin's top pick is the true optimum (same as raw regret == 0).
    1  -> the twin's top pick is the worst therapy on the grid.

    Scale-free across patients, so it can be averaged over a cohort with mixed
    dose multipliers without the most mis-dosed patients dominating.
    """
    names = [n for n in true_rewards if n in pred_rewards]
    if len(names) < 2:
        raise ValueError("need >= 2 shared policies to compute regret")
    t = {n: true_rewards[n] for n in names}
    p = {n: pred_rewards[n] for n in names}
    best_by_pred = max(names, key=lambda n: p[n])
    true_best = max(t.values())
    raw = true_best - t[best_by_pred]
    span = reward_span(t)
    return float(raw / (span + eps))


def baseline_gap_regret(true_rewards: dict[str, float],
                        pred_rewards: dict[str, float],
                        baseline_name: str = "bolus_x1.00",
                        eps: float = 1e-9) -> float:
    """Regret as a fraction of the baseline-to-optimum gap (the headroom a
    correct decision could recover from the mis-set baseline).

    This answers "of the improvement available over leaving the patient on their
    mis-dosed baseline, how much did the twin's choice forfeit?" -- often the
    most interpretable cohort metric for the carb-error study. Values can exceed
    1 only if the twin picks a therapy worse than baseline.
    """
    names = [n for n in true_rewards if n in pred_rewards]
    if baseline_name not in true_rewards:
        raise KeyError(f"baseline {baseline_name!r} not in true_rewards")
    p = {n: pred_rewards[n] for n in names}
    best_by_pred = max(names, key=lambda n: p[n])
    true_best = max(true_rewards.values())
    gap = true_best - true_rewards[baseline_name]      # available improvement
    forfeited = true_best - true_rewards[best_by_pred]  # raw regret
    return float(forfeited / (gap + eps))


def cohort_summary(per_patient: dict[str, dict]) -> dict:
    """Aggregate per-patient L1 dicts into scale-fair cohort means.

    ``per_patient`` maps patient name -> a dict that must contain at least
    ``spearman`` and ``norm_regret`` (and optionally ``top_k``). Returns the
    mean of each over the cohort plus the patient count.
    """
    keys = ("spearman", "norm_regret", "top_k")
    out: dict[str, float] = {"n_patients": float(len(per_patient))}
    for k in keys:
        vals = [d[k] for d in per_patient.values() if k in d]
        if vals:
            out[f"mean_{k}"] = float(np.mean(vals))
    return out

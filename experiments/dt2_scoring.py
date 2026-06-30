"""Shared DT2 scoring helpers — score an *in-memory* twin against the simglucose
ground truth for a subject, without round-tripping the twin through disk.

``compute_results.score_subject`` loads saved twin artifacts and scores all of
them; Phase 1 instead holds a freshly predicted ``PointTwin`` in memory and wants
to score it (and reuse the ground truth across the baselines), so these two
thin functions factor out exactly that: collect the ground truth once, then
score any number of twins against it.
"""
from __future__ import annotations

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import exp_common as C
from t1d_twin import value
from t1d_twin.evaluate import evaluate_twin, _row_from_result


def collect_truth(subject, hours, bolus_factors, basal_factors, seed=C.SEED):
    """Run the candidate therapies on the subject (simglucose) once.

    Returns ``(true_runs, true_ig, true_rewards)`` — reusable across any number
    of twins for this subject so the expensive ground-truth grid is computed once.
    """
    true_runs, _ = C.subject_ground_truth(subject, hours, bolus_factors,
                                           basal_factors, seed=seed)
    true_ig = {n: r.bg() for n, r in true_runs.items()}
    true_rewards = {n: value.reward(g) for n, g in true_ig.items()}
    return true_runs, true_ig, true_rewards


def score_twin(twin, truth) -> dict:
    """Score one twin against a precomputed ``truth`` bundle -> a metrics row
    (the same columns as ``evaluate.TABLE_COLUMNS``)."""
    true_runs, true_ig, true_rewards = truth
    res = evaluate_twin(twin, true_runs, true_rewards=true_rewards,
                        true_ig_by_policy=true_ig)
    return _row_from_result(res)
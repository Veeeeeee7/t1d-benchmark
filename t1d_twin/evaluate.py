"""
Evaluation metrics for the IG-vs-IG digital-twin benchmark.

Four metrics, all computed on **noise-free interstitial glucose (IG)**:

* ``spearman`` -- Spearman rank correlation between the twin's IG-reward
  ordering of the policy set Pi and the ground-truth ordering. +1 means the
  twin ranks the therapies exactly as the real system does.
* ``regret``   -- true IG-reward lost by following the twin's top-ranked
  policy (``true_best - true_reward(argmax_pred)``); >= 0, and 0 iff the twin's
  best pick is genuinely the true optimum.
* ``rmse``     -- mean over Pi of the per-policy RMSE [mg/dL] between the true
  and twin IG series.
* ``mard``     -- mean over Pi of the per-policy MARD [%] between the true and
  twin IG series.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.stats import spearmanr

from . import value
from .simglucose_adapter import RunResult

TABLE_COLUMNS = ("spearman", "regret", "rmse", "mard")


# ===========================================================================
# Helpers
# ===========================================================================

def _align(true: dict[str, float], pred: dict[str, float]):
    """
    Common keys (in ``true``'s order) and aligned value arrays.
    """
    names = [n for n in true if n in pred]
    if len(names) < 2:
        raise ValueError("need >= 2 shared policies to compare rankings")
    t = np.array([true[n] for n in names], dtype=float)
    p = np.array([pred[n] for n in names], dtype=float)
    return names, t, p


def ranking(rewards: dict[str, float]) -> list[str]:
    """
    Policy names sorted best -> worst by reward (descending).
    """
    return sorted(rewards, key=lambda n: rewards[n], reverse=True)


def ig_rewards(ig_by_policy: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Convert ``name -> IG series`` to ``name -> reward`` via ``value.reward``.
    """
    return {n: value.reward(g) for n, g in ig_by_policy.items()}


# ===========================================================================
# Decision-quality metrics (on two IG-reward dicts)
# ===========================================================================

def spearman_rank(true_rewards: dict[str, float],
                  pred_rewards: dict[str, float]) -> float:
    """
    Spearman rank correlation between true and predicted IG-reward orderings.
    """
    _, t, p = _align(true_rewards, pred_rewards)
    return float(spearmanr(t, p).correlation)


def decision_regret(true_rewards: dict[str, float],
                    pred_rewards: dict[str, float]) -> float:
    """
    True IG-reward lost by following the twin's top pick (>= 0; 0 is optimal).
    """
    names, t, p = _align(true_rewards, pred_rewards)
    best_by_pred = names[int(np.argmax(p))]
    true_best = float(np.max(t))
    return float(true_best - true_rewards[best_by_pred])


# ===========================================================================
# Trajectory-fidelity metrics (on two IG series)
# ===========================================================================

def trajectory_rmse(true_ig: np.ndarray, pred_ig: np.ndarray) -> float:
    """
    RMSE [mg/dL] between aligned IG series (assumed common length).
    """
    t = np.asarray(true_ig, float).ravel()
    p = np.asarray(pred_ig, float).ravel()
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mard(true_ig: np.ndarray, pred_ig: np.ndarray) -> float:
    """
    Mean Absolute Relative Difference [%], using true IG as reference (assumed common length).
    """
    t = np.asarray(true_ig, float).ravel()
    p = np.asarray(pred_ig, float).ravel()
    ref = np.clip(np.abs(t), value._G_FLOOR, None)
    return float(100.0 * np.mean(np.abs(p - t) / ref))


def trajectory_fidelity(true_ig_by_policy: dict[str, np.ndarray],
                        pred_ig_by_policy: dict[str, np.ndarray]) -> dict:
    """
    Mean RMSE / MARD across the shared policies, plus a per-policy breakdown.
    """
    names = [n for n in true_ig_by_policy if n in pred_ig_by_policy]
    if not names:
        raise ValueError("no shared policies for trajectory fidelity")
    per: dict[str, dict] = {}
    rmses, mards = [], []
    for n in names:
        r = trajectory_rmse(true_ig_by_policy[n], pred_ig_by_policy[n])
        m = mard(true_ig_by_policy[n], pred_ig_by_policy[n])
        per[n] = {"rmse": r, "mard": m}
        rmses.append(r)
        mards.append(m)
    return {"rmse": float(np.mean(rmses)), "mard": float(np.mean(mards)),
            "per_policy": per}


# ===========================================================================
# Ground truth + twin prediction (both noise-free IG)
# ===========================================================================

def collect_ground_truth(policies, scenario, hours: float, **run_kwargs
                         ) -> dict[str, RunResult]:
    """
    Run every policy on simglucose once; return ``name -> RunResult``.

    The RunResults are the single source of truth for both the ground-truth IG
    (``RunResult.bg()``) and the recorded inputs replayed on each twin.
    """
    from .simglucose_adapter import run_policy
    return {name: run_policy(ctrl, scenario, hours=hours, **run_kwargs)
            for name, ctrl in policies.items()}


def true_ig_from_runs(runs_by_policy: dict[str, RunResult]) -> dict[str, np.ndarray]:
    """
    Plant noise-free IG per policy (== the CGM signal before sensor noise).
    """
    return {n: r.bg() for n, r in runs_by_policy.items()}


def predict_ig_on_twin(twin, runs_by_policy: dict[str, RunResult]
                       ) -> dict[str, np.ndarray]:
    """
    Replay each policy's recorded inputs on the twin; return noise-free IG.
    """
    return {n: twin.replay_run(run, add_noise=False)
            for n, run in runs_by_policy.items()}


# ===========================================================================
# Twin evaluation + head-to-head comparison
# ===========================================================================

def evaluate_twin(
    twin,
    true_runs_by_policy: dict[str, RunResult],
    *,
    true_ig_by_policy: Optional[dict[str, np.ndarray]] = None,
    true_rewards: Optional[dict[str, float]] = None,
) -> dict:
    """
    Score one twin against ground truth on the four IG-vs-IG metrics.

    Parameters
    ----------
    twin : an identified set of parameters.
    true_runs_by_policy : ground-truth ``name -> RunResult`` (carries both the
        plant IG via ``bg()`` and the recorded inputs to replay on the twin).
    true_ig_by_policy, true_rewards : optional precomputed ground truth (to
        avoid recomputation when comparing several twins); derived from the runs
        if omitted.

    Returns
    -------
    dict with ``spearman``, ``regret``, ``rmse``, ``mard``, plus ``pred_rewards``,
    ``true_rewards``, ``pred_ranking``, ``true_ranking``, ``per_policy``, and
    ``n_posterior``.
    """
    if true_ig_by_policy is None:
        true_ig_by_policy = true_ig_from_runs(true_runs_by_policy)
    if true_rewards is None:
        true_rewards = ig_rewards(true_ig_by_policy)

    pred_ig_by_policy = predict_ig_on_twin(twin, true_runs_by_policy)
    pred_rewards = ig_rewards(pred_ig_by_policy)

    fid = trajectory_fidelity(true_ig_by_policy, pred_ig_by_policy)

    n_post = twin.n_posterior() if hasattr(twin, "n_posterior") else None
    return {
        "spearman": spearman_rank(true_rewards, pred_rewards),
        "regret": decision_regret(true_rewards, pred_rewards),
        "rmse": fid["rmse"],
        "mard": fid["mard"],
        "pred_rewards": pred_rewards,
        "true_rewards": true_rewards,
        "pred_ranking": ranking(pred_rewards),
        "true_ranking": ranking(true_rewards),
        "per_policy": fid["per_policy"],
        "n_posterior": n_post,
        # Full noise-free IG series per policy, kept so downstream code can draw
        # the per-patient IG overlay (plant solid vs twin dashed) without
        # re-running the replay. ``true_ig`` is shared across methods.
        "true_ig": true_ig_by_policy,
        "pred_ig": pred_ig_by_policy,
    }


def _row_from_result(res: dict) -> dict:
    """
    Flatten an ``evaluate_twin`` result into the headline table row.
    """
    return {c: res[c] for c in TABLE_COLUMNS}


def run_experiment(
    twin_methods: dict[str, Callable[[RunResult], object]],
    identification_run: RunResult,
    policies,
    scenario=None,
    hours: Optional[float] = None,
    *,
    true_runs_by_policy: Optional[dict[str, RunResult]] = None,
    verbose: bool = True,
    **run_kwargs,
):
    """
    Identify each twin from one run, evaluate it, return a comparison table.

    Returns a ``pandas.DataFrame`` indexed by method with columns
    :data:`TABLE_COLUMNS`. Per-twin ``evaluate_twin`` results, the true rewards,
    and the true ranking are attached on ``table.attrs`` (``details``,
    ``true_rewards``, ``true_ranking``) for downstream analysis.
    """
    import pandas as pd

    if true_runs_by_policy is None:
        if scenario is None or hours is None:
            raise ValueError(
                "provide true_runs_by_policy, or scenario and hours to collect it")
        if verbose:
            print(f"[run_experiment] collecting ground truth for "
                  f"{len(policies)} policies ...")
        true_runs_by_policy = collect_ground_truth(
            policies, scenario, hours, **run_kwargs)

    true_ig = true_ig_from_runs(true_runs_by_policy)
    true_rewards = ig_rewards(true_ig)

    rows: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for method, identify_fn in twin_methods.items():
        if verbose:
            print(f"[run_experiment] identifying + evaluating twin '{method}' ...")
        twin = identify_fn(identification_run)
        res = evaluate_twin(
            twin, true_runs_by_policy,
            true_ig_by_policy=true_ig, true_rewards=true_rewards)
        rows[method] = _row_from_result(res)
        details[method] = res

    table = pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=list(TABLE_COLUMNS))
    table.index.name = "method"
    table.attrs["details"] = details
    table.attrs["true_rewards"] = true_rewards
    table.attrs["true_ranking"] = ranking(true_rewards)
    return table
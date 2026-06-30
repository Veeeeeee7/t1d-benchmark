"""
Phase 1 — model-agnostic plumbing shared by every amortized baseline.

The TCN baseline (:mod:`experiments.tcn_cv`) and the linear baseline
(:mod:`experiments.linear_cv`) differ only in *which regressor* they fit; the
data handling around them is identical and lives here so the two baselines share
one methodology by construction:

* channel stacking into a ``(M, C, L)`` array,
* per-channel input standardization (fit on the train fold only),
* the leakage-free, **patient-grouped** K-fold splitter.

Keeping these here means a change to the CV protocol (e.g. the fold count or the
normalization) applies to all baselines at once.
"""
from __future__ import annotations

import numpy as np
import torch

# Default channel stacks (shared by all baselines).
CGM_ONLY = ("cgm",)
CGM_INS_CHO = ("cgm", "insulin", "cho")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device(device=None) -> torch.device:
    """
    Resolve the compute device: explicit value, else CUDA if available, else CPU.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Inputs / normalization
# ---------------------------------------------------------------------------
def stack_channels(dataset: dict, channels) -> np.ndarray:
    """
    Stack the requested channels into a ``(M, C, L)`` float32 array.
    """
    arrs = [np.asarray(dataset[c], dtype=np.float32) for c in channels]
    return np.stack(arrs, axis=1)


def norm_fit(X: np.ndarray):
    """
    Per-channel scalar mean/std over (M, L), shape ``(1, C, 1)``.
    """
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def norm_apply(X: np.ndarray, mean, std) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Grouped K-fold (split by patient; examples follow their patient)
# ---------------------------------------------------------------------------
def group_kfold_indices(groups: np.ndarray, k: int, seed: int):
    """
    Yield ``(train_idx, test_idx)`` example indices for a patient-grouped split.

    The *unique groups* (patients) are shuffled and partitioned into ``k`` folds;
    every example inherits its group's fold, so a patient's whole set of therapy
    traces is held out together.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    folds = np.array_split(uniq[rng.permutation(len(uniq))], k)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    for i in range(k):
        test_groups = set(folds[i].tolist())
        test = (np.concatenate([idx_by[g] for g in folds[i]])
                if len(folds[i]) else np.array([], dtype=int))
        train = np.concatenate([idx_by[g] for g in uniq if g not in test_groups])
        yield np.sort(train), np.sort(test)

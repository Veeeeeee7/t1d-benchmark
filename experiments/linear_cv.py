"""Phase 1 — pooled, cross-patient trace->theta *linear* regressor with grouped K-fold CV.

The simple-model counterpart of :mod:`experiments.tcn_cv`: the TCN is swapped for
a **ridge (L2-regularized linear) regressor**, everything else held fixed. It is
the deliberately-weak baseline that quantifies how much the TCN's nonlinearity and
temporal inductive bias actually buy on this task.

Identical methodology to the TCN baseline (shared via :mod:`experiments.cv_common`):

* same treatment-augmented dataset (one example per (patient, therapy) trace),
* same **patient-grouped** K-fold so train/test never share a patient,
* same per-channel input standardization (train-fold statistics only),
* same phi-space target (log-rates + linear Gb), standardized for the fit,
* same per-patient phi-space averaging of the held-out predictions.

The only differences are intrinsic to a linear model:

* the ``(C, L)`` trace is **flattened to a ``C*L`` feature vector** (a linear map
  has no global-pooling equivalent; the dataset truncates all traces to a common
  ``L``, so the flat dimension is fixed);
* the fit is the **closed-form ridge solution** (normal equations with an
  unpenalized intercept), so there is no SGD / early-stopping / validation split
  — the regularizer ``alpha`` plays the role the TCN's capacity controls play.

Implemented in NumPy (no scikit-learn dependency added).
"""
from __future__ import annotations

import numpy as np

from t1d_twin.identify_sbi import _theta_to_phi, _phi_to_theta
from experiments.cv_common import (
    CGM_ONLY, CGM_INS_CHO, stack_channels, norm_fit, norm_apply,
    group_kfold_indices,
)

__all__ = ["CGM_ONLY", "CGM_INS_CHO", "cross_val_predict", "fit_ridge"]

DEFAULT_ALPHA = 1.0


# ---------------------------------------------------------------------------
# Fit / predict one fold (closed-form ridge in standardized phi-space)
# ---------------------------------------------------------------------------
def fit_ridge(X: np.ndarray, Yphi: np.ndarray, alpha: float = DEFAULT_ALPHA):
    """Closed-form multi-output ridge of standardized phi on flattened inputs.

    ``X`` is ``(M, C, L)`` (already per-channel standardized); it is flattened to
    ``(M, C*L)`` and a bias column is appended. The targets are standardized to
    zero mean / unit std per phi-dimension before the fit. The intercept column is
    left unpenalized.

    Returns ``(W, ynorm)`` where ``W`` is ``(C*L + 1, 8)`` and ``ynorm`` holds the
    phi standardization for de-normalizing predictions.
    """
    M = X.shape[0]
    Xf = X.reshape(M, -1).astype(np.float64)                  # (M, F)
    Xb = np.concatenate([Xf, np.ones((M, 1))], axis=1)        # (M, F+1) bias col

    ymean = Yphi.mean(axis=0)
    ystd = Yphi.std(axis=0) + 1e-8
    Yn = (Yphi - ymean) / ystd                                # (M, 8)

    F1 = Xb.shape[1]
    reg = alpha * np.eye(F1)
    reg[-1, -1] = 0.0                                         # don't penalize the intercept
    A = Xb.T @ Xb + reg                                       # (F+1, F+1)
    W = np.linalg.solve(A, Xb.T @ Yn)                         # (F+1, 8)
    return W, {"ymean": ymean, "ystd": ystd}


def _predict_phi(W: np.ndarray, X: np.ndarray, ynorm: dict) -> np.ndarray:
    """Predict de-normalized phi ``(M, 8)`` for inputs ``(M, C, L)``."""
    M = X.shape[0]
    Xb = np.concatenate([X.reshape(M, -1).astype(np.float64),
                         np.ones((M, 1))], axis=1)
    phi_n = Xb @ W
    return phi_n * ynorm["ystd"] + ynorm["ymean"]


# ---------------------------------------------------------------------------
# Cross-validated prediction over all patients (grouped, treatment-augmented)
# ---------------------------------------------------------------------------
def cross_val_predict(dataset: dict, channels, k: int = 5, seed: int = 0,
                      cfg: dict | None = None, verbose: bool = True,
                      device=None) -> np.ndarray:
    """Patient-grouped K-fold CV with a ridge regressor. Returns held-out theta
    ``(N_patients, 8)`` in natural space, one row per patient (each predicted by
    the fold that excluded that patient, averaged in phi-space over the patient's
    held-out therapy traces).

    ``cfg`` may carry ``alpha`` (ridge strength); ``device`` is accepted and
    ignored so the signature is a drop-in match for ``tcn_cv.cross_val_predict``.

    Back-compatible: a dataset without ``patient_idx`` (one trace per patient) is
    treated as one group per row, recovering the original per-patient CV.
    """
    cfg = dict(cfg or {})
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    if verbose:
        print(f"  [cv] linear ridge (alpha={alpha:g})", flush=True)

    X = stack_channels(dataset, channels)                       # (M, C, L)
    theta_pp = np.asarray(dataset["theta"], dtype=np.float64)   # (N, 8) per patient
    M = X.shape[0]
    groups = np.asarray(dataset.get("patient_idx",
                                    np.arange(M)), dtype=np.int64)  # (M,)
    N = theta_pp.shape[0]
    Yphi_ex = _theta_to_phi(theta_pp[groups]).astype(np.float64)   # (M, 8) per example

    phi_sum = np.zeros((N, theta_pp.shape[1]), dtype=np.float64)
    counts = np.zeros(N, dtype=np.int64)
    for fi, (tr, te) in enumerate(group_kfold_indices(groups, k, seed)):
        if verbose:
            n_tr_p = len(np.unique(groups[tr])); n_te_p = len(np.unique(groups[te]))
            print(f"  [cv] fold {fi + 1}/{k}: train={len(tr)} ex / {n_tr_p} pts, "
                  f"test={len(te)} ex / {n_te_p} pts (channels={list(channels)})",
                  flush=True)
        mean, std = norm_fit(X[tr])
        W, ynorm = fit_ridge(norm_apply(X[tr], mean, std), Yphi_ex[tr], alpha=alpha)
        phi_te = _predict_phi(W, norm_apply(X[te], mean, std), ynorm)
        np.add.at(phi_sum, groups[te], phi_te)                  # accumulate per patient
        np.add.at(counts, groups[te], 1)

    counts = np.maximum(counts, 1)
    phi_hat = phi_sum / counts[:, None]                          # mean phi per patient
    return _phi_to_theta(phi_hat)                                # (N, 8) natural

"""Phase 1 — pooled, cross-patient trace->theta TCN regressor with grouped K-fold CV.

This is the *amortized TCN baseline*: one residual dilated 1-D CNN (TCN) trained
on many patients' ``(channels -> projected-theta)`` pairs, evaluated by K-fold CV
so every patient is predicted by a model that never saw that patient.

Treatment-augmented dataset
---------------------------
The dataset has **one input example per (patient, therapy)** trace, so the
training set is ``N_patients * T_therapies`` examples. ``theta`` is a *per-patient*
physiological target (the therapies are forcing inputs, not parameters), so every
one of a patient's traces shares that patient's single theta. Per-example arrays
(``cgm``/``insulin``/``cho``) carry a ``patient_idx`` group label; per-patient
arrays (``theta``/``Ib``/``names``) are length ``N``.

Leakage-free splitting: the K-fold is **grouped by patient** — all of a patient's
therapy traces fall in the same fold — so train and test never share a patient.
Held-out per-example predictions are averaged (in phi-space) back to one theta per
patient, which is what the downstream twin scoring consumes.

Input channels are selected via ``channels`` (e.g. ``("cgm","insulin","cho")``);
targets are regressed in phi-space (log-rates + linear Gb) and standardized.
Per-channel input standardization uses *train-fold* statistics only. The data
handling (stacking, normalization, the grouped splitter) is shared with the
linear baseline via :mod:`experiments.cv_common`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from t1d_twin.identify_tcn import TCNRegressor
from t1d_twin.identify_sbi import _theta_to_phi, _phi_to_theta
from experiments.cv_common import (
    CGM_ONLY, CGM_INS_CHO, pick_device, stack_channels,
    norm_fit, norm_apply, group_kfold_indices,
)

__all__ = ["CGM_ONLY", "CGM_INS_CHO", "cross_val_predict", "train_regressor"]


# ---------------------------------------------------------------------------
# Train / predict one fold
# ---------------------------------------------------------------------------
def train_regressor(X: np.ndarray, Yphi: np.ndarray, in_channels: int,
                    hidden: int = 128, n_epochs: int = 300, lr: float = 1e-3,
                    batch_size: int = 256, val_frac: float = 0.1,
                    seed: int = 0, device=None, verbose: bool = True):
    """Fit ``TCNRegressor`` (MSE in standardized phi-space) on ``device``.

    Returns ``(model, ynorm)`` with the phi standardization for de-normalizing
    predictions. The model is left on ``device`` (move with ``.cpu()`` if needed).
    """
    device = pick_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xt = torch.tensor(X, dtype=torch.float32)
    ymean = Yphi.mean(axis=0)
    ystd = Yphi.std(axis=0) + 1e-8
    Yn = torch.tensor((Yphi - ymean) / ystd, dtype=torch.float32)

    n = Xt.shape[0]
    n_val = max(1, int(val_frac * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = TCNRegressor(out_dim=Yphi.shape[1], hidden=hidden,
                         in_channels=in_channels, in_len=X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()

    # Keep a CPU copy of the validation tensors; move to device per evaluation.
    Xval, Yval = Xt[val_idx], Yn[val_idx]

    best_val, best_state = float("inf"), None
    for epoch in range(n_epochs):
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx))]
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            if idx.numel() < 2:        # BatchNorm needs >1 sample; skip a singleton tail
                continue
            opt.zero_grad()
            loss = lossfn(model(Xt[idx].to(device)), Yn[idx].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossfn(model(Xval.to(device)), Yval.to(device)).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"    [cv] epoch {epoch:>3}: val MSE={vloss:.4f} (best {best_val:.4f})",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"ymean": ymean, "ystd": ystd}


def _predict_phi(model, X: np.ndarray, ynorm: dict, device=None,
                 batch_size: int = 1024) -> np.ndarray:
    """Predict de-normalized phi ``(M, 8)`` for inputs ``(M, C, L)`` (batched)."""
    device = pick_device(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), batch_size):
            xb = torch.tensor(X[b:b + batch_size], dtype=torch.float32).to(device)
            outs.append(model(xb).cpu().numpy())
    phi_n = np.concatenate(outs, axis=0) if outs else np.zeros((0, len(ynorm["ymean"])))
    return phi_n * ynorm["ystd"] + ynorm["ymean"]


# ---------------------------------------------------------------------------
# Cross-validated prediction over all patients (grouped, treatment-augmented)
# ---------------------------------------------------------------------------
def cross_val_predict(dataset: dict, channels, k: int = 5, seed: int = 0,
                      cfg: dict | None = None, verbose: bool = True,
                      device=None) -> np.ndarray:
    """Patient-grouped K-fold CV. Returns held-out theta ``(N_patients, 8)`` in
    natural space, one row per patient (each predicted by the fold that excluded
    that patient, averaged in phi-space over the patient's held-out therapy traces).

    Back-compatible: a dataset without ``patient_idx`` (one trace per patient) is
    treated as one group per row, recovering the original per-patient CV.
    """
    cfg = dict(cfg or {})
    device = pick_device(device)
    if verbose:
        print(f"  [cv] device={device.type}", flush=True)

    X = stack_channels(dataset, channels)                       # (M, C, L)
    theta_pp = np.asarray(dataset["theta"], dtype=np.float64)   # (N, 8) per patient
    M, Cc = X.shape[0], X.shape[1]
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
        model, ynorm = train_regressor(
            norm_apply(X[tr], mean, std), Yphi_ex[tr], in_channels=Cc,
            seed=seed + fi, device=device, verbose=verbose, **cfg)
        phi_te = _predict_phi(model, norm_apply(X[te], mean, std), ynorm, device=device)
        np.add.at(phi_sum, groups[te], phi_te)                  # accumulate per patient
        np.add.at(counts, groups[te], 1)

    counts = np.maximum(counts, 1)
    phi_hat = phi_sum / counts[:, None]                          # mean phi per patient
    return _phi_to_theta(phi_hat)                                # (N, 8) natural
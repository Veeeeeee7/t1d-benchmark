"""TCN twin building blocks — the CGM->theta regressor and the point-estimate twin.

The benchmark's TCN baseline is the *cross-patient amortized* regressor trained
with K-fold CV in :mod:`experiments.tcn_cv` (Phase 1). This module provides the
two pieces that baseline (and any other point-estimate method) needs:

* :class:`TCNRegressor` -- a 1-D CNN (channel-mixing conv front-end + residual
  dilated blocks + global pooling + dense head) that maps an input trace to
  ReplayBG phi = [log-rates, Gb]; ``in_channels`` is the number of stacked input
  signals (CGM+insulin+CHO).
* :class:`PointTwin` -- a point-estimate twin: ReplayBG run at a single
  theta-hat. It subclasses :class:`SBITwin` with a one-row ``theta_post`` so
  replay / ranking / metrics are identical to the posterior twins; its prediction
  band is degenerate by design (a point estimate carries no parameter
  uncertainty). It is deliberately method-neutral: any regression baseline that
  emits a point estimate (the TCN, the linear baseline in
  :mod:`experiments.linear_cv`, ...) reuses it unchanged.

Whatever the regressor reads, it emits an 8-vector ReplayBG theta, so all input
information is bottlenecked through the ReplayBG family before it can reach any
downstream prediction -- Phi !in F is preserved by construction.

(The former *per-instance* regressor, which trained a fresh model on one
patient's ReplayBG sims, has been retired in favour of the amortized Phase 1
baseline.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .identify_sbi import SBITwin, DEFAULT_SIGMA
from .replaybg_model import THETA_NAMES


# ---------------------------------------------------------------------------
# Architecture: residual dilated 1-D CNN (TCN) + global pool + dense head
# ---------------------------------------------------------------------------
class _TCNBlock(nn.Module):
    """Residual dilated 1-D conv block: two same-width dilated convs with
    BatchNorm/ReLU/Dropout and a skip connection. ``padding`` keeps length, so
    the block is length-preserving and stackable."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, p_drop: float = 0.1) -> None:
        """Build the two dilated convs plus BatchNorm/Dropout/ReLU; padding is set so the block preserves sequence length and stacks."""
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(p_drop)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to (B, C, L) and return the same-shape activated sum, ReLU(x + f(x))."""
        y = self.drop(self.act(self.bn1(self.conv1(x))))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class TCNRegressor(nn.Module):
    """Residual dilated 1-D CNN (TCN) mapping ``(B, C, L)`` -> phi = [log-rates, Gb] ``(B, 8)``.

    A channel-mixing stem fuses the CGM / insulin / CHO signals (each Conv1d
    filter spans all ``C`` input channels = early fusion), then a stack of
    residual blocks with **exponentially increasing dilation** grows the
    receptive field to cover the whole multi-hour window with few parameters,
    global average pooling collapses time, and a small dense (MLP) head
    regresses to the 8-vector ReplayBG phi.

    Design choices for the treatment-augmented dataset (N patients x T therapy
    traces, C=3, L ~ 480):

    * **Residual + dilated convs** capture the long meal/insulin response over
      hours far more parameter-efficiently than stacking plain convs or
      flattening ``C*L``; residual connections keep the deeper stack trainable.
    * **Global average pooling** makes the head independent of ``L`` and
      regularizes; **BatchNorm + dropout** train stably on the larger augmented
      set.

    Inputs may arrive as ``(B, C, L)`` (normal), ``(B, L)`` or ``(L,)`` (treated
    as single-channel). ``in_len`` is accepted for API compatibility but unused
    (global pooling removes any dependence on ``L``).
    """

    def __init__(self, out_dim: int = 8, hidden: int = 128,
                 in_channels: int = 3, in_len: int | None = None,
                 channels: int = 64, n_blocks: int = 4, kernel_size: int = 3,
                 p_drop: float = 0.1) -> None:
        """Build the channel-mixing stem, the dilated residual stack (dilations 1,2,4,...), the global-average pool, and the dense head that regresses to 8-dim phi."""
        super().__init__()
        self.in_channels = int(in_channels)
        self.in_len = in_len  # unused (kept for caller compatibility)
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            _TCNBlock(channels, kernel_size=kernel_size,
                      dilation=2 ** i, p_drop=p_drop)
            for i in range(n_blocks)                       # dilations 1,2,4,8,...
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)                # (B, channels, 1)
        self.head = nn.Sequential(
            nn.Flatten(),                                  # (B, channels)
            nn.Linear(channels, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the TCN on x (accepts (L,), (B, L) or (B, C, L)) and return the regressed phi = [log-rates, Gb], shape (B, out_dim)."""
        # Normalize the input to (B, C, L) regardless of how it arrives.
        if x.dim() == 1:            # (L,)     single, single-channel
            x = x.view(1, 1, -1)
        elif x.dim() == 2:          # (B, L)   batch, single-channel
            x = x.unsqueeze(1)      #       -> (B, 1, L)
        # x.dim() == 3              # (B, C, L) already channel-stacked
        x = self.blocks(self.stem(x))
        return self.head(self.pool(x))                     # -> (B, out_dim)


# ---------------------------------------------------------------------------
# Twin: ReplayBG run at the point estimate theta-hat
# ---------------------------------------------------------------------------
class PointTwin(SBITwin):
    """A point-estimate twin: ReplayBG at the single regressed theta-hat.

    Subclasses SBITwin with a one-row ``theta_post`` so replay/predict/metrics
    are shared with the other twins; the prediction band is degenerate by design
    (a point estimate carries no posterior uncertainty). Method-neutral: used by
    the TCN baseline and the linear baseline alike.
    """

    def __init__(self, theta_point, Ib, sample_time, dt=1.0,
                 sensor_name="Dexcom", sigma=DEFAULT_SIGMA):
        """Wrap a single regressed theta-hat as a one-row posterior so replay/predict/metrics are shared with SBITwin; the prediction band is degenerate by design."""
        theta_point = np.asarray(theta_point, dtype=float).reshape(1, -1)
        super().__init__(posterior=None, theta_post=theta_point, Ib=Ib,
                         sample_time=sample_time, dt=dt, log_space=False,
                         sensor_name=sensor_name, sigma=sigma)
        self.theta_point = theta_point.ravel()

    def summary(self) -> dict:
        """Per-parameter point estimate (no posterior spread), keyed by THETA_NAMES."""
        return {name: {"point": float(self.theta_point[i])}
                for i, name in enumerate(THETA_NAMES)}
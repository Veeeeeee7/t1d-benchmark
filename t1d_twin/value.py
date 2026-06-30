"""
Reward and clinical glucose metrics

Pure functions on glucose series (mg/dL).

Risk kernel
-----------
All risk-based quantities derive from the Magni et al. (2007) blood-glucose
risk function [Magni et al., J Diabetes Sci Technol 1(6):804-812, 2007],
defined as

    risk(IG) = 10 * (g * (ln(IG)**a - b))**2

where the constants are derived from the paper's symmetry constraints (p.807):

    rl(70)  = rl(280) = 25   (equal hypo/hyper penalty at the 70/280 pair, **log symmetry**)
    rl(50)  = rl(400)        (second symmetry pair)

Solving these constraints numerically gives:

    a       = 0.835334
    b       = 3.793357
    g       = 3.550313

The transform f(IG) = g * (ln(IG)**a - b) is negative in hypoglycemia
and positive in hyperglycemia, zeroing at the euglycemic centre g* ~= 138.9
mg/dL. This gives the paper's calibration anchors:

    risk(70 mg/dL)  = 25.0   (hypo)
    risk(138.9)     ~= 0     (zero-risk centre -> max reward)
    risk(280 mg/dL) = 25.0   (hyper)

The directional halves of this same kernel are the standard glycemic indices:
LBGI is the mean of the low (f < 0) part and HBGI is the mean of the high
(f > 0) part, so ``risk = low + high`` sample-wise and
``LBGI + HBGI = mean(risk)``. Using one kernel keeps the reward (B1), the
clinical-metric fidelity (L2 in B3), and the LBGI/HBGI report mutually
consistent.

``reward(g) = -sum(magni_risk(g))`` is the default objective: it is maximized
(= 0) by a constant ~138.9 mg/dL trace and penalizes both hypo and hyper
excursions on the clinically calibrated risk scale.
"""
from __future__ import annotations

import numpy as np


_R_G = 3.550313
_R_A = 0.835334
_R_B = 3.793357

TIR_LOW = 70.0
TIR_HIGH = 180.0

# Smallest glucose passed to log(); guards against zeros / sensor floor.
_G_FLOOR = 1.0


def _as_array(glucose) -> np.ndarray:
    """
    Coerce input to a 1-D float array and validate it is non-empty.
    """
    g = np.asarray(glucose, dtype=float).ravel()
    if g.size == 0:
        raise ValueError("glucose series is empty")
    return g


def _risk_transform(glucose) -> np.ndarray:
    """
    f(g) = _R_G * (ln(g)**_R_A - _R_B).
    """
    g = np.clip(_as_array(glucose), _G_FLOOR, None)
    return _R_G * (np.log(g) ** _R_A - _R_B)


def magni_risk(glucose) -> np.ndarray:
    """
    Per-sample symmetrized glucose risk ``10 * f(g)**2`` [risk units].
    """
    f = _risk_transform(glucose)
    return 10 * f * f


def risk_low_high(glucose) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-sample (low_risk, high_risk) split of :func:`magni_risk`.
    """
    f = _risk_transform(glucose)
    r = 10 * f * f
    low = np.where(f < 0.0, r, 0.0)
    high = np.where(f > 0.0, r, 0.0)
    return low, high


def lbgi(glucose) -> float:
    """
    Low Blood Glucose Index: mean of the hypoglycemic risk component.
    """
    low, _ = risk_low_high(glucose)
    return float(np.mean(low))


def hbgi(glucose) -> float:
    """
    High Blood Glucose Index: mean of the hyperglycemic risk component.
    """
    _, high = risk_low_high(glucose)
    return float(np.mean(high))


def mean_glucose(glucose) -> float:
    """
    Mean glucose [mg/dL].
    """
    return float(np.mean(_as_array(glucose)))


def time_in_range(glucose, low: float = TIR_LOW, high: float = TIR_HIGH) -> float:
    """
    Percent of samples with ``low <= g <= high`` (default 70-180 mg/dL).
    """
    g = _as_array(glucose)
    return float(100.0 * np.mean((g >= low) & (g <= high)))


def time_below_range(glucose, low: float = TIR_LOW) -> float:
    """
    Percent of samples with ``g < low`` (default <70 mg/dL).
    """
    g = _as_array(glucose)
    return float(100.0 * np.mean(g < low))


def time_above_range(glucose, high: float = TIR_HIGH) -> float:
    """
    Percent of samples with ``g > high`` (default >180 mg/dL).
    """
    g = _as_array(glucose)
    return float(100.0 * np.mean(g > high))


def reward(glucose) -> float:
    """
    Default scalar objective: ``-sum(magni_risk(glucose))``.

    Higher is better; maximized (= 0) by a constant ~138.9 mg/dL trace
    (the Magni 2007 zero-risk centre). Summed (not averaged) so it has the
    units of total accumulated risk over the horizon; policies are always
    compared over a common horizon, so the sum is a valid ordering objective.
    """
    return float(-np.sum(magni_risk(glucose)))


def clinical_metrics(glucose) -> dict:
    """
    Keys: ``mean_glucose``, ``tir``, ``tbr``, ``tar`` (percent), ``lbgi``,
    ``hbgi``, ``magni_risk_sum``, ``magni_risk_mean``, ``reward``.
    """
    g = _as_array(glucose)
    low, high = risk_low_high(g)
    risk = low + high
    return {
        "mean_glucose": float(np.mean(g)),
        "tir": time_in_range(g),
        "tbr": time_below_range(g),
        "tar": time_above_range(g),
        "lbgi": float(np.mean(low)),
        "hbgi": float(np.mean(high)),
        "magni_risk_sum": float(np.sum(risk)),
        "magni_risk_mean": float(np.mean(risk)),
        "reward": float(-np.sum(risk)),
    }
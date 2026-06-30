"""CGM sensor models for twin output generation.

Converts a noise-free IG trajectory (the model output) into a noisy CGM trace,
so that twin output is directly comparable to ground-truth CGM on the *same*
(CGM vs CGM) footing.

The primary model, ``add_cgm_noise``, is the UVA/Padova sensor error model of
Breton & Kovatchev (2008) [1], as implemented by simglucose's ``CGMNoise``.
This is the exact sensor model that simglucose applies when it generates the
ground-truth CGM, so re-applying it to twin IG gives an apples-to-apples
comparison. The model has three stages:

1. AR(1) on white noise at a fixed 15-min model grid (temporal correlation):
       e_0 = N(0,1);  e_n = PACF * (e_{n-1} + N(0,1))           (PACF = 0.7)
2. Johnson S_U transform (gives the skewed, heavy-tailed CGM error shape):
       eps = xi + lambda * sinh((e_n - gamma) / delta)
   Dexcom params: xi=-5.47, lambda=15.9574, gamma=-0.5444, delta=1.6898.
3. Cubic interpolation from the 15-min grid down to the sensor ``sample_time``,
   then  CGM = clip(IG + eps, sensor_min, sensor_max).

Because stage 3 depends on ``sample_time``, the IG samples passed in MUST be
spaced at the same ``sample_time`` used here — otherwise the noise correlation
timescale is wrong. ``add_cgm_noise`` therefore takes an explicit ``sample_time``
and configures the generator to match (the ``Twin`` layer always passes its own
``self.sample_time``).

``gaussian_noise`` is a simple i.i.d. fallback for likelihood calculations and
quick tests where a known, white noise scale is required.

[1] Breton M, Kovatchev B. "Analysis, Modeling, and Simulation of the Accuracy
    of Continuous Glucose Sensors." J Diabetes Sci Technol, 2008.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pkg_resources

from simglucose.sensor.noise_gen import CGMNoise

_SENSOR_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/sensor_params.csv"
)


def _sensor_params(sensor_name: str, sample_time: float | None = None) -> pd.Series:
    """Load simglucose sensor params, optionally overriding ``sample_time``.

    The CSV ships one row per sensor (Dexcom, GuardianRT, Navigator) with the
    Johnson-transform parameters (PACF, gamma, lambda, delta, xi), the native
    ``sample_time``, and the physical ``min``/``max`` range. We override
    ``sample_time`` so the noise model's 15-min->sample_time interpolation grid
    matches the spacing of the IG samples we feed it.
    """
    df = pd.read_csv(_SENSOR_PARA_FILE)
    rows = df.loc[df.Name == sensor_name]
    if rows.empty:
        raise ValueError(
            f"unknown sensor {sensor_name!r}; available: {list(df.Name)}"
        )
    p = rows.squeeze().copy()
    if sample_time is not None:
        p["sample_time"] = float(sample_time)
    return p


def add_cgm_noise(
    ig_sampled: np.ndarray,
    seed: int = 1,
    sensor_name: str = "Dexcom",
    sample_time: float | None = None,
) -> np.ndarray:
    """Apply the UVA/Padova (Breton-Kovatchev) sensor model to sampled IG.

    Parameters
    ----------
    ig_sampled : (n,) IG values at the CGM sample times [mg/dL]. These must be
        extracted from the dt-grid via ``replaybg_model.sample_indices`` at the
        *same* ``sample_time`` passed here.
    seed : RNG seed (reproduces an exact noise realization). Use distinct seeds
        across posterior draws / vs the ground truth to get independent noise.
    sensor_name : simglucose sensor (``"Dexcom"``, ``"GuardianRT"``,
        ``"Navigator"``).
    sample_time : CGM sampling interval [min]. If ``None``, the sensor's native
        value is used (Dexcom 3, GuardianRT 5, Navigator 1). The ``Twin`` layer
        passes ``self.sample_time`` so the noise correlation grid is correct.

    Returns
    -------
    cgm : (n,) CGM values [mg/dL], clipped to the sensor's physical range.
    """
    ig_sampled = np.asarray(ig_sampled, dtype=float)
    p = _sensor_params(sensor_name, sample_time)
    lo, hi = float(p["min"]), float(p["max"])

    noise_gen = CGMNoise(p, seed=seed)
    cgm = np.empty_like(ig_sampled)
    for i, ig_val in enumerate(ig_sampled):
        cgm[i] = float(np.clip(ig_val + next(noise_gen), lo, hi))
    return cgm


def gaussian_noise(
    ig_sampled: np.ndarray,
    sigma: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a sampled IG trace (white-noise fallback).

    Useful for likelihood calculations and quick tests where a known noise
    scale is required. Not the UVA/Padova model -- use ``add_cgm_noise`` for
    realistic, temporally correlated sensor error.
    """
    ig_sampled = np.asarray(ig_sampled, dtype=float)
    rng = np.random.default_rng(seed)
    return ig_sampled + rng.normal(0.0, sigma, size=ig_sampled.shape)
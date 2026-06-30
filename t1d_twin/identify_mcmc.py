"""Bayesian MCMC identification of a ReplayBG digital twin (Cappon et al., 2023).

Fits the 8-parameter ReplayBG model to a CGM trace plus its insulin/CHO inputs
using an ensemble MCMC sampler (``emcee``), producing a posterior over the free
parameters and an ``MCMCTwin`` that implements the common ``Twin`` interface
(``t1d_twin.twin.Twin``).

Parameter layout (follows ``replaybg_model.THETA_NAMES``):
    [ka2, kd, kempt, kabs, SG, SI, p2, Gb]

Sampling strategy
-----------------
The seven rate constants (all but Gb) are sampled in **log-space** because they
span orders of magnitude and have strictly positive support. Gb is sampled
linearly. The internal sampler works in the 8-D space::

    phi = [log(ka2), log(kd), log(kempt), log(kabs), log(SG), log(SI), log(p2), Gb]

All prior/likelihood functions accept ``phi`` vectors. The public API always
speaks in natural (un-transformed) theta space.

Prior (informative; published ReplayBG prior, Cappon et al. 2023)
-----------------------------------------------------------------
The prior is the published informative prior defined in ``replaybg_priors``
(lognormal rate constants, Gamma SI, truncated-normal Gb, sqrt-normal p2) with
the ordering constraints ka2<kd and kabs<kempt. It is shared with the MAP fit
(``population.fit_replaybg``) and the SBI twin, so all three reference one
distribution. ``log_prior`` delegates to ``replaybg_priors.log_prior_phi``.
``PRIOR_LO/HI`` here are only a generous *support* box (installed by
``population.install_published_prior``) used to clamp walkers and short-circuit
out-of-support proposals — they are not the prior itself.

emcee integration
-----------------
``emcee.EnsembleSampler(vectorize=True)`` is used so each proposal step is a
*single batched forward pass* over all walkers — exactly matching the batched
``simulate(theta_batch, ...)`` API of the integrator. Out-of-prior walkers are
detected before calling the ODE, short-circuiting expensive computation.
"""
from __future__ import annotations

import warnings
from typing import Optional

import emcee
import numpy as np

from .replaybg_model import (
    THETA_NAMES, get_pop_theta,
    theta_to_array, simulate, sample_indices,
)
from . import replaybg_priors as _priors
from .simglucose_adapter import RunResult
from .twin import Twin

# ---------------------------------------------------------------------------
# Prior bounds in natural (un-transformed) space
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Support envelope (natural space). The prior itself is now the informative
# published one in ``replaybg_priors`` (see ``log_prior`` below); PRIOR_LO/HI
# serve only to clamp walkers / bound the support, and are (re)installed by
# ``experiments.population.install_published_prior``.
# ---------------------------------------------------------------------------
PRIOR_LO = _priors.SUPPORT_LO.copy()
PRIOR_HI = _priors.SUPPORT_HI.copy()

# Keep old private names as aliases for test-file compatibility
_PRIOR_LO = PRIOR_LO
_PRIOR_HI = PRIOR_HI

# Default noise std [mg/dL] — matches Dexcom empirical std (~10 mg/dL)
DEFAULT_SIGMA: float = 10.0

# Default MCMC hyperparameters
DEFAULT_NWALKERS: int = 32
DEFAULT_BURN: int = 300
DEFAULT_NSAMPLE: int = 500
DEFAULT_N_POSTERIOR: int = 1000


# ---------------------------------------------------------------------------
# Parameter transforms  phi <-> theta
# ---------------------------------------------------------------------------
# Canonical definition lives in ``replaybg_priors`` (the phi-convention
# authority). Re-exported here under the historical private names so existing
# imports (``from t1d_twin.identify_mcmc import _theta_to_phi, _phi_to_theta``,
# e.g. in ``experiments.population``) keep working unchanged. Numerically these
# ARE the canonical numpy transforms — no separate copy to drift.

_theta_to_phi = _priors.theta_to_phi
_phi_to_theta = _priors.phi_to_theta


# ---------------------------------------------------------------------------
# Log-prior  (vectorised over batch B)
# ---------------------------------------------------------------------------

def log_prior(phi_batch: np.ndarray) -> np.ndarray:
    """Informative published (Cappon et al.) log-prior in phi-space; (B,8)->(B,).

    Delegates to ``replaybg_priors.log_prior_phi``: lognormal rates, gamma SI,
    truncnorm Gb, sqrt-normal p2, plus the ``ka2<kd`` / ``kabs<kempt`` ordering
    gates and the support box. Replaces the former uniform box prior, so the
    posterior is now properly regularised toward the published distribution.
    """
    return _priors.log_prior_phi(np.atleast_2d(phi_batch))


# ---------------------------------------------------------------------------
# Log-likelihood  (vectorised, batched simulate)
# ---------------------------------------------------------------------------

def log_likelihood(
    phi_batch: np.ndarray,
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Gaussian CGM log-likelihood for a batch of phi vectors.

    Parameters
    ----------
    phi_batch : (B, 8) sampler coordinates.
    cgm : (n,) observed CGM [mg/dL].
    insulin, cho : (T,) input grids [mU/kg/min, mg/kg/min].
    Ib : basal insulin [mU/kg/min] for steady-state ICs.
    sample_time : CGM sampling interval [min].
    dt : integration step [min].
    sigma : sensor noise std dev [mg/dL].

    Returns
    -------
    (B,) log-likelihood values; -inf for non-finite simulations.
    """
    theta_b = _phi_to_theta(np.atleast_2d(phi_batch))
    n = len(cgm)
    obs_idx = sample_indices(n, sample_time, dt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, ig_b = simulate(theta_b, insulin, cho, Ib=Ib, dt=dt)   # (B, T)

    ig_obs = ig_b[:, obs_idx]                                      # (B, n)
    finite_mask = np.all(np.isfinite(ig_obs), axis=1)              # (B,)

    resid = ig_obs - cgm[None, :]
    logL = (-0.5 * np.sum((resid / sigma) ** 2, axis=1)
            - n * np.log(sigma))
    return np.where(finite_mask, logL, -np.inf)


# ---------------------------------------------------------------------------
# Joint log-probability  (vectorised callable for emcee)
# ---------------------------------------------------------------------------

def make_log_prob(
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
):
    """Return a vectorised ``log_prob`` callable for ``emcee``.

    ``emcee`` calls ``log_prob(phi_batch)`` with shape (B, ndim) and expects
    (B,).  Out-of-prior walkers are short-circuited before the ODE runs.
    """
    def log_prob(phi_batch: np.ndarray) -> np.ndarray:
        phi_batch = np.atleast_2d(phi_batch)
        lp = log_prior(phi_batch)
        out = np.full(lp.shape, -np.inf)
        in_prior = np.isfinite(lp)
        if np.any(in_prior):
            ll = log_likelihood(
                phi_batch[in_prior],
                cgm, insulin, cho, Ib, sample_time, dt, sigma,
            )
            out[in_prior] = lp[in_prior] + ll
        return out

    return log_prob


# ---------------------------------------------------------------------------
# Walker initialisation
# ---------------------------------------------------------------------------

def _init_walkers(
    nwalkers: int,
    Gb_init: float,
    rng: np.random.Generator,
    jitter: float = 0.05,
) -> np.ndarray:
    """Initialise walkers near the population prior centre in phi space.

    ``Gb_init`` overrides the population Gb with the data-derived fasting
    median.  Each walker is perturbed by a small additive Gaussian jitter in
    phi-space (multiplicative in natural space), keeping walkers inside the
    prior for small jitter.
    """
    pop_theta = theta_to_array(get_pop_theta()).copy()
    pop_theta[7] = float(np.clip(Gb_init, PRIOR_LO[7] * 1.01, PRIOR_HI[7] * 0.99))
    pop_theta = np.clip(pop_theta, PRIOR_LO * 1.01, PRIOR_HI * 0.99)
    pop_phi = _theta_to_phi(pop_theta)

    p0 = pop_phi[None, :] + jitter * rng.standard_normal((nwalkers, 8))

    # Clamp back into the prior after jitter
    theta_trial = _phi_to_theta(p0)
    theta_trial = np.clip(theta_trial, PRIOR_LO * 1.01, PRIOR_HI * 0.99)
    return _theta_to_phi(theta_trial)


# ---------------------------------------------------------------------------
# MCMC driver
# ---------------------------------------------------------------------------

def run_mcmc(
    cgm: np.ndarray,
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    nwalkers: int = DEFAULT_NWALKERS,
    nburn: int = DEFAULT_BURN,
    nsample: int = DEFAULT_NSAMPLE,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    seed: int = 42,
    progress: bool = True,
) -> np.ndarray:
    """Run ensemble MCMC and return ``n_posterior`` posterior theta samples.

    Parameters
    ----------
    cgm : (n,) observed CGM trace [mg/dL].
    insulin, cho : (T,) input arrays on the dt-grid.
    Ib : basal insulin [mU/kg/min].
    sample_time : CGM sampling period [min].
    dt : integration step [min].
    sigma : fixed sensor noise std [mg/dL].
    nwalkers : ensemble walkers (>= 2 x ndim = 16; default 32).
    nburn : burn-in steps (discarded).
    nsample : production steps after burn-in.
    n_posterior : posterior samples to subsample for the Twin.
    seed : RNG seed for reproducibility.
    progress : show emcee progress bar.

    Returns
    -------
    theta_post : (n_posterior, 8) posterior samples in natural space.
    """
    rng = np.random.default_rng(seed)
    ndim = 8

    Gb_init = float(np.median(cgm[:max(1, len(cgm) // 4)]))
    p0 = _init_walkers(nwalkers, Gb_init, rng)

    log_prob_fn = make_log_prob(cgm, insulin, cho, Ib, sample_time, dt, sigma)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, vectorize=True)

    state = sampler.run_mcmc(
        p0, nburn, progress=progress,
        progress_kwargs={"desc": "MCMC burn-in"},
    )
    sampler.reset()
    sampler.run_mcmc(
        state, nsample, progress=progress,
        progress_kwargs={"desc": "MCMC sampling"},
    )

    flat_phi = sampler.get_chain(flat=True)
    flat_lp = sampler.get_log_prob(flat=True)

    finite = np.isfinite(flat_lp)
    flat_phi = flat_phi[finite]

    if len(flat_phi) < n_posterior:
        warnings.warn(
            f"Only {len(flat_phi)} finite-prob samples available "
            f"(requested {n_posterior}); using all.",
            RuntimeWarning, stacklevel=2,
        )
        chosen_phi = flat_phi
    else:
        idx = rng.choice(len(flat_phi), size=n_posterior, replace=False)
        chosen_phi = flat_phi[idx]

    return _phi_to_theta(chosen_phi)


# ---------------------------------------------------------------------------
# MCMCTwin — implements the common Twin interface
# ---------------------------------------------------------------------------

class MCMCTwin(Twin):
    """ReplayBG digital twin identified by Bayesian MCMC.

    Holds a posterior sample ``theta_post`` (R, 8) and exposes the standard
    ``Twin`` interface: ``predict_ig``, ``generate_cgm``, ``replay_run``,
    ``summary``.

    Parameters
    ----------
    theta_post : (R, 8) posterior samples in natural theta space.
    Ib : basal insulin [mU/kg/min] used for steady-state ICs.
    sample_time : CGM sampling interval [min].
    dt : integration step [min].
    sensor_name : simglucose sensor for CGM noise (default ``"Dexcom"``).
    sigma : noise std used during fitting (stored for reference).
    """

    def __init__(
        self,
        theta_post: np.ndarray,
        Ib: float,
        sample_time: float,
        dt: float = 1.0,
        sensor_name: str = "Dexcom",
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        self.theta_post: np.ndarray = np.asarray(theta_post, dtype=float)
        self.theta_median: np.ndarray = np.median(self.theta_post, axis=0)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.sensor_name = sensor_name
        self.sigma = float(sigma)

    # ------------------------------------------------------------------
    # Twin ABC implementation
    # ------------------------------------------------------------------

    def predict_ig(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate IG ensemble over the posterior, return (median, IQR).

        Parameters
        ----------
        insulin, cho : (T,) inputs on the dt-grid.
        n_samples : subsample this many posterior draws (default: use all).

        Returns
        -------
        ig_median : (T,) posterior-median IG [mg/dL].
        ig_iqr : (2, T) [25th, 75th] percentile band.
        """
        ig_ens = self._ig_ensemble(insulin, cho, n_samples)
        ig_median = np.median(ig_ens, axis=0)
        ig_iqr = np.percentile(ig_ens, [25, 75], axis=0)
        return ig_median, ig_iqr

    def _ig_ensemble(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Return (R, T) IG ensemble from batched posterior simulate."""
        theta_b = self.theta_post
        if n_samples is not None and n_samples < len(theta_b):
            idx = np.random.default_rng(0).choice(
                len(theta_b), size=n_samples, replace=False
            )
            theta_b = theta_b[idx]
        _, ig_ens = simulate(theta_b, insulin, cho, Ib=self.Ib, dt=self.dt)
        if n_samples is not None and n_samples > len(ig_ens):
            idx = np.random.default_rng(0).choice(
                len(ig_ens), size=n_samples, replace=True)
            ig_ens = ig_ens[idx]
        return ig_ens   # (R, T)

    def summary(self) -> dict:
        """Per-parameter posterior medians and 95% credible intervals."""
        result = {}
        for i, name in enumerate(THETA_NAMES):
            col = self.theta_post[:, i]
            lo, med, hi = np.percentile(col, [2.5, 50, 97.5])
            result[name] = {
                "median": float(med),
                "ci95": (float(lo), float(hi)),
            }
        return result

    # ------------------------------------------------------------------
    # Extra MCMC-specific properties
    # ------------------------------------------------------------------

    def n_posterior(self) -> int:
        """Number of posterior samples stored."""
        return self.theta_post.shape[0]


# ---------------------------------------------------------------------------
# High-level glue
# ---------------------------------------------------------------------------

def identify_twin_from_run(
    run: RunResult,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    sensor_name: str = "Dexcom",
    nwalkers: int = DEFAULT_NWALKERS,
    nburn: int = DEFAULT_BURN,
    nsample: int = DEFAULT_NSAMPLE,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    seed: int = 42,
    progress: bool = True,
) -> MCMCTwin:
    """Identify an MCMCTwin from a simglucose ``RunResult``.

    Pulls CGM, insulin/CHO grids, and metadata from the run, runs MCMC,
    and returns an ``MCMCTwin`` implementing the common ``Twin`` interface.

    Parameters
    ----------
    run : completed simglucose ``RunResult``.
    dt : ODE integration step [min].
    sigma : fixed sensor noise std [mg/dL] (used in the likelihood).
    sensor_name : simglucose sensor for CGM generation (default ``"Dexcom"``).
    nwalkers, nburn, nsample : MCMC hyperparameters.
    n_posterior : posterior samples stored in the Twin.
    seed : RNG seed.
    progress : show emcee progress bars.

    Returns
    -------
    MCMCTwin
    """
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])

    theta_post = run_mcmc(
        cgm=cgm,
        insulin=insulin,
        cho=cho,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sigma=sigma,
        nwalkers=nwalkers,
        nburn=nburn,
        nsample=nsample,
        n_posterior=n_posterior,
        seed=seed,
        progress=progress,
    )

    return MCMCTwin(
        theta_post=theta_post,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sensor_name=sensor_name,
        sigma=sigma,
    )
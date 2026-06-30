"""SBI (Simulation-Based Inference) twin identification (Hoang et al., 2025).

Implements a Neural Posterior Estimator (NPE) using a Masked Autoregressive
Flow (MAF) trained on (θ, y) pairs generated from the ReplayBG forward model.
This is the SBI twinning method described in:

    Hoang et al., "A Real-Time Digital Twin for T1D using Simulation-Based
    Inference", 2025.

Architecture overview
---------------------
Phase A3 — Prior, simulator, and training-data generation (this module,
            section 1):
  * ``make_prior()``:   the published informative prior (``replaybg_priors``),
                        shared with MCMC and the MAP fit (replaces the old
                        BoxUniform). Defined in phi (log) space.
  * ``make_simulator()``: wraps ``replaybg_model.simulate`` for a fixed
                          identification therapy; returns the IG sub-sampled
                          at CGM observation times as the observation vector y.
  * ``generate_training_data()``: draws N θ from the prior, simulates y,
                                  rejection-samples to [40, 400] mg/dL.

Phase A4 — NPE training + SBITwin wrapper (this module, section 2):
  * ``train_npe()``:    trains SNPE-C (MAF) on (θ, y) pairs.
  * ``SBITwin(Twin)``:  holds the trained posterior; conditions on an observed
                         CGM and draws samples; exposes predict_ig / generate_cgm.
  * ``identify_twin_from_run()``:  high-level glue from RunResult → SBITwin.

Joint IC inference (flag: ``infer_ic=False`` default, v1 steady-state)
-----------------------------------------------------------------------
The paper (section 3.1) proposes extending θ to θ̂ = [θ, x0] ∈ R^17,
sampling x0 by simulating from steady-state and extracting a random window.
This is implemented but kept behind ``infer_ic=False`` to stay compatible
with the v1 steady-state assumption.

Parameter layout (follows ``replaybg_model.THETA_NAMES``):
    [ka2, kd, kempt, kabs, SG, SI, p2, Gb]   — 8 free params.
    + [G, Isc1, Isc2, Ip, Qsto1, Qsto2, Qgut, X, IG] initial states if
      ``infer_ic=True``.

Prior bounds — same as MCMC (A2)
---------------------------------
    ka2   : [1e-3, 5e-2]
    kd    : [1e-3, 8e-2]
    kempt : [2e-2, 6e-1]
    kabs  : [1e-3, 8e-2]
    SG    : [1e-3, 5e-2]
    SI    : [1e-5, 5e-4]
    p2    : [1e-3, 5e-2]
    Gb    : [80,   200 ]

SBI training details (§3.2 of the paper)
-----------------------------------------
    * Rejection sampling: keep only sims with all CGM in [40, 400] mg/dL.
    * N = 5 000 valid samples (default; set N_TRAIN_DEFAULT).
    * MAF density estimator (SNPE-C) from the `sbi` library.
    * Training: batch_size=200, lr=5e-4, clip_max_norm=5.0, val_fraction=0.10,
                early stopping after 20 epochs of no improvement.
    * Observation y: IG at CGM sample times (length n_obs).
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from sbi.inference import SNPE, simulate_for_sbi

from .replaybg_model import (
    THETA_NAMES,
    simulate, sample_indices, steady_state, theta_to_array,
)
from . import replaybg_priors as _priors
from .simglucose_adapter import RunResult
from .twin import Twin

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Prior bounds — same as MCMC A2 (identify_mcmc.PRIOR_LO / PRIOR_HI)
# Support envelope (natural space) — the prior itself is the informative
# published one in ``replaybg_priors`` (see ``make_prior``). These bounds are
# only a physical support box, (re)installed by population.install_published_prior.
PRIOR_LO = _priors.SUPPORT_LO.copy()
PRIOR_HI = _priors.SUPPORT_HI.copy()

# Default training-data set size (paper: 5 000)
N_TRAIN_DEFAULT: int = 5_000

# CGM physiological rejection bounds [mg/dL]
CGM_LO: float = 40.0
CGM_HI: float = 400.0

# Noise std added to simulator output during training [mg/dL]
# Matches Dexcom empirical std used in MCMC A2.
DEFAULT_SIGMA: float = 10.0

# Default NPE posterior samples drawn per identification
DEFAULT_N_POSTERIOR: int = 1_000

# Fixed log-space transforms for rate params (indices 0-6) same as MCMC
_LOG_IDX = list(range(_priors.N_LOG_RATES))   # leading rate params are log-transformed


# ---------------------------------------------------------------------------
# Log-space <-> natural transforms (same convention as MCMC)
# ---------------------------------------------------------------------------
# Canonical definitions live in ``replaybg_priors`` (the single phi-convention
# authority). Two flavours are surfaced from this torch-facing module:
#
#   * ``_theta_to_phi`` / ``_phi_to_theta``  — NumPy. Re-exported under the
#     historical private names so existing imports keep working unchanged
#     (``tcn_cv`` / ``linear_cv`` import these; ``run_sbi`` / ``run_sbi0`` call
#     them as ``S._phi_to_theta``). EVERY current SBI call site converts its
#     tensors to numpy *before* transforming (e.g. ``phi.detach().cpu().numpy()``
#     in the simulator and posterior glue), so these stay numpy to be exactly
#     behaviour-preserving.
#   * ``theta_to_phi_torch`` / ``phi_to_theta_torch``  — Torch-native. Use these
#     when you want to transform a ``torch.Tensor`` without a numpy round-trip
#     (preserves device / dtype / autograd). Provided for the SBI path per the
#     log-rates standardisation; numerically identical to the numpy pair.

_theta_to_phi = _priors.theta_to_phi
_phi_to_theta = _priors.phi_to_theta
theta_to_phi_torch = _priors.theta_to_phi_torch
phi_to_theta_torch = _priors.phi_to_theta_torch


# ---------------------------------------------------------------------------
# A3 — Part 1: Prior
# ---------------------------------------------------------------------------

def make_prior(log_space: bool = True):
    """Return the informative published (Cappon et al.) prior over phi-space.

    Replaces the former ``BoxUniform``. The prior — lognormal rates, gamma SI,
    truncnorm Gb, sqrt-normal p2, with ``ka2<kd`` / ``kabs<kempt`` ordering
    rejection — lives in ``replaybg_priors`` and is shared with the MAP fit and
    the MCMC log-prior, so all three reference one distribution.

    ``log_space`` must be True: the prior is defined in phi = [log(rates), Gb].
    A linear-space variant is no longer supported.
    """
    if not log_space:
        raise NotImplementedError(
            "The published prior is defined in phi (log) space; "
            "call make_prior(log_space=True).")
    prior = _priors.make_sbi_prior()
    # Best-effort adapt to the installed sbi version; the raw object already
    # exposes .sample/.log_prob if process_prior is unavailable.
    try:
        from sbi.utils.user_input_checks import process_prior
        prior, _, _ = process_prior(prior)
    except Exception:
        pass
    return prior


# ---------------------------------------------------------------------------
# A3 — Part 2: Simulator
# ---------------------------------------------------------------------------

def make_simulator(
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    log_space: bool = True,
    seed: Optional[int] = None,
):
    """Build a ``sbi``-compatible simulator wrapping the ReplayBG ODE.

    The returned callable maps one phi (or theta) vector → observation y.

    Parameters
    ----------
    insulin, cho : (T,) identification-therapy inputs [mU/kg/min, mg/kg/min].
    Ib : basal insulin [mU/kg/min] for steady-state ICs.
    sample_time : CGM sampling interval [min].
    dt : ODE integration step [min].
    sigma : Gaussian noise std added to IG at observation times [mg/dL].
        Set to 0 to skip noise.
    log_space : whether phi is in log-space (must match ``make_prior``).
    seed : global numpy RNG seed for reproducibility; ``None`` = no seeding.

    Returns
    -------
    simulator : callable(phi: Tensor[8]) -> Tensor[n_obs]
        Returns the noisy CGM observation vector for the given parameters.
        Returns a tensor of NaNs if the simulation is non-physical.
    """
    T = len(insulin)
    n_obs = T // int(round(sample_time / dt))
    obs_idx = sample_indices(n_obs, sample_time, dt)

    rng_state = np.random.default_rng(seed)

    def simulator(phi_t: torch.Tensor) -> torch.Tensor:
        phi = phi_t.detach().cpu().numpy().astype(float)
        theta = _phi_to_theta(phi) if log_space else phi

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                _, ig = simulate(theta, insulin, cho, Ib=Ib, dt=dt)
            except Exception:
                return torch.full((n_obs,), float("nan"), dtype=torch.float32)

        if not np.all(np.isfinite(ig)):
            return torch.full((n_obs,), float("nan"), dtype=torch.float32)

        ig_obs = ig[obs_idx]
        if sigma > 0.0:
            ig_obs = ig_obs + rng_state.normal(0.0, sigma, size=n_obs)
        return torch.tensor(ig_obs, dtype=torch.float32)

    return simulator, n_obs


# ---------------------------------------------------------------------------
# A3 — Part 3: Training-data generation with rejection sampling
# ---------------------------------------------------------------------------

def generate_training_data(
    insulin: np.ndarray,
    cho: np.ndarray,
    Ib: float,
    sample_time: float,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    n_train: int = N_TRAIN_DEFAULT,
    cgm_lo: float = CGM_LO,
    cgm_hi: float = CGM_HI,
    log_space: bool = True,
    seed: int = 42,
    batch_size: int = 512,
    max_attempts: int = 20,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate (θ, y) training pairs with rejection sampling.

    Draws parameter vectors φ from the prior, simulates CGM observations y,
    and retains only samples where **all** CGM values are in
    [``cgm_lo``, ``cgm_hi``] mg/dL (physiological rejection filter per §3.2).

    Parameters
    ----------
    insulin, cho : (T,) identification inputs.
    Ib : basal insulin for steady-state ICs.
    sample_time : CGM sample interval [min].
    dt : integration step [min].
    sigma : sensor noise std added to IG [mg/dL].
    n_train : number of valid samples to collect.
    cgm_lo, cgm_hi : rejection bounds [mg/dL].
    log_space : whether phi is log-space (must match ``make_prior``).
    seed : RNG seed.
    batch_size : how many samples to simulate per round.
    max_attempts : stop after this many rounds even if n_train not reached.
    verbose : print progress.

    Returns
    -------
    theta_tensor : (n_train, 8) accepted phi/theta samples.
    obs_tensor   : (n_train, n_obs) accepted observation vectors.
    """
    rng_phi = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)

    prior = make_prior(log_space=log_space)
    simulator, n_obs = make_simulator(
        insulin, cho, Ib, sample_time, dt, sigma=0.0,  # we add noise below
        log_space=log_space, seed=seed + 2,
    )

    T = len(insulin)
    obs_idx = sample_indices(n_obs, sample_time, dt)

    accepted_phi: list[np.ndarray] = []
    accepted_obs: list[np.ndarray] = []
    n_accepted = 0
    n_attempts = 0

    if verbose:
        print(f"  [SBI A3] Generating {n_train} training pairs "
              f"(rejection bounds [{cgm_lo}, {cgm_hi}] mg/dL) ...")

    while n_accepted < n_train and n_attempts < max_attempts:
        n_attempts += 1
        # Draw batch of phi from prior
        phi_batch_t = prior.sample((batch_size,))          # (B, 8)
        phi_batch = phi_batch_t.numpy()                    # (B, 8)

        # Convert to theta for batched simulate
        theta_batch = _phi_to_theta(phi_batch) if log_space else phi_batch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                _, ig_batch = simulate(theta_batch, insulin, cho,
                                       Ib=Ib, dt=dt)     # (B, T)
            except Exception as exc:
                if verbose:
                    print(f"  [warn] simulate batch failed: {exc}")
                continue

        ig_obs_batch = ig_batch[:, obs_idx]                # (B, n_obs)

        # Add sensor noise
        if sigma > 0.0:
            noise = rng_noise.normal(0.0, sigma,
                                     size=ig_obs_batch.shape)
            ig_obs_batch = ig_obs_batch + noise

        # Rejection filter: physiological range + finite
        finite_mask = np.all(np.isfinite(ig_obs_batch), axis=1)
        range_mask = (
            np.all(ig_obs_batch >= cgm_lo, axis=1) &
            np.all(ig_obs_batch <= cgm_hi, axis=1)
        )
        keep = finite_mask & range_mask                    # (B,)

        n_kept = int(keep.sum())
        if n_kept == 0:
            continue

        accepted_phi.append(phi_batch[keep])
        accepted_obs.append(ig_obs_batch[keep])
        n_accepted += n_kept

        if verbose:
            accept_rate = n_kept / batch_size * 100.0
            print(f"    round {n_attempts:2d}: kept {n_kept}/{batch_size} "
                  f"({accept_rate:.1f}%),  total {min(n_accepted, n_train)}/{n_train}")

        if n_accepted >= n_train:
            break

    if n_accepted < n_train:
        warnings.warn(
            f"Only {n_accepted}/{n_train} training samples collected after "
            f"{n_attempts} rounds (each of size {batch_size}). "
            "Consider increasing max_attempts or relaxing rejection bounds.",
            RuntimeWarning, stacklevel=2,
        )

    phi_all = np.vstack(accepted_phi)[:n_train]
    obs_all = np.vstack(accepted_obs)[:n_train]

    theta_tensor = torch.tensor(phi_all, dtype=torch.float32)
    obs_tensor = torch.tensor(obs_all, dtype=torch.float32)

    if verbose:
        print(f"  [SBI A3] Done: {len(theta_tensor)} pairs collected.")

    return theta_tensor, obs_tensor


# ---------------------------------------------------------------------------
# A4 — Part 1: Train NPE (MAF)
# ---------------------------------------------------------------------------

def train_npe(
    theta_tensor: torch.Tensor,
    obs_tensor: torch.Tensor,
    prior,
    hidden_features: int = 50,
    num_transforms: int = 5,
    batch_size: int = 200,
    learning_rate: float = 5e-4,
    clip_max_norm: float = 5.0,
    validation_fraction: float = 0.10,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 200,
    verbose: bool = True,
) -> object:
    """Train a Masked Autoregressive Flow (MAF) posterior estimator.

    Uses SNPE-C from the `sbi` library with the training configuration from
    the paper (§3.2): batch_size=200, lr=5e-4, clip_max_norm=5, val=10%,
    early stopping after 20 epochs without improvement.

    Parameters
    ----------
    theta_tensor : (N, d_theta) parameter samples (phi-space if log_space).
    obs_tensor : (N, n_obs) corresponding observation vectors.
    prior : the prior used to generate the data (published prior; replaybg_priors).
    hidden_features : units per MAF hidden layer.
    num_transforms : number of MAF transform blocks.
    batch_size : training mini-batch size.
    learning_rate : Adam learning rate.
    clip_max_norm : gradient norm clipping.
    validation_fraction : fraction of data held out for early stopping.
    stop_after_epochs : patience for early stopping (epochs of no improvement).
    max_num_epochs : hard cap on training epochs.
    verbose : print training progress.

    Returns
    -------
    posterior : trained ``sbi`` ``DirectPosterior`` estimator.
    """
    if verbose:
        print(f"  [SBI A4] Training NPE (MAF) on {len(theta_tensor)} samples ...")

    # The sbi library writes TensorBoard logs via a SummaryWriter that, by
    # default, lands in a `sbi_logs`/`sbi-logs` folder in the current working
    # directory. Point it under the unified output root instead (env-driven so
    # this module stays decoupled from the experiments package), with a per-job
    # subdir so concurrent per-patient fits never collide.
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        root = os.environ.get("T1D_OUTPUT_ROOT", "/scratch/vmli3/t1d_experiment")
        if os.environ.get("SLURM_ARRAY_JOB_ID"):
            tag = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
        elif os.environ.get("SLURM_JOB_ID"):
            tag = os.environ["SLURM_JOB_ID"]
        else:
            from datetime import datetime
            tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_dir = os.path.join(root, "sbi_logs", tag)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    except Exception:
        writer = None  # fall back to sbi's default location if TB is unavailable

    try:
        inference = SNPE(prior=prior, density_estimator="maf", summary_writer=writer)
    except TypeError:
        # very different sbi version without the summary_writer kwarg
        inference = SNPE(prior=prior, density_estimator="maf")
    inference = inference.append_simulations(theta_tensor, obs_tensor)

    density_estimator = inference.train(
        training_batch_size=batch_size,
        learning_rate=learning_rate,
        clip_max_norm=clip_max_norm,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
        show_train_summary=verbose,
    )

    posterior = inference.build_posterior(density_estimator)
    if writer is not None:
        writer.close()
    if verbose:
        print("  [SBI A4] NPE training complete.")
    return posterior


# ---------------------------------------------------------------------------
# A4 — Part 2: SBITwin wrapper
# ---------------------------------------------------------------------------

class SBITwin(Twin):
    """ReplayBG digital twin identified by Simulation-Based Inference (NPE/MAF).

    Stores the trained amortized posterior and, once conditioned on an
    observed CGM trace, draws ``n_posterior`` samples from p(θ | y_obs).
    Exposes the standard ``Twin`` interface: ``predict_ig``, ``generate_cgm``,
    ``replay_run``, ``summary``.

    Parameters
    ----------
    posterior : trained ``sbi`` posterior (from ``train_npe``).
    theta_post : (R, 8) posterior samples in phi-space (converted to natural θ
                 before simulation).
    Ib : basal insulin [mU/kg/min] for steady-state ICs.
    sample_time : CGM sampling interval [min].
    dt : integration step [min].
    log_space : if True, ``theta_post`` is in phi-space and must be exponentiated.
    sensor_name : simglucose sensor for CGM noise.
    sigma : noise std used during training (stored for reference).
    """

    def __init__(
        self,
        posterior,
        theta_post: np.ndarray,
        Ib: float,
        sample_time: float,
        dt: float = 1.0,
        log_space: bool = True,
        sensor_name: str = "Dexcom",
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        self._posterior = posterior
        # Store phi_post (log-space) internally; expose natural theta for summaries
        phi_post = np.asarray(theta_post, dtype=float)
        self._phi_post = phi_post
        self.theta_post = _phi_to_theta(phi_post) if log_space else phi_post
        self.theta_median = np.median(self.theta_post, axis=0)
        self.Ib = float(Ib)
        self.sample_time = float(sample_time)
        self.dt = float(dt)
        self.log_space = log_space
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
        """Return (R, T) IG ensemble from batched posterior simulate.

        Rows with non-finite values (from theta drawn outside the valid
        physics regime) are replaced by the ensemble median column-wise so
        that downstream statistics stay finite.
        """
        theta_b = self.theta_post
        if n_samples is not None and n_samples < len(theta_b):
            idx = np.random.default_rng(0).choice(
                len(theta_b), size=n_samples, replace=False,
            )
            theta_b = theta_b[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, ig_ens = simulate(theta_b, insulin, cho, Ib=self.Ib, dt=self.dt)
        # Replace non-finite rows with column medians
        bad = ~np.all(np.isfinite(ig_ens), axis=1)
        if np.any(bad):
            col_med = np.nanmedian(ig_ens[~bad], axis=0) if np.any(~bad) else np.full(ig_ens.shape[1], 120.0)
            ig_ens[bad] = col_med
        # If more rows are requested than we have (e.g. a point-estimate twin
        # with a single theta), resample rows with replacement up to n_samples
        # so the band code receives a full ensemble. With one unique trajectory
        # the band then reflects sensor noise alone -- the honest band for a
        # method that carries no parameter uncertainty.
        if n_samples is not None and n_samples > len(ig_ens):
            idx = np.random.default_rng(0).choice(
                len(ig_ens), size=n_samples, replace=True)
            ig_ens = ig_ens[idx]
        return ig_ens  # (R, T)

    def summary(self) -> dict:
        """Per-parameter posterior medians and 95% credible intervals (natural θ)."""
        result = {}
        for i, name in enumerate(THETA_NAMES):
            col = self.theta_post[:, i]
            lo, med, hi = np.percentile(col, [2.5, 50, 97.5])
            result[name] = {
                "median": float(med),
                "ci95": (float(lo), float(hi)),
            }
        return result

    def n_posterior(self) -> int:
        """Number of posterior samples stored."""
        return self.theta_post.shape[0]


# ---------------------------------------------------------------------------
# A4 — Part 3: identify_twin_from_run  (high-level glue)
# ---------------------------------------------------------------------------

def identify_twin_from_run(
    run: RunResult,
    dt: float = 1.0,
    sigma: float = DEFAULT_SIGMA,
    sensor_name: str = "Dexcom",
    n_train: int = N_TRAIN_DEFAULT,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    log_space: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    batch_size_train: int = 200,
    learning_rate: float = 5e-4,
    clip_max_norm: float = 5.0,
    validation_fraction: float = 0.10,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 200,
    seed: int = 42,
    infer_ic: bool = False,
    verbose: bool = True,
) -> "SBITwin":
    """Identify an SBITwin from a simglucose ``RunResult``.

    Full pipeline:
      1. Extract CGM, insulin/CHO grids, and Ib from the run.
      2. Generate training data (prior + rejection-sampled simulations).
      3. Train the NPE (MAF) posterior estimator.
      4. Condition the posterior on the observed CGM → draw n_posterior samples.
      5. Return an ``SBITwin`` wrapping the posterior samples.

    Parameters
    ----------
    run : completed simglucose ``RunResult``.
    dt : ODE integration step [min].
    sigma : sensor noise std [mg/dL] (used in simulator).
    sensor_name : simglucose sensor for CGM generation.
    n_train : training data size (default 5 000; use fewer for fast tests).
    n_posterior : posterior samples drawn per identification.
    log_space : use log-space prior/simulator (recommended).
    hidden_features, num_transforms : MAF architecture.
    batch_size_train, learning_rate, clip_max_norm, validation_fraction,
    stop_after_epochs, max_num_epochs : NPE training config.
    seed : master RNG seed.
    infer_ic : if True, jointly infer initial conditions (17-dim θ̂; A4 ext).
               Currently not implemented; raises NotImplementedError.
    verbose : print progress.

    Returns
    -------
    SBITwin
    """
    if infer_ic:
        raise NotImplementedError(
            "Joint IC inference (infer_ic=True) is reserved for A4 extension. "
            "Use infer_ic=False (default) for v1 steady-state identification."
        )

    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])

    # --- build prior ---
    prior = make_prior(log_space=log_space)

    # --- generate training data ---
    theta_tensor, obs_tensor = generate_training_data(
        insulin=insulin,
        cho=cho,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        sigma=sigma,
        n_train=n_train,
        log_space=log_space,
        seed=seed,
        verbose=verbose,
    )

    # --- train NPE ---
    posterior = train_npe(
        theta_tensor=theta_tensor,
        obs_tensor=obs_tensor,
        prior=prior,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        batch_size=batch_size_train,
        learning_rate=learning_rate,
        clip_max_norm=clip_max_norm,
        validation_fraction=validation_fraction,
        stop_after_epochs=stop_after_epochs,
        max_num_epochs=max_num_epochs,
        verbose=verbose,
    )

    # --- condition posterior on observed CGM ---
    # The observation vector is the full CGM as a flat Tensor
    y_obs = torch.tensor(cgm, dtype=torch.float32)

    # Condition and sample.
    # reject_outside_prior=False avoids the slow rejection loop when the
    # flow density has low overlap with the prior box (common for small N).
    # The resulting samples may occasionally drift slightly outside the prior,
    # which is acceptable for inference; the simulate call clips bad theta.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        posterior_samples = posterior.sample(
            (n_posterior,), x=y_obs, show_progress_bars=verbose,
            reject_outside_prior=False,
        )  # (n_posterior, 8) in phi-space

    phi_post = posterior_samples.detach().cpu().numpy()

    return SBITwin(
        posterior=posterior,
        theta_post=phi_post,
        Ib=Ib,
        sample_time=run.sample_time,
        dt=dt,
        log_space=log_space,
        sensor_name=sensor_name,
        sigma=sigma,
    )
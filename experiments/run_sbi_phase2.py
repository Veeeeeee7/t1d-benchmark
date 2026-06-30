"""Experiment 2/3 — identify the **SBI / NPE** ReplayBG twin and save it.

Trains a Neural Posterior Estimator (MAF) on (theta, y) pairs simulated from the
ReplayBG forward model, conditions it on the 24 h baseline CGM, and stores the
posterior samples in ``experiments/artifacts/sbi_twin.npz``.

Observation handling
--------------------
The identification window is a single 24 h day (3 meals), i.e. ~480 CGM points
at 3 min cadence. That is short enough to feed the raw trace **directly** to the
MAF (with sbi's per-dimension z-scoring), so there is no 1-D CNN embedding net
any more — the flow conditions on the CGM series itself. The prior, simulator,
rejection-sampled training data, and the ``SBITwin`` wrapper are reused from the
verified ``t1d_twin.identify_sbi`` module.

The population/training simulations come from ReplayBG, never simglucose, so the
SBI twin gets no unfair information about the ground-truth system.

Run (from the repo root):
    python -m experiments.run_sbi            # 24 h window, 20k sims
    python -m experiments.run_sbi --smoke    # 24 h window, 300 sims (plumbing)
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import warnings

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from sbi.inference import SNPE

from . import exp_common as C
from . import output_paths as OP        # standardized layout (phase-tagged)
from t1d_twin import identify_sbi as S
from t1d_twin.identify_sbi import make_prior, generate_training_data, SBITwin

# Production vs smoke hyperparameters -------------------------------------------------
PROD = dict(n_train=20_000, max_num_epochs=400, n_posterior=2000,
            hidden=64, transforms=8,
            train_batch=512, sim_batch=1024, max_attempts=400, stop_after=30)
SMOKE = dict(n_train=300, max_num_epochs=20, n_posterior=100,
             hidden=32, transforms=4,
             train_batch=50, sim_batch=256, max_attempts=40, stop_after=10)
SIGMA = 10.0
LR = 5e-4
CLIP = 5.0


def _posterior_nn(hidden: int, transforms: int):
    """MAF density estimator conditioned on the raw (24 h) CGM, no embedding net.

    sbi z-scores the observation per-dimension by default, so the ~480-point CGM
    series is standardized and fed straight to the flow's conditioner.
    """
    try:
        from sbi.neural_nets import posterior_nn
    except Exception:                              # older/newer layout
        from sbi.utils import posterior_nn
    return posterior_nn(model="maf", hidden_features=hidden, num_transforms=transforms)


def main() -> None:
    # Each patient's SBI fit runs as its own process; when many run concurrently
    # on one node (phase-2 packs 32 patients per job, one per core), torch must
    # stay single-threaded or 32 fits each spawn a full intra-op threadpool and
    # thrash the node. OMP/MKL=1 is set in the job, but pin torch explicitly too.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast 1-day plumbing run")
    C.add_subject_args(ap)
    C.add_population_arg(ap)
    args = ap.parse_args()

    subject = C.resolve_subject(args.patient, args.patients)
    C.apply_population(args)
    hours = C.hours_for(args.smoke)
    cfg = SMOKE if args.smoke else PROD
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    print(f"[sbi] subject={subject.name}; identification run: "
          f"{hours:.0f} h @ {C.SAMPLE_TIME:.0f}-min CGM")
    run = C.subject_identification_run(subject, hours)
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=C.DT)
    Ib = float(insulin[0])
    print(f"[sbi] obs dim = {len(cgm)} CGM points; simulating {cfg['n_train']} (theta,y) pairs")

    # 1. Rejection-sampled training data from the ReplayBG simulator.
    t0 = time.time()
    theta_t, obs_t = generate_training_data(
        insulin=insulin, cho=cho, Ib=Ib, sample_time=run.sample_time, dt=C.DT,
        sigma=SIGMA, n_train=cfg["n_train"], log_space=True, seed=C.SEED,
        batch_size=cfg["sim_batch"], max_attempts=cfg["max_attempts"], verbose=True)
    print(f"[sbi] training data ready ({len(theta_t)} pairs) in {time.time() - t0:.1f} s")

    # 2. Train the NPE directly on the short raw CGM observation (no CNN).
    prior = make_prior(log_space=True)
    inference = SNPE(prior=prior,
                     density_estimator=_posterior_nn(cfg["hidden"], cfg["transforms"]))
    inference.append_simulations(theta_t, obs_t)
    t0 = time.time()
    density_estimator = inference.train(
        training_batch_size=cfg["train_batch"], learning_rate=LR,
        clip_max_norm=CLIP, validation_fraction=0.10,
        stop_after_epochs=cfg["stop_after"], max_num_epochs=cfg["max_num_epochs"],
        show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)
    print(f"[sbi] NPE trained in {time.time() - t0:.1f} s")

    # 3. Condition on the observed 24 h of CGM and draw posterior samples.
    y_obs = torch.tensor(cgm, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi = posterior.sample((cfg["n_posterior"],), x=y_obs,
                               reject_outside_prior=False,
                               show_progress_bars=True).detach().cpu().numpy()
    theta_nat = S._phi_to_theta(phi)               # log-space -> natural theta

    twin = SBITwin(posterior=None, theta_post=theta_nat, Ib=Ib,
                   sample_time=run.sample_time, dt=C.DT, log_space=False,
                   sensor_name=C.SENSOR, sigma=SIGMA)
    s = twin.summary()
    print(f"[sbi] Gb median = {s['Gb']['median']:.1f} mg/dL, "
          f"SI median = {s['SI']['median']:.2e}")
    print(f"[sbi] fit RMSE vs identification CGM = {C.fit_rmse(twin, run):.2f} mg/dL")

    path = C.save_sbi(twin, OP.twin_artifact_paths(OP.PHASE2, subject.safe_name)["sbi"])
    print(f"[sbi] saved -> {path}")


if __name__ == "__main__":
    main()
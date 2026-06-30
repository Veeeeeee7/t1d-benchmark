"""Experiment 1/3 — identify the **MCMC** ReplayBG twin and save it.

Fits the 8-parameter ReplayBG model to the 24 h baseline run by ensemble
MCMC (emcee) at production settings, then serializes the posterior to
``experiments/artifacts/mcmc_twin.npz`` for ``compute_results.py`` to reload.

Run (from the repo root):
    python -m experiments.run_mcmc            # full 24 h window, production chain
    python -m experiments.run_mcmc --smoke    # 1 day, tiny chain (plumbing)
"""
from __future__ import annotations

import os
import sys
import time
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from . import exp_common as C
from t1d_twin.identify_mcmc import identify_twin_from_run

# Production vs smoke hyperparameters -------------------------------------------------
PROD = dict(nwalkers=64, nburn=2000, nsample=6000, n_posterior=2000)
SMOKE = dict(nwalkers=16, nburn=40, nsample=80, n_posterior=100)
SIGMA = 10.0    # fixed CGM noise std in the Gaussian likelihood [mg/dL]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast 1-day plumbing run")
    C.add_subject_args(ap)
    C.add_population_arg(ap)
    args = ap.parse_args()

    subject = C.resolve_subject(args.patient, args.patients)
    C.apply_population(args)
    hours = C.hours_for(args.smoke)
    cfg = SMOKE if args.smoke else PROD

    print(f"[mcmc] subject={subject.name}; identification run: "
          f"{hours:.0f} h @ {C.SAMPLE_TIME:.0f}-min CGM")
    run = C.subject_identification_run(subject, hours)
    print(f"[mcmc] CGM samples = {len(run.cgm())}; "
          f"chain = {cfg['nwalkers']}w x ({cfg['nburn']}+{cfg['nsample']}) steps")

    t0 = time.time()
    twin = identify_twin_from_run(
        run, dt=C.DT, sigma=SIGMA, sensor_name=C.SENSOR,
        seed=C.SEED, progress=True, **cfg)
    print(f"[mcmc] identified in {time.time() - t0:.1f} s")

    s = twin.summary()
    print(f"[mcmc] Gb median = {s['Gb']['median']:.1f} mg/dL, "
          f"SI median = {s['SI']['median']:.2e}")
    print(f"[mcmc] fit RMSE vs identification CGM = {C.fit_rmse(twin, run):.2f} mg/dL")

    path = C.save_mcmc(twin, C.artifact_paths(subject)["mcmc"])
    print(f"[mcmc] saved -> {path}")


if __name__ == "__main__":
    main()
"""Phase 0, stage 1/2 — identify the **MCMC** twin on the matched-model plant.

Identical to ``run_mcmc.py`` except the identification run comes from the
self-consistent ReplayBG plant (``replaybg_plant``) instead of simglucose, so
the twin fits data drawn from its own model class (no plant<->twin mismatch).
The posterior is serialized to the per-subject artifact for ``compute_results0``.

Run (from the repo root):
    python -m experiments.run_mcmc0 --patients patients0.csv --patient rbg0001
    python -m experiments.run_mcmc0 --patients patients0.csv --patient rbg0001 --smoke
"""
from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import exp_common as C            # artifact paths + (de)serialization
from experiments import replaybg_plant as P        # the ReplayBG plant
from experiments import population as POP          # cached LS-fit population
from experiments import phase0_paths as P0        # phase0-namespaced artifact paths
from t1d_twin.identify_mcmc import identify_twin_from_run

# Same hyperparameters as the Phase 2 run_mcmc (so the only difference is the plant)
PROD = dict(nwalkers=64, nburn=2000, nsample=6000, n_posterior=2000)
SMOKE = dict(nwalkers=16, nburn=40, nsample=80, n_posterior=100)


def _resolve(patient: str, patients_csv: str) -> P.Phase0Subject:
    subs = {s.name: s for s in P.load_phase0_cohort(patients_csv)}
    if patient not in subs:
        raise KeyError(f"patient {patient!r} not in {patients_csv}")
    return subs[patient]


def _fit_rmse_ig(twin, run) -> float:
    """RMSE of the twin's noise-free IG vs the plant's noise-free IG [mg/dL]."""
    pred = twin.replay_run(run, add_noise=False)
    obs = run.bg()
    n = min(len(pred), len(obs))
    return float(np.sqrt(np.mean((pred[:n] - obs[:n]) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast tiny-chain plumbing run")
    ap.add_argument("--patients", required=True, help="phase0 cohort CSV (patients0.csv)")
    ap.add_argument("--patient", required=True, help="patient Name (a row in --patients)")
    ap.add_argument("--population", default=None,
                    help="legacy population.npz (prior is now published; see replaybg_priors) "
                         "(default: <artifacts>/population.npz)")
    args = ap.parse_args()

    # Phase 0 installs the SAME published prior as every other phase
    # (``install_published_prior`` via install_for_phase0); the population.npz
    # path is ignored for the prior now. Phase 0's controlled centre is
    # P.PHASE0_CENTER, used as the fallback centre here.
    POP.install_for_phase0(args.population or os.path.join(C.ARTIFACT_DIR, "population.npz"),
                           P.PHASE0_CENTER)

    subject = _resolve(args.patient, args.patients)
    hours = P.hours_for(args.smoke)
    cfg = SMOKE if args.smoke else PROD

    print(f"[mcmc0] subject={subject.name} (m={subject.dose_mult}); "
          f"identification run: {hours:.0f} h @ {P.SAMPLE_TIME:.0f}-min CGM")
    run = P.identification_run(subject, hours)
    print(f"[mcmc0] CGM samples = {len(run.cgm())}; "
          f"chain = {cfg['nwalkers']}w x ({cfg['nburn']}+{cfg['nsample']}) steps")

    t0 = time.time()
    twin = identify_twin_from_run(
        run, dt=P.DT, sigma=P.SIGMA, sensor_name=P.SENSOR,
        seed=P.SEED, progress=True, **cfg)
    print(f"[mcmc0] identified in {time.time() - t0:.1f} s")

    s = twin.summary()
    print(f"[mcmc0] Gb median = {s['Gb']['median']:.1f} mg/dL "
          f"(true {subject.theta[7]:.1f}); SI median = {s['SI']['median']:.2e} "
          f"(true {subject.theta[5]:.2e})")
    print(f"[mcmc0] fit RMSE vs identification IG = {_fit_rmse_ig(twin, run):.2f} mg/dL")

    path = C.save_mcmc(twin, P0.artifact_paths(subject)["mcmc"])
    print(f"[mcmc0] saved -> {path}")


if __name__ == "__main__":
    main()
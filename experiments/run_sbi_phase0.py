"""Phase 0, stage 2/2 — identify the **SBI / NPE** twin on the matched-model plant.

Identical to ``run_sbi_phase2.py`` except the identification CGM comes from the
self-consistent ReplayBG plant. The training simulations already come from the
ReplayBG forward model, so in Phase 0 the NPE is trained on exactly the
data-generating process and conditioned on an observation from it — the cleanest
possible amortized-inference case.

Run (from the repo root):
    python -m experiments.run_sbi_phase0 --patients patients0.csv --patient rbg0001
    python -m experiments.run_sbi_phase0 --patients patients0.csv --patient rbg0001 --smoke
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import warnings

import numpy as np
import torch

from sbi.inference import SNPE

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

from experiments import exp_common as C
from experiments import replaybg_plant as P
from experiments import population as POP
from experiments import output_paths as OP        # standardized layout (phase-tagged)
from t1d_twin import identify_sbi as S
from t1d_twin.identify_sbi import make_prior, generate_training_data, SBITwin

# Same hyperparameters as the Phase 2 run_sbi_phase2 (only the plant differs)
PROD = dict(n_train=20_000, max_num_epochs=400, n_posterior=2000,
            hidden=64, transforms=8,
            train_batch=512, sim_batch=1024, max_attempts=400, stop_after=30)
SMOKE = dict(n_train=300, max_num_epochs=20, n_posterior=100,
             hidden=32, transforms=4,
             train_batch=50, sim_batch=256, max_attempts=40, stop_after=10)
LR = 5e-4
CLIP = 5.0


def _resolve(patient: str, patients_csv: str) -> P.Phase0Subject:
    """Look up one Phase-0 subject by name in the cohort CSV (KeyError if absent)."""
    subs = {s.name: s for s in P.load_phase0_cohort(patients_csv)}
    if patient not in subs:
        raise KeyError(f"patient {patient!r} not in {patients_csv}")
    return subs[patient]


def _posterior_nn(hidden: int, transforms: int):
    """Build a MAF posterior-net factory for sbi, tolerant of the installed sbi version's import path."""
    try:
        from sbi.neural_nets import posterior_nn
    except Exception:
        from sbi.utils import posterior_nn
    return posterior_nn(model="maf", hidden_features=hidden, num_transforms=transforms)


def _fit_rmse_ig(twin, run) -> float:
    """RMSE between the twin's replayed noise-free IG and the observed identification IG, over their common length."""
    pred = twin.replay_run(run, add_noise=False)
    obs = run.bg()
    n = min(len(pred), len(obs))
    return float(np.sqrt(np.mean((pred[:n] - obs[:n]) ** 2)))


def main() -> None:
    """CLI: fit the Phase-0 SBI twin for one patient (ReplayBG plant) and save the posterior artifact."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast tiny-NPE plumbing run")
    ap.add_argument("--patients", required=True, help="phase0 cohort CSV (patients0.csv)")
    ap.add_argument("--patient", required=True, help="patient Name (a row in --patients)")
    ap.add_argument("--population", default=None,
                    help="legacy population.npz (prior is now published; see replaybg_priors) "
                         "(default: <artifacts>/population.npz)")
    args = ap.parse_args()

    # Phase 0 installs the SAME published prior as every other phase, so the
    # SBI training data is drawn from the published prior — consistent with
    # Phase 2. Must run before generate_training_data / make_prior read the
    # module-level support box. P.PHASE0_CENTER is the fallback centre.
    POP.install_for_phase0(args.population or os.path.join(C.ARTIFACT_DIR, "population.npz"),
                           P.PHASE0_CENTER)

    subject = _resolve(args.patient, args.patients)
    hours = P.hours_for(args.smoke)
    cfg = SMOKE if args.smoke else PROD
    torch.manual_seed(P.SEED)
    np.random.seed(P.SEED)

    print(f"[sbi0] subject={subject.name} (m={subject.dose_mult}); "
          f"identification run: {hours:.0f} h @ {P.SAMPLE_TIME:.0f}-min CGM")
    run = P.identification_run(subject, hours)
    cgm = run.cgm()
    _, insulin, cho = run.replaybg_inputs(dt=P.DT)
    Ib = float(insulin[0])
    print(f"[sbi0] obs dim = {len(cgm)} CGM points; simulating {cfg['n_train']} (theta,y) pairs")

    # 1. Rejection-sampled training data from the ReplayBG simulator.
    t0 = time.time()
    theta_t, obs_t = generate_training_data(
        insulin=insulin, cho=cho, Ib=Ib, sample_time=run.sample_time, dt=P.DT,
        sigma=P.SIGMA, n_train=cfg["n_train"], log_space=True, seed=P.SEED,
        batch_size=cfg["sim_batch"], max_attempts=cfg["max_attempts"], verbose=True)
    print(f"[sbi0] training data ready ({len(theta_t)} pairs) in {time.time() - t0:.1f} s")

    # 2. Train the NPE on the raw 24 h CGM observation.
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
    print(f"[sbi0] NPE trained in {time.time() - t0:.1f} s")

    # 3. Condition on the observed CGM and draw posterior samples.
    y_obs = torch.tensor(cgm, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi = posterior.sample((cfg["n_posterior"],), x=y_obs,
                               reject_outside_prior=False,
                               show_progress_bars=True).detach().cpu().numpy()
    theta_nat = S._phi_to_theta(phi)

    twin = SBITwin(posterior=None, theta_post=theta_nat, Ib=Ib,
                   sample_time=run.sample_time, dt=P.DT, log_space=False,
                   sensor_name=P.SENSOR, sigma=P.SIGMA)
    s = twin.summary()
    print(f"[sbi0] Gb median = {s['Gb']['median']:.1f} mg/dL "
          f"(true {subject.theta[7]:.1f}); SI median = {s['SI']['median']:.2e} "
          f"(true {subject.theta[5]:.2e})")
    print(f"[sbi0] fit RMSE vs identification IG = {_fit_rmse_ig(twin, run):.2f} mg/dL")

    path = C.save_sbi(twin, OP.twin_artifact_paths(OP.PHASE0, subject.safe_name)["sbi"])
    print(f"[sbi0] saved -> {path}")


if __name__ == "__main__":
    main()
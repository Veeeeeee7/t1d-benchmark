"""Shared configuration, scenario builder, and twin (de)serialization for the
production decision-targeted twinning experiment.

Data cadence (chosen here, used everywhere)
-------------------------------------------
CGM is sampled at simglucose's native Dexcom rate: **sample_time = 3 min**.
The identification window is **24 h** (1 day, 3 meals), therefore

    24 h * 60 min / 3 min = 480 CGM samples,

driven on a dt = 1 min ReplayBG integration grid (1440 steps). All three
twinning methods identify from the *same* 24 h baseline run, and every
candidate therapy is evaluated on the *same* meal schedule (only the controller
changes), so any difference in the results is attributable to the twinning
method, not to the data. (``WEEK_HOURS`` is a backward-compat alias for this
window; the horizon used to be a full week.)

This module is imported by ``run_mcmc.py``, ``run_sbi.py``, ``compute_results.py``
and the Phase 0/1/2 drivers; it never trains anything itself.
"""
from __future__ import annotations

import os
import sys
import argparse
import datetime

# --- path bootstrap so `t1d_twin` and `experiments` import from the repo root,
#     whether a script is run as `python -m experiments.x` or `python x.py` ---
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from simglucose.simulation.scenario import CustomScenario

from t1d_twin.policies import ModulatedBBController
from t1d_twin.experiment import make_experiment_policies
from t1d_twin.simglucose_adapter import run_policy

# ---------------------------------------------------------------------------
# Cadence / horizon / identity
# ---------------------------------------------------------------------------
SAMPLE_TIME = 3.0        # CGM sampling interval [min] — Dexcom native
DT = 1.0                 # ReplayBG integration step [min]
WINDOW_HOURS = 24.0      # production identification window: 1 day, 3 meals
SMOKE_HOURS = 24.0       # 1 day (used by --smoke for fast plumbing checks)
WEEK_HOURS = WINDOW_HOURS  # backward-compat alias (now the 24h window, not a week)
SEED = 1
PATIENT = "adult#001"
SENSOR = "Dexcom"
START = datetime.datetime(2024, 1, 1, 0, 0, 0)

# ---------------------------------------------------------------------------
# Output locations
#
# The standardized on-disk layout lives in ``experiments.output_paths`` (a
# pure-stdlib module the simglucose-free Phase 0 path can also import). We
# re-export the constants/helpers here so the simglucose-side code can keep
# using ``C.ARTIFACT_DIR`` / ``C.results_dir_for`` etc. unchanged. Every
# generated output (artifacts, results, logs) lives under one root; override
# with T1D_OUTPUT_ROOT, e.g.:  T1D_OUTPUT_ROOT=./_out python -m experiments.run_phase1
# ---------------------------------------------------------------------------
from . import output_paths as _OP

OUTPUT_ROOT = _OP.OUTPUT_ROOT
ARTIFACT_DIR = _OP.ARTIFACT_DIR
RESULTS_DIR = _OP.RESULTS_DIR
LOG_DIR = _OP.LOG_DIR

# Phase tags + layout helpers, re-exported for convenience (C.PHASE2, etc.).
PHASE0, PHASE0_ML, PHASE1, PHASE2, PREP = (
    _OP.PHASE0, _OP.PHASE0_ML, _OP.PHASE1, _OP.PHASE2, _OP.PREP)
population_path = _OP.population_path
dataset_path = _OP.dataset_path
summary_path = _OP.summary_path
per_patient_path = _OP.per_patient_path
log_dir = _OP.log_dir

# Single-subject legacy defaults (per-subject paths are passed explicitly in
# practice via ``artifact_paths``; these only back the no-arg save/load calls).
MCMC_PATH = os.path.join(ARTIFACT_DIR, "mcmc_twin.npz")
SBI_PATH = os.path.join(ARTIFACT_DIR, "sbi_twin.npz")


def hours_for(smoke: bool) -> float:
    return SMOKE_HOURS if smoke else WINDOW_HOURS


# ---------------------------------------------------------------------------
# Meal schedule + scenario + identification run
# ---------------------------------------------------------------------------
def meal_plan(hours: float, seed: int = SEED):
    """Deterministic 3-meals-a-day schedule as ``(hour, grams)`` tuples.

    Breakfast/lunch/dinner anchored at 7:00 / 12:30 / 19:00 with small seeded
    jitter in timing (+/- 0.5 h) and size (+/- 15 %), repeated each day. The
    midnight start leaves a fasting lead-in before the first meal so the
    steady-state initial condition at t=0 is valid.
    """
    rng = np.random.default_rng(seed)
    anchors = [(7.0, 50.0), (12.5, 70.0), (19.0, 80.0)]
    meals = []
    n_days = int(np.ceil(hours / 24.0))
    for d in range(n_days):
        for h0, g0 in anchors:
            h = d * 24.0 + h0 + float(rng.uniform(-0.5, 0.5))
            g = float(g0 * rng.uniform(0.85, 1.15))
            if 0.0 < h < hours:
                meals.append((round(h, 3), round(g, 1)))
    return sorted(meals)


def weekly_scenario(hours: float, seed: int = SEED):
    """Return ``(CustomScenario, meal_list)`` for the given horizon."""
    meals = meal_plan(hours, seed)
    return CustomScenario(start_time=START, scenario=meals), meals


def identification_run(hours: float, seed: int = SEED):
    """Run the baseline therapy on the 24 h scenario -> the RunResult every twin
    is identified from. Deterministic, so each script reconstructs the same run.
    """
    scenario, _ = weekly_scenario(hours, seed)
    base = ModulatedBBController(bolus_factor=1.0, basal_factor=1.0)
    return run_policy(base, scenario, hours=hours, patient_name=PATIENT,
                      sensor_name=SENSOR, sensor_seed=seed)


def policy_set(smoke: bool = False):
    """The candidate set Pi (production: 10 policies, interior optimum ~bolus_x2.0)."""
    if smoke:
        return make_experiment_policies(
            bolus_factors=(0.85, 1.0, 1.5, 2.0, 2.5), basal_factors=())
    return make_experiment_policies()


# ---------------------------------------------------------------------------
# Twin (de)serialization — fit once, reload for scoring
# ---------------------------------------------------------------------------
# Reconstructs a *working* twin (one that can predict_ig / generate_cgm) from a
# minimal artifact, without re-running identification. For the posterior twins
# the conditioned posterior samples are all predict_ig needs; the SBI flow is
# not required at scoring time (the identification observation is fixed).

def save_mcmc(twin, path: str = MCMC_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, theta_post=twin.theta_post, Ib=twin.Ib,
             sample_time=twin.sample_time, dt=twin.dt, sigma=twin.sigma,
             sensor_name=twin.sensor_name)
    return path


def load_mcmc(path: str = MCMC_PATH):
    from t1d_twin.identify_mcmc import MCMCTwin
    d = np.load(path, allow_pickle=True)
    return MCMCTwin(theta_post=d["theta_post"], Ib=float(d["Ib"]),
                    sample_time=float(d["sample_time"]), dt=float(d["dt"]),
                    sensor_name=str(d["sensor_name"]), sigma=float(d["sigma"]))


def save_sbi(twin, path: str = SBI_PATH) -> str:
    # twin.theta_post is in NATURAL theta space (SBITwin converts on init).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, theta_post=twin.theta_post, Ib=twin.Ib,
             sample_time=twin.sample_time, dt=twin.dt, sigma=twin.sigma,
             sensor_name=twin.sensor_name)
    return path


def load_sbi(path: str = SBI_PATH):
    from t1d_twin.identify_sbi import SBITwin
    d = np.load(path, allow_pickle=True)
    # stored in natural space -> log_space=False so it is used as-is
    return SBITwin(posterior=None, theta_post=d["theta_post"], Ib=float(d["Ib"]),
                   sample_time=float(d["sample_time"]), dt=float(d["dt"]),
                   log_space=False, sensor_name=str(d["sensor_name"]),
                   sigma=float(d["sigma"]))


def fit_rmse(twin, run) -> float:
    """RMSE of the twin's replayed CGM vs the identification CGM [mg/dL]."""
    pred = twin.replay_run(run, seed=SEED)
    obs = run.cgm()
    n = min(len(pred), len(obs))
    return float(np.sqrt(np.mean((pred[:n] - obs[:n]) ** 2)))


# ---------------------------------------------------------------------------
# Candidate-therapy factor sets (shared by every method, via the Subject)
# ---------------------------------------------------------------------------
# Grid rationale:
#   * x2.5 / x3.0 dropped: both sit on the deep-hypo reward floor, redundant
#     with x2.0 as the "clearly over-dosed" anchor and they inflate the spread.
#   * basal 0.85 / 1.2 dropped: their reward collides with bolus 0.85 / 1.2
#     (within sensor noise), creating non-informative ties; kept basal points
#     are clearly off-baseline so they separate.
#   * bolus 1.3 / 1.6 added: the over-dose flank is steep and well-resolved.
#   * Brackets the carb-counting-error optima. With the dose multiplier
#     m in {0.75, 0.85, 1.25, 1.50} (patients.DEFAULT_CARB_ERROR_CHOICES) the
#     optimal corrective bolus is f* ~= m, so every optimum lands strictly
#     inside (0.7, 2.0) with grid neighbours on BOTH sides:
#         m=0.75 -> between 0.70 and 0.85 ; m=0.85 -> on node (0.70 | 1.00)
#         m=1.25 -> between 1.20 and 1.30 ; m=1.50 -> on node (1.30 | 1.60)
#     This is the interiority discipline the truncation bug violated; keep any
#     change to m and to this grid in lock-step so no optimum hits an edge.
PROD_BOLUS = (0.7, 0.85, 1.0, 1.2, 1.3, 1.5, 1.6, 2.0)
PROD_BASAL = (0.7, 1.3)
SMOKE_BOLUS = (0.85, 1.0, 1.5, 2.0, 2.5)
SMOKE_BASAL = ()
SEEN_BAND = 0.2


def factors_for(smoke: bool):
    return (SMOKE_BOLUS, SMOKE_BASAL) if smoke else (PROD_BOLUS, PROD_BASAL)


def seen_unseen(bolus_factors, basal_factors):
    """Names split into seen (within SEEN_BAND of baseline) / unseen."""
    seen, unseen = [], []
    for f in bolus_factors:
        (seen if abs(f - 1.0) <= SEEN_BAND else unseen).append(f"bolus_x{f:.2f}")
    for f in basal_factors:
        (seen if abs(f - 1.0) <= SEEN_BAND else unseen).append(f"basal_x{f:.2f}")
    return seen, unseen


# ---------------------------------------------------------------------------
# Subject resolution + per-subject output locations
# ---------------------------------------------------------------------------
def add_subject_args(ap: argparse.ArgumentParser) -> None:
    """Standard CLI flags so every script can target a patient from patients.csv."""
    ap.add_argument("--patients", default=None,
                    help="path to a patients.csv (synthetic subjects)")
    ap.add_argument("--patient", default=None,
                    help="patient Name to run (a row in --patients, or a base "
                         "simglucose name); default adult#001")


def add_population_arg(ap: argparse.ArgumentParser) -> None:
    """Prior + centre controls.

    By default the **published** ReplayBG prior is installed automatically
    (``population.install_published_prior``), so no run silently uses a hardcoded
    centre. The legacy cohort/``--population`` flags below still load or build a
    (now ignored) population artifact but no longer determine the prior; they are
    escape hatches, mainly for tests / debugging.
    """
    ap.add_argument("--population", default=None,
                    help="population.npz to load (or to cache a fresh derivation "
                         "into). Default: <ARTIFACT_DIR>/population.npz")
    ap.add_argument("--pop-theta", default=None,
                    help="bypass derivation and install this theta directly: a "
                         "JSON dict, or 8 comma/space-separated values in order "
                         f"{','.join(_THETA_ORDER)}. For tests / debugging.")
    ap.add_argument("--no-population", action="store_true",
                    help="bypass derivation without installing a theta; a theta "
                         "must already be set (via --pop-theta earlier or "
                         "replaybg_model.set_pop_theta) or the run errors.")
    ap.add_argument("--pop-hours", type=float, default=WINDOW_HOURS,
                    help="baseline horizon used only when fitting from scratch "
                         "(no prep cache); prep itself fits at 24 h")
    ap.add_argument("--pop-jobs", type=int, default=0,
                    help="worker processes for the population fit (0 = auto)")


_THETA_ORDER = ("ka2", "kd", "kempt", "kabs", "SG", "SI", "p2", "Gb")


def _parse_pop_theta(raw: str) -> dict:
    """Parse --pop-theta (JSON dict or 8 comma/space-separated values)."""
    raw = raw.strip()
    if raw.startswith("{"):
        import json
        return json.loads(raw)
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if len(vals) != len(_THETA_ORDER):
        raise SystemExit(
            f"--pop-theta needs {len(_THETA_ORDER)} values "
            f"({','.join(_THETA_ORDER)}); got {len(vals)}")
    return dict(zip(_THETA_ORDER, vals))


def _population_cohort(args):
    """Resolve the cohort to derive the population from.

    Uses ``--patients`` CSV if given, otherwise the base simglucose cohort
    (``--pop-kind`` if present, else adults).
    """
    from . import patients as PT
    csv = getattr(args, "patients", None)
    if csv:
        return PT.load_subjects_csv(csv)
    kind = getattr(args, "pop_kind", None) or "adult"
    return [PT.subject_from_base(n) for n in PT.list_patients(kind)]


def apply_population(args, subjects=None) -> bool:
    """Install the population centre+prior before any twin fit.

    Default behaviour is automatic: install the published prior (the cohort
    load/build still runs for back-compat but is ignored for the prior).
    Escape hatches: ``--pop-theta`` installs a theta directly and
    ``--no-population`` requires a theta to be set already (both for
    tests / debugging). Returns True if the prior was installed here.
    """
    import t1d_twin.replaybg_model as RB
    from . import population as POP

    # 1. explicit manual theta (tests / debugging)
    raw = getattr(args, "pop_theta", None)
    if raw:
        RB.set_pop_theta(_parse_pop_theta(raw))
        return False

    # 2. explicit opt-out: a theta must already be installed
    if getattr(args, "no_population", False):
        if not RB.pop_theta_is_set():
            raise SystemExit(
                "--no-population was given but no theta is set; pass --pop-theta "
                "or call replaybg_model.set_pop_theta(...) first.")
        return False

    # 3. automatic load-or-build (default): prefer the prep population.npz, then
    #    the cached rbg_* fits in patients.csv; only fit from scratch with no prep.
    path = getattr(args, "population", None) or population_path()
    csv = getattr(args, "patients", None)
    if subjects is None and not csv:
        subjects = _population_cohort(args)      # one-off run with no cohort CSV
    return POP.ensure_population(
        path=path, patients_csv=csv, subjects=subjects,
        hours=getattr(args, "pop_hours", WINDOW_HOURS),
        jobs=(getattr(args, "pop_jobs", 0) or None),
    )


def resolve_subject(patient: str | None = None, patients_csv: str | None = None):
    """Return a ``patients.Subject`` for the requested patient.

    * ``--patients CSV`` + ``--patient NAME`` -> that synthetic subject.
    * ``--patient NAME`` only -> a built-in simglucose patient.
    * neither -> the default (adult#001), which reproduces the original runs.
    """
    from . import patients as PT
    if patients_csv:
        subs = {s.name: s for s in PT.load_subjects_csv(patients_csv)}
        if patient is None:
            raise ValueError("--patients given without --patient; pass a Name "
                             "(or use run_suite to iterate all rows)")
        if patient not in subs:
            raise KeyError(f"patient {patient!r} not in {patients_csv}")
        return subs[patient]
    return PT.subject_from_base(patient or PATIENT)


def artifact_paths(subject) -> dict:
    """Per-subject Phase 2 twin-artifact paths under ``artifacts/phase2/<name>/``.

    Phase 0 has its own (``phase0_paths.artifact_paths``); both delegate to the
    shared ``output_paths`` layout so the two phases sit side by side under
    ``artifacts/`` and never clobber each other (they share patient names).
    """
    return _OP.twin_artifact_paths(_OP.PHASE2, subject.safe_name)


def results_dir_for(subject) -> str:
    """Per-subject Phase 2 results dir: ``results/phase2/<name>/``."""
    return _OP.results_dir(_OP.PHASE2, subject.safe_name)


# ---------------------------------------------------------------------------
# Subject-aware identification run + ground truth (ExplicitBBController therapies)
# ---------------------------------------------------------------------------
def subject_identification_run(subject, hours: float, seed: int = SEED):
    """The baseline 24 h run for this subject (every twin identifies from it)."""
    scenario, _ = weekly_scenario(hours, seed)
    return subject.run(subject.baseline_controller(), scenario, hours, sensor_seed=seed)


def subject_ground_truth(subject, hours: float, bolus_factors, basal_factors=(),
                         seed: int = SEED):
    """Run the candidate therapies on this subject -> ``(true_runs, controllers)``."""
    scenario, _ = weekly_scenario(hours, seed)
    controllers = subject.therapy_controllers(bolus_factors, basal_factors)
    true_runs = {name: subject.run(c, scenario, hours, sensor_seed=seed)
                 for name, c in controllers.items()}
    return true_runs, controllers
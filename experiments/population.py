"""
Per-patient ReplayBG MAP fit and prior install.

Each simglucose patient is mapped **into** ReplayBG parameter space by a
**MAP fit** of the 8-parameter ReplayBG model to that patient's baseline CGM
(``fit_replaybg`` — penalised least-squares == maximum-a-posteriori under a
Gaussian CGM likelihood and the published prior in ``replaybg_priors``). That
per-patient fit is computed **once** in prep by ``derive_replaybg_params`` (the
``rbg_*`` columns of ``patients.csv``, at the 24 h identification horizon) and
reused everywhere — the Phase 0 matched plant, the Phase 1 regression target,
and any twin warm-start.

Prior install (changed 2026-06-27 — see ``MAP_published_prior_refactor_6_27.md``)
--------------------------------------------------------------------------------
The prior is no longer derived from the cohort. ``install_published_prior``
installs the published ReplayBG prior (``replaybg_priors``) into every twin:
the centre theta is the published median (``set_pop_theta``), and the modules'
``PRIOR_LO/HI`` are set to the published *support* envelope (used only to clamp
walkers / bound rejection). The informative prior itself lives in
``replaybg_priors`` and is consumed directly by the MCMC log-prior and the SBI
prior. ``install_population`` is now a back-compat shim that delegates here and
ignores any cohort ``Population``.

Leakage: the prior now carries **no** cohort or simglucose information (it is
literature), and every twin still trains only on ReplayBG forward-model
simulations — none sees simglucose. The per-patient label is a MAP projection of
that patient's own CGM, which discards the "Phi not in F" residual, so only
ReplayBG-coordinate info crosses.

VESTIGIAL: the cohort-aggregation machinery below (``Population.bounds`` /
``mean_theta`` / ``median_theta``, ``population_from_fits``,
``derive_population``, the ``population.npz`` save/load and CLI ``main``) no
longer determines the prior or centre. It is retained so the phase runners do
not break on import and can be removed once tests pass — see the removal
checklist in ``MAP_published_prior_refactor_6_27.md``.

Run (from the repo root):
    python -m experiments.derive_replaybg_params --patients patients.csv   # MAP fits
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

import warnings

import numpy as np
from scipy.optimize import least_squares

from t1d_twin import replaybg_model as M
from t1d_twin.replaybg_model import THETA_NAMES, sample_indices
from t1d_twin.identify_mcmc import _theta_to_phi, _phi_to_theta
from t1d_twin import replaybg_priors as PR

from experiments import exp_common as C

# Optimiser support envelope (generous; NOT a prior). The published informative
# prior in ``replaybg_priors`` does the regularising — this only bounds the MAP
# search so a fit can never leave the physical region. Kept under the old names
# for backward compatibility with any external import.
WIDE_LO = PR.SUPPORT_LO
WIDE_HI = PR.SUPPORT_HI


# ===========================================================================
# Per-patient ReplayBG fit (MAP / penalised least-squares)
# ===========================================================================
# Residual scale [mg/dL] that sets the data-vs-prior balance in the MAP fit.
# The fit targets the noise-free IG (``RunResult.bg()``), so this is NO LONGER
# the sensor-noise std — it is the assumed residual scale, now dominated by
# ReplayBG model mismatch ("Phi not in F") rather than CGM noise. Kept at 10 so
# the prior's influence on the label is unchanged from the CGM-target version;
# tune deliberately if you want the label to track the clean signal more tightly.
MAP_SIGMA = 10.0


def _residual(phi, insulin, cho, Ib, target, dt, obs_idx, sigma):
    """
    MAP residual: sigma-scaled IG misfit stacked with the published-prior
    penalty, so least-squares minimises the negative-log-posterior. ``target`` is
    the noise-free interstitial glucose (``RunResult.bg()``), so the fit is
    IG-vs-IG — consistent with the IG-vs-IG evaluation metrics.
    """
    theta = _phi_to_theta(phi)
    _, ig = M.simulate(theta, insulin, cho, Ib, dt=dt)
    data_res = (ig[obs_idx] - target) / sigma
    return np.concatenate([data_res, PR.prior_residuals(theta)])


def fit_replaybg(run, dt: float = 1.0, max_nfev: int = 80, sigma: float = MAP_SIGMA):
    """
    MAP ReplayBG parameters for one patient run -> (theta(8,), ig_rmse).

    Penalised least-squares == MAP under a Gaussian likelihood and the published
    (Cappon et al.) prior in ``replaybg_priors``. The fit targets the **noise-free
    interstitial glucose** (``RunResult.bg()``), not the noisy CGM, so the label
    is an IG-vs-IG projection — consistent with the evaluation in ``evaluate.py``
    (which scores twins on ``bg()`` with sensor noise stripped). The optimiser
    starts at the published prior centre, which regularises the non-identified
    directions and gives a physiological, reproducible label.

    The returned RMSE is the **IG-only** fit error in mg/dL (data residuals, prior
    penalty excluded) so it stays comparable across the cohort.
    """
    _, insulin, cho = run.replaybg_inputs(dt=dt)
    Ib = float(insulin[0])
    target = np.asarray(run.bg(), dtype=float)   # noise-free IG (matches evaluate.py)
    T = len(insulin)
    obs_idx = sample_indices(len(target), run.sample_time, dt)
    keep = obs_idx < T
    obs_idx, target = obs_idx[keep], target[keep]

    lo, hi = _theta_to_phi(WIDE_LO), _theta_to_phi(WIDE_HI)
    theta0 = np.clip(M.theta_to_array(PR.prior_center()), WIDE_LO, WIDE_HI)
    phi0 = _theta_to_phi(theta0)
    res = least_squares(_residual, phi0, bounds=(lo, hi), method="trf",
                        max_nfev=max_nfev,
                        args=(insulin, cho, Ib, target, dt, obs_idx, sigma))
    theta = _phi_to_theta(res.x)
    n_obs = len(target)
    ig_resid = res.fun[:n_obs] * sigma           # undo sigma scaling -> mg/dL
    rmse = float(np.sqrt(np.mean(ig_resid ** 2)))
    if not PR.order_ok(theta):
        warnings.warn(f"MAP fit violated ordering constraints: theta={theta}",
                      RuntimeWarning, stacklevel=2)
    return theta, rmse


# ===========================================================================
# Population container
# ===========================================================================
@dataclass
class Population:
    thetas: np.ndarray            # (N, 8) fitted ReplayBG params, natural space
    members: list                 # patient names
    rmses: np.ndarray             # (N,) fit RMSE per patient

    def median_theta(self) -> dict:
        med = np.median(self.thetas, axis=0)
        return {n: float(v) for n, v in zip(THETA_NAMES, med)}

    def mean_theta(self, log_rates: bool = True) -> dict:
        """
        [VESTIGIAL] Cohort centre = average of the per-patient MAP fits.

        No longer used for the prior/centre (the published prior centre is
        installed instead; see module docstring). Retained for diagnostics.

        This is the centre that ``install_population`` installs via
        ``replaybg_model.set_pop_theta``:
        for each of the 8 ReplayBG parameters we average the per-patient
        regression fits across the cohort ("averaged across").

        ``log_rates`` controls how the seven rate parameters (indices 0-6) are
        averaged. With ``log_rates=True`` (default) they are averaged in log
        space — i.e. a geometric mean, ``exp(mean(log(theta)))`` — which is the
        consistent notion of "average" everywhere else in this pipeline (the
        prior, the MCMC/SBI transform ``_theta_to_phi``, and ``bounds()`` all
        treat the rates in log space) and avoids the upward bias of arithmetic-
        averaging rate constants that span an order of magnitude. ``Gb`` (index
        7) is always averaged arithmetically. Pass ``log_rates=False`` for a
        plain arithmetic mean of every parameter.
        """
        if log_rates:
            avg = self.thetas.mean(axis=0).copy()                # arithmetic (Gb)
            n = PR.N_LOG_RATES
            avg[:n] = np.exp(np.log(self.thetas[:, :n]).mean(axis=0))  # geometric (rates)
        else:
            avg = self.thetas.mean(axis=0)
        return {n: float(v) for n, v in zip(THETA_NAMES, avg)}

    def bounds(self, q=(5.0, 95.0), margin: float = 0.25):
        """
        [VESTIGIAL] Per-parameter (lo, hi) from cohort percentiles, widened by
        ``margin`` (in log space for rates) and clipped to the wide envelope.

        No longer feeds the prior/centre after the MAP refactor; retained only
        for the diagnostic SI/Gb range print in ``main()``.
        """
        lo = np.percentile(self.thetas, q[0], axis=0)
        hi = np.percentile(self.thetas, q[1], axis=0)
        # widen: rates (leading N_LOG_RATES) multiplicatively, Gb additively
        n = PR.N_LOG_RATES
        lo7 = lo[:n] * np.exp(-margin)
        hi7 = hi[:n] * np.exp(+margin)
        gb_lo = lo[n] - margin * (hi[n] - lo[n] + 1.0)
        gb_hi = hi[n] + margin * (hi[n] - lo[n] + 1.0)
        out_lo = np.concatenate([lo7, [gb_lo]])
        out_hi = np.concatenate([hi7, [gb_hi]])
        return np.clip(out_lo, WIDE_LO, WIDE_HI), np.clip(out_hi, WIDE_LO, WIDE_HI)


def save_population(pop: Population, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, thetas=pop.thetas, members=np.array(pop.members, dtype=object),
             rmses=pop.rmses)
    return path


def load_population(path: str) -> Population:
    d = np.load(path, allow_pickle=True)
    return Population(thetas=d["thetas"], members=list(d["members"]), rmses=d["rmses"])


# Columns written by derive_replaybg_params (the single per-patient fit pass).
RBG_THETA_COLS = [f"rbg_{n}" for n in THETA_NAMES]
RBG_RMSE_COL = "rbg_fit_rmse"


def csv_has_fits(patients_csv: str) -> bool:
    """
    Whether ``patients_csv`` carries the cached ReplayBG fit columns.
    """
    import csv as _csv
    try:
        with open(patients_csv, newline="") as fh:
            header = next(_csv.reader(fh), [])
    except OSError:
        return False
    return all(c in header for c in RBG_THETA_COLS)


def population_from_fits(patients_csv: str, exclude: str | None = None,
                         limit: int | None = None, verbose: bool = True) -> Population:
    """
    Build a Population from the cached per-patient ReplayBG fits.

    Reads the ``rbg_*`` columns that ``derive_replaybg_params`` appended to
    ``patients.csv`` and aggregates them — **no re-fitting**. This is the single
    source of per-patient fits for the whole experiment: prep fits each patient
    once (``derive_replaybg_params``, 24 h identification horizon), and every
    consumer reuses those same parameters — the Phase 0 matched plant, the
    Phase 1 regression target, and this population (centre + prior).
    """
    import pandas as pd
    df = pd.read_csv(patients_csv)
    missing = [c for c in RBG_THETA_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{patients_csv} is missing {missing}. Run the prep fit first: "
            "python -m experiments.derive_replaybg_params --patients "
            f"{patients_csv}")
    df = df[df[RBG_THETA_COLS].notna().all(axis=1)].copy()   # keep fitted rows
    if exclude:
        df = df[df["Name"].astype(str) != str(exclude)]
    if limit:
        df = df.iloc[:limit]
    if len(df) == 0:
        raise ValueError(f"{patients_csv} has no fitted patients (all rbg_* NaN)")

    thetas = df[RBG_THETA_COLS].to_numpy(dtype=float)
    members = df["Name"].astype(str).tolist()
    rmses = (df[RBG_RMSE_COL].to_numpy(dtype=float)
             if RBG_RMSE_COL in df.columns else np.full(len(members), np.nan))
    if verbose:
        print(f"[population] aggregating {len(members)} cached ReplayBG fits "
              f"from {patients_csv} (no re-fit)")
    return Population(thetas=thetas, members=members, rmses=rmses)


def n_workers(requested: int | None = None) -> int:
    """
    Resolve worker count: explicit --jobs > SLURM_CPUS_PER_TASK > cpu_count.

    ``requested <= 0`` (the default) auto-detects; on a SLURM node this picks up
    the whole allocation via ``SLURM_CPUS_PER_TASK``.
    """
    if requested and requested > 0:
        return int(requested)
    for var in ("SLURM_CPUS_PER_TASK", "POP_JOBS"):
        v = os.environ.get(var)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return os.cpu_count() or 1


def _fit_one_subject(task):
    """
    Worker: one subject's baseline run + ReplayBG MAP fit.

    Module-scope so the process pool can pickle it. The scenario is rebuilt
    inside the worker (it is deterministic in ``(hours, seed)``), which avoids
    shipping a CustomScenario across the process boundary.
    """
    s, hours, seed, dt = task
    from experiments.exp_common import weekly_scenario
    scenario, _ = weekly_scenario(hours, seed)
    run = s.run(s.baseline_controller(), scenario, hours, sensor_seed=seed)
    theta, rmse = fit_replaybg(run, dt=dt)
    return (s.name, np.asarray(theta, dtype=float), float(rmse))


def derive_population(subjects, hours: float = 12.0, seed: int = 1,
                      dt: float = 1.0, verbose: bool = True,
                      jobs: int | None = None) -> Population:
    """
    Fit ReplayBG to each subject's baseline run -> Population.

    Each patient is independent, so the per-patient sim+fit is fanned across
    ``jobs`` worker processes (``jobs<=0`` -> auto from SLURM/cpu_count). Results
    are gathered in input order, so ``thetas``/``members`` line up with
    ``subjects``.
    """
    n = len(subjects)
    workers = n_workers(jobs)
    tasks = [(s, hours, seed, dt) for s in subjects]
    results = [None] * n
    t0 = time.time()

    def _note(i, res):
        if verbose and ((i + 1) % 10 == 0 or n <= 12):
            print(f"  [{i + 1}/{n}] {res[0]}: fit RMSE={res[2]:.1f} mg/dL "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    if workers <= 1 or n <= 1:
        for i, task in enumerate(tasks):
            results[i] = _fit_one_subject(task)
            _note(i, results[i])
    else:
        if verbose:
            print(f"  [parallel] fitting {n} patients across {workers} worker(s)", flush=True)
        ctx = mp.get_context("fork")          # Linux: children inherit imports + globals
        with ctx.Pool(processes=workers) as pool:
            # imap preserves input order, so results[i] <-> subjects[i]
            for i, res in enumerate(pool.imap(_fit_one_subject, tasks, chunksize=1)):
                results[i] = res
                _note(i, res)

    members = [r[0] for r in results]
    thetas  = [r[1] for r in results]
    rmses   = [r[2] for r in results]
    return Population(thetas=np.array(thetas), members=members, rmses=np.array(rmses))


# ===========================================================================
# Install the population into the twin modules (mutates module-level globals)
# ===========================================================================
def install_published_prior(verbose: bool = True) -> None:
    """
    Install the published (Cappon et al.) prior + centre into the twin modules.

    Replaces the former cohort-derived empirical-Bayes prior. The informative
    prior itself lives in ``t1d_twin.replaybg_priors`` and is consumed directly
    by the MCMC log-prior and the SBI prior; here we only (a) set the population
    centre to the published median (MCMC walker init / Phase-0 fallback) and
    (b) point each module's SUPPORT box at the published support envelope so
    walker clamping and rejection share the same physical bounds.
    """
    import t1d_twin.replaybg_model as RB
    import t1d_twin.identify_mcmc as MC
    import t1d_twin.identify_sbi as SB

    RB.set_pop_theta(PR.prior_center())
    MC.PRIOR_LO, MC.PRIOR_HI = PR.SUPPORT_LO.copy(), PR.SUPPORT_HI.copy()
    if hasattr(MC, "_PRIOR_LO"):
        MC._PRIOR_LO, MC._PRIOR_HI = PR.SUPPORT_LO.copy(), PR.SUPPORT_HI.copy()
    SB.PRIOR_LO, SB.PRIOR_HI = PR.SUPPORT_LO.copy(), PR.SUPPORT_HI.copy()
    if verbose:
        c = PR.prior_center()
        print(f"[population] installed PUBLISHED prior "
              f"(SI={c['SI']:.2e}, Gb={c['Gb']:.0f}); "
              f"cohort no longer derives the prior")


def install_population(pop: "Population | None" = None, margin: float = 0.25,
                       verbose: bool = True) -> None:
    """
    Back-compat shim. The prior is now the published one (``replaybg_priors``),
    independent of any cohort, so ``pop`` and ``margin`` are ignored — kept only
    so existing callers (the phase runners) don't break.
    """
    install_published_prior(verbose=verbose)


def maybe_install(path: str | None, margin: float = 0.25) -> bool:
    """
    Install a saved population if ``path`` is given; return whether it was.
    """
    if not path:
        return False
    install_population(load_population(path), margin=margin)
    return True


def install_for_phase0(path: str | None, fallback_center, margin: float = 0.25,
                       verbose: bool = True) -> bool:
    """
    Phase-0 prior install (centre + support bounds).

    Now installs the **published** prior (centre = published median, support box)
    via ``install_population``/``install_published_prior``, the same prior every
    other phase uses — the ``path``/``population.npz`` argument no longer affects
    the prior and is kept only for signature compatibility. If ``path`` is absent
    it sets ``fallback_center`` directly (a no-prep escape hatch); note Phase 0's
    own controlled centre is normally ``replaybg_plant.PHASE0_CENTER``, so verify
    which centre is in force before relying on Phase 0 output.
    """
    import t1d_twin.replaybg_model as RB
    if path and os.path.exists(path):
        install_population(load_population(path), margin=margin, verbose=verbose)
        return True
    if verbose:
        print("[population] no population.npz; Phase 0 using fixed centre + wide bounds")
    RB.set_pop_theta(fallback_center)
    return False


def ensure_population(path: str | None = None, patients_csv: str | None = None,
                      subjects=None, margin: float = 0.25,
                      hours: float = 24.0, jobs: int | None = None,
                      force: bool = False, verbose: bool = True) -> bool:
    """
    Guarantee a prior + centre is installed before any twin fit.

    Since 2026-06-27 the **installed prior is always the published one**
    (``install_population`` is a shim for ``install_published_prior``); the
    cohort load/aggregate steps below still run but no longer determine the
    prior or centre. The resolution order only decides *whether* a (now
    ignored) ``Population`` object gets built/cached along the way:
      1. A manual theta already set and ``force`` is False -> leave it
         (respects tests / Phase-0 overrides), nothing installed.
      2. ``path`` exists and ``force`` is False -> load (cache hit), then
         install the published prior.
      3. ``patients_csv`` carries cached ``rbg_*`` fits -> aggregate them, then
         install the published prior.
      4. Otherwise derive by fitting ``subjects`` (last resort), then install.

    Returns True if the prior was installed here, False if a manual theta was
    already in force.
    """
    import t1d_twin.replaybg_model as RB

    if not force and RB.pop_theta_is_set():
        if verbose:
            print("[population] a theta is already set; skipping derivation")
        return False

    if path and os.path.exists(path) and not force:
        if verbose:
            print(f"[population] loading cached population: {path}")
        install_population(load_population(path), margin=margin, verbose=verbose)
        return True

    if patients_csv and csv_has_fits(patients_csv):
        pop = population_from_fits(patients_csv, verbose=verbose)
    elif subjects:
        if verbose:
            print(f"[population] no cached fits; deriving from {len(subjects)} "
                  f"patients (fit horizon {hours:.0f} h) across "
                  f"{n_workers(jobs)} worker(s) ...")
        pop = derive_population(subjects, hours=hours, jobs=jobs, verbose=verbose)
    else:
        raise ValueError(
            "ensure_population: nothing to install — no manual theta, no cached "
            f"population at {path!r}, no rbg_* fits in {patients_csv!r}, and no "
            "cohort to fit. Run run_prep.sh (derive_replaybg_params) first.")

    if path:
        # atomic write so concurrent sweep workers never read a half-written file
        tmp = f"{path}.{os.getpid()}.tmp"
        save_population(pop, tmp)
        os.replace(tmp, path)
        if verbose:
            print(f"[population] cached -> {path}")
    install_population(pop, margin=margin, verbose=verbose)
    return True


# ===========================================================================
# CLI: derive + save a population
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default=None, help="patients.csv (else base cohort)")
    ap.add_argument("--kind", default="adult", help="base cohort if no --patients")
    ap.add_argument("--limit", type=int, default=None, help="cap patients used")
    ap.add_argument("--exclude", default=None, help="patient name to leave out (LOO)")
    ap.add_argument("--refit", action="store_true",
                    help="ignore cached rbg_* columns and re-fit from scratch "
                         "(otherwise the cached per-patient fits are aggregated)")
    ap.add_argument("--hours", type=float, default=C.WINDOW_HOURS,
                    help="baseline horizon for the re-fit path (--refit / no cache)")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker processes for the re-fit path "
                         "(0 = auto: SLURM_CPUS_PER_TASK, else all cores)")
    ap.add_argument("--out", default=os.path.join(C.ARTIFACT_DIR, "population.npz"))
    args = ap.parse_args()

    # Canonical path: aggregate the per-patient fits that derive_replaybg_params
    # already wrote into patients.csv (single 24 h fit pass, reused everywhere).
    if args.patients and not args.refit and csv_has_fits(args.patients):
        pop = population_from_fits(args.patients, exclude=args.exclude,
                                   limit=args.limit)
    else:
        from . import patients as PT
        if args.patients:
            subjects = PT.load_subjects_csv(args.patients)
        else:
            subjects = [PT.subject_from_base(n) for n in PT.list_patients(args.kind)]
        if args.exclude:
            subjects = [s for s in subjects if s.name != args.exclude]
        if args.limit:
            subjects = subjects[:args.limit]
        why = "forced --refit" if args.refit else "no cached rbg_* columns"
        print(f"[population] {why}: deriving from {len(subjects)} patients "
              f"(fit horizon {args.hours:.0f} h) across {n_workers(args.jobs)} worker(s) ...")
        pop = derive_population(subjects, hours=args.hours, jobs=args.jobs)

    save_population(pop, args.out)

    lo, hi = pop.bounds()
    print(f"[population] mean theta (installed centre): "
          + ", ".join(f"{n}={v:.3g}" for n, v in pop.mean_theta().items()))
    print(f"[population] SI range [{lo[5]:.2e}, {hi[5]:.2e}], "
          f"Gb range [{lo[7]:.0f}, {hi[7]:.0f}]")
    if np.isfinite(pop.rmses).any():
        print(f"[population] fit RMSE = {np.nanmean(pop.rmses):.1f} mg/dL "
              f"(worst {np.nanmax(pop.rmses):.1f})")
    print(f"[population] wrote {args.out} (from {len(pop.members)} patients)")


if __name__ == "__main__":
    main()
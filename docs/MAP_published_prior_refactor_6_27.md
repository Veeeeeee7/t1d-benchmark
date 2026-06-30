# MAP fit + published-prior refactor

**Date:** 2026-06-27
**Scope:** replace the least-squares fit and the empirical-Bayes (cohort-derived)
prior with a single **MAP fit under the published ReplayBG prior** (Cappon et al.,
2023). The published prior becomes the one distribution used everywhere — the
per-patient label fit, the MCMC prior, and the SBI prior. Supersedes
`population_prior_refactor_6_26.md`.

---

## 1. Why this change

We are building a **dataset / benchmark**, not making a contribution about
updating the literature. So we dropped the two-stage *empirical-Bayes* pipeline
(published prior → update with the cohort LS fits → population posterior → use as
prior) in favour of using the published prior **directly**. This is simpler, fully
reproducible (anyone with the same prior + the same simglucose patient recovers
the same label), and gives the whole benchmark a single citable backbone.

**No circularity.** Phase 2 scores each twin against the **simglucose plant**
(`dt2_scoring.collect_truth` runs the candidate therapies on the UVA/Padova
subject and compares the twin's therapy rewards/CGM to that). `rbg_theta` is *not*
the Phase 2 evaluation target — it is only the Phase 1 regression label and the
Phase 0 matched-plant parameters. So defining `rbg_theta` as a MAP estimate under
the published prior does not let any method "see the answer" in Phase 2.

**MAP vs LS for the label.** Under one CGM trace ReplayBG is non-identifiable in
several directions (`SI`/`SG`/`p2`, `ka2`/`kd`, `kabs`/`kempt`). Plain LS wanders
to arbitrary local optima / box edges in those directions. The prior penalty
regularises exactly those flat directions, so the labels are better-conditioned
and reproducible, while the likelihood still dominates the well-identified
directions.

---

## 2. The published prior (single source of truth)

New module **`t1d_twin/replaybg_priors.py`** holds the upstream priors and is the
only place the prior is defined. Coordinate convention matches the rest of the
project: `phi = [log(ka2..p2), Gb]` (seven rates in log space, `Gb` linear).

| Param | Form | Hyper-parameters |
|---|---|---|
| `ka2,kd,kempt,kabs,SG` | LogNormal(θ) = Normal(φ) | μ,σ per param (`_LOGN_MU/_SIGMA`) |
| `p2` | √p2 ~ Normal | (0.11, 0.004) — pins p2 ≈ 0.0121 |
| `SI` | (SI·VG) ~ Gamma | shape 3.3, scale 5e-4 |
| `Gb` | TruncNormal on [70,180] | (119.13, 7.11) |

Ordering constraints `ka2 < kd` and `kabs < kempt` are enforced as hard gates.

### Public API

- `prior_center()` → published medians (dict). Installed as the population centre
  (MCMC walker init / Phase-0 fallback).
- `prior_residuals(theta)` → (10,) residual vector for the **MAP fit**; appended
  to the sigma-scaled CGM residuals so penalised LS == MAP. Ordering enforced via
  soft hinges (weight `_ORDER_W=50`) plus a post-fit `order_ok` check.
- `log_prior_phi(phi_batch)` → (B,) informative **MCMC** log-prior, with the
  proper change-of-variables Jacobians for `p2`/`SI` and `-inf` outside the
  support box or on ordering violation.
- `make_sbi_prior()` → torch-facing **SBI** prior with `.sample()` (ordering
  rejection) and `.log_prob()`.
- `SUPPORT_LO/HI` → a deliberately **generous** physical envelope (NOT a prior).
  It only bounds the optimiser / clamps MCMC walkers; the informative prior does
  the regularising.

### Convention note (intentional, documented)
The MAP point-estimate residuals omit the small `p2`/`SI` φ-Jacobian terms
(`p2` is pinned and `SI`'s gamma is broad, so the shift is negligible). The
**MCMC** `log_prior_phi` includes them, so the sampled posterior is the proper
density. `Gb` (truncnorm) and the five lognormal rates are exact in both paths.

---

## 3. Per-file changes

### `t1d_twin/replaybg_priors.py` — **new**
The module above.

### `experiments/population.py`
- `fit_replaybg`: **LS → MAP**. `_residual` now returns the sigma-scaled CGM
  misfit `(ig-cgm)/σ` stacked with `PR.prior_residuals(theta)`; `σ = MAP_SIGMA =
  10` mg/dL sets the data-vs-prior balance and matches the MCMC/SBI likelihood.
  The optimiser starts at the published prior centre. Returned RMSE is still the
  **CGM-only** error in mg/dL (prior penalty excluded), comparable across the
  cohort. A `RuntimeWarning` fires if a fit violates ordering (should be ~never).
- `WIDE_LO/HI`: now aliases of `PR.SUPPORT_LO/HI` (a generous support box, not the
  old prior). Name kept for backward compatibility.
- **EB prior removed**: `install_population` is now a back-compat shim that
  delegates to the new `install_published_prior()` (sets the centre to the
  published median and points each module's support box at `SUPPORT_LO/HI`).
  `pop`/`margin` are ignored.

### `t1d_twin/identify_mcmc.py`
- `log_prior`: uniform box → `replaybg_priors.log_prior_phi` (informative).
- `PRIOR_LO/HI`: now the support envelope (clamp walkers only), (re)installed by
  `install_published_prior`. `_init_walkers` still reads the centre via
  `get_pop_theta` (now the published median) — unchanged.

### `t1d_twin/identify_sbi.py`
- `make_prior`: `BoxUniform` → `replaybg_priors.make_sbi_prior()` (best-effort
  `process_prior` wrap for the installed sbi version). `log_space=False` now
  raises. Training-data generation is otherwise unchanged: `prior.sample()`
  already enforces ordering, and the CGM-range rejection still runs.
- `PRIOR_LO/HI`: support envelope, as in MCMC.

---

## 4. Legacy-removal checklist (do against the test suite)

These are now **dead** as a prior source and can be deleted once tests pass. They
were left in place so the phase runners don't break on import; verify callers
first with the greps below.

- `experiments/population.py`: `Population.bounds`, `Population.mean_theta`,
  `Population.median_theta`, `population_from_fits`, `derive_population`,
  `_fit_one_subject`, `ensure_population`, `maybe_install`, the `npz`
  save/load + CLI `main()` (the population artifact no longer defines the prior).
  Keep `fit_replaybg`, `n_workers`, and `install_published_prior`.
- Drop the `--population population.npz` plumbing from `run_phase0/1/2.py` and
  `phase_runner.py` (or leave the flag as a no-op).

```bash
grep -rn "install_population\|ensure_population\|population_from_fits\|\.bounds(\|mean_theta\|\.npz" experiments/ t1d_twin/ tests/
grep -rn "PRIOR_LO\|PRIOR_HI\|make_prior\|BoxUniform" t1d_twin/ tests/
```

**Phase 0 note:** `install_for_phase0` still sets a Phase-0 fallback centre. Its
prior install now routes through the published prior; confirm Phase 0's
controlled centre (`replaybg_plant.PHASE0_CENTER`) is still applied where you
expect before relying on Phase 0 outputs.

---

## 5. Migration / re-run order

The labels change (LS → MAP), so prep must be re-derived and everything
downstream rebuilt:

```bash
# 1. re-derive rbg_* labels (now MAP) and re-merge into patients.csv
python -m experiments.derive_replaybg_params --patients patients.csv

# 2. rebuild the Phase 1 regression dataset (target = new MAP rbg_theta)
python -m experiments.build_phase1_dataset --patients patients.csv --overwrite

# 3. re-run phases (twins now use the published prior directly)
python -m experiments.run_phase1 --patients patients.csv
python -m experiments.run_phase2 --patients patients.csv
```

---

## 6. Caveats to validate / document for the benchmark

1. **SBI/torch integration is the one untested piece.** `make_sbi_prior` exposes
   `.sample()`/`.log_prob()` and is wrapped with `process_prior` when available,
   but it was not run against the live `sbi`/`torch` here. Smoke-test
   `generate_training_data` and one `SBITwin` fit first; if your `sbi` version
   rejects the custom prior, wrap/adapt at `make_prior`.
2. **`p2` is effectively fixed** by its prior (penalty ≈13 at 0.008 vs ≈0 at
   0.0121). So "predict p2" is a trivial target. Characterise, per parameter,
   which labels are likelihood-driven (genuinely patient-specific: `SI`, `Gb`,
   `kabs`) vs prior-driven (near-constant: `p2`, partly `SG`).
3. **simglucose vs real-patient centre.** The published prior was built from real
   patients; our subjects are UVA/Padova projections. MAP shrinks toward the
   real-patient centre, so labels sit partway between the simglucose projection
   and the literature. Run the overlay diagnostic (LS/MAP fit histograms vs the
   published densities) and **report the shrinkage** — it is the trust argument.
4. **`VI` divergence.** `replaybg_model.VI = 0.126` vs upstream `0.135`. The
   priors were calibrated against upstream's `VI`; this shifts steady-state
   `Ipb = Ib/(VI·ke)` by ~7%. Left unchanged here — note it, or align `VI` if you
   want strict consistency with the source of the priors.
5. **Ordering is soft in the MAP fit, hard in MCMC/SBI.** The MAP hinge (weight
   50) plus the post-fit `order_ok` warning should keep all labels valid; if any
   warning fires, raise `_ORDER_W` or switch that fit to a constrained optimiser
   (`scipy.optimize.minimize` SLSQP with `kd-ka2>0`, `kempt-kabs>0`).

---

## 7. Files touched

```
t1d_twin/replaybg_priors.py     NEW   published prior: constants + 3 consumer APIs
experiments/population.py       EDIT  MAP fit_replaybg; install_published_prior; EB removed
t1d_twin/identify_mcmc.py       EDIT  log_prior -> informative
t1d_twin/identify_sbi.py        EDIT  make_prior -> informative
```

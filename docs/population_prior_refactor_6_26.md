# Population centre & prior refactor

**Date:** 2026-06-26
**Scope:** how the ReplayBG population centre (`POP_THETA`) and the MCMC/SBI prior
are derived, removing a hardcoded default and a redundant fit, and making Phase 0
consistent with Phase 2.

---

## 1. Background: what "population" means here

The ReplayBG forward model (`replaybg_model.py`) has two parameter groups:

- **8 free parameters** (`THETA_NAMES = ka2, kd, kempt, kabs, SG, SI, p2, Gb`) —
  identified per patient.
- **6 fixed structural parameters** (`VI, KE, BETA, F, VG, ALPHA`) — held constant
  at the Cappon et al. (2023) population values. **These remain fixed** (see §6).

Every twinning method needs a *population* in ReplayBG parameter space:

- a **centre** theta — the MCMC walker initialisation, and the Phase-0 plant
  perturbation centre;
- a **prior box** (`PRIOR_LO/HI`) — the MCMC prior and the SBI `BoxUniform` from
  which training thetas are drawn.

That population lives in ReplayBG space, not UVA/Padova space, so it is obtained by
least-squares fitting the 8-parameter ReplayBG model to each simglucose patient's
baseline CGM and aggregating the per-patient fits.

---

## 2. Problems addressed

1. **The centre was hardcoded, not derived.** `replaybg_model.POP_THETA` was a fixed
   "plausible point". The derivation machinery existed but was opt-in, and the
   aggregation used the **median** while the code labelled it `mean`.
2. **The aggregation was a median, not an average.** The intended centre is the
   per-patient fits *averaged across* the cohort.
3. **The per-patient fit was computed twice.** `derive_replaybg_params` fit each
   patient (24 h) for the `rbg_*` columns, and `population.py` independently re-fit
   each patient (12 h) for the prior — duplicated work at mismatched horizons.
4. **Phase 0 used artificially wide prior bounds**, inconsistent with Phase 2, even
   though the matched Phase-0 patients *are* the simglucose LS-fit population.
5. **Horizons were not uniform** — one runner used 8 h / 6 h.

---

## 3. The model now

### 3.1 Single source of per-patient fits

There is exactly **one** per-patient least-squares pass, done once in prep by
`derive_replaybg_params` at the **24 h** identification horizon, written to the
`rbg_*` columns of `patients.csv`. Three consumers reuse those same parameters:

- Phase 0 matched plant (`subjects_from_patients_csv` reads `rbg_theta`),
- Phase 1 regression target (`build_phase1_dataset` reads `rbg_theta`),
- the population centre + prior (`population.population_from_fits` aggregates the
  `rbg_*` columns — **no re-fitting**).

### 3.2 Population centre = average of the per-patient fits

`Population.mean_theta()` averages the fits across the cohort. Rates (the first 7
params) are averaged in **log space** (geometric mean), consistent with how the
prior, the `_theta_to_phi` transform, and the bounds-widening all treat rates;
`Gb` is averaged arithmetically. `log_rates=False` gives a plain arithmetic mean.
The prior box (`Population.bounds`) is the per-parameter 5–95th percentile range,
log-widened for rates and additively widened for `Gb`, clipped to the wide envelope.

### 3.3 No hardcoded centre — a registry that must be set

`replaybg_model.POP_THETA` (the hardcoded dict) is gone. The centre is a
process-level registry:

```python
replaybg_model.set_pop_theta(theta)   # dict or 8-array; None clears
replaybg_model.get_pop_theta()        # returns the centre, or RAISES if unset
replaybg_model.pop_theta_is_set()
replaybg_model.clear_pop_theta()
```

`get_pop_theta()` raises rather than returning a default, so no run can silently
use fabricated parameters. The MCMC walker init (`identify_mcmc`) and the Phase-0
plant (`replaybg_plant.draw_theta`) resolve the centre through `get_pop_theta()`
at call time; `install_population` is the single writer. The firewall holds:
`t1d_twin` only *consumes* the centre; all derivation lives in `experiments`.

### 3.4 Installation paths

- `population.install_population(pop)` — sets the centre via `set_pop_theta(mean)`
  and the prior box (`MC.PRIOR_LO/HI`, `SB.PRIOR_LO/HI`) to `pop.bounds()`.
- `population.ensure_population(path, patients_csv, subjects)` — the per-run entry,
  resolving in order: a manual theta already set → cached `population.npz` →
  cached `rbg_*` fits in `patients.csv` → fit from scratch (last resort, no prep).
- `population.install_for_phase0(path, fallback_center)` — Phase 0; full install
  (centre + bounds) from `population.npz`, falling back to a fixed centre + wide
  bounds only if prep has not run.
- `exp_common.apply_population(args)` — wires the above into `run_mcmc`/`run_sbi`.

### 3.5 Manual override (tests / debugging)

```python
from t1d_twin import replaybg_model as RB
RB.set_pop_theta({...})   # or an 8-element array
# ... run a fit; no derivation happens ...
RB.clear_pop_theta()
```

CLI equivalents on the run scripts: `--pop-theta "ka2,kd,kempt,kabs,SG,SI,p2,Gb"`
(or a JSON dict) installs a theta directly; `--no-population` requires a theta to
be set already and otherwise errors.

### 3.6 Phase 0 is consistent with Phase 2

Matched Phase-0 patients are the per-patient LS fits of the simglucose cohort, so
Phase 0 now uses the **same** population centre **and** prior bounds as every other
phase (`install_for_phase0` → full `install_population`), not the old wide bounds.
`run_mcmc0` and `run_sbi0` install it before the fit; both honour `--population`
(default `<artifacts>/population.npz`).

### 3.7 24 h everywhere

Every experiment runs at the 24 h horizon (`WINDOW_HOURS = SMOKE_HOURS = 24` in
both `exp_common` and `replaybg_plant`; the dataset builders, identifiers, and
evaluators all resolve to it). The population inherits 24 h from the cached `rbg_*`
fits. The lone exception, `run_all.py`, was moved from 8 h / 6 h to 24 h.

---

## 4. Data flow

**Prep (`run_prep.sh`, once):**

```
generate_patients            -> patients.csv
derive_replaybg_params (24h) -> patients.csv + rbg_* columns   [THE single LS fit pass]
population (aggregate rbg_*) -> population.npz                  [mean centre + percentile prior]
build_phase1_dataset         -> phase1_dataset.npz
build_phase0_dataset         -> phase0_dataset.npz
```

**Per run (loads, does not re-derive):**

```
apply_population / install_for_phase0
  -> load population.npz
  -> set_pop_theta(mean)              (MCMC walker init; Phase-0 plant centre)
  -> PRIOR_LO/HI = bounds             (MCMC prior; SBI BoxUniform)
```

`run_phase2.sh`'s embedded `setup` mirrors prep (derive `rbg_*` once at 24 h, then
aggregate). For sweeps, prep should run first so per-patient subprocesses load the
cached `population.npz` rather than each deriving.

---

## 5. Files changed

| File | Change |
|---|---|
| `replaybg_model.py` | Removed hardcoded `POP_THETA`; added `set_pop_theta` / `get_pop_theta` (raises) / `pop_theta_is_set` / `clear_pop_theta`. Fixed structural params unchanged. |
| `population.py` | `mean_theta` centre (geometric-mean rates); `population_from_fits` / `csv_has_fits` (aggregate cached fits, no re-fit); `ensure_population`; `install_for_phase0`; `install_population` writes via `set_pop_theta`; `main` aggregates by default (`--refit` to fit). |
| `identify_mcmc.py` | Walker init resolves the centre via `get_pop_theta()`. |
| `identify_sbi.py` | Removed the unused `POP_THETA` import. |
| `replaybg_plant.py` | `draw_theta` uses `get_pop_theta()`; added `PHASE0_CENTER` fallback constant (the former hardcoded values). |
| `exp_common.py` | `apply_population` loads-or-builds the population automatically (cache → `rbg_*` → fit); `--pop-theta` / `--no-population` escape hatches; `--pop-hours` default 24. |
| `run_mcmc0.py` | Installs the full Phase-0 population (centre + prior bounds); honours `--population`. |
| `run_sbi0.py` | Installs the full Phase-0 population (prior box) before training-data generation; honours `--population`. |
| `generate_phase0_patients.py` | Installs `PHASE0_CENTER` before generating the synthetic cohort. |
| `run_all.py` | Horizon 8 h / 6 h → 24 h. |
| `run_prep.sh` | Step 3 aggregates step 2's single 24 h fit pass (no separate 12 h fit); `POP_HOURS` removed. |
| `run_phase2.sh` | `setup` derives `rbg_*` once (24 h) then aggregates; `POP_HOURS` → `DERIVE_HOURS`. |

---

## 6. Decision: fixed structural parameters stay at Cappon values

`VI, KE, BETA, F, VG, ALPHA` are **not** fitted and remain fixed at the Cappon et
al. (2023) values:

```python
VI = 0.126   # insulin distribution volume [l/kg]
KE = 0.127   # insulin fractional clearance [1/min]
BETA = 8.0   # insulin appearance delay [min]
F = 0.9      # fraction of intestinal glucose absorbed [-]
VG = 1.45    # glucose distribution volume [dl/kg]
ALPHA = 7.0  # plasma->interstitium delay [min]
```

These are deliberately **not** added to the per-patient fit because they are
structurally non-identifiable from CGM alone:

- **`F` and `VG`** enter only through glucose appearance `F·kabs·Qgut/VG`, so only
  the ratio `F/VG` affects IG — fitting both is a pure degeneracy.
- **`VI`** is an insulin input gain (`Ip ∝ 1/VI`) reaching glucose only via
  `SI·(Ip − Ipb)`, so glucose depends on `VI` and `SI` only through `SI/(VI·KE)` —
  `VI` and `SI` trade off perfectly. `KE` couples on the same magnitude.

Fitting these would land per-patient estimates on arbitrary points of those
degenerate ridges, making any population average meaningless. The free parameters
(`SI`, `kabs`) absorb individual variation, which is why ReplayBG fixes them.

If population-specific structural values are ever wanted, the sound route is to
read the analogues from the UVA/Padova patient table (`patients.patient_params`,
i.e. `vpatient_params.csv`) and map `Vi→VI`, `Vg→VG`, `ke→KE` — reading ground
truth rather than inferring it — not a CGM fit. Not done here.

---

## 7. Verification status / TODO

- All files AST / `bash -n` checked; the registry (`set`/`get`/`clear`/validation)
  and the cached-fit aggregation (header detection, NaN-row drop, mean, `--exclude`)
  were unit-tested in isolation.
- **Not** run end-to-end (no simglucose/torch in the editing environment). Before
  relying on this, smoke-test: `run_prep.sh` (or `run_phase2.sh setup`), then a
  `run_mcmc0 --smoke` and `run_sbi0 --smoke`, and confirm the population installs
  and the fits converge.
- Confirm the population now living at the 24 h horizon (inherited from `rbg_*`) is
  intended — it replaces the previous 12 h prior fit.

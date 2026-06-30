# Phase 0 — Matched-Model Baseline — Implementation

**Status:** implemented · core paths validated in a sandbox (fit recovery,
matched-cohort load, MCMC identify→score)
**New:** `experiments/derive_replaybg_params.py`, `experiments/replaybg_plant.py`,
`experiments/generate_phase0_patients.py`, `experiments/build_phase0_dataset.py`,
`experiments/run_phase0_ml.py`, `experiments/run_mcmc0.py`,
`experiments/run_sbi0.py`, `experiments/compute_results0.py`,
`experiments/run_phase0.py`, `experiments/phase0_paths.py`,
`run_phase0_twins.sh`, `run_phase0_ml.sh`, `run_prep.sh`
**Updated:** `experiments/phase_runner.py`, `experiments/patients.py`,
`experiments/build_phase1_dataset.py`
**One-line summary:** a new phase whose ground-truth plant _is_ the ReplayBG
model, so the MCMC/SBI twins fit data from their own model class. Each Phase 0
patient is the **matched-model counterpart of a real Phase 2 patient**: its true
ReplayBG physiology is the best-fit ReplayBG parameters for that simglucose
patient, cached once in `patients.csv` and reused everywhere.

---

## 1. Why this phase exists

Phase 2 scores the twins against a simglucose / UVA-Padova plant while the twins
fit with a ReplayBG ODE — two different models. In that setting MCMC collapsed
to Spearman ≈ 0 and SBI showed catastrophic regret outliers, which _looked_ like
a harness bug. A controlled experiment (ReplayBG as both plant and twin, running
the real `identify_mcmc`) showed the opposite: with the model correctly
specified, the same MCMC pipeline scores Spearman ≈ 0.95–1.0 and recovers SI to
within a few percent. The Phase 2 degradation is the designed plant↔twin model
mismatch (the fidelity-vs-decision dissociation DT² is about), not a defect.

Phase 0 makes that a first-class baseline. To make it a _controlled, per-patient_
comparison rather than a generic one, each Phase 0 patient is matched to a real
Phase 2 patient: same identity, same carb error, and a true physiology equal to
the closest ReplayBG approximation of that patient. The only thing that differs
between Phase 0 and Phase 2 for patient _i_ is whether the plant is simglucose or
its best-fit ReplayBG surrogate — so a paired Phase 0 vs Phase 2 comparison
isolates the structural mismatch.

Phase 0 covers **both** method families: the per-instance twins (MCMC/SBI, §4–5)
and the amortized regressors (TCN/ridge, §6), so every method has a matched-model
ceiling to read its Phase 1/2 numbers against.

---

## 2. Where the ReplayBG patient parameters come from

Two cohort sources are supported; **patient-matched is the default and intended
one**.

### (a) Patient-matched (default) — fit ReplayBG to each simglucose patient

For every patient in `patients.csv` we run its baseline identification run (the
same one the twins identify from) and least-squares-fit the 8-parameter ReplayBG
model with the **same** `population.fit_replaybg` Phase 1 uses. That best-fit
`theta` becomes the patient's true Phase 0 physiology. The Phase 0 plant then
runs that `theta` at the patient's own body weight and basal, with a calibrated
matched carb ratio and the patient's **existing** carb error `dose_mult` (`m`).
Net effect: Phase 0 patient `synth0001_k2` is the matched-model twin of Phase 2
patient `synth0001_k2` — same name, same `m`, same decision target, ReplayBG
physiology fit to that patient.

These fits are derived **once** and cached as `rbg_*` columns in `patients.csv`
(see §3), so Phase 0/1/2 reuse them without refitting.

A note on self-consistency and the (intended) circularity: the fitted `theta` is
only used to choose a realistic, patient-aligned physiology. The Phase 0
identification CGM and candidate ground truth are then **regenerated from that
theta through the ReplayBG plant** (not reused from the simglucose trace), so the
twins still fit clean in-class data. The plant being the closest ReplayBG model
to each real patient is exactly what "matched-model baseline of this cohort"
means.

### (b) Independent prior draws (alternative) — `generate_phase0_patients`

The original mode is still available: draw each `theta` from the ReplayBG prior
centered on `POP_THETA` (deterministic per name), calibrate `CR_true`, assign a
fresh carb error. This is a simglucose-free, population-level baseline ("averaged
over the prior, how well do the methods rank?"). It is **not** patient-matched
and cannot be paired with Phase 2. Use it when you want a model-free reference;
use (a) for the controlled comparison.

### Same decision problem as Phase 2 (both modes)

Same cadence/horizon/grid (3-min CGM, 1-min ODE, 24 h, 3 meals,
`PROD_BOLUS`/`PROD_BASAL`); same carb-counting-error mechanism so the optimum is
`f* ~= m` interior on the grid; and the runs quack like
`simglucose_adapter.RunResult`, so `identify_mcmc`, `identify_sbi`, and
`evaluate_twin` are reused **unchanged**. Phase 0 adds Gaussian `N(0, 10)` sensor
noise that matches the likelihood/simulator — a clean "everything matches" lower
bound; swap `_add_noise` for `t1d_twin.sensor.add_cgm_noise` for Dexcom realism.

---

## 3. Deriving and caching the fit (`derive_replaybg_params.py`)

The derivation appends these columns to `patients.csv`, next to the UVA/Padova
columns, keyed by `Name`:

| column               | meaning                                                               |
| -------------------- | --------------------------------------------------------------------- |
| `rbg_ka2 ... rbg_Gb` | best-fit ReplayBG `theta` (8 params), from `population.fit_replaybg`  |
| `rbg_Ib`             | patient's baseline basal [mU/kg/min] for the ReplayBG plant           |
| `rbg_CR_true`        | Phase-0-plant matched carb ratio (calibrated so `f* = m` is interior) |
| `rbg_fit_rmse`       | least-squares fit RMSE [mg/dL] (diagnostic)                           |

Because the theta is produced by the very function Phase 1 calls, the cached
`rbg_theta` is **identical** to Phase 1's regression target. The script shards
(`--start`/`--limit`, resumable per-patient parts) and merges (`--merge`), the
same running method as the other phases.

### How the cache is reused (the "without refitting" win)

- **Phase 0** — `replaybg_plant.subjects_from_patients_csv` builds the
  matched-model cohort straight from these columns; no refit, no recalibration.
- **Phase 1** — `build_phase1_dataset` now reads `subject.rbg_theta` as the
  per-patient target when present, instead of re-running the least-squares on
  every dataset build. (`patients.load_subjects_csv` surfaces the cache as
  `Subject.rbg_theta` / `rbg_Ib` / `rbg_cr_true` / `rbg_fit_rmse`.)
- **Phase 2** — the same `Subject.rbg_theta` is available as an MCMC walker
  warm-start / a reference projection to compare the posteriors against (left as
  an opt-in; Phase 2 still fits its own twins, which is its job).

`patients.py`'s loader ignores unknown columns, so a `patients.csv` **without**
the `rbg_*` columns still loads exactly as before — the change is fully
back-compatible.

---

## 4. Files and where they go

Paths are relative to the repo root (the directory containing `t1d_twin/` and
`experiments/`).

```
<repo-root>/
|-- run_phase0_twins.sh                     NEW    (CPU SLURM: MCMC/SBI + derive; next to run_phase2.sh)
|-- run_phase0_ml.sh                         NEW    (GPU SLURM: TCN/ridge CV + score)
|-- run_prep.sh                              NEW    (CPU SLURM: one-shot prep for phases 0/1/2)
|-- phase_0_implementation.md              NEW    (this doc; with the other design docs)
`-- experiments/
    |-- derive_replaybg_params.py          NEW    fit ReplayBG per patient -> rbg_* cols in patients.csv
    |-- replaybg_plant.py                  NEW    matched-model plant + cohort loaders + therapy grid
    |-- generate_phase0_patients.py        NEW    independent prior-draw cohort (alternative)
    |-- build_phase0_dataset.py            NEW    amortized dataset (ReplayBG inputs, exact theta labels)
    |-- run_phase0_ml.py                   NEW    amortized driver (TCN/ridge CV + score vs ReplayBG truth)
    |-- run_mcmc0.py                        NEW    per-patient MCMC fit (ReplayBG plant)
    |-- run_sbi0.py                         NEW    per-patient SBI fit  (ReplayBG plant)
    |-- compute_results0.py                NEW    score vs ReplayBG truth -> comparison_table.csv
    |-- run_phase0.py                       NEW    per-instance driver (mirror of run_phase2.py)
    |-- phase0_paths.py                     NEW    phase0-namespaced results/ + artifacts/ paths
    |-- phase_runner.py                     UPDATE stages/results/loader + results_dir_fn (back-compatible)
    |-- patients.py                         UPDATE Subject.rbg_theta + read/write rbg_* cols (back-compatible)
    `-- build_phase1_dataset.py            UPDATE reuse cached rbg_theta instead of refitting
```

Generated at runtime: the augmented `patients.csv` (in place), per-patient fit
parts under `$T1D_OUTPUT_ROOT/artifacts/rbg_fits/`, the per-subject
`comparison_table.{csv,md}`, and `phase0_{summary,per_patient}.csv`.

### File-by-file (new/changed roles)

| File                                                         | Role                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `derive_replaybg_params.py`                                  | Fit ReplayBG to each simglucose patient (reusing `population.fit_replaybg`), calibrate the plant carb ratio, append `rbg_*` columns to `patients.csv`. Shardable / resumable / `--merge`.                                                                                                                       |
| `replaybg_plant.py`                                          | The plant. `Phase0Subject`, `RBGRun` (RunResult-compatible), `identification_run` / `ground_truth`, `calibrate_cr_true`, the candidate therapy grid (`factors_for`); loaders `subjects_from_patients_csv` (matched), `load_phase0_csv` (native), and `load_phase0_cohort` (auto-detects which).                 |
| `build_phase0_dataset.py`                                    | Amortized dataset builder. One `(cgm, insulin, cho)` trace per therapy from the ReplayBG plant, with the **known true theta** as the exact per-patient label. Format-identical to `build_phase1_dataset` (drop-in for the CV code). Simglucose-free.                                                            |
| `run_phase0_ml.py`                                           | Amortized driver. Reuses `tcn_cv` / `linear_cv` / `cv_common` CV unchanged, scores the predicted `PointTwin`s against the ReplayBG ground truth, writes `results/phase0_ml/{per_patient,summary}.csv`.                                                                                                          |
| `patients.py` (updated)                                      | `Subject` gains cached `rbg_theta` / `rbg_Ib` / `rbg_cr_true` / `rbg_fit_rmse`; `load_subjects_csv` / `write_patients_csv` read/write the `rbg_*` columns. Absent columns -> old behavior.                                                                                                                      |
| `build_phase1_dataset.py` (updated)                          | Uses `s.rbg_theta` when present (skips the per-patient refit); falls back to `fit_replaybg` otherwise.                                                                                                                                                                                                          |
| `run_mcmc0` / `run_sbi0` / `compute_results0` / `run_phase0` | Identical to Phase 2's siblings except the plant is ReplayBG and the cohort loader is `load_phase0_cohort`.                                                                                                                                                                                                     |
| `phase0_paths.py`                                            | Phase-0-namespaced `artifact_paths` / `results_dir_for` (under `artifacts/phase0/` and `results/phase0/`). **Needed because the matched cohort shares patient names with Phase 2** — without it the two phases would overwrite each other's `comparison_table.csv` and trip each other's resumable skip-checks. |
| `phase_runner.py` (updated)                                  | `run_phase` takes optional `stages` / `results_module` / `load_subjects` / `results_dir_fn` (defaults = Phase 2), so Phase 2 is unchanged and Phase 0 passes its own.                                                                                                                                           |

---

## 5. How to run

On the cluster, do all shared prep once, then launch the phases (each phase's own
prep then skips):

```bash
sbatch run_prep.sh        # builds patients.csv (+rbg_*), population.npz,
                          # phase1_dataset.npz, phase0_dataset.npz — for phases 0/1/2
```

Or run the Phase 0 steps directly:

```bash
# 1. Derive + cache the per-patient ReplayBG fits into patients.csv (once).
python -m experiments.derive_replaybg_params --patients patients.csv          # all + merge
#    sharded on a cluster:
#    python -m experiments.derive_replaybg_params --patients patients.csv --start 0 --limit 256
#    python -m experiments.derive_replaybg_params --patients patients.csv --merge

# 2. Fit + score the matched-model baseline (same running method as Phase 2).
python -m experiments.run_phase0 --patients patients.csv --start 0 --limit 256
python -m experiments.run_phase0 --patients patients.csv --aggregate-only

# 3. Amortized (ML) baselines: build the dataset, then CV + score (reuses phase 1's models).
python -m experiments.build_phase0_dataset --patients patients.csv
python -m experiments.run_phase0_ml --patients patients.csv      # -> results/phase0_ml/{per_patient,summary}.csv

# Fast plumbing check end-to-end:
python -m experiments.derive_replaybg_params --patients patients.csv --limit 4 --smoke
python -m experiments.run_phase0 --patients patients.csv --smoke --limit 4
python -m experiments.run_phase0_ml --patients patients.csv --smoke --limit 4 --build

# Alternative: independent prior-draw cohort (no simglucose, not patient-matched)
python -m experiments.generate_phase0_patients --n 1013 --out patients0.csv
python -m experiments.run_phase0 --patients patients0.csv --start 0 --limit 256
```

`run_phase0.py` auto-detects the cohort: a `patients.csv` carrying `rbg_*`
columns -> matched cohort; a native `patients0.csv` -> prior-draw cohort. Outputs
land at `phase0_summary.csv` / `phase0_per_patient.csv` in the same schema as
`phase2_summary.csv`, so they drop straight into the existing analysis.

On the cluster the two halves match the Phase 1/2 scripts you launch:
**`run_phase0_twins.sh`** (CPU, `c64-m512`) is self-submitting like `run_phase2.sh`
— launch it from the login node with `bash run_phase0_twins.sh` and it chains
`setup` (generate + derive the rbg*\* columns, fanned across one node) ->
`patient` array (PER_TASK patients/task, matched-model MCMC+SBI) -> `aggregate`,
via `afterok` dependencies. **`run_phase0_ml.sh`** (GPU, `b200`) mirrors
`run_phase1.sh` — a single `sbatch run_phase0_ml.sh` that builds the dataset and
runs the TCN/ridge CV (GPU) + scoring (CPU). Run the derivation once (the twins
`setup` stage, or just let whichever script runs first do it — both guard on the
rbg*\* columns already being present), then the CPU twins sweep and the GPU ML job
can run in parallel. Phase 0 results are namespaced under `results/phase0/` and
`results/phase0_ml/`, so they sit beside Phase 2 / Phase 1 and are paired by name
at analysis time.

---

## 6. Amortized baselines (the ML methods)

Phase 1's amortized regressors — a TCN and a ridge linear model that map a
patient's `(CGM, insulin, CHO)` traces to ReplayBG `theta` — get the same
matched-model treatment. Two things change from Phase 1, both removing mismatch:

- **Inputs** come from the ReplayBG plant (`build_phase0_dataset` runs each
  candidate therapy through `replaybg_plant.ground_truth`), not simglucose.
- **Labels are exact.** In Phase 1 the per-patient target is a least-squares
  _projection_ of a simglucose run into ReplayBG space (it carries projection
  error, reported as `proj_rmse`). In Phase 0 the target is the patient's _true_
  generating `theta` (`subject.theta`), so `proj_rmse = 0` by construction.

Everything else is reused **unchanged**: `cv_common` (channel stacking,
train-fold normalization, patient-grouped K-fold), `tcn_cv.cross_val_predict` /
`linear_cv.cross_val_predict` (the models only ever see the dataset arrays, so
they are plant-agnostic), `identify_tcn.PointTwin` (the point-estimate twin), and
`phase_runner.summarize`. The dataset `.npz` is byte-for-byte the same shape and
keys as `build_phase1_dataset`, so it is a drop-in.

So Phase 0's amortized run answers: "with in-class data and exact labels, how well
can a TCN / ridge recover ReplayBG `theta` from CGM, and how well do those
recovered twins rank therapies?" — the amortized ceiling, to be read against
Phase 1 (simglucose plant + projected labels), exactly as the per-instance Phase 0
is read against Phase 2. Outputs land in `results/phase0_ml/{per_patient,
summary}.csv`, the same schema as every other phase, so they concatenate with the
per-instance Phase 0 results (MCMC/SBI) for a single combined table.

The amortized run needs `torch` (and `tcn_cv`/`linear_cv`); it does **not** need
simglucose — the cohort comes from `patients.csv` and the plant is ReplayBG.

---

## 7. Baseline result (live demo)

A 7-patient demo (independent prior-draw cohort, MCMC, reduced 24-walker chains
vs PROD's 64w/8000-step) was run through the real pipeline. With the model
correctly specified the same code that scored Spearman ~ 0 in Phase 2 is
near-ceiling:

| metric       | Phase 0 (matched-model)    | Phase 2 (mismatch) |
| ------------ | -------------------------- | ------------------ |
| Spearman     | **0.96 +/- 0.03**          | 0.02               |
| regret       | **15** (6/7 are exactly 0) | 465                |
| RMSE [mg/dL] | **3.3**                    | 37                 |
| MARD [%]     | **2.8**                    | 26                 |

This establishes the methods are sound and the Phase 2 degradation is the
designed mismatch. The patient-matched cohort run (step 1+2 above, full settings,
both methods) is the production deliverable and should be run on the cluster; the
demo numbers are MCMC-only at reduced settings.

---

## 8. Verification performed

- `population.fit_replaybg`'s least-squares recovers a known ReplayBG `theta`
  from a 24 h run (SI to ~2%, Gb to <1 mg/dL; fit RMSE ~ noise floor).
- A synthetic augmented `patients.csv` loads via `load_phase0_cohort` ->
  `subjects_from_patients_csv` and drives the plant with `f* ~= m` interior.
- The full per-instance Phase 0 path (cohort -> plant runs ->
  `identify_twin_from_run` -> `evaluate_twin`) ran end-to-end for 7 patients (§7).
- `build_phase0_dataset` produces a dataset byte-compatible with
  `build_phase1_dataset` (same keys/shapes), labels exact (`proj_rmse = 0`), and
  `patient_idx` groups therapies by patient — validated on a 2-patient cohort.
- `patients.py` / `build_phase1_dataset.py` edits are additive and pass
  `py_compile`; a `patients.csv` without `rbg_*` columns loads unchanged.
- All new modules pass `python -m py_compile`; `run_phase0_twins.sh` and `run_phase0_ml.sh` pass `bash -n`, and every CLI flag the scripts invoke exists in the corresponding Python entry point;
- Phase 0 writes under `results/phase0/` and `artifacts/phase0/` (via `phase0_paths`), so the matched cohort never collides with Phase 2 despite sharing patient names;
  `phase_runner.run_phase` is back-compatible with the Phase 2 call.

---

## 9. Open items

- Run derivation + Phase 0 over the full 1013-patient cohort at PROD settings on
  the cluster; record `phase0_summary.csv` next to `phase2_summary.csv` and do
  the **paired** per-patient Phase 0 vs Phase 2 comparison (now possible because
  the cohorts share identities).
- The `patients.py` and `build_phase1_dataset.py` edits compile but were not
  runtime-tested here (they import simglucose, unavailable in the sandbox); give
  them one smoke run in-repo before the full sweep.
- SBI and the amortized baselines were not executed here (need torch / sbi);
  `run_sbi0.py` and `run_phase0_ml.py` are wired to the validated pieces. Run the
  amortized sweep (`build_phase0_dataset` + `run_phase0_ml`) on the cluster and
  record `results/phase0_ml/summary.csv` next to the per-instance Phase 0 and the
  Phase 1 numbers.
- Decide the headline noise model (Gaussian sigma=10 vs Dexcom) and optionally
  pair Phase 0 with the sigma-inference / expected-reward-ranking variants from
  the Phase 2 analysis to attribute the Phase 2 gap across mismatch vs
  assumptions.

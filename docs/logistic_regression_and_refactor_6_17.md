# Logistic-Regression Baseline & MLP→TCN Refactor — 2025-06-17

This document records four changes to the experiment code:

1. Renaming the old **MLP** code/identifiers to **TCN** (the architecture you switched to).
2. Adding a **linear (ridge) regression baseline** alongside the TCN, on the same methodology.
3. Listing and removing **unused / legacy functions**.
4. **Restructuring the phases**: removing the old Phase 2 (100-patient subset),
   promoting Phase 3 → Phase 2, and trimming the shell scripts to two.

> **Note on naming.** You asked for a "logistic regression" baseline. The Phase 1 target is a
> continuous 8-vector ReplayBG `theta` (a *regression* task), so a literal logistic regression
> (classification) doesn't map onto it. Per your selection, this was implemented as a **ridge
> (L2-regularized linear) regression** to phi — the same pipeline as the TCN with the network
> swapped for a linear model. The deliverable filename keeps your original wording.

---

## 1. MLP → TCN rename

### Files renamed

| Old | New | Contents |
|-----|-----|----------|
| `t1d_twin/identify_mlp.py` | `t1d_twin/identify_tcn.py` | the regressor + the point-estimate twin |
| `experiments/mlp_cv.py` | `experiments/tcn_cv.py` | the TCN cross-validation driver |

### Symbols renamed

| Old | New | Why |
|-----|-----|-----|
| `CGMRegressor` | `TCNRegressor` | makes the architecture explicit and parallels the new `LinearRegressor`-style baseline |
| `MLPTwin` | `PointTwin` | **method-neutral** rename (see below) |
| `mlp_cgm_ins_cho` (CLI/baseline key) | `tcn_cgm_ins_cho` | result rows / summaries now read `tcn_…` |

**Why `PointTwin` and not `TCNTwin`.** `MLPTwin` was only ever a *generic point-estimate twin*
— "ReplayBG run at a single `theta`-hat," with a degenerate (zero-width) prediction band. It is
not TCN-specific, and it is now shared unchanged by **both** the TCN baseline and the new linear
baseline. Calling it `TCNTwin` would be a misnomer the moment the linear baseline wraps its own
point estimate in it. I renamed it to the method-neutral `PointTwin`. If you'd rather it read
`TCNTwin`, it's a one-symbol change in `identify_tcn.py` plus its two import sites
(`run_phase1.py`, and this doc).

### Legitimate "MLP" terms left in place

`TCNRegressor`'s docstring still refers to its final dense layers as the "dense (MLP) head."
That's an accurate description of a sub-component of the TCN, not a reference to the old method,
so it was kept intentionally.

### Every file touched by the rename

`identify_tcn.py` (new), `tcn_cv.py` (new), `run_phase1.py`, `exp_common.py`, `dt2_scoring.py`,
`phase_runner.py`, `population.py`, `run_suite.py`, `build_phase1_dataset.py`, `requirements.txt`,
and the shell scripts `run_phase1.sh`, `run_all_phases.sh`, `run_experiments.sh`,
`run_experiments_smoke.sh`.

### Design docs NOT touched (your call)

`MLP` still appears in `HANDOFF_2.md`, `HANDOFF_3.md`, `IMPLEMENTATION_PLAN.md`, and
`carb_error_perturbation.md`. These are point-in-time design/handoff records, so I left them as
historical snapshots rather than rewrite them. Say the word and I'll update them for consistency.

---

## 2. Linear (ridge) regression baseline

### New files

| File | Role |
|------|------|
| `experiments/cv_common.py` | **model-agnostic** Phase 1 plumbing shared by both baselines |
| `experiments/linear_cv.py` | the ridge baseline's CV driver |

To guarantee the linear baseline follows *exactly* the same methodology as the TCN, the shared,
non-model parts of the old `mlp_cv.py` were factored into `cv_common.py`:

- `stack_channels` — build the `(M, C, L)` input array,
- `norm_fit` / `norm_apply` — per-channel input standardization (train-fold stats only),
- `group_kfold_indices` — the leakage-free, **patient-grouped** K-fold splitter,
- `pick_device`, and the `CGM_ONLY` / `CGM_INS_CHO` channel stacks.

`tcn_cv.py` and `linear_cv.py` both import these, so a change to the CV protocol now applies to
all baselines at once.

### What is identical to the TCN baseline

Treatment-augmented dataset (one example per (patient, therapy) trace) · patient-grouped K-fold ·
per-channel train-fold input standardization · phi-space targets (`log`-rates + linear `Gb`),
standardized for the fit · per-patient phi-space averaging of held-out predictions · scoring
through the shared `PointTwin` against the same simglucose ground-truth grid.

### What differs (intrinsic to a linear model)

- The `(C, L)` trace is **flattened to a `C*L` feature vector** (a linear map has no
  global-pooling analogue; the dataset already truncates traces to a common `L`, so the flat
  dimension is fixed).
- The fit is the **closed-form ridge solution** (normal equations, intercept left unpenalized)
  — no SGD, early stopping, or validation split. The regularizer `alpha` plays the role the
  TCN's capacity/dropout controls play.
- Implemented in **NumPy only** — no scikit-learn dependency was added (`requirements.txt`
  unchanged for this).

### Running it

No new commands. `run_phase1.py` now registers both baselines:

```python
BASELINES = {
    "tcn_cgm_ins_cho":    (TCN.cross_val_predict, CGM_INS_CHO),
    "linear_cgm_ins_cho": (LIN.cross_val_predict, CGM_INS_CHO),
}
```

Both predictors share the `cross_val_predict(dataset, channels, k, seed, cfg, device)` signature,
so the scoring loop is unchanged and both methods land in `per_patient.csv` / `summary.csv` as
`tcn_…` and `linear_…` rows.

The ridge strength defaults to `alpha = 1.0`. It reads from the same `cfg` dict
(`cfg["alpha"]`), and `linear_cv` ignores the TCN-only keys (`hidden`, `n_epochs`, …) so the
shared `PROD_CFG` / `SMOKE_CFG` work for both. If you want per-fold `alpha` selection later,
that's a small addition inside `linear_cv.cross_val_predict`.

---

## 3. Unused / legacy functions

### Removed

| Symbol | File | Reason |
|--------|------|--------|
| `save_mlp` | `exp_common.py` | zero call sites; already flagged "not in the pipeline"; legacy MLP name |
| `load_mlp` | `exp_common.py` | zero call sites; "not in the pipeline"; legacy MLP name (imported the deleted `MLPTwin`) |
| `predict_theta_batch` | `mlp_cv.py` → not carried into `tcn_cv.py` | zero call sites; a convenience wrapper superseded by `cross_val_predict` |

### Found unused but **retained** (with reasoning)

Static analysis shows no in-repo callers for the items below, but each looks like a deliberate
public/library surface where callers live outside what the scan can see (notebooks, ad-hoc figure
scripts, documented APIs). I did **not** delete these — flagging them so you can decide:

| Symbol(s) | File | Why I left it |
|-----------|------|---------------|
| `plot_candidate_runs`, `plot_cgm_overlay`, `plot_decision_vs_fidelity`, `plot_forward_trajectory`, `plot_ranking_scatter`, `plot_regret_bars`, `plot_twin_ig_cgm`, `plot_twin_overlay` | `plotting.py` | the **entire module** is unreferenced in the pipeline — almost certainly your interactive figure-generation library. Deleting it unprompted would be the kind of overreach that breaks a paper-figure workflow. |
| `baseline_gap_regret`, `cohort_summary` | `dt2_fairness.py` | `baseline_gap_regret` is explicitly recommended as a cohort-reporting metric in `carb_error_perturbation.md` — documented, optional reporting path. |
| `clinical_metrics` | `value.py` | clinical reporting helper (TIR/LBGI/HBGI bundle); reads as intended public API. |
| `baseline_policy` | `policies.py` | listed as part of the `policies` API in `IMPLEMENTATION_PLAN.md`. |
| `mU_per_kg_min_to_insulin_U_per_min`, `mg_per_kg_min_to_cho_g_per_min` | `units.py` | the inverse-direction half of the two converters that *are* used; a symmetric unit API. |
| `fig_dir_for` | `exp_common.py` | per-subject figure-path helper paired with the (retained) plotting workflow. |
| `rewards_for_subject` | `inspect_reward_separation.py` | a helper inside a standalone diagnostic script. |

If you want any of these removed too, point me at them and I'll take them out.

---

## 4. Phase restructuring — old Phase 2 removed, Phase 3 → Phase 2

Before this change the experiment had three phases: Phase 1 = the amortized baselines
(TCN / linear); **old Phase 2** = the per-instance twins (MCMC / SBI) on a *fixed random
100-patient subset*; **old Phase 3** = the same per-instance twins on *all* patients (the heavy,
shardable sweep). You asked to drop the 100-patient phase, promote the full sweep to be "Phase 2,"
and keep only two shell scripts.

### Files removed (8)

| File | What it was |
|------|-------------|
| `run_phase2.py` *(old)* | old Phase 2 driver — the fixed 100-patient subset run |
| `sample_phase2_patients.py` | sampled & froze the 100-patient list; **only** the old Phase 2 used it |
| `run_phase23.sh` | combined Phase 2 + Phase 3 as one SLURM job per patient |
| `run_all_phases.sh` | inline "run every phase" convenience script |
| `run_experiments.sh` | full three-phase SLURM benchmark |
| `run_experiments_smoke.sh` | fast plumbing smoke test across all three phases |
| `run_phase3.py` | renamed (see below), so the old path is gone |
| `run_phase3.sh` | renamed (see below), so the old path is gone |

### Files renamed (Phase 3 → Phase 2)

| Old | New | What changed inside |
|-----|-----|---------------------|
| `run_phase3.py` | `run_phase2.py` | the all-patients sweep; `[phase3]`→`[phase2]` logging, `label="phase2"`, outputs now `phase2_summary.csv` / `phase2_per_patient.csv`, and the docstring's stale `run_experiments.slurm` reference now points at `run_phase2.sh` |
| `run_phase3.sh` | `run_phase2.sh` | header text, job names `t1d_p3_*`→`t1d_p2_*`, and the aggregate stage now calls `run_phase2` and reports `results/phase2_summary.csv` |

### Files edited

| File | Change |
|------|--------|
| `phase_runner.py` | docstring now describes a *single* per-instance phase ("Phase 2 = this over all patients") instead of the old Phase 2 & 3 split |
| `run_sbi.py` | one comment: "phase-3 packs 32 patients per job" → "phase-2 …" |

### Net effect

- **Shell scripts are now exactly two:** `run_phase1.sh` and `run_phase2.sh`.
- The new Phase 2 keeps the **sharding interface** from old Phase 3 (`--start` / `--limit` /
  `--aggregate-only`); the old Phase 2's subset args (`--list` / `--n` / `--list-seed`) are gone
  with it.
- `HANDOFF_2.md` / `HANDOFF_3.md` still use the old phase numbering — left as historical
  snapshots, same as the MLP references. Say the word and I'll renumber them.

---

## Verification performed

- All modified/new modules pass `python -m py_compile`.
- `group_kfold_indices` verified leakage-free (no patient in both train and test) and to cover
  every patient exactly once across folds.
- The ridge solver verified to recover a known linear target (R² ≈ 1.0) and confirmed to add no
  scikit-learn dependency.
- Repo-wide grep confirms **no** remaining references to `identify_mlp`, `mlp_cv`, `MLPTwin`,
  `CGMRegressor`, `predict_theta_batch`, `save_mlp`, or `load_mlp` in any `.py`/`.sh` file.
- Both kept shell scripts (`run_phase1.sh`, `run_phase2.sh`) pass `bash -n`; `run_phase2.py`
  passes `python -m py_compile`.
- Repo-wide grep confirms **no** remaining references to `run_phase3`, `sample_phase2_patients`,
  `run_phase23`, `run_all_phases`, `run_experiments`, or "phase 3" in any `.py`/`.sh` file.

> Not run end-to-end here: the full Phase 1 pipeline (`torch`, `sbi`, `simglucose`, and the
> `t1d_twin`/`experiments` package layout) wasn't executed in this environment, so a real
> `run_phase1` smoke run on your side is the final confirmation.
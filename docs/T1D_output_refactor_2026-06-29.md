# T1D twinning — output-layout standardization & per-patient IG figures

**Date:** 2026-06-29
**Scope:** `t1d_twin/` (library) and `experiments/` (runners) + the five SLURM
shell scripts at the repo root.

This refactor makes the four experiments (Phase 0 twins, Phase 0 ML, Phase 1,
Phase 2) write to **one consistent directory layout**, adds a **per-patient
IG-overlay figure**, renames each patient's `comparison_table.md` to a richer
`notes.md`, and **splits the SLURM logs by phase**. It also fixes a latent
skip-check bug in `run_phase2.sh`.

---

## 1. The standardized layout

Root is `$T1D_OUTPUT_ROOT` (default `/scratch/vmli3/t1d_experiment`).
`<phase>` is one of `phase0`, `phase0_ml`, `phase1`, `phase2`; `<name>` is a
patient's `safe_name`.

```
$T1D_OUTPUT_ROOT/
├── artifacts/
│   ├── population.npz                       # COMMON to all phases
│   ├── phase1/dataset.npz                   # experiment-specific dataset
│   ├── phase0_ml/dataset.npz                #   "
│   ├── phase2/<name>/{mcmc_twin,sbi_twin}.npz   # per-patient fitted twins
│   └── phase0/<name>/{mcmc_twin,sbi_twin}.npz   #   "
├── results/
│   ├── <phase>_summary.csv                  # per-phase aggregate (flat, all 4 phases)
│   ├── <phase>_per_patient.csv              # per-phase per-patient roll-up (flat)
│   └── <phase>/<name>/                      # per-patient detail (twin phases)
│       ├── comparison_table.csv
│       ├── notes.md
│       ├── ig_overlay_mcmc.png
│       └── ig_overlay_sbi.png
└── logs/
    └── <phase>/<jobname>_<jobid>.{out,err}  # e.g. logs/phase2/t1d_p2_patient_12345_3.out
```

There is **no `figures/` tree any more** — each patient's figures live beside
that patient's table and notes.

### What changed vs. before

| Concern | Before | After |
|---|---|---|
| Common prior | `artifacts/population.npz` | unchanged |
| Datasets | `artifacts/phase1_dataset.npz`, `artifacts/phase0_dataset.npz` | `artifacts/phase1/dataset.npz`, `artifacts/phase0_ml/dataset.npz` |
| Per-patient twins | `artifacts/<name>/…` (Phase 2, un-namespaced) vs `artifacts/phase0/<name>/…` | `artifacts/phase2/<name>/…` and `artifacts/phase0/<name>/…` (both namespaced) |
| Phase summary | `results/phase1/summary.csv` (amortized) vs `results/phase2_summary.csv` (twins) | `results/<phase>_summary.csv` for **all four** |
| Per-patient roll-up | `results/phase1/per_patient.csv` vs `results/phase2_per_patient.csv` | `results/<phase>_per_patient.csv` for all four |
| Per-patient detail | `results/<name>/…` (Phase 2) vs `results/phase0/<name>/…` | `results/<phase>/<name>/…` for both twin phases |
| Per-patient notes | `comparison_table.md` | `notes.md` (richer) |
| Per-patient figures | (none; `figures/` was created but never written) | `results/<phase>/<name>/ig_overlay_<method>.png` |
| Logs | `logs/<jobname>_<jobid>.…` (flat) | `logs/<phase>/<jobname>_<jobid>.…` |

The most consequential structural fix: Phase 2 used to drop per-patient dirs
straight under `results/` (un-namespaced), which is the only reason Phase 0 had
to defensively namespace itself. Now both twin phases namespace by phase, so
nothing collides and the two cohorts (which share patient names) pair cleanly by
`results/phase0/<name>/` ↔ `results/phase2/<name>/`.

---

## 2. New file — where it goes

- **`output_paths.py` → put it in `experiments/`.**

This is the single source of truth for the layout above. It is intentionally
**pure-stdlib** (no numpy / simglucose / `t1d_twin` imports) so the deliberately
simglucose-free Phase 0 ML / dataset-build path can import it without dragging in
the simulator. `exp_common` re-exports its constants and helpers, so simglucose-
side code keeps using `C.ARTIFACT_DIR`, `C.results_dir_for(...)`, etc. unchanged.

Public API: `OUTPUT_ROOT`, `ARTIFACT_DIR`, `RESULTS_DIR`, `LOG_DIR`; phase tags
`PHASE0`, `PHASE0_ML`, `PHASE1`, `PHASE2`, `PREP`; helpers `population_path()`,
`artifact_dir(phase, safe_name=None)`, `dataset_path(phase)`,
`twin_artifact_paths(phase, safe_name)`, `summary_path(phase)`,
`per_patient_path(phase)`, `results_dir(phase, safe_name)`, `log_dir(phase)`.

---

## 3. The per-patient IG figure

For each twinning method present (`mcmc`, `sbi`) a figure
`ig_overlay_<method>.png` is written into `results/<phase>/<name>/`. For **every
candidate therapy** it overlays:

- **solid line** = the plant's noise-free IG (`RunResult.bg()`, the exact series
  used in scoring) — labeled *simglucose* for Phase 2, *ReplayBG plant* for
  Phase 0;
- **dashed line** = the twin's replayed IG (`twin.replay_run(add_noise=False)`),

in **one color per therapy** over the 24 h window. The baseline therapy
(`bolus_x1.00`) is drawn in black and slightly heavier. The legend has two
blocks: therapy→color, and a solid/dashed key (plant vs twin). This mirrors the
published ReplayBG "Data (solid) vs Replay (dashed)" figure.

Implementation:
- `t1d_twin/plotting.py` gains `plot_therapy_ig_overlay(...)` (the figure) and
  `write_therapy_overlays(out_dir, true_ig, pred_ig_by_method, …)` (writes one
  PNG per method, building the shared time axis from `sample_time`).
- `t1d_twin/evaluate.py`: `evaluate_twin(...)` now also returns `true_ig` and
  `pred_ig` (the per-policy noise-free IG series) so the figure reuses the
  already-computed replays instead of recomputing them. Additive change — no
  existing key was removed.
- `experiments/compute_results.py` / `compute_results0.py`: `write_outputs(...)`
  now writes `comparison_table.csv` + `notes.md` + the overlays.

---

## 4. `notes.md` (was `comparison_table.md`)

Per patient, `notes.md` now summarizes the whole run: horizon/grid, methods
scored, the true therapy ranking, the comparison table, the decision-vs-fidelity
(DT2) read, and a list of the figures. `comparison_table.csv` is unchanged and is
still what the phase aggregator reads.

---

## 5. SLURM logs split by phase + a gotcha

Each script now writes to `logs/<phase>/%x_%j.{out,err}` (arrays:
`%x_%A_%a.…`), keeping your `<jobname>_<jobid>` naming. Because SLURM reads
`#SBATCH --output` **before** the job body runs and does **not** create the
directory, the per-phase log dirs must pre-exist:

- `run_prep.sh` now creates **all** of them up front
  (`logs/{prep,phase0,phase0_ml,phase1,phase2}`).
- The self-submitting scripts (`run_phase2.sh`, `run_phase0_twins.sh`) also
  `mkdir` their phase log dir on the login node before the inner `sbatch`.
- The directly-submitted scripts (`run_phase1.sh`, `run_phase0_ml.sh`) `mkdir`
  their dir in the body too, but their `#SBATCH --output` header still needs the
  dir to exist at submit time — so **run `run_prep.sh` first** (the documented
  workflow), or `mkdir -p $T1D_OUTPUT_ROOT/logs/<phase>` once before submitting.

---

## 6. Bug fix — `run_phase2.sh` resume skip-check

The Phase 2 per-patient skip-check tested a **relative, un-namespaced** path
(`results/${safe}/comparison_table.csv`) while `compute_results` actually writes
an absolute, phase-namespaced path. As written it would not match, so finished
patients were not skipped on resume. It now tests
`"$T1D_OUTPUT_ROOT/results/phase2/${safe}/comparison_table.csv"`, matching what
is written (`run_phase0_twins.sh` was already correct and is unchanged here apart
from the log/dir edits).

---

## 7. Full list of changed files

**`t1d_twin/` (2):**
- `evaluate.py` — `evaluate_twin` also returns `true_ig` / `pred_ig`.
- `plotting.py` — `plot_therapy_ig_overlay` + `write_therapy_overlays` (and a
  `matplotlib.lines.Line2D` import).

**`experiments/` (11, one of them new):**
- `output_paths.py` — **NEW**, the layout module.
- `exp_common.py` — paths now come from `output_paths`; `artifact_paths` /
  `results_dir_for` delegate to the Phase 2 layout; `FIG_DIR` / `fig_dir_for`
  removed; phase tags + `summary_path` / `per_patient_path` / `dataset_path` /
  `population_path` re-exported.
- `phase0_paths.py` — now delegates to `output_paths` (same function signatures,
  so its callers are untouched).
- `compute_results.py`, `compute_results0.py` — write `notes.md` + IG overlays.
- `build_phase1_dataset.py`, `build_phase0_dataset.py` — datasets now at
  `artifacts/<phase>/dataset.npz`.
- `run_phase1.py`, `run_phase0_ml.py` — write flat `results/<phase>_summary.csv`
  and `results/<phase>_per_patient.csv` (replacing the old `--out-dir` scheme).
- `run_phase2.py`, `run_phase0.py` — use the shared `summary_path` /
  `per_patient_path` helpers (same on-disk result as before).

**Repo-root SLURM scripts (5):**
- `run_prep.sh`, `run_phase1.sh`, `run_phase2.sh`, `run_phase0_twins.sh`,
  `run_phase0_ml.sh` — per-phase log dirs, new dataset paths, dropped `figures/`,
  plus the `run_phase2.sh` skip-check fix.

**Not modified** (they inherit the new layout for free through the delegating
path helpers): `run_mcmc.py`, `run_sbi.py`, `run_mcmc0.py`, `run_sbi0.py`,
`phase_runner.py`, `run_suite.py`, `population.py`.

---

## 8. Migration notes

- The layout keys off `$T1D_OUTPUT_ROOT`; existing artifacts at the **old**
  paths won't be found by the new code. Either re-run `run_prep.sh` (cheap steps
  are guarded/skipped) or move existing files:
  `artifacts/phase1_dataset.npz → artifacts/phase1/dataset.npz`,
  `artifacts/phase0_dataset.npz → artifacts/phase0_ml/dataset.npz`,
  `artifacts/<name>/ → artifacts/phase2/<name>/`,
  `results/<name>/ → results/phase2/<name>/`,
  `results/phase1/{summary,per_patient}.csv → results/phase1_{summary,per_patient}.csv`,
  `results/phase0_ml/{summary,per_patient}.csv → results/phase0_ml_{summary,per_patient}.csv`.
- Override the root anytime (e.g. a local dry-run):
  `T1D_OUTPUT_ROOT=./_out python -m experiments.run_phase1 …`.

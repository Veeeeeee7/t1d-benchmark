# T1D twinning — structural cleanup: dead-code removal, de-duplication & consistent naming

*Companion to `T1D_output_refactor_2026-06-29.md`. That refactor standardized the
on-disk output layout (the `output_paths` module, the flat `results/` CSVs, the
per-patient IG figure). This one is a pure **source-tree** cleanup applied on top
of that state: nothing about the output layout, the metrics, or the science
changes. The goal was a smaller, more legible directory — no dead files, one home
for each shared helper, and filenames that say what they are.*

All edits are correctness-preserving by construction: every `.py` passes
`python -m py_compile`, every `.sh` passes `bash -n`, and an exhaustive grep
confirms zero references to any removed or renamed module remain (no module name
is ever built dynamically — every `python -m experiments.X`, every `stages`
entry, and every `RESULTS_MODULE` is a literal string — so the textual renames
are complete and safe).

---

## 1. Files removed (6)

| File | Why it's gone |
|---|---|
| `t1d_twin/run_all.py` | Legacy single-subject driver from before the phased pipeline. Its `main()` was superseded by `run_phase2` / `run_phase0_twins`; the only piece still in use was the `_df_to_markdown` table formatter, which has moved to `evaluate.py` (see §2). |
| `t1d_twin/scenarios.py` | `single_meal_scenario` was imported by **only** `run_all.py`. Dead once `run_all` is gone. |
| `experiments/run_suite.py` | Older "iterate every patient on one machine" driver. Superseded by `phase_runner` + the `run_phase2` / `run_phase0_twins` aggregate runners, which do the same sweep with subprocess isolation and resume support. Was referenced only in stale comments. |
| `experiments/run_multipatient.py` | Earlier multi-patient experiment script, imported by nobody and not in any shell pipeline. Fully superseded. |
| `experiments/dt2_fairness.py` | One-off fairness-aggregation script, imported by nobody and invoked by nothing. Dead. |
| `experiments/phase0_paths.py` | Thin Phase-0 path-namespacing shim. Folded into `output_paths` (see §3) — the layout module already takes a phase tag, so a separate Phase-0 module was redundant. |

Each removal was verified against the full import graph **and** the shell
pipeline (`python -m …` invocations) before deletion.

---

## 2. One home for shared report/metric helpers → `t1d_twin/evaluate.py`

Several small helpers were copy-pasted across modules or stranded in `run_all`.
They now live once in `evaluate.py`, next to `TABLE_COLUMNS` and the metric code
they belong with:

- **`DECISION_COLS` / `FIDELITY_COLS`** — the split of the four metrics into
  decision-quality (`spearman`, `regret`) vs trajectory-fidelity (`rmse`,
  `mard`). These were defined **identically in three places** (`phase_runner.py`,
  `run_phase1.py`, `run_phase0_ml.py`). Now defined once and imported.
- **`df_to_markdown(table, floatfmt)`** — DataFrame→GitHub-Markdown renderer,
  extracted from the deleted `run_all._df_to_markdown`. Imported by both
  per-patient results modules.
- **`dt2_summary(table)`** — the decision-vs-fidelity "dissociation" check
  (the headline DT2 result). This was **byte-identical** in
  `compute_results.py` and `compute_results0.py`; both now import the one copy.

Net effect: the two results modules dropped ~20 lines each and no longer reach
into `run_all`.

---

## 3. One set of path helpers, parameterized by phase (not one per phase)

This addresses the "duplicate functions doing the same thing for different
systems" smell directly. Per-subject artifact/results paths used to come from
**two parallel pairs** of functions that differed only by phase:

```
exp_common.artifact_paths(subject)    / .results_dir_for(subject)   # Phase 2
phase0_paths.artifact_paths(subject)  / .results_dir_for(subject)   # Phase 0
```

Both pairs already delegated to the same `output_paths` layout, so they were
pure duplication. They are removed; every caller now goes straight to the
phase-tagged layout helpers:

```python
OP.twin_artifact_paths(OP.PHASE2, subject.safe_name)   # or OP.PHASE0
OP.results_dir(OP.PHASE2, subject.safe_name)           # or OP.PHASE0
```

Callers updated: `run_mcmc_phase2`, `run_sbi_phase2`, `compute_results_phase2`
(→ `PHASE2`); `run_mcmc_phase0`, `run_sbi_phase0`, `compute_results_phase0`,
`run_phase0_twins` (→ `PHASE0`); and `phase_runner`'s default
`results_dir_fn`. `exp_common` still re-exports the `output_paths` **constants**
(`ARTIFACT_DIR`, `OUTPUT_ROOT`, …) so existing `C.ARTIFACT_DIR`-style references
keep working; only the redundant per-subject wrapper functions are gone.

`output_paths` remains pure-stdlib (no numpy / simglucose / t1d_twin imports) so
the deliberately simglucose-free Phase-0 dataset path
(`build_phase0_dataset`, `replaybg_plant`) can keep importing it.

---

## 4. `plotting.py` trimmed: 675 → 153 lines

Eight plotting functions were orphaned (defined, never called by any runner —
left over from earlier exploratory notebooks):

`plot_candidate_runs`, `plot_forward_trajectory`, `plot_twin_overlay`,
`plot_cgm_overlay`, `plot_twin_ig_cgm`, `plot_ranking_scatter`,
`plot_regret_bars`, `plot_decision_vs_fidelity`.

All removed. The module now contains only the two functions the pipeline
actually calls — `plot_therapy_ig_overlay` and `write_therapy_overlays` (the
per-patient plant-vs-twin IG overlay introduced in the output refactor) — plus
the shared header/imports and the `TARGET_LOW/TARGET_HIGH` band constants.

---

## 5. Consistent filenames (the cryptic `0` suffix is gone)

The per-patient stage scripts mixed two conventions: Phase 2 files had **no**
suffix while their Phase 0 twins carried a bare `0`, and the Phase-0 aggregate
runner's filename didn't match its own shell script. Every phase-specific script
is now suffixed with its phase tag, so the filename states which phase it drives.

| Old name | New name | Role |
|---|---|---|
| `experiments/run_mcmc.py` | `experiments/run_mcmc_phase2.py` | per-patient MCMC fit, Phase 2 (simglucose plant) |
| `experiments/run_sbi.py` | `experiments/run_sbi_phase2.py` | per-patient SBI fit, Phase 2 |
| `experiments/compute_results.py` | `experiments/compute_results_phase2.py` | per-patient scoring, Phase 2 |
| `experiments/run_mcmc0.py` | `experiments/run_mcmc_phase0.py` | per-patient MCMC fit, Phase 0 (matched ReplayBG plant) |
| `experiments/run_sbi0.py` | `experiments/run_sbi_phase0.py` | per-patient SBI fit, Phase 0 |
| `experiments/compute_results0.py` | `experiments/compute_results_phase0.py` | per-patient scoring, Phase 0 |
| `experiments/run_phase0.py` | `experiments/run_phase0_twins.py` | Phase 0 aggregate runner — now matches `run_phase0_twins.sh` and parallels `run_phase0_ml.py` |

The aggregate runners now read cleanly as a set:
`run_phase0_twins.py`, `run_phase0_ml.py`, `run_phase1.py`, `run_phase2.py`,
each paired with a like-named `run_*.sh`. The prep/dataset builders
(`build_phase0_dataset.py`, `build_phase1_dataset.py`,
`derive_replaybg_params.py`, `generate_patients.py`) were already consistent and
are unchanged.

Every reference was updated in lock-step: the `python -m experiments.X` lines in
`run_phase2.sh` / `run_phase0_twins.sh`, the `STAGES` / `RESULTS_MODULE` literals
in `phase_runner.py` (Phase 2) and `run_phase0_twins.py` (Phase 0), and all
docstring/comment mentions. Word-boundary-aware substitution was used so the
`0` → `_phase0` change never touched `build_phase0_dataset`, and the
`run_phase0` → `run_phase0_twins` change never touched `run_phase0_ml`.

---

## 6. Tools kept on purpose (not legacy)

Three standalone scripts look peripheral but are live or genuinely useful, so
they stay:

- **`experiments/dt2_scoring.py`** — used by `run_phase1` (`SC.collect_truth`,
  `SC.score_twin`). It is the Phase-1-specific truth-collection + scoring helper
  (simglucose truth, cached per patient). Live.
- **`experiments/generate_phase0_patients.py`** — generates a *native* (prior-
  draw) Phase-0 cohort, an alternative to the matched cohort built by
  `generate_patients` + `derive_replaybg_params`. Referenced by name in a
  `replaybg_plant.py` error message, and a valid manual workflow. Kept.
- **`experiments/inspect_reward_separation.py`** — standalone diagnostic for
  reward separation; imports only stable modules (`exp_common`, `patients`,
  `value`). Harmless and handy. Kept.

---

## 7. Comments

Stale comments left dangling by the moves above were corrected, not just the
renamed-module mentions:

- Removed every reference to the deleted `run_suite` (e.g. the
  `compute_results_phase2` docstring, the `exp_common` argparse help text — now
  "use `run_phase2` / `run_phase0_twins` to iterate all rows" — and the
  `phase_runner` "crash-tolerance" note).
- Updated the `exp_common` layout comment to describe per-subject paths coming
  from `output_paths` directly (parameterized by phase) rather than the removed
  wrappers.
- Refreshed the `plotting` module docstring to describe its now-two functions.
- Rewrote the `phase_runner` `results_dir_fn` docstring (no more
  `phase0_paths`).

---

## 8. Where each file goes

Drop the package folders in over your repo; copy the shell scripts and this doc
to the repo root.

**`t1d_twin/`** (changed): `evaluate.py`, `plotting.py`.
**`experiments/`** (changed / renamed / new): `exp_common.py`, `output_paths.py`,
`phase_runner.py`, `compute_results_phase2.py`, `compute_results_phase0.py`,
`run_mcmc_phase2.py`, `run_sbi_phase2.py`, `run_mcmc_phase0.py`,
`run_sbi_phase0.py`, `run_phase0_twins.py`, `run_phase1.py`, `run_phase0_ml.py`.
**repo root** (changed): `run_phase2.sh`, `run_phase0_twins.sh` (+ the unchanged
`run_phase1.sh`, `run_phase0_ml.sh`, `run_prep.sh` for completeness).

### Files to DELETE from your repo

```
t1d_twin/run_all.py
t1d_twin/scenarios.py
experiments/run_suite.py
experiments/run_multipatient.py
experiments/dt2_fairness.py
experiments/phase0_paths.py
experiments/run_mcmc.py            # renamed -> run_mcmc_phase2.py
experiments/run_sbi.py             # renamed -> run_sbi_phase2.py
experiments/compute_results.py     # renamed -> compute_results_phase2.py
experiments/run_mcmc0.py           # renamed -> run_mcmc_phase0.py
experiments/run_sbi0.py            # renamed -> run_sbi_phase0.py
experiments/compute_results0.py    # renamed -> compute_results_phase0.py
experiments/run_phase0.py          # renamed -> run_phase0_twins.py
```

### Migration as `git mv` (preserves history for the renames)

```bash
# renames (history-preserving)
git mv experiments/run_mcmc.py        experiments/run_mcmc_phase2.py
git mv experiments/run_sbi.py         experiments/run_sbi_phase2.py
git mv experiments/compute_results.py experiments/compute_results_phase2.py
git mv experiments/run_mcmc0.py       experiments/run_mcmc_phase0.py
git mv experiments/run_sbi0.py        experiments/run_sbi_phase0.py
git mv experiments/compute_results0.py experiments/compute_results_phase0.py
git mv experiments/run_phase0.py      experiments/run_phase0_twins.py

# deletions
git rm t1d_twin/run_all.py t1d_twin/scenarios.py \
       experiments/run_suite.py experiments/run_multipatient.py \
       experiments/dt2_fairness.py experiments/phase0_paths.py
```

Then copy the changed file contents from this delivery over the (now-renamed)
files. Re-running `run_prep.sh` is **not** required — no output paths changed in
this refactor; only the source tree did.

---

## 9. Verification performed

- `python -m py_compile` on all 40 modules — clean.
- `bash -n` on all 5 shell scripts — clean.
- Grep sweeps confirming: no references to the 6 removed modules; no references
  to any old (pre-rename) module name in either `python -m` strings, `STAGES` /
  `RESULTS_MODULE` literals, or prose; no double-suffix corruption
  (`_phase2_phase0` etc.); no remaining duplicate `DECISION_COLS` / `dt2_summary`
  definitions.

(The simulation itself can't be executed in this environment — no simglucose /
torch / sbi / network — so correctness here is by construction + static checks,
same basis as the companion output refactor.)

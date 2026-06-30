# T1D Benchmark — Refactor & Tuning Summary

A record of the changes made to the phase-1 / phase-3 pipeline: the SLURM
orchestration, parallelism, output locations, job sizing, and the move to a
short 24 h identification window without a 1-D CNN.

---

## 1. Launch model & orchestration

The original single `run_experiments.sh` was split into two launchers.

- **`run_phase1.sh`** — one CPU job for the amortized MLP baseline (dataset
  build + K-fold CV + scoring). Submitted directly with `sbatch run_phase1.sh`.
- **`run_phase3.sh`** — a login-node *submitter* (run with `bash run_phase3.sh`,
  **not** `sbatch`) that fires off a dependency chain:
  `setup → patient array → aggregate`. Phase 2 is skipped; phase 3 runs on all
  patients.

### `sbatch` vs `bash` bug (the original error)
`run_phase23.sh` failed with *"did not specify a partition / missed both memory
and GPU"* because it was launched with `sbatch`. It is a **submitter** meant for
`bash`; under `sbatch`, SLURM scheduled the submitter itself and read its header,
which lacked `--partition`/`--mem`. Fix: launch with `bash`, and a safety-net
`#SBATCH` header (partition + small mem) was added so an accidental `sbatch`
no longer crashes. The real per-stage resources live on the inner `sbatch` lines.

---

## 2. Parallelism (CPU)

Three embarrassingly-parallel per-patient loops were fanned across cores with a
`fork` process pool. Each worker stays **single-threaded** (`OMP/OPENBLAS/MKL/
NUMEXPR=1`, plus `torch.set_num_threads(1)`) so N processes don't each spawn N
threads and oversubscribe the node.

| Loop | File | What was parallelized |
|------|------|------------------------|
| Phase-1 dataset build | `build_phase1_dataset.py` | per-patient baseline sim + ReplayBG LS projection |
| Phase-1 scoring | `run_phase1.py` | per-patient `collect_truth` + twin scoring (CV stays serial — it's cheap) |
| Population prior fit | `population.py` | per-patient ReplayBG least-squares fit (was the multi-hour serial step) |

Each gained a `-j/--jobs` flag and an `n_workers()` resolver
(`--jobs` > `SLURM_CPUS_PER_TASK` > `cpu_count`). The GPU idea for phase 1 was
dropped: the MLP/CV is CPU-bound and cheap; the real cost is serial simglucose,
which the pool addresses.

---

## 3. SLURM job sizing (the 50-job limit)

To stay under a **50-job** cap, the phase-3 per-patient array was repacked:

- **`PER_TASK=32`** — 32 patients per array task ⇒ 1013 patients → **32 array
  tasks** (+ setup + aggregate = 34 jobs submitted).
- Each array task: **`--cpus-per-task=32 --mem=128G --time=7-00:00:00`**
  (4 GB/patient, 7-day walltime).
- The task's 32 patients run **concurrently, one per core** (backgrounded with a
  `wait -n` throttle and process substitution so the loop sees its jobs).
- `CONCURRENCY` default lowered to 48 to respect the cap.
- The population `setup` job uses a whole node (`--cpus-per-task=64 --mem=480G`)
  for the parallel prior fit.

Resumability preserved: a patient with an existing `comparison_table.csv` is
skipped, so re-launching only redoes failures. (Trade-off: a node/OOM failure
now takes down all 32 patients in that task at once — drop `PER_TASK` if you hit
OOM.)

---

## 4. Output locations → scratch

All generated outputs now live under one root, driven by an env var with a
default of `/scratch/vmli3/t1d_experiment`:

```
$T1D_OUTPUT_ROOT/
├── artifacts/   # population.npz, phase1_dataset.npz, per-patient twins
├── results/     # comparison tables, phase summaries
├── figures/     # per-patient figures
├── logs/        # SLURM .out/.err
└── sbi_logs/    # SBI TensorBoard logs (per job/task subdir)
```

- **`exp_common.py`** — `ARTIFACT_DIR/RESULTS_DIR/FIG_DIR/LOG_DIR` derive from
  `OUTPUT_ROOT = os.environ.get("T1D_OUTPUT_ROOT", "/scratch/vmli3/t1d_experiment")`.
  Because `artifact_paths`/`results_dir_for`/`fig_dir_for` build on these, this
  one edit relocates outputs for the whole codebase.
- **`population.py`** — `--out` default now uses `C.ARTIFACT_DIR`.
- **`run_phase1.sh` / `run_phase3.sh`** — `DATASET`/`POP` defaults and SLURM log
  paths moved under the root; scripts `mkdir -p` the tree and `export
  T1D_OUTPUT_ROOT` so the Python side agrees.
- **`identify_sbi.py`** — the `sbi` library's TensorBoard `SummaryWriter` (which
  defaulted to a `sbi_logs` folder in the CWD) is now pointed at
  `$T1D_OUTPUT_ROOT/sbi_logs/<job-or-timestamp>`. Done by reading the env var
  (not importing `exp_common`), keeping the `t1d_twin` library decoupled.

Override for local runs: `T1D_OUTPUT_ROOT=./_out python -m experiments...`.
Note: the two `#SBATCH --output=` *header* lines are literal scratch paths (SLURM
can't read shell vars), so they only matter for an accidental direct `sbatch`.

---

## 5. Modeling change: 24 h window, no 1-D CNN

The CGM was too long/dense (a week = 3360 points at 3-min), which was the reason
both phases used a 1-D CNN to compress it. We switched to a short window and fed
the CGM directly.

### Shorter identification window
- **`exp_common.py`** — production horizon changed from `WEEK_HOURS=168` to
  **`WINDOW_HOURS=24.0`** (1 day, exactly 3 meals; ~480 CGM points). `hours_for`
  now returns it; a `WEEK_HOURS = WINDOW_HOURS` alias keeps old references valid.
  MCMC needed no change — it identifies on `hours_for(...)` automatically.

### Removed the 1-D CNN (both phases)
- **Phase 1 — `identify_mlp.py`:** `CGMRegressor` was a Conv1d front-end; it is
  now a plain **MLP** that flattens `(B, C, L)` and runs
  `Linear(C·L → hidden → hidden → out)`. It takes a new `in_len` arg (the fixed
  per-channel length); `mlp_cv.py` passes `X.shape[-1]` (= 480) at the one call
  site.
- **Phase 3 — `run_sbi.py`:** removed the `CGMEmbedding` CNN; the MAF now
  conditions on the **raw ~480-point CGM** directly (sbi's per-dimension
  z-scoring handles the mg/dL scale). Dropped the dead `emb_out` config and
  `torch.nn` import.

---

## 6. How to run

```bash
# one-time: create the scratch output tree (needed for direct-sbatch logs)
mkdir -p /scratch/vmli3/t1d_experiment/{artifacts,results,figures,logs}

# Phase 1 (single CPU job, all cores)
sbatch run_phase1.sh

# Phase 3 (parallel population prior fit -> per-patient array -> aggregate)
bash run_phase3.sh
```

Useful env overrides: `T1D_OUTPUT_ROOT`, `PER_TASK`, `CONCURRENCY`, `POP_HOURS`,
`POP_LIMIT`, `JOBS`/`-j`.

### ⚠️ Clear stale caches before rerunning at 24 h
Previous artifacts were produced at the 168 h horizon and will be silently
reused:

```bash
rm -f  $T1D_OUTPUT_ROOT/artifacts/phase1_dataset.npz   # rebuild dataset at 24 h
rm -rf $T1D_OUTPUT_ROOT/results/*                       # drop old per-patient tables
# also clear old per-patient twins under $T1D_OUTPUT_ROOT/artifacts/<patient>/
```

---

## 7. Files changed

| File | Change |
|------|--------|
| `exp_common.py` | output root → scratch (env-driven); horizon → 24 h window |
| `population.py` | parallel per-patient prior fit (`--jobs`); `--out` under `ARTIFACT_DIR` |
| `build_phase1_dataset.py` | parallel build (`--jobs`); 24 h default horizon |
| `run_phase1.py` | parallel scoring loop (`--jobs`); 24 h fallback |
| `mlp_cv.py` | pass `in_len` to the MLP regressor |
| `identify_mlp.py` | `CGMRegressor`: 1-D CNN → plain MLP |
| `run_sbi.py` | removed CNN embedding (raw CGM → MAF); `torch.set_num_threads(1)`; 24 h docstring |
| `identify_sbi.py` | `sbi_logs` → scratch via `SummaryWriter(log_dir=...)` |
| `run_phase1.sh` | CPU partition, all cores, scratch paths/logs |
| `run_phase3.sh` | submitter chain; parallel prior fit; 32×(32-CPU/128 GB/7-day) array; scratch paths |
| `run_phase3_fixedpop.sh` | (alternate) skip population fit, assert the 6 fixed params, phase 3 on all |

---

## 8. Validation status & open items

- **Validated here:** `bash -n` on all shell scripts; `py_compile` on all Python;
  path/tag resolution for the scratch root and `sbi_logs`; the in-task parallel
  throttle; MLP forward-shape logic.
- **Not executed** (simglucose / torch / sbi absent in the dev sandbox): no
  end-to-end run, training step, or memory profile. Recommended smoke checks on
  an interactive node before a full submit:
  ```bash
  python -m experiments.run_sbi  --smoke --patient <name> --patients patients.csv
  python -m experiments.run_phase1 --smoke --limit 8 -j 4
  POP_LIMIT=8 python -m experiments.population --patients patients.csv -j 4 --out /tmp/pop.npz
  ```
- **Watch on first real run:** per-patient `MaxRSS` at 32-way concurrency
  (`sacct -j <id> --format=MaxRSS`) to confirm < 128 GB; that `c64-m512` allows a
  7-day walltime; and that the installed `sbi` accepts the embedding-free
  `posterior_nn` (it does for the classic `SNPE` API in use).

### Optional follow-ups (offered, not yet done)
- Propagate the scratch-path + log edits to `run_phase23.sh` and
  `run_phase3_fixedpop.sh` for consistency.
- Make the population prior fit use the same 24 h window (`POP_HOURS=24`) by
  default.
- Add `sbi_logs` to the up-front `mkdir` so the empty dir is visible pre-run.

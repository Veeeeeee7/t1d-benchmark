#!/bin/bash
# =============================================================================
# run_prep.sh — one-shot PREPARATION for Phases 0, 1 and 2.
#
# Builds every shared, generated prerequisite ONCE on a single c64-m512 node
# (all cores), so the per-phase run scripts just consume them and their own
# prep steps become guarded no-ops. CPU only (no GPU needed for prep).
#
# Produces (under $T1D_OUTPUT_ROOT/artifacts unless noted):
#   1. patients.csv                 the UVA/Padova cohort + carb errors   [all phases]
#   2. patients.csv  + rbg_* cols   best-fit ReplayBG params per patient  [phase 0 plant;
#                                                                          phase 1 target;
#                                                                          phase 2 aggregates]
#   3. population.npz               population centre + prior (aggregates  [all phases]
#                                   the step-2 rbg_* fits; NO re-fit)
#   4. phase1/dataset.npz           amortized dataset (simglucose plant)   [phase 1]
#   5. phase0_ml/dataset.npz        amortized dataset (ReplayBG plant)     [phase 0 ML]
#
# Ordering matters: the rbg_* fit (step 2) is the ONE per-patient least-squares
# pass, run BEFORE the population (step 3, which just aggregates those cached
# fits), before build_phase1_dataset (step 4, reads the cached fit as its target),
# and before build_phase0_dataset (step 5, the matched cohort). Every step is
# guarded, so re-running skips whatever already exists.
#
# After this finishes, launch the phases (their own prep will skip):
#       sbatch run_phase1.sh
#       bash   run_phase2.sh
#       bash   run_phase0_twins.sh
#       sbatch run_phase0_ml.sh
#
# Submit:
#       sbatch run_prep.sh
#
# Env overrides: CONDA_ENV, PATIENTS, POP, DS1, DS0, JOBS, POP_LIMIT,
#                DERIVE_HOURS.
# =============================================================================

#SBATCH --job-name=t1d_prep
#SBATCH --account=ai-gpu
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/vmli3/t1d_experiment/logs/prep/%x_%j.out
#SBATCH --error=/scratch/vmli3/t1d_experiment/logs/prep/%x_%j.err
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

set -uo pipefail

CONDA_ENV="${CONDA_ENV:-t1d_benchmark}"
PATIENTS="${PATIENTS:-patients.csv}"

# All generated outputs (artifacts/results/logs) live under one root.
# Exported so the Python side (experiments.exp_common) writes to the same place.
# NOTE: the #SBATCH --output/--error paths above are read by SLURM *before* this
# runs, so if you override T1D_OUTPUT_ROOT also edit those two header lines.
export T1D_OUTPUT_ROOT="${T1D_OUTPUT_ROOT:-/scratch/vmli3/t1d_experiment}"
LOG_DIR="$T1D_OUTPUT_ROOT/logs"

# Match the default output paths the per-phase scripts look for.
POP="${POP:-$T1D_OUTPUT_ROOT/artifacts/population.npz}"            # phase 2 prior
DS1="${DS1:-$T1D_OUTPUT_ROOT/artifacts/phase1/dataset.npz}"        # phase 1 dataset
DS0="${DS0:-$T1D_OUTPUT_ROOT/artifacts/phase0_ml/dataset.npz}"     # phase 0 ML dataset

POP_LIMIT="${POP_LIMIT:-}"          # empty = aggregate the prior over ALL patients
DERIVE_HOURS="${DERIVE_HOURS:-24}"  # horizon for rbg_* fit; MUST match build_phase1's hours

# /etc/bashrc + conda activation reference unset vars; relax -u around sourcing.
set +u
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u

cd "${SLURM_SUBMIT_DIR:-.}"

# One thread per worker; parallelism comes from the per-patient process pools.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$USER"
mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs}
# Per-phase SLURM log dirs must pre-exist (SLURM reads #SBATCH --output before
# the job body runs and will NOT create them). Prep makes ALL of them once so
mkdir -p "$T1D_OUTPUT_ROOT"/logs/{prep,phase0,phase0_ml,phase1,phase2}

JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
echo "[prep] $JOBS worker process(es) | patients=$PATIENTS | $(date)"

if [[ ! -d experiments ]]; then echo "run from the repo root" >&2; exit 1; fi

# --- 1. cohort --------------------------------------------------------
if [[ -f "$PATIENTS" ]]; then
    echo "[prep 1/5] cohort exists: $PATIENTS"
else
    echo "[prep 1/5] generating cohort -> $PATIENTS"
    python -m experiments.generate_patients --out "$PATIENTS"
fi
N=$(( $(wc -l < "$PATIENTS") - 1 ))
echo "[prep] cohort N=$N patients"

# --- 2. per-patient ReplayBG fit -> rbg_* columns (phase 0; phase 1 reuses) ---
if head -1 "$PATIENTS" | grep -q 'rbg_SI'; then
    echo "[prep 2/5] $PATIENTS already has rbg_* columns — skipping derivation"
else
    echo "[prep 2/5] deriving best-fit ReplayBG params (${DERIVE_HOURS} h) -> rbg_* cols"
    python -m experiments.derive_replaybg_params --patients "$PATIENTS" \
        --hours "$DERIVE_HOURS" --jobs "$JOBS"
fi

# --- 3. population prior (all phases: centre + prior box) -------------
# Aggregates the cached per-patient rbg_* fits from step 2 — NO re-fitting — so
# the population centre and prior live at the SAME 24 h LS parameters used by the
# Phase 0 matched plant and the Phase 1 regression target. (Requires step 2.)
if [[ -f "$POP" ]]; then
    echo "[prep 3/5] population prior exists: $POP"
else
    echo "[prep 3/5] aggregating cached ReplayBG fits -> $POP"
    LIM=(); [[ -n "$POP_LIMIT" ]] && LIM=(--limit "$POP_LIMIT")
    python -m experiments.population --patients "$PATIENTS" \
        "${LIM[@]}" --out "$POP"
fi

# --- 4. phase 1 amortized dataset (simglucose plant; reuses rbg_* cache) ---
if [[ -f "$DS1" ]]; then
    echo "[prep 4/5] phase 1 dataset exists: $DS1"
else
    echo "[prep 4/5] building phase 1 dataset -> $DS1"
    python -m experiments.build_phase1_dataset \
        --patients "$PATIENTS" --out "$DS1" --jobs "$JOBS"
fi

# --- 5. phase 0 amortized dataset (ReplayBG plant; matched cohort) ----
if [[ -f "$DS0" ]]; then
    echo "[prep 5/5] phase 0 dataset exists: $DS0"
else
    echo "[prep 5/5] building phase 0 dataset -> $DS0"
    python -m experiments.build_phase0_dataset \
        --patients "$PATIENTS" --out "$DS0" --jobs "$JOBS"
fi

echo ""
echo "[prep] DONE $(date). Prerequisites ready:"
echo "         cohort + rbg_* : $PATIENTS"
echo "         population prior: $POP"
echo "         phase 1 dataset : $DS1"
echo "         phase 0 dataset : $DS0"
echo "[prep] now launch: sbatch run_phase1.sh ; bash run_phase2.sh ; "
echo "                   bash run_phase0_twins.sh ; sbatch run_phase0_ml.sh"
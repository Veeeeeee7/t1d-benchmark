#!/bin/bash
# =============================================================================
# run_phase0_ml.sh — Phase 0 AMORTIZED baselines (TCN + ridge), matched model.
#
# The Phase 0 analogue of run_phase1.sh: a single job on one b200 GPU node. The
# CV that trains the baselines uses the GPU; the two expensive CPU parts are
# fanned across all cores of the node:
#     * build_phase0_dataset : run each therapy through the ReplayBG plant / patient
#     * run_phase0_ml scoring : collect ReplayBG ground truth + score / patient
# Difference from Phase 1: the plant is ReplayBG (no simglucose) and the labels
# are the exact true theta (no projection error).
#
# Prereq: patients.csv must carry the rbg_* columns (the matched cohort). This
# script derives them if missing (guarded), but if you are also running
# run_phase0_twins.sh, let its setup stage derive them once first and this job
# will skip straight to the dataset + CV.
#
# Submit:
#     sbatch run_phase0_ml.sh
#
# Env overrides: CONDA_ENV, PATIENTS, DATASET, JOBS (worker count; default = all
# allocated cores), DERIVE_HOURS.
# =============================================================================

#SBATCH --job-name=t1d_phase0_ml
#SBATCH --account=ai-gpu
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/vmli3/t1d_experiment/logs/phase0_ml/%x_%j.out
#SBATCH --error=/scratch/vmli3/t1d_experiment/logs/phase0_ml/%x_%j.err
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
DATASET="${DATASET:-$T1D_OUTPUT_ROOT/artifacts/phase0_ml/dataset.npz}"
DERIVE_HOURS="${DERIVE_HOURS:-24}"

# /etc/bashrc + conda activation reference unset vars; relax -u around sourcing.
set +u
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u

cd "${SLURM_SUBMIT_DIR:-.}"

# CRITICAL: one thread per worker. Parallelism comes from the process pool (one
# process per patient, ~64 at a time) for the CPU parts and from the GPU for CV
# training — so pin every math library to a single thread to avoid 64x64
# oversubscription on the dataset build / scoring.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$USER"
mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs}
mkdir -p "$T1D_OUTPUT_ROOT/logs/phase0_ml"   # this phase's SLURM log dir

# worker count = all allocated cores unless JOBS overrides it
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
echo "[phase0-ml] using $JOBS worker process(es) on partition b200 (GPU for CV)"

if [[ ! -d experiments ]]; then echo "run from the repo root" >&2; exit 1; fi

# prep: cohort + matched-model rbg_* columns. The derivation is CPU+simglucose;
# it is skipped if patients.csv already has the columns (e.g. from the twins
# setup stage), so re-runs and a parallel twins sweep cost nothing here.
[[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
if ! head -1 "$PATIENTS" | grep -q 'rbg_SI'; then
    echo "[phase0-ml] deriving ReplayBG fits -> rbg_* columns in $PATIENTS"
    python -m experiments.derive_replaybg_params --patients "$PATIENTS" \
        --hours "$DERIVE_HOURS" --jobs "$JOBS"
fi

# dataset build is parallel and guarded so re-runs skip it
[[ -f "$DATASET" ]] || python -m experiments.build_phase0_dataset \
    --patients "$PATIENTS" --out "$DATASET" --jobs "$JOBS"

# phase 0 (amortized): K-fold CV for both baselines (GPU) + parallel per-patient
# scoring against the ReplayBG ground truth (CPU).
python -m experiments.run_phase0_ml --patients "$PATIENTS" --dataset "$DATASET" \
    --device cuda --jobs "$JOBS"
echo "[phase0-ml] done -> results/phase0_ml_summary.csv"
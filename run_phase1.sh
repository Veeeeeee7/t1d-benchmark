#!/bin/bash
# =============================================================================
# run_phase1.sh — Phase 1 only (amortized baselines: K-fold CV + scoring).
#
# Single CPU job on one whole c64-m512 node. The two expensive parts of phase 1
# are pure simglucose and are now fanned across all cores of the node:
#     * build_phase1_dataset : one baseline sim + ReplayBG projection / patient
#     * run_phase1 scoring    : collect_truth (|Pi| x 168 h) + score / patient
# The K-fold CV that trains the baselines is cheap (~seconds) and stays serial.
#
# Parallelism model: ONE process per core (set by --cpus-per-task), each pinned
# to a SINGLE BLAS/OMP thread so 64 worker processes don't each spawn 64 threads
# and oversubscribe the node. The Python side auto-detects the core count from
# SLURM_CPUS_PER_TASK, so there is nothing else to tune.
#
# Submit:
#     sbatch run_phase1.sh
#
# Env overrides: CONDA_ENV, PATIENTS, DATASET, JOBS (worker count; default = all
# allocated cores).
# =============================================================================

#SBATCH --job-name=t1d_phase1
#SBATCH --account=ai-gpu
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/vmli3/t1d_experiment/logs/phase1/%x_%j.out
#SBATCH --error=/scratch/vmli3/t1d_experiment/logs/phase1/%x_%j.err
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
DATASET="${DATASET:-$T1D_OUTPUT_ROOT/artifacts/phase1/dataset.npz}"

# /etc/bashrc + conda activation reference unset vars; relax -u around sourcing.
set +u
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u

cd "${SLURM_SUBMIT_DIR:-.}"

# CRITICAL: one thread per worker. Parallelism comes from the process pool (one
# process per patient, ~64 at a time), NOT from threaded BLAS — so pin every
# math library to a single thread to avoid 64x64 oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$USER"
mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs}
mkdir -p "$T1D_OUTPUT_ROOT/logs/phase1"   # this phase's SLURM log dir

# worker count = all allocated cores unless JOBS overrides it
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
echo "[phase1] using $JOBS worker process(es) on partition c64-m512"

if [[ ! -d experiments ]]; then echo "run from the repo root" >&2; exit 1; fi

# prep (cohort cheap; dataset build is parallel and guarded so re-runs skip it)
[[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
[[ -f "$DATASET" ]]  || python -m experiments.build_phase1_dataset \
    --patients "$PATIENTS" --out "$DATASET" --jobs "$JOBS"

# phase 1: K-fold CV for both baselines (serial) + parallel per-patient scoring
python -m experiments.run_phase1 --patients "$PATIENTS" --dataset "$DATASET" --jobs "$JOBS"
echo "[phase1] done -> results/phase1_summary.csv"
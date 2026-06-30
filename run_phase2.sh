#!/bin/bash
# =============================================================================
# run_phase2.sh — population prior fit (PARALLEL, all patients) -> Phase 2 on
# all patients.
#
# Pipeline (afterok dependency chain):
#
#     setup (1 node, all cores)   ->   patient array   ->   aggregate
#
#   setup     : generate_patients + derive the ReplayBG population prior by
#               least-squares-fitting EVERY patient's baseline, FANNED across all
#               cores of one c64-m512 node (this is the step that used to take
#               many hours serially). Writes population.npz.
#   patient   : one SLURM array task per patient (1 core + 4 GB) — MCMC + SBI
#               twin fit (using --population) + scoring. Resumable.
#   aggregate : phase-2 mean +/- std summary once every patient is scored.
#
# The 6 structural ReplayBG parameters stay fixed in t1d_twin/replaybg_model.py;
# this fits the PRIOR over the 8 free parameters from the patient cohort.
#
# Launch from the LOGIN node (NOT sbatch):
#       bash run_phase2.sh
#
# Env overrides: CONDA_ENV, PATIENTS, POP, PARTITION, CONCURRENCY, PER_TASK,
#                DERIVE_HOURS, POP_LIMIT.
#   DERIVE_HOURS  per-patient rbg_* fit horizon, reused by the prior (default 24)
#   POP_LIMIT   cap patients used to BUILD the prior (default: all = 1013).
#               Leave unset to fit the prior on the whole cohort.
#   CONCURRENCY max patient jobs running at once (default 200)
#   PER_TASK    patients per array task (default 1 = one job/patient)
# =============================================================================

# NOTE: run with `bash run_phase2.sh` (it submits the real jobs itself). The
# #SBATCH block is only a safety net so an accidental `sbatch` still allocates
# something; real per-stage resources are on the inner sbatch lines below.
#SBATCH --account=ai-gpu
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/vmli3/t1d_experiment/logs/phase2/%x_%j.out
#SBATCH --error=/scratch/vmli3/t1d_experiment/logs/phase2/%x_%j.err
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type FAIL

set -uo pipefail

CONDA_ENV="${CONDA_ENV:-t1d_benchmark}"
PATIENTS="${PATIENTS:-patients.csv}"
# All generated outputs (artifacts/results/logs) live under one root.
# Exported so the Python side (experiments.exp_common) writes to the same place.
export T1D_OUTPUT_ROOT="${T1D_OUTPUT_ROOT:-/scratch/vmli3/t1d_experiment}"
LOG_DIR="$T1D_OUTPUT_ROOT/logs/phase2"   # per-phase SLURM log dir
POP="${POP:-$T1D_OUTPUT_ROOT/artifacts/population.npz}"
PARTITION="${PARTITION:-c64-m512}"
CONCURRENCY="${CONCURRENCY:-48}"       # keep concurrent array tasks under the 50-job cap
PER_TASK="${PER_TASK:-32}"            # 32 patients per array task (one per core)
DERIVE_HOURS="${DERIVE_HOURS:-24}"    # per-patient rbg_* fit horizon (match build_phase1)
POP_LIMIT="${POP_LIMIT:-}"             # empty = aggregate the prior over ALL patients

# ============================ submission mode ===============================
if [[ $# -eq 0 ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "note: this is the submitter — run it on the login node with:  bash $0"
    fi
    [[ -d experiments ]] || { echo "run from the repo root" >&2; exit 1; }
    # create the whole output tree up front so SLURM can write every job's logs
    # ($LOG_DIR = logs/phase2 must exist before the inner sbatch --output lines)
    mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs} "$LOG_DIR"
    # cohort needed now to size the array
    [[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
    N=$(( $(wc -l < "$PATIENTS") - 1 ))
    NTASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
    LAST=$(( NTASKS - 1 ))
    echo "patients=$N | prior on ${POP_LIMIT:-all} patients | $PER_TASK patients/job (one per core)"
    echo "  -> $NTASKS array task(s) @ 32 CPU / 128 GB / 7-day, <=$CONCURRENCY running at once"
    if (( NTASKS > 50 )); then
        echo "WARNING: $NTASKS array tasks exceeds your 50-job limit. Raise PER_TASK"
        echo "         (e.g. PER_TASK=$(( (N + 49) / 50 )) packs the cohort into <=50 tasks)."
    fi

    # setup: a whole node so the parallel population fit uses all 64 cores
    j_setup=$(sbatch --parsable --job-name=t1d_p2_setup --partition="$PARTITION" \
        --cpus-per-task=64 --mem=480G --time=12:00:00 \
        --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" "$0" setup)
    j_arr=$(sbatch --parsable --job-name=t1d_p2_patient --partition="$PARTITION" \
        --array=0-${LAST}%${CONCURRENCY} --cpus-per-task=32 --mem=128G --time=7-00:00:00 \
        --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
        --dependency=afterok:"$j_setup" "$0" patient)
    j_agg=$(sbatch --parsable --job-name=t1d_p2_aggregate --partition="$PARTITION" \
        --cpus-per-task=2 --mem=16G --time=02:00:00 \
        --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
        --dependency=afterok:"$j_arr" "$0" aggregate)

    echo "submitted: setup=$j_setup  patients(array)=$j_arr  aggregate=$j_agg"
    echo "track:  squeue -u \"\$USER\""
    exit 0
fi

# ============================ execution mode ================================
STAGE="$1"
set +u
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate "$CONDA_ENV"
set -u
cd "${SLURM_SUBMIT_DIR:-.}"
# one thread per process: the population fit gets its parallelism from the pool
# (one process per patient), and each array task is a single core.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$USER"
mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs} "$LOG_DIR"

# fit both per-instance twins + score one patient; resumable + crash-tolerant.
# Uses --population so every twin shares the patient-derived prior.
fit_one() {
    local name="$1" safe
    safe=$(printf '%s' "$name" | sed 's#[^A-Za-z0-9._+-]#_#g')
    if [[ -f "$T1D_OUTPUT_ROOT/results/phase2/${safe}/comparison_table.csv" ]]; then echo "[skip] $name"; return 0; fi
    python -m experiments.run_mcmc --patient "$name" --patients "$PATIENTS" \
        --population "$POP"  || echo "[warn] $name: mcmc nonzero"
    python -m experiments.run_sbi  --patient "$name" --patients "$PATIENTS" \
        --population "$POP"  || echo "[warn] $name: sbi nonzero"
    python -m experiments.compute_results --patient "$name" --patients "$PATIENTS" \
        || echo "[warn] $name: compute_results nonzero"
}

case "$STAGE" in
  setup)
    echo "[setup] $(date)"
    [[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
    # one per-patient least-squares fit pass (rbg_* cols), reused by the prior
    if head -1 "$PATIENTS" | grep -q 'rbg_SI'; then
        echo "[setup] $PATIENTS already has rbg_* columns — skipping fit"
    else
        python -m experiments.derive_replaybg_params --patients "$PATIENTS" \
            --hours "$DERIVE_HOURS" --jobs "${SLURM_CPUS_PER_TASK:-1}"
    fi
    if [[ -f "$POP" ]]; then
        echo "[setup] $POP exists — reusing (delete it to rebuild the prior)"
    else
        # aggregate the cached rbg_* fits into the population (no re-fit);
        # POP_LIMIT empty -> all patients.
        LIM=(); [[ -n "$POP_LIMIT" ]] && LIM=(--limit "$POP_LIMIT")
        python -m experiments.population --patients "$PATIENTS" \
            "${LIM[@]}" --out "$POP"
    fi
    echo "[setup] done $(date)"
    ;;

  patient)
    : "${SLURM_ARRAY_TASK_ID:?the patient stage must run as an array task}"
    N=$(( $(wc -l < "$PATIENTS") - 1 ))
    start=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
    end=$(( start + PER_TASK )); (( end > N )) && end=$N
    (( start >= N )) && { echo "task $SLURM_ARRAY_TASK_ID: nothing to do"; exit 0; }
    NPROC="${SLURM_CPUS_PER_TASK:-1}"
    echo "=== task $SLURM_ARRAY_TASK_ID: patients [$start,$end) across $NPROC core(s) $(date) ==="
    # Fan this task's patients across the cores: each patient's fit is itself
    # single-threaded (OMP/MKL=1, torch pinned in run_sbi), so we run up to NPROC
    # patients at once. Process substitution keeps the loop in THIS shell so the
    # background jobs are visible to `wait`.
    while IFS= read -r nm; do
        ( echo "--- $nm (task $SLURM_ARRAY_TASK_ID) start $(date) ==="; fit_one "$nm" ) &
        # throttle to NPROC concurrent patients
        while (( $(jobs -rp | wc -l) >= NPROC )); do wait -n; done
    done < <(awk -F, -v s="$start" -v e="$end" 'NR>=s+2 && NR<=e+1{print $1}' "$PATIENTS")
    wait
    echo "=== task $SLURM_ARRAY_TASK_ID: all patients done $(date) ==="
    ;;

  aggregate)
    echo "[aggregate] building the phase 2 summary over ALL patients"
    python -m experiments.run_phase2 --patients "$PATIENTS" --aggregate-only
    echo "summary: results/phase2_summary.csv"
    ;;

  *)
    echo "unknown stage '$STAGE' (expected setup|patient|aggregate)" >&2; exit 2;;
esac
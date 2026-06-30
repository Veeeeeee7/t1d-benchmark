#!/bin/bash
# =============================================================================
# run_phase0_twins.sh — Phase 0 PER-INSTANCE twins (MCMC + SBI), matched model.
#
# The Phase 0 analogue of run_phase2.sh: same self-submitting setup -> patient
# array -> aggregate chain, but the plant is the patient's best-fit ReplayBG
# model (no plant<->twin mismatch). CPU only.
#
# Pipeline (afterok dependency chain):
#
#     setup (1 node, all cores)   ->   patient array   ->   aggregate
#
#   setup     : generate_patients (if needed) + derive_replaybg_params — fit
#               ReplayBG to EVERY patient's baseline and append rbg_* columns to
#               patients.csv, FANNED across all cores of one c64-m512 node. This
#               is the Phase 0 analogue of the population-prior fit. Idempotent:
#               skipped if patients.csv already has the rbg_* columns.
#   patient   : one SLURM array task per PER_TASK patients (32 cores) — the
#               matched-model MCMC + SBI twin fit + scoring. Resumable.
#   aggregate : phase-0 mean +/- std summary once every patient is scored.
#
# Phase 0 writes under results/phase0/ and artifacts/phase0/ (namespaced) so it
# never clobbers Phase 2's per-patient tables — the two are paired by name later.
#
# Launch from the LOGIN node (NOT sbatch):
#       bash run_phase0_twins.sh
#
# Env overrides: CONDA_ENV, PATIENTS, PARTITION, CONCURRENCY, PER_TASK,
#                DERIVE_HOURS (baseline horizon fit for the rbg_* params, def 24).
# =============================================================================

# NOTE: run with `bash run_phase0_twins.sh` (it submits the real jobs itself).
# The #SBATCH block is only a safety net so an accidental `sbatch` still
# allocates something; real per-stage resources are on the inner sbatch lines.
#SBATCH --account=general
#SBATCH --partition=c64-m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/vmli3/t1d_experiment/logs/phase0/%x_%j.out
#SBATCH --error=/scratch/vmli3/t1d_experiment/logs/phase0/%x_%j.err
#SBATCH --mail-user victor.li@emory.edu
#SBATCH --mail-type FAIL

set -uo pipefail

CONDA_ENV="${CONDA_ENV:-t1d_benchmark}"
PATIENTS="${PATIENTS:-patients.csv}"
# All generated outputs (artifacts/results/logs) live under one root.
# Exported so the Python side (experiments.exp_common) writes to the same place.
export T1D_OUTPUT_ROOT="${T1D_OUTPUT_ROOT:-/scratch/vmli3/t1d_experiment}"
LOG_DIR="$T1D_OUTPUT_ROOT/logs/phase0"   # per-phase SLURM log dir
PARTITION="${PARTITION:-c64-m512}"
CONCURRENCY="${CONCURRENCY:-48}"       # keep concurrent array tasks under the 50-job cap
PER_TASK="${PER_TASK:-32}"             # 32 patients per array task (one per core)
DERIVE_HOURS="${DERIVE_HOURS:-24}"     # baseline horizon fit for the rbg_* params

# ============================ submission mode ===============================
if [[ $# -eq 0 ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "note: this is the submitter — run it on the login node with:  bash $0"
    fi
    [[ -d experiments ]] || { echo "run from the repo root" >&2; exit 1; }
    # create the whole output tree up front so SLURM can write every job's logs
    # ($LOG_DIR = logs/phase0 must exist before the inner sbatch --output lines)
    mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs} "$LOG_DIR"
    # cohort needed now to size the array
    [[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
    N=$(( $(wc -l < "$PATIENTS") - 1 ))
    NTASKS=$(( (N + PER_TASK - 1) / PER_TASK ))
    LAST=$(( NTASKS - 1 ))
    echo "patients=$N | matched-model MCMC+SBI | $PER_TASK patients/job (one per core)"
    echo "  -> $NTASKS array task(s) @ 32 CPU / 128 GB / 7-day, <=$CONCURRENCY running at once"
    if (( NTASKS > 50 )); then
        echo "WARNING: $NTASKS array tasks exceeds your 50-job limit. Raise PER_TASK"
        echo "         (e.g. PER_TASK=$(( (N + 49) / 50 )) packs the cohort into <=50 tasks)."
    fi

    # setup: a whole node so the parallel ReplayBG-fit derivation uses all 64 cores
    j_setup=$(sbatch --parsable --job-name=t1d_p0_setup --partition="$PARTITION" \
        --cpus-per-task=64 --mem=480G --time=12:00:00 \
        --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" "$0" setup)
    j_arr=$(sbatch --parsable --job-name=t1d_p0_patient --partition="$PARTITION" \
        --array=0-${LAST}%${CONCURRENCY} --cpus-per-task=32 --mem=128G --time=7-00:00:00 \
        --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" \
        --dependency=afterok:"$j_setup" "$0" patient)
    j_agg=$(sbatch --parsable --job-name=t1d_p0_aggregate --partition="$PARTITION" \
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
# one thread per process: the derivation gets its parallelism from the pool (one
# process per patient), and each array task fans patients across its cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_$USER"
mkdir -p "$T1D_OUTPUT_ROOT"/{artifacts,results,logs} "$LOG_DIR"

# fit both matched-model twins + score one patient; resumable + crash-tolerant.
# Phase 0 paths are namespaced under results/phase0/ so they don't collide with
# Phase 2's results/phase2/<name>/ tables (the cohorts share patient names).
fit_one() {
    local name="$1" safe
    safe=$(printf '%s' "$name" | sed 's#[^A-Za-z0-9._+-]#_#g')
    if [[ -f "$T1D_OUTPUT_ROOT/results/phase0/${safe}/comparison_table.csv" ]]; then
        echo "[skip] $name"; return 0
    fi
    python -m experiments.run_mcmc_phase0 --patient "$name" --patients "$PATIENTS" \
        || echo "[warn] $name: mcmc0 nonzero"
    python -m experiments.run_sbi_phase0  --patient "$name" --patients "$PATIENTS" \
        || echo "[warn] $name: sbi0 nonzero"
    python -m experiments.compute_results_phase0 --patient "$name" --patients "$PATIENTS" \
        || echo "[warn] $name: compute_results_phase0 nonzero"
}

case "$STAGE" in
  setup)
    echo "[setup] $(date)"
    [[ -f "$PATIENTS" ]] || python -m experiments.generate_patients --out "$PATIENTS"
    if head -1 "$PATIENTS" | grep -q 'rbg_SI'; then
        echo "[setup] $PATIENTS already has rbg_* columns — reusing (delete them to refit)"
    else
        # parallel ReplayBG-fit derivation across this node's cores (auto-detects
        # SLURM_CPUS_PER_TASK); appends rbg_* columns to patients.csv in place.
        python -m experiments.derive_replaybg_params --patients "$PATIENTS" \
            --hours "$DERIVE_HOURS" --jobs "${SLURM_CPUS_PER_TASK:-1}"
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
    # single-threaded (OMP/MKL=1, torch pinned in run_sbi_phase0), so we run up to NPROC
    # patients at once. Process substitution keeps the loop in THIS shell so the
    # background jobs are visible to `wait`.
    while IFS= read -r nm; do
        ( echo "--- $nm (task $SLURM_ARRAY_TASK_ID) start $(date) ==="; fit_one "$nm" ) &
        # throttle to NPROC concurrent patients
        while (( $(jobs -rp | wc -l) >= NPROC )); do wait -n; done
    done < <(awk -F, -v s="$start" -v e="$end" '
        NR==1 { for (i=1;i<=NF;i++) if ($i=="Name") col=i;
                if (!col) { print "ERROR: no Name column in " FILENAME > "/dev/stderr"; exit 3 }
                next }
        NR>=s+2 && NR<=e+1 { print $col }' "$PATIENTS")
    wait
    echo "=== task $SLURM_ARRAY_TASK_ID: all patients done $(date) ==="
    ;;

  aggregate)
    echo "[aggregate] building the phase 0 summary over ALL patients"
    python -m experiments.run_phase0_twins --patients "$PATIENTS" --aggregate-only
    echo "summary: results/phase0_summary.csv"
    ;;

  *)
    echo "unknown stage '$STAGE' (expected setup|patient|aggregate)" >&2; exit 2;;
esac
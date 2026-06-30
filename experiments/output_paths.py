"""Single source of truth for the standardized output directory layout.

Every generated output lives under one root (``$T1D_OUTPUT_ROOT``, default the
project scratch space) in four top-level trees::

    $T1D_OUTPUT_ROOT/
      artifacts/                         generated inputs / fitted twins
        population.npz                   COMMON prior (shared by all phases)
        <phase>/dataset.npz              amortized dataset for that phase
        <phase>/<name>/mcmc_twin.npz     per-patient fitted twins
        <phase>/<name>/sbi_twin.npz
      results/
        <phase>_summary.csv              per-phase aggregate (flat, in results/)
        <phase>_per_patient.csv          per-phase per-patient roll-up (flat)
        <phase>/<name>/comparison_table.csv   per-patient table
        <phase>/<name>/notes.md               per-patient human-readable notes
        <phase>/<name>/ig_overlay_<method>.png   per-patient figures
      logs/
        <phase>/<jobname>_<jobid>.{out,err}   SLURM logs, split by phase
      (no figures/ tree — per-patient figures live beside their results)

Rationale for centralizing here
-------------------------------
The on-disk layout used to be inconsistent across the four experiments (Phase 2
wrote per-patient dirs straight under ``results/`` while Phase 0 namespaced its
own under ``results/phase0/``; the amortized phases wrote ``results/<phase>/
summary.csv`` while the twin phases wrote ``results/<phase>_summary.csv``). All
of that now derives from the helpers below, so the four experiments share one
shape and nothing has to defensively namespace itself.

This module is intentionally **pure stdlib** (no numpy / simglucose / ``t1d_twin``
imports) so the simglucose-free Phase 0 ML and dataset-build paths
(``build_phase0_dataset``, ``run_phase0_ml``) can import it without dragging in
the simulator. ``exp_common`` re-exports the constants/helpers for the
simglucose-side code, so callers can use either ``output_paths`` directly or via
``exp_common``.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Root + top-level trees
# ---------------------------------------------------------------------------
# Override with T1D_OUTPUT_ROOT for local runs, e.g.:
#   T1D_OUTPUT_ROOT=./_out python -m experiments.run_phase1
OUTPUT_ROOT = os.environ.get("T1D_OUTPUT_ROOT", "/scratch/vmli3/t1d_experiment")
ARTIFACT_DIR = os.path.join(OUTPUT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
# NOTE: there is deliberately no FIG_DIR any more — per-patient figures live in
# results/<phase>/<name>/ next to that patient's table and notes.

# ---------------------------------------------------------------------------
# Phase tags (the four experiments). Use these everywhere instead of literals.
# ---------------------------------------------------------------------------
PHASE0 = "phase0"          # matched-model per-instance twins (ReplayBG plant)
PHASE0_ML = "phase0_ml"    # amortized baselines (ReplayBG plant)
PHASE1 = "phase1"          # amortized baselines (simglucose plant)
PHASE2 = "phase2"          # population-prior per-instance twins (simglucose plant)
PREP = "prep"              # shared prerequisite build (logs only)

PHASES = (PHASE0, PHASE0_ML, PHASE1, PHASE2)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def population_path() -> str:
    """The ONE prior shared by every phase: ``artifacts/population.npz``."""
    return os.path.join(ARTIFACT_DIR, "population.npz")


def artifact_dir(phase: str, safe_name: str | None = None) -> str:
    """``artifacts/<phase>/`` (experiment-wide) or ``artifacts/<phase>/<name>/``."""
    d = os.path.join(ARTIFACT_DIR, phase)
    return d if safe_name is None else os.path.join(d, safe_name)


def dataset_path(phase: str) -> str:
    """Amortized dataset for ``phase``: ``artifacts/<phase>/dataset.npz``."""
    return os.path.join(artifact_dir(phase), "dataset.npz")


def twin_artifact_paths(phase: str, safe_name: str) -> dict:
    """Per-patient fitted-twin paths under ``artifacts/<phase>/<name>/``."""
    d = artifact_dir(phase, safe_name)
    return {"dir": d,
            "mcmc": os.path.join(d, "mcmc_twin.npz"),
            "sbi": os.path.join(d, "sbi_twin.npz")}


# ---------------------------------------------------------------------------
# Results (+ per-patient figures, which now live alongside the tables)
# ---------------------------------------------------------------------------
def summary_path(phase: str) -> str:
    """Per-phase aggregate, flat in results/: ``results/<phase>_summary.csv``."""
    return os.path.join(RESULTS_DIR, f"{phase}_summary.csv")


def per_patient_path(phase: str) -> str:
    """Per-phase per-patient roll-up: ``results/<phase>_per_patient.csv``."""
    return os.path.join(RESULTS_DIR, f"{phase}_per_patient.csv")


def results_dir(phase: str, safe_name: str) -> str:
    """Per-patient results dir: ``results/<phase>/<name>/`` (table + notes + figures)."""
    return os.path.join(RESULTS_DIR, phase, safe_name)


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
def log_dir(phase: str) -> str:
    """Per-phase SLURM log dir: ``logs/<phase>/``."""
    return os.path.join(LOG_DIR, phase)

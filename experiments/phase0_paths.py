"""Phase-0-namespaced artifact / results paths.

The matched Phase 0 cohort shares patient *names* with Phase 2 (both are the
same ``patients.csv``), so Phase 0 must NOT write to Phase 2's per-subject paths
or the two phases would clobber each other's ``comparison_table.csv`` (and the
resumable skip-checks would fire on the wrong phase). These helpers mirror
``exp_common.artifact_paths`` / ``results_dir_for`` but for the ``phase0`` tag,
so Phase 0 and Phase 2 results sit side by side under ``artifacts/`` /
``results/`` and are paired by name at analysis time.

Both these helpers and the Phase 2 ones in ``exp_common`` now delegate to the
single layout defined in ``experiments.output_paths``; this module remains as
the phase-0 entry point (callers do ``phase0_paths.artifact_paths(subject)``),
and its signatures are unchanged. Kept in its own tiny module so the
simglucose-free plant / dataset / ML path can reach the phase-0 layout via
``output_paths`` directly without importing ``exp_common`` (which pulls in
simglucose).
"""
from __future__ import annotations

from experiments import output_paths as _OP

PHASE0_TAG = _OP.PHASE0


def artifact_paths(subject) -> dict:
    """Per-subject Phase 0 twin-artifact paths under ``artifacts/phase0/<name>/``."""
    return _OP.twin_artifact_paths(PHASE0_TAG, subject.safe_name)


def results_dir_for(subject) -> str:
    """Per-subject Phase 0 results dir under ``results/phase0/<name>/``."""
    return _OP.results_dir(PHASE0_TAG, subject.safe_name)
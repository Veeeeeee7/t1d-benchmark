"""Candidate therapy set Pi and the seen/unseen split (Phase B, step B2).

This module assembles the policy set the whole evaluation ranks. Two design
choices encode the research question:

Interior optimum (non-trivial ranking)
--------------------------------------
The bolus axis is widened past the point where over-dosing causes
hypoglycemia, so reward is a *bowl* in dosing: under-dosing leaves the patient
hyperglycemic, over-dosing drives them hypoglycemic, and the best therapy sits
in the interior (empirically near a 2.0x bolus on adult#001 with the default
single-meal scenario). A monotone "max bolus is always best" set would make the
ranking trivial and the decision-transfer question vacuous; B3's ground-truth
test asserts the optimum is interior, and this set is built to satisfy it.

Seen vs unseen (the generalization test)
----------------------------------------
Twins are identified from a single baseline run (bolus=1.0, basal=1.0). The
``seen`` subset is the band of baseline-adjacent therapies (within
``seen_band`` of 1.0 on a single axis) -- small perturbations the twin can be
expected to interpolate. The ``unseen`` subset is the far modulations the twin
never saw context for; this is where decision transfer actually matters.
Notably the *optimal* therapy lands in ``unseen``, so picking it correctly is a
genuine extrapolation test, not a memorized in-distribution lookup.

Returns
-------
``make_experiment_policies`` returns a :class:`PolicySet`, which carries the
``name -> Controller`` mapping plus the ``seen`` / ``unseen`` partition (the
"`{name: controller}` plus the partition" the plan calls for). It also proxies
the common dict reads (``len``, ``in``, ``[]``, ``items``, iteration) so it can
be passed anywhere a plain policy dict is expected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from simglucose.controller.base import Controller

from .policies import make_candidate_policies

# Default widened sweeps. Bolus brackets the interior (hypo-inducing) optimum;
# basal is a secondary axis. 1.0 is intentionally omitted from BASAL so the
# baseline therapy appears exactly once, as "bolus_x1.00".
DEFAULT_BOLUS_FACTORS = (0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
DEFAULT_BASAL_FACTORS = (0.7, 0.85, 1.2)
DEFAULT_SEEN_BAND = 0.2
BASELINE_NAME = "bolus_x1.00"


@dataclass
class PolicySet:
    """A named candidate set with a seen/unseen partition.

    Attributes
    ----------
    policies : dict[str, Controller]
        Name -> controller for every member of Pi.
    seen : list[str]
        Baseline-adjacent therapy names (within ``seen_band`` of factor 1.0).
    unseen : list[str]
        Far-from-baseline therapy names (the generalization test).
    baseline_name : str
        The identification therapy's name (factors 1.0/1.0).
    """
    policies: dict[str, Controller]
    seen: list[str]
    unseen: list[str]
    baseline_name: str = BASELINE_NAME
    meta: dict = field(default_factory=dict)

    # --- dict-like proxies so a PolicySet drops in for a plain dict ----------
    def names(self) -> list[str]:
        return list(self.policies.keys())

    def items(self):
        return self.policies.items()

    def keys(self):
        return self.policies.keys()

    def values(self):
        return self.policies.values()

    def __getitem__(self, name: str) -> Controller:
        return self.policies[name]

    def __contains__(self, name: str) -> bool:
        return name in self.policies

    def __iter__(self):
        return iter(self.policies)

    def __len__(self) -> int:
        return len(self.policies)

    def subset(self, which: str) -> dict[str, Controller]:
        """Return the ``"seen"`` or ``"unseen"`` controllers as a dict."""
        if which == "seen":
            names = self.seen
        elif which == "unseen":
            names = self.unseen
        else:
            raise ValueError("which must be 'seen' or 'unseen'")
        return {n: self.policies[n] for n in names}


def _classify(factor: float, seen_band: float) -> str:
    """'seen' if the modulation is within ``seen_band`` of baseline, else 'unseen'."""
    return "seen" if abs(factor - 1.0) <= seen_band else "unseen"


def make_experiment_policies(
    bolus_factors=DEFAULT_BOLUS_FACTORS,
    basal_factors=DEFAULT_BASAL_FACTORS,
    target: float = 140.0,
    seen_band: float = DEFAULT_SEEN_BAND,
) -> PolicySet:
    """Build the candidate set Pi with its seen/unseen partition.

    Names follow ``make_candidate_policies``: ``bolus_x{f:.2f}`` and
    ``basal_x{f:.2f}``. A factor is ``seen`` iff it lies within ``seen_band`` of
    1.0 on its axis. The partition is disjoint and covers Pi by construction.

    Parameters
    ----------
    bolus_factors, basal_factors : multiplicative therapy sweeps; defaults give
        an interior reward optimum on adult#001 (best ~ bolus_x2.00).
    target : BBController setpoint passed through to every controller.
    seen_band : half-width (in factor units) of the baseline-adjacent band.

    Returns
    -------
    PolicySet
    """
    policies = make_candidate_policies(bolus_factors, basal_factors, target=target)

    seen: list[str] = []
    unseen: list[str] = []
    for f in bolus_factors:
        name = f"bolus_x{f:.2f}"
        (seen if _classify(f, seen_band) == "seen" else unseen).append(name)
    for f in basal_factors:
        name = f"basal_x{f:.2f}"
        (seen if _classify(f, seen_band) == "seen" else unseen).append(name)

    # Invariants: partition is disjoint and covers Pi; baseline is present/seen.
    assert set(seen).isdisjoint(unseen), "seen/unseen overlap"
    assert set(seen) | set(unseen) == set(policies), "partition does not cover Pi"
    assert BASELINE_NAME in policies, "baseline therapy missing from Pi"

    return PolicySet(
        policies=policies,
        seen=seen,
        unseen=unseen,
        baseline_name=BASELINE_NAME,
        meta={
            "bolus_factors": tuple(bolus_factors),
            "basal_factors": tuple(basal_factors),
            "seen_band": seen_band,
            "target": target,
        },
    )

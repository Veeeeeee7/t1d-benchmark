"""Control policies for the twinning platform.

Every policy is a simglucose ``Controller`` (it implements
``policy(observation, reward, done, **info) -> Action`` and ``reset()``).
Keeping a single interface means the *same* policy object can later drive both
simglucose (ground truth) and a twin (prediction) through the identical run
loop in ``simglucose_adapter.run_policy`` -- the two paths can never silently
diverge.

The interface ``policy(observation, ...)`` is general enough for both open- and
closed-loop control: an open-loop policy simply ignores ``observation.CGM``.
The candidate set below is open-loop (v1): basal-bolus with the bolus and/or
basal scaled by a constant factor, matching ReplayBG's Scenario-1 therapy
modulations. Genuinely closed-loop controllers (e.g. PID) can be added later
behind the same interface without touching the run loop.
"""
from __future__ import annotations

from dataclasses import dataclass

from simglucose.controller.base import Controller, Action
from simglucose.controller.basal_bolus_ctrller import BBController


@dataclass
class ModulatedBBController(Controller):
    """Basal-bolus controller with constant multiplicative therapy modulation.

    ``bolus_factor`` / ``basal_factor`` scale the corresponding components of
    the underlying :class:`BBController` action. ``bolus_factor=1.2`` is the
    "+20% bolus" candidate; ``basal_factor`` modulates the basal rate.
    """
    bolus_factor: float = 1.0
    basal_factor: float = 1.0
    target: float = 140.0

    def __post_init__(self):
        self._base = BBController(target=self.target)

    def policy(self, observation, reward, done, **info):
        a = self._base.policy(observation, reward, done, **info)
        return Action(basal=a.basal * self.basal_factor,
                      bolus=a.bolus * self.bolus_factor)

    def reset(self):
        self._base.reset()


def baseline_policy(target: float = 140.0) -> Controller:
    """The behavioural policy used to collect the identification dataset."""
    return ModulatedBBController(bolus_factor=1.0, basal_factor=1.0, target=target)


def make_candidate_policies(
    bolus_factors=(0.5, 0.7, 0.85, 1.0, 1.2, 1.5),
    basal_factors=(0.7, 0.85, 1.2),
    target: float = 140.0,
) -> dict[str, Controller]:
    """Return a named set of candidate therapies (the policy set Pi).

    Bolus and basal sweeps are kept on separate axes so the set spans
    under-/over-dosing in both, giving a non-trivial ground-truth ranking.
    The +20% bolus member is ``bolus_x1.20``.
    """
    policies: dict[str, Controller] = {}
    for f in bolus_factors:
        policies[f"bolus_x{f:.2f}"] = ModulatedBBController(bolus_factor=f, target=target)
    for f in basal_factors:
        policies[f"basal_x{f:.2f}"] = ModulatedBBController(basal_factor=f, target=target)
    return policies

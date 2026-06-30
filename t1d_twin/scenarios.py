"""Scenario builders.

For v1 we use the single-meal stationary window the steady-state initial
condition assumption requires (per ReplayBG): a quiet fasting lead-in followed
by one announced meal, with the window long enough for the excursion to resolve.
The candidate therapies (in ``policies.py``) keep this meal/scenario fixed and
only modulate the controller, mirroring ReplayBG Scenario 1.
"""
from __future__ import annotations

import datetime

from simglucose.simulation.scenario import CustomScenario

DEFAULT_START = datetime.datetime(2024, 1, 1, 0, 0, 0)


def single_meal_scenario(meal_time_h: float = 1.0,
                         meal_g: float = 50.0,
                         start_time: datetime.datetime = DEFAULT_START) -> CustomScenario:
    """One ``meal_g`` gram meal at ``meal_time_h`` hours after start.

    ``meal_time_h`` provides a short fasting baseline before the meal so the
    run begins near steady state. simglucose expects ``scenario`` as a list of
    ``(time_in_hours, grams)`` tuples.
    """
    return CustomScenario(start_time=start_time, scenario=[(meal_time_h, meal_g)])

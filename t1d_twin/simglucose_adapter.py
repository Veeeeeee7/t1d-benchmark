"""simglucose adapter.

``run_policy`` drives a simglucose patient with any controller for a fixed
horizon and returns a standardized :class:`RunResult`: a per-sample timeseries
(CGM, BG, the delivered insulin/CHO, plus those inputs already converted to
ReplayBG units) together with the metadata (body weight, sample time, sensor
seed, patient name) needed to (a) replay the same therapy on a twin and (b)
reuse the same sensor model on twin output later.

We drive the environment loop directly rather than via simglucose's ``sim``
engine because ``sim`` insists on writing results to disk; the manual loop also
gives the single shared run interface used for both ground truth and twins.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Controller

from . import units


@dataclass
class RunResult:
    """Standardized output of one simulation run.

    ``df`` columns (one row per ``sample_time`` minutes):
        t_min            elapsed minutes from start
        datetime         wall-clock timestamp
        CGM              sensor glucose [mg/dL]
        BG               true plasma glucose [mg/dL]
        CHO_g_min        meal carbohydrate rate [g/min]   (simglucose units)
        basal_U_min      basal insulin rate [U/min]
        bolus_U_min      bolus insulin rate [U/min]
        insulin_U_min    total insulin rate [U/min]
        CHO_mg_kg_min    meal rate in ReplayBG units [mg/kg/min]
        insulin_mU_kg_min total insulin in ReplayBG units [mU/kg/min]
        lbgi, hbgi, risk simglucose per-step risk indices
    """
    df: pd.DataFrame
    BW: float
    sample_time: float
    patient_name: str
    sensor_name: str
    sensor_seed: int
    meta: dict = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------------
    def cgm(self) -> np.ndarray:
        return self.df["CGM"].to_numpy()

    def bg(self) -> np.ndarray:
        return self.df["BG"].to_numpy()

    def replaybg_inputs(self, dt: float = 1.0):
        """Inputs for the ReplayBG forward model on a regular ``dt``-minute grid.

        Sample-rate inputs are held piecewise-constant across each sample
        interval. Returns ``(t_grid, insulin_mU_kg_min, cho_mg_kg_min)`` with
        all three arrays the same length. Requires ``sample_time`` to be an
        integer multiple of ``dt`` (true for the defaults: sample_time=3, dt=1).
        """
        reps = self.sample_time / dt
        if abs(reps - round(reps)) > 1e-9:
            raise ValueError(
                f"sample_time ({self.sample_time}) must be an integer multiple of dt ({dt})")
        reps = int(round(reps))
        insulin = np.repeat(self.df["insulin_mU_kg_min"].to_numpy(), reps)
        cho = np.repeat(self.df["CHO_mg_kg_min"].to_numpy(), reps)
        t_grid = np.arange(len(insulin)) * dt
        return t_grid, insulin, cho


def run_policy(controller: Controller,
               scenario,
               hours: float,
               patient_name: str = "adult#001",
               sensor_name: str = "Dexcom",
               pump_name: str = "Insulet",
               sensor_seed: int = 1) -> RunResult:
    """Run ``controller`` on ``patient_name`` under ``scenario`` for ``hours``."""
    patient = T1DPatient.withName(patient_name)
    BW = float(patient._params.BW)
    sensor = CGMSensor.withName(sensor_name, seed=sensor_seed)
    pump = InsulinPump.withName(pump_name)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    sample_time = float(env.sample_time)

    if hasattr(controller, "reset"):
        controller.reset()
    obs, reward, done, info = env.reset()
    start_time = info["time"]

    n_steps = int(round(hours * 60.0 / sample_time))
    rows = []
    for _ in range(n_steps):
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)

        cho_g_min = float(info["meal"])                 # g/min
        basal = float(action.basal)                     # U/min
        bolus = float(action.bolus)                     # U/min
        insulin_U_min = basal + bolus
        t_min = (info["time"] - start_time).total_seconds() / 60.0

        rows.append({
            "t_min": t_min,
            "datetime": info["time"],
            "CGM": float(obs.CGM),
            "BG": float(info["bg"]),
            "CHO_g_min": cho_g_min,
            "basal_U_min": basal,
            "bolus_U_min": bolus,
            "insulin_U_min": insulin_U_min,
            "CHO_mg_kg_min": units.cho_g_per_min_to_mg_per_kg_min(cho_g_min, BW),
            "insulin_mU_kg_min": units.insulin_U_per_min_to_mU_per_kg_min(insulin_U_min, BW),
            "lbgi": float(info["lbgi"]),
            "hbgi": float(info["hbgi"]),
            "risk": float(info["risk"]),
        })
        if done:
            break

    df = pd.DataFrame(rows)
    return RunResult(df=df, BW=BW, sample_time=sample_time,
                     patient_name=patient_name, sensor_name=sensor_name,
                     sensor_seed=sensor_seed,
                     meta={"hours": hours, "start_time": start_time})

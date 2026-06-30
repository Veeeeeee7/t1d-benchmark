"""Custom / synthetic virtual patients for the multi-subject ranking experiment.

Builds simglucose ``T1DPatient`` objects from **manually entered** or
**averaged** UVA/Padova parameters and runs any controller on them to generate
CGM. The output is a ``RunResult`` identical in shape to
``t1d_twin.simglucose_adapter.run_policy``, so the rest of the pipeline (twins,
metrics, ranking) is unchanged.

Patient generation
-------------------
* ``average_patient(["adult#003", "adult#006"])`` -> the element-wise mean of
  those patients' UVA/Padova parameter vectors (a new synthetic subject).
* ``permutation_set([...])`` -> every pairwise average of a list of base
  patients (e.g. all 45 unordered pairs of the 10 adults).
* ``manual_patient(base, overrides={...})`` -> start from a base patient (or the
  population mean) and override individual hyperparameters by hand.

Why a custom controller
-----------------------
simglucose's ``BBController`` looks up carb-ratio (CR) / correction-factor (CF)
by patient *name* in ``Quest.csv`` and silently falls back to nonsensical
defaults (CR = 1/15) for any unknown name. A synthetic patient is not in
``Quest.csv``, so we use :class:`ExplicitBBController`, which takes CR / CF /
u2ss / BW directly (averaged from the parents) and reproduces ``BBController``'s
exact basal+bolus formula. The basal/bolus *factor* knobs match
``ModulatedBBController`` so the same 5 candidate therapies carry over verbatim.
"""
from __future__ import annotations

import os
import re
import hashlib
import datetime
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import pkg_resources

from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.base import Controller, Action

from t1d_twin import units
from t1d_twin.replaybg_model import THETA_NAMES
from t1d_twin.simglucose_adapter import RunResult

# --- simglucose parameter tables (read once) -------------------------------
_VP_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")
_Q_FILE = pkg_resources.resource_filename("simglucose", "params/Quest.csv")
_VP = pd.read_csv(_VP_FILE)
_Q = pd.read_csv(_Q_FILE)


# ===========================================================================
# Parameter access + synthesis
# ===========================================================================
def list_patients(kind: str | None = None) -> list[str]:
    """All base patient names, optionally filtered by ``kind`` prefix
    (``"adult"``, ``"adolescent"``, ``"child"``)."""
    names = list(_VP["Name"])
    return [n for n in names if kind is None or n.startswith(kind)]


def patient_params(name: str) -> pd.Series:
    """The raw UVA/Padova parameter row for a base patient (a pandas Series)."""
    row = _VP.loc[_VP["Name"] == name]
    if row.empty:
        raise KeyError(f"unknown patient {name!r}; see list_patients()")
    return row.squeeze().copy()


def quest_params(name: str) -> dict:
    """``{CR, CF, TDI}`` therapy constants for a base patient."""
    row = _Q.loc[_Q["Name"] == name]
    if row.empty:
        raise KeyError(f"{name!r} not in Quest.csv")
    r = row.squeeze()
    return {"CR": float(r["CR"]), "CF": float(r["CF"]), "TDI": float(r["TDI"])}


# ===========================================================================
# Carb-counting-error perturbation (a.k.a. mis-set carb ratio)
# ---------------------------------------------------------------------------
# Synthetic patients are subset-averages of well-tuned UVA/Padova adults, so
# their averaged carb ratio (CR_true) is matched to their averaged physiology
# and the unmodulated bolus (x1.00) is already near-optimal. That makes the
# decision problem trivial ("do nothing") and hides the fidelity-vs-decision
# dissociation DT2 is meant to expose. To restore a real, individualized
# decision we deliberately mis-set each patient's *programmed* carb ratio by a
# per-patient factor m:
#
#     CR_prog = CR_true * m,     bolus_carb = meal / CR_prog = (meal/CR_true)/m
#
# i.e. the carb->insulin mapping is scaled by 1/m. This is exactly a persistent
# carb-counting error (announcing 1/m times the true carbs each meal): m > 1
# under-doses (programmed CR too high), m < 1 over-doses. The corrective bolus
# multiplier that restores the physiologically correct carb dose is f* ~= m, so
# choosing m places each patient's optimum on the candidate grid. Physiology is
# untouched, so the population (physiology) prior stays valid.
#
# Choices are balanced over-/under-dose, pushed away from m~=1 (which collapses
# back to the trivial case), and kept inside [0.75, 1.5] so that with the
# production grid (0.7 ... 2.0) every optimum is interior with neighbours on
# both sides (no edge/truncation artifacts) and no baseline run crashes into the
# severe-hypo padding regime. See doc: docs/carb_error_perturbation.md.
DEFAULT_CARB_ERROR_CHOICES = (0.75, 0.85, 1.25, 1.50)


def carb_error_multiplier(name: str,
                          choices=DEFAULT_CARB_ERROR_CHOICES) -> float:
    """Deterministic per-patient dose multiplier ``m`` (CR_prog = CR_true * m).

    Drawn from ``name`` via a stable SHA-256 hash (NOT Python's ``hash()``,
    which is salted per process), so the same patient always gets the same m and
    the cohort is fixed/auditable. The balanced default set yields a ~50/50
    over-/under-dose split. The optimal corrective bolus multiplier is f* ~= m.
    """
    h = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest(), 16)
    return float(choices[h % len(choices)])


def average_patient(names: list[str], new_name: str | None = None):
    """Element-wise average of several base patients -> ``(params, quest)``.

    Averages every UVA/Padova parameter (including the initial-state columns
    ``x0_*``, ``BW``, ``u2ss``, ``Gb`` ...) and the Quest therapy constants
    (CR / CF / TDI). ``params`` is a Series ready for :func:`make_patient`.
    """
    if len(names) < 1:
        raise ValueError("need at least one patient to average")
    nm = new_name or ("avg(" + ",".join(names) + ")")

    # vpatient: average all numeric columns (everything except the Name string)
    numeric = [patient_params(n).drop(labels=["Name"]).astype(float) for n in names]
    avg_num = sum(numeric) / len(numeric)
    params = pd.Series(index=_VP.columns, dtype=object)
    for c in _VP.columns:
        params[c] = avg_num[c] if c in avg_num.index else None
    params["Name"] = nm

    # quest: average CR / CF / TDI
    qs = [quest_params(n) for n in names]
    quest = {k: float(np.mean([q[k] for q in qs])) for k in ("CR", "CF", "TDI")}
    return params, quest


def permutation_set(base_names: list[str]) -> dict[str, tuple]:
    """Every unordered pairwise average of ``base_names``.

    Returns ``{name -> (params, quest)}`` with ``C(n, 2)`` synthetic patients
    (e.g. 45 for the 10 adults).
    """
    out: dict[str, tuple] = {}
    for a, b in combinations(base_names, 2):
        nm = f"avg({a},{b})"
        out[nm] = average_patient([a, b], nm)
    return out


def manual_patient(base: str | None = None,
                   overrides: dict | None = None,
                   quest_overrides: dict | None = None,
                   new_name: str = "manual#001"):
    """Hand-built patient: start from ``base`` (or the adult population mean) and
    override individual hyperparameters.

    Parameters
    ----------
    base : a base patient name to start from; ``None`` -> mean of all adults.
    overrides : ``{param_name: value}`` applied to the vpatient parameters
        (e.g. ``{"BW": 80.0, "Vmx": 0.05, "kabs": 0.08}``). Use
        :data:`PARAM_NAMES` to see what is settable.
    quest_overrides : ``{CR/CF/TDI: value}`` for the therapy constants.
    new_name : name for the synthetic subject.
    """
    if base is None:
        params, quest = average_patient(list_patients("adult"), new_name)
    else:
        params = patient_params(base).copy()
        params["Name"] = new_name
        quest = quest_params(base)
    for k, v in (overrides or {}).items():
        if k not in params.index:
            raise KeyError(f"{k!r} is not a patient parameter; see PARAM_NAMES")
        params[k] = float(v)
    quest = dict(quest)
    quest.update(quest_overrides or {})
    return params, quest


# Settable parameter names (everything but the Name label).
PARAM_NAMES = [c for c in _VP.columns if c != "Name"]


def make_patient(params: pd.Series, init_state=None, seed: int | None = None) -> T1DPatient:
    """Construct a ``T1DPatient`` from a parameter Series."""
    return T1DPatient(params, init_state=init_state, seed=seed)


# ===========================================================================
# Controller that works for patients not in Quest.csv
# ===========================================================================
class ExplicitBBController(Controller):
    """Basal-bolus controller with explicit CR / CF / u2ss / BW.

    Reproduces simglucose ``BBController._bb_policy`` exactly, but takes the
    therapy constants directly instead of looking them up by patient name, and
    applies the same multiplicative ``bolus_factor`` / ``basal_factor`` as
    ``ModulatedBBController`` so the candidate therapy set is identical.
    """

    def __init__(self, CR: float, CF: float, u2ss: float, BW: float,
                 target: float = 140.0,
                 bolus_factor: float = 1.0, basal_factor: float = 1.0) -> None:
        self.CR = float(CR)
        self.CF = float(CF)
        self.u2ss = float(u2ss)
        self.BW = float(BW)
        self.target = float(target)
        self.bolus_factor = float(bolus_factor)
        self.basal_factor = float(basal_factor)

    def policy(self, observation, reward, done, **kwargs):
        sample_time = kwargs.get("sample_time", 1)
        meal = kwargs.get("meal", 0)          # g/min
        glucose = observation.CGM
        basal = self.u2ss * self.BW / 6000.0  # U/min
        if meal > 0:
            bolus = ((meal * sample_time) / self.CR
                     + (glucose > 150) * (glucose - self.target) / self.CF)
            bolus = bolus / sample_time       # U -> U/min
        else:
            bolus = 0.0
        return Action(basal=basal * self.basal_factor,
                      bolus=bolus * self.bolus_factor)

    def reset(self):
        pass


def therapy_controllers(params: pd.Series, quest: dict, target: float = 140.0,
                        bolus_factors=(0.85, 1.0, 1.5, 2.0, 2.5)) -> dict:
    """The candidate-therapy set for a given (synthetic) patient.

    Mirrors ``experiment.make_experiment_policies`` smoke set (bolus axis only),
    but bound to this patient's CR / CF / u2ss / BW.
    """
    u2ss = float(params["u2ss"])
    BW = float(params["BW"])
    return {
        f"bolus_x{f:.2f}": ExplicitBBController(
            quest["CR"], quest["CF"], u2ss, BW, target=target, bolus_factor=f)
        for f in bolus_factors
    }


# ===========================================================================
# Run loop (mirrors simglucose_adapter.run_policy, custom patient + controller)
# ===========================================================================
def run_on_patient(params: pd.Series, controller: Controller, scenario, hours: float,
                   sensor_name: str = "Dexcom", pump_name: str = "Insulet",
                   sensor_seed: int = 1, init_state=None) -> RunResult:
    """Drive ``controller`` on the synthetic patient ``params`` -> ``RunResult``."""
    patient = make_patient(params, init_state=init_state)
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
    truncated = False
    for _ in range(n_steps):
        action = controller.policy(obs, reward, done, **info)
        obs, reward, done, info = env.step(action)

        cho_g_min = float(info["meal"])
        basal = float(action.basal)
        bolus = float(action.bolus)
        insulin_U_min = basal + bolus
        t_min = (info["time"] - start_time).total_seconds() / 60.0
        rows.append({
            "t_min": t_min, "datetime": info["time"],
            "CGM": float(obs.CGM), "BG": float(info["bg"]),
            "CHO_g_min": cho_g_min, "basal_U_min": basal, "bolus_U_min": bolus,
            "insulin_U_min": insulin_U_min,
            "CHO_mg_kg_min": units.cho_g_per_min_to_mg_per_kg_min(cho_g_min, BW),
            "insulin_mU_kg_min": units.insulin_U_per_min_to_mU_per_kg_min(insulin_U_min, BW),
            "lbgi": float(info["lbgi"]), "hbgi": float(info["hbgi"]),
            "risk": float(info["risk"]),
        })
        if done:
            truncated = True
            break

    # Worst-case padding for early-terminated episodes. simglucose sets done=True
    # when BG leaves its safe range (severe hypo/hyper); breaking there leaves a
    # SHORT trajectory, and because the candidate reward SUMS per-sample risk a
    # crash would accumulate LESS total risk and rank artificially best, inverting
    # the objective. Hold the patient at the terminal danger glucose for the rest
    # of the fixed horizon so a crash integrates to the WORST reward. CGM/BG/risk
    # carry forward from the terminal step; further delivery is zeroed. Runs that
    # complete the horizon are untouched.
    completed_steps = len(rows)
    if truncated and completed_steps < n_steps and rows:
        last = rows[-1]
        for k in range(1, n_steps - completed_steps + 1):
            pad = dict(last)
            pad["t_min"] = last["t_min"] + k * sample_time
            pad["datetime"] = last["datetime"] + datetime.timedelta(minutes=k * sample_time)
            pad["CHO_g_min"] = 0.0
            pad["basal_U_min"] = 0.0
            pad["bolus_U_min"] = 0.0
            pad["insulin_U_min"] = 0.0
            pad["CHO_mg_kg_min"] = 0.0
            pad["insulin_mU_kg_min"] = 0.0
            rows.append(pad)

    df = pd.DataFrame(rows)
    return RunResult(df=df, BW=BW, sample_time=sample_time,
                     patient_name=str(params["Name"]), sensor_name=sensor_name,
                     sensor_seed=sensor_seed,
                     meta={"hours": hours, "start_time": start_time,
                           "synthetic": True, "n_steps": n_steps,
                           "completed_steps": completed_steps, "truncated": truncated})


# ===========================================================================
# Subject abstraction + patients.csv I/O
# ===========================================================================
_QUEST_COLS = ("CR", "CF", "TDI")

# Cached best-fit ReplayBG parameters (written by derive_replaybg_params.py and
# read back here so Phase 0/1/2 reuse the fit instead of recomputing it). The
# theta columns are ``rbg_<param>``; plus the per-patient basal, the calibrated
# matched carb ratio for the ReplayBG plant, and the MAP fit RMSE.
RBG_THETA_COLS = [f"rbg_{n}" for n in THETA_NAMES]
RBG_EXTRA_COLS = ["rbg_Ib", "rbg_CR_true", "rbg_fit_rmse"]
RBG_ALL_COLS = RBG_THETA_COLS + RBG_EXTRA_COLS


@dataclass
class Subject:
    """A (synthetic or base) virtual patient: UVA/Padova params + therapy constants.

    Everything the experiment needs to run *this* patient lives here, so the
    twin scripts and the multi-patient driver can treat every subject uniformly.

    Carb-counting-error audit fields
    --------------------------------
    ``dose_mult`` (m) and ``cr_true`` record the perturbation applied by
    :func:`apply_carb_error`. ``quest["CR"]`` always holds the *programmed*
    (possibly mis-set) ratio that every controller doses from -- so the
    identification baseline and the candidate grid are automatically consistent
    -- while ``cr_true`` keeps the physiologically matched ratio for analysis
    (e.g. locating the true optimum / normalising regret). For an unperturbed
    subject ``dose_mult == 1.0`` and ``cr_true == quest["CR"]``.
    """
    name: str
    params: "pd.Series"
    quest: dict
    members: tuple = field(default_factory=tuple)
    cr_true: float | None = None
    dose_mult: float = 1.0
    # Cached best-fit ReplayBG parameters (None until derive_replaybg_params is
    # run). ``rbg_theta`` is the 8-vector phase 1/2 fit as a regression target /
    # warm start; ``rbg_Ib`` / ``rbg_cr_true`` are the ReplayBG-plant basal and
    # matched carb ratio Phase 0 uses; ``rbg_fit_rmse`` is the fit diagnostic.
    rbg_theta: "np.ndarray | None" = None
    rbg_Ib: float | None = None
    rbg_cr_true: float | None = None
    rbg_fit_rmse: float | None = None

    def __post_init__(self):
        if self.cr_true is None:
            self.cr_true = float(self.quest["CR"])

    @property
    def cr_prog(self) -> float:
        """The programmed carb ratio actually used for dosing (= quest['CR'])."""
        return float(self.quest["CR"])

    @property
    def safe_name(self) -> str:
        """Filesystem-safe identifier for per-subject artifact/result folders."""
        return re.sub(r"[^A-Za-z0-9._+-]", "_", self.name)

    def baseline_controller(self, target: float = 140.0) -> "ExplicitBBController":
        return ExplicitBBController(self.quest["CR"], self.quest["CF"],
                                    float(self.params["u2ss"]),
                                    float(self.params["BW"]), target=target)

    def therapy_controllers(self, bolus_factors=(0.85, 1.0, 1.5, 2.0, 2.5),
                            basal_factors=(), target: float = 140.0) -> dict:
        u2ss = float(self.params["u2ss"])
        BW = float(self.params["BW"])
        CR, CF = self.quest["CR"], self.quest["CF"]
        d = {}
        for f in bolus_factors:
            d[f"bolus_x{f:.2f}"] = ExplicitBBController(CR, CF, u2ss, BW, target,
                                                        bolus_factor=f)
        for f in basal_factors:
            d[f"basal_x{f:.2f}"] = ExplicitBBController(CR, CF, u2ss, BW, target,
                                                        basal_factor=f)
        return d

    def run(self, controller, scenario, hours, sensor_seed: int = 1) -> RunResult:
        return run_on_patient(self.params, controller, scenario, hours,
                              sensor_seed=sensor_seed)


def subject_from_base(name: str) -> Subject:
    """Wrap a built-in simglucose patient as a Subject (uses its real CR/CF)."""
    return Subject(name=name, params=patient_params(name),
                   quest=quest_params(name), members=(name,))


def averaged_subject(names: list[str], new_name: str | None = None) -> Subject:
    """Subject from the element-wise average of several base patients."""
    params, quest = average_patient(names, new_name)
    return Subject(name=str(params["Name"]), params=params, quest=quest,
                   members=tuple(names))


def apply_carb_error(subject: Subject, m: float | None = None,
                     choices=DEFAULT_CARB_ERROR_CHOICES) -> Subject:
    """Return a copy of ``subject`` with a mis-set programmed carb ratio.

    ``CR_prog = CR_true * m`` is written into ``quest["CR"]`` (the value every
    controller doses from); ``cr_true`` and ``dose_mult`` are recorded for audit.
    If ``m`` is None it is drawn deterministically from the subject name via
    :func:`carb_error_multiplier`. Idempotent in intent: it perturbs relative to
    the subject's ``cr_true`` (the matched ratio), so re-applying with the same m
    is a no-op rather than compounding.
    """
    if m is None:
        m = carb_error_multiplier(subject.name, choices)
    cr_true = float(subject.cr_true if subject.cr_true is not None
                    else subject.quest["CR"])
    new_quest = dict(subject.quest)
    new_quest["CR"] = cr_true * float(m)
    return Subject(name=subject.name, params=subject.params, quest=new_quest,
                   members=subject.members, cr_true=cr_true, dose_mult=float(m))


def write_patients_csv(subjects, path: str) -> str:
    """Serialize a list of Subjects to a patients.csv (all params + CR/CF/TDI +
    carb-error audit columns + members).

    ``CR`` is the *programmed* (possibly mis-set) ratio used for dosing;
    ``CR_true`` is the physiologically matched ratio and ``dose_mult`` is the
    applied multiplier m (CR == CR_true * dose_mult). For unperturbed cohorts
    dose_mult == 1.0 and CR == CR_true.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    records = []
    for s in subjects:
        rec = {c: s.params[c] for c in _VP.columns}
        rec["CR"], rec["CF"], rec["TDI"] = s.quest["CR"], s.quest["CF"], s.quest["TDI"]
        rec["CR_true"] = float(s.cr_true if s.cr_true is not None else s.quest["CR"])
        rec["dose_mult"] = float(s.dose_mult)
        rec["members"] = "+".join(s.members)
        if getattr(s, "rbg_theta", None) is not None:
            for n, v in zip(THETA_NAMES, s.rbg_theta):
                rec[f"rbg_{n}"] = float(v)
            if s.rbg_Ib is not None:
                rec["rbg_Ib"] = float(s.rbg_Ib)
            if s.rbg_cr_true is not None:
                rec["rbg_CR_true"] = float(s.rbg_cr_true)
            if s.rbg_fit_rmse is not None:
                rec["rbg_fit_rmse"] = float(s.rbg_fit_rmse)
        records.append(rec)
    cols = (list(_VP.columns) + list(_QUEST_COLS)
            + ["CR_true", "dose_mult", "members"])
    if any(getattr(s, "rbg_theta", None) is not None for s in subjects):
        cols += RBG_ALL_COLS
    pd.DataFrame.from_records(records).reindex(columns=cols).to_csv(path, index=False)
    return path


def load_subjects_csv(path: str) -> list[Subject]:
    """Read a patients.csv back into a list of Subjects.

    Back-compatible: a CSV without the ``CR_true`` / ``dose_mult`` columns loads
    as an unperturbed cohort (dose_mult = 1.0, cr_true = CR).
    """
    df = pd.read_csv(path)
    subs = []
    for _, row in df.iterrows():
        params = row[list(_VP.columns)].copy()
        params["Name"] = str(row["Name"])
        quest = {c: float(row[c]) for c in _QUEST_COLS}
        mem = str(row["members"]) if "members" in df.columns and pd.notna(row.get("members")) else ""
        members = tuple(mem.split("+")) if mem else ()
        cr_true = (float(row["CR_true"]) if "CR_true" in df.columns
                   and pd.notna(row.get("CR_true")) else float(quest["CR"]))
        dose_mult = (float(row["dose_mult"]) if "dose_mult" in df.columns
                     and pd.notna(row.get("dose_mult")) else 1.0)
        # Cached ReplayBG fit (present only after derive_replaybg_params has run).
        rbg_theta = None
        if all(c in df.columns for c in RBG_THETA_COLS):
            vals = [row.get(c) for c in RBG_THETA_COLS]
            if all(pd.notna(v) for v in vals):
                rbg_theta = np.array([float(v) for v in vals], dtype=float)

        def _opt(col):
            return (float(row[col]) if col in df.columns and pd.notna(row.get(col))
                    else None)
        subs.append(Subject(name=str(row["Name"]), params=params,
                            quest=quest, members=members,
                            cr_true=cr_true, dose_mult=dose_mult,
                            rbg_theta=rbg_theta, rbg_Ib=_opt("rbg_Ib"),
                            rbg_cr_true=_opt("rbg_CR_true"),
                            rbg_fit_rmse=_opt("rbg_fit_rmse")))
    return subs
"""Plotting utilities for the twinning platform."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

TARGET_LOW, TARGET_HIGH = 70.0, 180.0


def plot_candidate_runs(runs: dict,
                        save_path: str = "figures/step1_candidates.png",
                        meal_time_h: float | None = None,
                        title: str = "Candidate controller CGM datasets") -> str:
    """Overlay CGM traces (top) and insulin delivery (bottom) for each candidate.

    Parameters
    ----------
    runs : dict[str, RunResult]
        Mapping of candidate name -> run, as returned by ``run_policy``.
    save_path : str
        Where to write the PNG (directories are created as needed).
    meal_time_h : float, optional
        If given, a dashed vertical line marks the meal.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, (ax_cgm, ax_ins) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 2]})

    for name, res in runs.items():
        t_h = res.df["t_min"].to_numpy() / 60.0
        ax_cgm.plot(t_h, res.df["CGM"].to_numpy(), label=name, linewidth=1.4)
        ax_ins.plot(t_h, res.df["insulin_mU_kg_min"].to_numpy(), linewidth=1.1)

    # clinical reference range on the CGM panel
    ax_cgm.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.06)
    ax_cgm.axhline(TARGET_HIGH, color="orange", linewidth=0.8, linestyle=":")
    ax_cgm.axhline(TARGET_LOW, color="red", linewidth=0.8, linestyle=":")

    if meal_time_h is not None:
        ax_cgm.axvline(meal_time_h, color="gray", linestyle="--", linewidth=1,
                       alpha=0.8, label="meal")
        ax_ins.axvline(meal_time_h, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    ax_cgm.set_ylabel("CGM [mg/dL]")
    ax_cgm.set_title(title)
    ax_cgm.legend(fontsize=8, ncol=2, loc="upper right")
    ax_cgm.margins(x=0)

    ax_ins.set_ylabel("insulin [mU/kg/min]")
    ax_ins.set_xlabel("time [h]")
    ax_ins.margins(x=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_forward_trajectory(t_min, ig, insulin, cho, cgm=None, t_cgm_min=None,
                            Gb=None, meal_time_h=None,
                            save_path="figures/step2_forward_trajectory.png",
                            title="ReplayBG forward model - sample trajectory"):
    """Plot one forward-model run: glucose (top) and the driving inputs (bottom).

    If ``cgm`` (with optional ``t_cgm_min``) is supplied it is overlaid as the
    ground-truth signal the model output is being compared against.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t_h = np.asarray(t_min, float) / 60.0

    fig, (ax_g, ax_u) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 2]})

    ax_g.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.06)
    if cgm is not None:
        tc = np.asarray(t_cgm_min, float) / 60.0 if t_cgm_min is not None else t_h
        ax_g.plot(tc, cgm, color="0.45", linewidth=1.0,
                  label="simglucose CGM (ground truth)")
    ax_g.plot(t_h, ig, color="C3", linewidth=1.6, label="ReplayBG IG (population theta)")
    if Gb is not None:
        ax_g.axhline(Gb, color="C0", linestyle="--", linewidth=0.8, label=f"Gb = {Gb:.0f}")
    if meal_time_h is not None:
        ax_g.axvline(meal_time_h, color="gray", linestyle="--", linewidth=1,
                     alpha=0.8, label="meal")
    ax_g.set_ylabel("glucose [mg/dL]")
    ax_g.set_title(title)
    ax_g.legend(fontsize=8, loc="upper right")
    ax_g.margins(x=0)

    ax_u.plot(t_h, insulin, color="C4", linewidth=1.2)
    ax_u.set_ylabel("insulin [mU/kg/min]", color="C4")
    ax_u.tick_params(axis="y", labelcolor="C4")
    ax_u.margins(x=0)
    ax_cho = ax_u.twinx()
    ax_cho.plot(t_h, cho, color="C2", linewidth=1.2)
    ax_cho.set_ylabel("CHO [mg/kg/min]", color="C2")
    ax_cho.tick_params(axis="y", labelcolor="C2")
    if meal_time_h is not None:
        ax_u.axvline(meal_time_h, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    ax_u.set_xlabel("time [h]")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)

def plot_twin_overlay(
    t_min_grid: np.ndarray,
    ig_median: np.ndarray,
    ig_iqr: np.ndarray | None,
    cgm: np.ndarray,
    t_cgm_min: np.ndarray,
    Gb: float | None = None,
    meal_time_h: float | None = None,
    save_path: str = "figures/step2_mcmc_twin_overlay.png",
    title: str = "Twin: posterior IG + CGM vs observed CGM",
    cgm_twin: np.ndarray | None = None,
    t_twin_min: np.ndarray | None = None,
    twin=None,
    run=None,
    twin_seed: int = 3,
) -> str:
    """Overlay twin IG (blue) + posterior band + twin CGM (orange) vs observed CGM.

    The orange twin-CGM trace is the noise-free IG pushed through the UVA/Padova
    sensor model. There are three ways to supply it (checked in order):

    1. pass ``cgm_twin`` (and ``t_twin_min``) directly;
    2. pass ``twin`` + ``run`` and the trace is generated via
       ``twin.replay_run(run, seed=twin_seed)`` (times taken from ``run``);
    3. pass nothing -> no orange trace (legacy IG-only behavior).

    The IG 25-75%% band is drawn only when ``ig_iqr`` is given *and* has real
    spread, so point-estimate twins don't get a misleading flat band.

    Parameters
    ----------
    t_min_grid : (T,) time axis for the IG traces [minutes].
    ig_median : (T,) twin median IG [mg/dL].
    ig_iqr : (2, T) [25th, 75th] percentile IG, or ``None`` for no band.
    cgm : (n,) observed (ground-truth) CGM [mg/dL].
    t_cgm_min : (n,) observed CGM times [minutes].
    Gb, meal_time_h : optional reference markers.
    save_path, title : output path and base title.
    cgm_twin : (m,) twin CGM [mg/dL] (orange), optional.
    t_twin_min : (m,) twin CGM times [minutes]; defaults to ``t_cgm_min``.
    twin, run : optional; if ``cgm_twin`` is not given, generate it from these.
    twin_seed : sensor seed used when generating from ``twin``.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t_h = np.asarray(t_min_grid, float) / 60.0
    tc_h = np.asarray(t_cgm_min, float) / 60.0

    # Resolve the orange twin-CGM trace.
    if cgm_twin is None and twin is not None and run is not None:
        cgm_twin = twin.replay_run(run, seed=twin_seed, use_sensor_model=True)
        if t_twin_min is None:
            t_twin_min = run.df["t_min"].to_numpy()
    tt_h = (np.asarray(t_twin_min, float) / 60.0
            if t_twin_min is not None else tc_h)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.06, zorder=0)

    # IG band -- only when present and non-degenerate
    if ig_iqr is not None:
        ig_iqr = np.asarray(ig_iqr, float)
        if float(np.max(np.abs(ig_iqr[1] - ig_iqr[0]))) > 1e-6:
            ax.fill_between(t_h, ig_iqr[0], ig_iqr[1],
                            alpha=0.22, color="C0", zorder=1,
                            label="Twin IG 25–75 %ile")

    ax.plot(t_h, ig_median, color="C0", linewidth=1.8, zorder=2,
            label="Twin median IG")

    # orange twin CGM (sensor model)
    if cgm_twin is not None:
        cgm_twin = np.asarray(cgm_twin, float)
        lbl = "Twin CGM (sensor model)"
        n = min(len(cgm_twin), len(cgm))
        if n > 0:
            resid = cgm_twin[:n] - np.asarray(cgm, float)[:n]
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            lbl = f"Twin CGM (RMSE={rmse:.1f} mg/dL)"
        ax.scatter(tt_h, cgm_twin, s=12, color="C1", zorder=3, label=lbl)

    ax.scatter(tc_h, cgm, s=12, color="0.35", zorder=3, label="Observed CGM")

    if Gb is not None:
        ax.axhline(Gb, color="C3", linestyle="--", linewidth=0.9,
                   label=f"Gb = {Gb:.0f} mg/dL")
    if meal_time_h is not None:
        ax.axvline(meal_time_h, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.8, label="meal")

    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)

def plot_cgm_overlay(
    t_cgm_min: np.ndarray,
    cgm_median: np.ndarray,
    cgm_iqr: np.ndarray,
    cgm_obs: np.ndarray,
    t_obs_min: np.ndarray | None = None,
    Gb: float | None = None,
    meal_time_h: float | None = None,
    save_path: str = "figures/step2_cgm_overlay.png",
    title: str = "Twin CGM vs observed CGM",
) -> str:
    """Standardized CGM-vs-CGM comparison figure for any twinning method.

    Every method is plotted the same way: the twin's predicted CGM median and
    25-75%% band (both produced by passing twin IG through the UVA/Padova sensor
    model via ``Twin.generate_cgm_band``) overlaid on the observed CGM. This
    replaces the inconsistent mix of IG medians, IG bands, and CGM scatter.

    The RMSE and MARD of the median CGM vs the observed CGM are annotated in the
    title so figures are directly comparable across methods.

    Parameters
    ----------
    t_cgm_min : (n,) twin CGM times [minutes] (from ``generate_cgm_band``).
    cgm_median : (n,) twin median CGM [mg/dL].
    cgm_iqr : (2, n) [25th, 75th] percentile band [mg/dL].
    cgm_obs : (m,) observed (ground-truth) CGM [mg/dL].
    t_obs_min : (m,) observed CGM times [minutes]; defaults to ``t_cgm_min``.
    Gb, meal_time_h : optional reference markers.
    save_path, title : output path and base title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t_h = np.asarray(t_cgm_min, float) / 60.0
    to_h = (np.asarray(t_obs_min, float) / 60.0
            if t_obs_min is not None else t_h)

    # RMSE / MARD on the overlap (median vs observed)
    n = min(len(cgm_median), len(cgm_obs))
    resid = np.asarray(cgm_median[:n]) - np.asarray(cgm_obs[:n])
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mard = float(np.mean(np.abs(resid) / np.maximum(cgm_obs[:n], 1e-6)) * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.06, zorder=0)

    ax.fill_between(t_h, cgm_iqr[0], cgm_iqr[1],
                    alpha=0.25, color="C0", label="Twin CGM 25-75 %ile")
    ax.plot(t_h, cgm_median, color="C0", linewidth=1.8, label="Twin median CGM")
    ax.scatter(to_h, cgm_obs, s=12, color="0.35", zorder=3,
               label="Observed CGM")

    if Gb is not None:
        ax.axhline(Gb, color="C3", linestyle="--", linewidth=0.9,
                   label=f"Gb = {Gb:.0f} mg/dL")
    if meal_time_h is not None:
        ax.axvline(meal_time_h, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.8, label="meal")

    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_title(f"{title}\nmedian CGM vs observed: RMSE={rmse:.1f} mg/dL, "
                 f"MARD={mard:.1f}%")
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_twin_ig_cgm(
    t_ig_min: np.ndarray,
    ig_median: np.ndarray,
    cgm_twin: np.ndarray,
    t_cgm_min: np.ndarray,
    cgm_obs: np.ndarray | None = None,
    t_obs_min: np.ndarray | None = None,
    ig_iqr: np.ndarray | None = None,
    Gb: float | None = None,
    meal_time_h: float | None = None,
    save_path: str = "figures/step2_twin_ig_cgm.png",
    title: str = "Twin: IG (model) vs CGM (sensor model)",
) -> str:
    """Combined view: smooth twin IG, noisy twin CGM, prob band, observed CGM.

    Shows on one axis:
      * blue line  -- twin IG median (noise-free model output, on its dt-grid);
      * light-blue band -- IG 25-75%% probability band, drawn *only if* the twin
        has posterior spread (auto-detected; point-estimate twins skip it);
      * orange dots -- twin CGM (IG passed through the UVA/Padova sensor model);
      * gray dots   -- observed (ground-truth) CGM, if supplied.

    RMSE / MARD of twin CGM vs observed CGM are annotated when ``cgm_obs`` is
    given.

    Parameters
    ----------
    t_ig_min : (T,) IG time axis [minutes] (the dt-grid).
    ig_median : (T,) twin median IG [mg/dL].
    cgm_twin : (n,) twin CGM [mg/dL] (e.g. ``twin.generate_cgm`` /
        median of ``generate_cgm_band``).
    t_cgm_min : (n,) twin CGM times [minutes].
    cgm_obs : (m,) observed CGM [mg/dL], optional.
    t_obs_min : (m,) observed CGM times [minutes]; defaults to ``t_cgm_min``.
    ig_iqr : (2, T) [25th, 75th] IG percentile band, optional. The band is
        drawn only when its spread is non-negligible.
    Gb, meal_time_h : optional reference markers.
    save_path, title : output path and base title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t_ig_h = np.asarray(t_ig_min, float) / 60.0
    t_cgm_h = np.asarray(t_cgm_min, float) / 60.0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.06, zorder=0,
               label="TIR 70-180")
    ax.axhline(TARGET_HIGH, color="orange", linewidth=0.8, linestyle=":")
    ax.axhline(TARGET_LOW, color="red", linewidth=0.8, linestyle=":")

    # IG probability band -- only "if applicable" (posterior with real spread)
    if ig_iqr is not None:
        ig_iqr = np.asarray(ig_iqr, float)
        spread = float(np.max(np.abs(ig_iqr[1] - ig_iqr[0])))
        if spread > 1e-6:
            ax.fill_between(t_ig_h, ig_iqr[0], ig_iqr[1],
                            alpha=0.20, color="C0", zorder=1,
                            label="Twin IG 25-75 %ile")

    # smooth IG (blue) and noisy CGM (orange)
    ax.plot(t_ig_h, ig_median, color="C0", linewidth=1.8, zorder=2,
            label="Twin IG (median)")

    rmse_lbl = "Twin CGM (sensor model)"
    if cgm_obs is not None:
        n = min(len(cgm_twin), len(cgm_obs))
        resid = np.asarray(cgm_twin[:n]) - np.asarray(cgm_obs[:n])
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        rmse_lbl = f"Twin CGM (RMSE={rmse:.1f} mg/dL)"

    ax.scatter(t_cgm_h, cgm_twin, s=12, color="C1", zorder=3, label=rmse_lbl)

    if cgm_obs is not None:
        to_h = (np.asarray(t_obs_min, float) / 60.0
                if t_obs_min is not None else t_cgm_h)
        ax.scatter(to_h, cgm_obs, s=12, color="0.35", zorder=3,
                   label="Observed CGM")

    if Gb is not None:
        ax.axhline(Gb, color="C3", linestyle="--", linewidth=0.9,
                   label=f"Gb = {Gb:.0f} mg/dL")
    if meal_time_h is not None:
        ax.axvline(meal_time_h, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.8, label="meal")

    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)

# ===========================================================================
# Phase B step B5: comparison figures
# ===========================================================================
# These consume the outputs of ``evaluate.run_experiment``: the per-method
# ``details`` dict (each entry an ``evaluate_twin`` result) and the headline
# comparison table (a DataFrame indexed by method).

def plot_ranking_scatter(
    details: dict,
    save_path: str = "figures/b5_ranking_scatter.png",
    title: str = "Decision transfer: twin-predicted vs true reward",
) -> str:
    """Per-method scatter of twin-predicted reward vs true reward across Pi.

    One panel per method. Each point is a candidate therapy; the true reward is
    on x and the twin's predicted reward on y. The true-best and twin-best
    therapies are highlighted, and the L1 metrics (Spearman / regret / top-k)
    are annotated. Perfect rank transfer => points are monotonically increasing.

    Parameters
    ----------
    details : ``{method -> evaluate_twin result}`` (from
        ``run_experiment(...).attrs['details']``).
    save_path, title : output path and base title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    methods = list(details)
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), squeeze=False)

    for ax, m in zip(axes[0], methods):
        res = details[m]
        names = list(res["true_rewards"])
        x = np.array([res["true_rewards"][k] for k in names], float)
        y = np.array([res["pred_rewards"][k] for k in names], float)
        ax.scatter(x, y, s=36, color="C0", zorder=2, label="therapies")

        true_best = res["true_ranking"][0]
        twin_best = res["pred_ranking"][0]
        ax.scatter([res["true_rewards"][true_best]], [res["pred_rewards"][true_best]],
                   s=120, facecolors="none", edgecolors="C2", linewidths=2.0,
                   zorder=3, label=f"true best ({true_best})")
        ax.scatter([res["true_rewards"][twin_best]], [res["pred_rewards"][twin_best]],
                   marker="x", s=90, color="C3", linewidths=2.0,
                   zorder=4, label=f"twin best ({twin_best})")

        l1 = res["l1"]
        ax.set_title(
            f"{m}\nSpearman={l1['spearman']:.2f}  "
            f"regret={l1['regret']:.0f}  top{l1['k']}={l1['top_k']:.2f}")
        ax.set_xlabel("true reward (simglucose)")
        ax.set_ylabel("twin-predicted reward")
        ax.legend(fontsize=7, loc="best")
        ax.margins(0.08)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_regret_bars(
    table,
    save_path: str = "figures/b5_regret_bars.png",
    title: str = "Decision regret by twinning method",
) -> str:
    """Bar chart of decision regret per method (lower is better; 0 = optimal).

    Parameters
    ----------
    table : the comparison DataFrame from ``run_experiment`` (needs a ``regret``
        column, indexed by method).
    save_path, title : output path and base title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    methods = list(table.index)
    regrets = [float(table.loc[m, "regret"]) for m in methods]

    fig, ax = plt.subplots(figsize=(1.6 * max(len(methods), 3), 4.4))
    bars = ax.bar(methods, regrets, color="C3", alpha=0.85)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_ylabel("decision regret  (true reward lost)")
    ax.set_xlabel("twinning method")
    ax.set_title(title)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)


def plot_decision_vs_fidelity(
    table,
    fidelity_col: str = "l3_rmse",
    decision_col: str = "spearman",
    save_path: str = "figures/b5_decision_vs_fidelity.png",
    title: str = "Decision quality vs trajectory fidelity (DT2 thesis)",
) -> str:
    """Scatter of ranking quality (L1) against trajectory error (L3) per method.

    The DT2 thesis is that fidelity is not the same as decision quality: a twin
    with larger trajectory error (worse L3 RMSE) can still rank therapies better
    (higher Spearman). Plotting decision quality on y against trajectory error on
    x makes that decoupling visible -- the desirable corner is top-left (good
    decisions, and incidentally low error), but methods may legitimately land
    top-right (good decisions despite high error), which is the whole point.

    Parameters
    ----------
    table : comparison DataFrame from ``run_experiment``.
    fidelity_col : L3 column for the x-axis (default ``l3_rmse``, lower better).
    decision_col : L1 column for the y-axis (default ``spearman``, higher better).
    save_path, title : output path and base title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    methods = list(table.index)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for i, m in enumerate(methods):
        xrmse = float(table.loc[m, fidelity_col])
        ydec = float(table.loc[m, decision_col])
        if not (np.isfinite(xrmse) and np.isfinite(ydec)):
            continue
        ax.scatter([xrmse], [ydec], s=90, color=f"C{i % 10}", zorder=3)
        ax.annotate(m, (xrmse, ydec), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)

    ax.set_xlabel(f"trajectory error  ({fidelity_col}, mg/dL)  -- lower is better")
    ax.set_ylabel(f"decision quality  ({decision_col})  -- higher is better")
    ax.set_title(title)
    ax.axhline(0.0, color="0.7", linewidth=0.7, linestyle=":")
    ax.annotate("better decisions",
                xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=8, color="0.4", va="top")
    ax.margins(0.18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)

# ===========================================================================
# Per-patient therapy IG overlay (plant vs twin, one color per therapy)
# ===========================================================================
# This is the standardized per-patient figure written next to each patient's
# comparison_table.csv (results/<phase>/<name>/ig_overlay_<method>.png). For
# every candidate therapy it overlays the plant's noise-free IG (solid) against
# the twin's replayed IG (dashed) in a single color, so rank/level agreement is
# read off at a glance. Modeled on the published ReplayBG "Data vs Replay"
# figure (solid = data, dashed = replay, one hue per perturbation).

def plot_therapy_ig_overlay(
    t_min: np.ndarray,
    true_ig_by_policy: dict,
    twin_ig_by_policy: dict,
    *,
    order: list | None = None,
    baseline_name: str | None = "bolus_x1.00",
    plant_label: str = "simglucose",
    twin_label: str = "twin",
    save_path: str = "results/ig_overlay.png",
    title: str = "Per-therapy IG: plant (solid) vs twin (dashed)",
) -> str:
    """Overlay plant IG (solid) vs twin IG (dashed) for every therapy.

    One color per therapy; the baseline therapy (if present) is drawn in black
    and slightly heavier. The legend maps each color to its therapy and, in a
    second block, the solid/dashed convention (plant vs twin).

    Parameters
    ----------
    t_min : (T,) shared time axis [minutes] for both series (the 24 h window).
    true_ig_by_policy : ``name -> (T,) plant noise-free IG`` [mg/dL] (solid).
    twin_ig_by_policy : ``name -> (T,) twin replayed IG`` [mg/dL] (dashed).
    order : optional explicit therapy order (controls color assignment);
        defaults to ``true_ig_by_policy``'s insertion order (the factor sweep).
    baseline_name : therapy drawn in black; ``None`` to disable.
    plant_label : label for the solid series (e.g. ``"simglucose"`` for Phase 2,
        ``"ReplayBG plant"`` for Phase 0).
    twin_label : label for the dashed series (e.g. ``"MCMC twin"``).
    save_path, title : output path and figure title.

    Returns the absolute path of the written figure.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    t_h = np.asarray(t_min, float) / 60.0

    names = list(order) if order is not None else list(true_ig_by_policy)
    cmap = plt.get_cmap("turbo")
    n = max(len(names), 1)
    palette = [cmap(x) for x in np.linspace(0.05, 0.95, n)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.05, zorder=0)

    therapy_handles = []
    for i, name in enumerate(names):
        is_base = (name == baseline_name)
        color = "black" if is_base else palette[i]
        lw = 2.0 if is_base else 1.3

        tg = np.asarray(true_ig_by_policy[name], float)
        k = min(len(t_h), len(tg))
        ax.plot(t_h[:k], tg[:k], color=color, linestyle="-", linewidth=lw, zorder=2)

        if name in twin_ig_by_policy:
            tw = np.asarray(twin_ig_by_policy[name], float)
            m = min(len(t_h), len(tw))
            ax.plot(t_h[:m], tw[:m], color=color, linestyle="--", linewidth=lw, zorder=2)

        label = f"{name} (baseline)" if is_base else name
        therapy_handles.append(Line2D([0], [0], color=color, lw=2.0, label=label))

    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.8, linestyle="-",
               label=f"{plant_label} (data)"),
        Line2D([0], [0], color="0.3", lw=1.8, linestyle="--",
               label=f"{twin_label} (replay)"),
    ]

    ax.set_xlabel("time [h]")
    ax.set_ylabel("glucose [mg/dL]")
    ax.set_title(title)
    ax.margins(x=0)
    ax.grid(True, alpha=0.25)

    # Two legends: therapy->color (upper) and line-style key (lower), both to
    # the right of the axes so they don't cover the traces.
    leg_therapy = ax.legend(handles=therapy_handles, title="therapy",
                            fontsize=8, loc="upper left",
                            bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.add_artist(leg_therapy)
    ax.legend(handles=style_handles, fontsize=8, loc="lower left",
              bbox_to_anchor=(1.01, 0.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(save_path)


def write_therapy_overlays(
    out_dir: str,
    true_ig_by_policy: dict,
    pred_ig_by_method: dict,
    *,
    sample_time: float = 3.0,
    order: list | None = None,
    baseline_name: str | None = "bolus_x1.00",
    plant_label: str = "simglucose",
) -> list[str]:
    """Write one ``ig_overlay_<method>.png`` per twinning method into ``out_dir``.

    Convenience wrapper used by ``compute_results`` / ``compute_results0``: it
    builds the shared time axis from the series length and ``sample_time`` (the
    runs all share the 24 h scenario, so any policy's length works), then calls
    :func:`plot_therapy_ig_overlay` once per method.

    Parameters
    ----------
    out_dir : the per-patient results dir (``results/<phase>/<name>/``).
    true_ig_by_policy : shared plant IG ``name -> (T,)``.
    pred_ig_by_method : ``method -> {name -> (T,) twin IG}``.
    sample_time : CGM sample interval [min] used to build the time axis.
    order, baseline_name, plant_label : forwarded to the plot function.

    Returns the list of written figure paths.
    """
    if not true_ig_by_policy:
        return []
    length = len(next(iter(true_ig_by_policy.values())))
    t_min = (np.arange(length) + 1) * float(sample_time)

    paths = []
    for method, pred_ig in pred_ig_by_method.items():
        save_path = os.path.join(out_dir, f"ig_overlay_{method}.png")
        title = (f"Per-therapy IG — {plant_label} (solid) vs "
                 f"{method.upper()} twin (dashed)")
        paths.append(plot_therapy_ig_overlay(
            t_min, true_ig_by_policy, pred_ig,
            order=order, baseline_name=baseline_name,
            plant_label=plant_label, twin_label=f"{method.upper()} twin",
            save_path=save_path, title=title))
    return paths
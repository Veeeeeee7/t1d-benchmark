"""Plotting utilities for the twinning platform.

Currently the per-patient therapy IG overlay (plant IG solid vs twin IG dashed,
one color per therapy) written next to each patient's results, plus a small
wrapper that emits one figure per twinning method.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

TARGET_LOW, TARGET_HIGH = 70.0, 180.0


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

    Convenience wrapper used by ``compute_results_phase2`` / ``compute_results_phase0``: it
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
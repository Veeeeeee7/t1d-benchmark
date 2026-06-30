"""Full experiment runner + reproducibility (Phase B, step B5).

Ties the whole platform together: from one configuration it identifies each
twinning method from a baseline run, evaluates all methods against the
simglucose ground truth over the candidate set Pi, and writes a results table
(CSV + Markdown).

Reproducibility
---------------
``run_all`` fixes every RNG it can reach (Python ``random``, NumPy, and Torch
if present) from ``cfg.seed`` before doing anything, and the downstream
identification / sensor noise are all seeded, so a given config reproduces its
table exactly.

Method dependencies
-------------------
Only the requested methods are imported, so an MCMC-only run needs neither
``torch`` nor ``sbi``. ``"sbi"`` requires ``torch`` and ``sbi``
for the former); they are imported lazily inside :func:`make_identify_fns`.

Usage
-----
    from t1d_twin.run_all import run_all, RunAllConfig, fast_config
    out = run_all(fast_config())          # quick MCMC-only smoke run
    out = run_all(RunAllConfig())          # full three-method comparison
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

from . import evaluate as E
from .experiment import (make_experiment_policies, DEFAULT_BOLUS_FACTORS,
                          DEFAULT_BASAL_FACTORS)
from .scenarios import single_meal_scenario
from .simglucose_adapter import run_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RunAllConfig:
    """Everything needed to reproduce one comparison run."""
    seed: int = 1

    # scenario / horizon
    meal_time_h: float = 1.0
    meal_g: float = 50.0
    hours: float = 24.0

    # candidate set Pi
    bolus_factors: tuple = DEFAULT_BOLUS_FACTORS
    basal_factors: tuple = DEFAULT_BASAL_FACTORS

    # which twins to compare
    methods: tuple = ("mcmc", "sbi")

    # per-method identification hyperparameters
    mcmc: dict = field(default_factory=lambda: dict(
        nwalkers=32, nburn=300, nsample=500, n_posterior=1000))
    sbi: dict = field(default_factory=lambda: dict(
        n_train=5000, n_posterior=1000))

    # outputs
    out_dir: str = "results"
    verbose: bool = True


def fast_config(**overrides) -> RunAllConfig:
    """A minimal, fast, MCMC-only config (used by the B5 smoke test).

    Small Pi, short chain. Any field can be overridden, e.g.
    ``fast_config(out_dir="results_smoke")``.
    """
    cfg = RunAllConfig(
        bolus_factors=(0.85, 1.0, 1.5, 2.0, 2.5),
        basal_factors=(),
        hours=24.0,
        methods=("mcmc",),
        mcmc=dict(nwalkers=16, nburn=20, nsample=40, n_posterior=60),
    )
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility + method wiring
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Seed Python ``random``, NumPy, and Torch (if importable)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_identify_fns(cfg: RunAllConfig) -> dict:
    """Build ``{method -> identify_fn(run) -> Twin}`` for the requested methods.

    Imports are lazy and per-method so unused heavy deps (torch, sbi) are never
    required. Each identify_fn closes over the config's seed and hyperparameters.
    """
    fns: dict = {}
    for m in cfg.methods:
        if m == "mcmc":
            from .identify_mcmc import identify_twin_from_run as f_mcmc
            p = dict(cfg.mcmc)
            fns["mcmc"] = (lambda run, _f=f_mcmc, _p=p:
                           _f(run, seed=cfg.seed, progress=False, **_p))
        elif m == "sbi":
            from .identify_sbi import identify_twin_from_run as f_sbi
            p = dict(cfg.sbi)
            fns["sbi"] = (lambda run, _f=f_sbi, _p=p:
                          _f(run, seed=cfg.seed, verbose=False, **_p))
        else:
            raise ValueError(f"unknown method '{m}' (expected mcmc/sbi)")
    return fns


# ---------------------------------------------------------------------------
# Markdown writer (no tabulate dependency)
# ---------------------------------------------------------------------------

def _df_to_markdown(table, floatfmt: str = "{:.3g}") -> str:
    """Render a DataFrame (indexed by method) as a GitHub Markdown table."""
    cols = list(table.columns)
    header = "| method | " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    lines = [header, sep]
    for method, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—" if v is None else "nan")
            elif isinstance(v, (int, float, np.floating, np.integer)):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all(cfg: RunAllConfig = None) -> dict:
    """Run the full comparison and write the table.

    Returns a dict with: ``table`` (the DataFrame), ``table_csv`` /
    ``table_md`` (paths), ``policy_set``, and ``true_ranking``.
    """
    cfg = cfg or RunAllConfig()
    set_all_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1. Candidate set Pi + scenario + baseline identification run.
    ps = make_experiment_policies(bolus_factors=cfg.bolus_factors,
                                   basal_factors=cfg.basal_factors)
    scenario = single_meal_scenario(meal_time_h=cfg.meal_time_h, meal_g=cfg.meal_g)
    identification_run = run_policy(ps[ps.baseline_name], scenario, hours=cfg.hours)

    # 2. Identify + evaluate every requested twin -> comparison table.
    twin_methods = make_identify_fns(cfg)
    table = E.run_experiment(
        twin_methods, identification_run, ps, scenario, cfg.hours,
        verbose=cfg.verbose)

    # 3. Write the results table (CSV + Markdown).
    table_csv = os.path.join(cfg.out_dir, "comparison_table.csv")
    table_md = os.path.join(cfg.out_dir, "comparison_table.md")
    table.to_csv(table_csv)
    with open(table_md, "w") as fh:
        fh.write(f"# Twinning-method comparison (seed={cfg.seed})\n\n")
        fh.write(f"Scenario: {cfg.meal_g:.0f} g meal at {cfg.meal_time_h:.1f} h, "
                 f"{cfg.hours:.0f} h horizon; |Pi|={len(ps)} "
                 f"({len(ps.seen)} seen / {len(ps.unseen)} unseen).\n\n")
        fh.write(f"True ranking (best->worst): "
                 f"{', '.join(table.attrs['true_ranking'])}\n\n")
        fh.write(_df_to_markdown(table))

    if cfg.verbose:
        print(f"[run_all] wrote table -> {table_csv}")
        print(f"[run_all] wrote table -> {table_md}")

    return {
        "table": table,
        "table_csv": os.path.abspath(table_csv),
        "table_md": os.path.abspath(table_md),
        "policy_set": ps,
        "true_ranking": table.attrs["true_ranking"],
    }


if __name__ == "__main__":
    # Default: full three-method comparison. For a quick run use fast_config().
    run_all(RunAllConfig())
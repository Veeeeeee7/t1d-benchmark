"""Inspect how separated the candidate therapies' TRUE rewards are.

The DT2 decision metrics (spearman / regret) are only meaningful if the
ground-truth rewards over the candidate set Pi are actually well separated and
stably rankable. If two therapies are within sensor noise of each other, the
"true ranking" between them is a coin flip and regret/spearman just measure that
noise. This script answers, per patient:

  * the true reward of every candidate therapy (same path the scorer uses:
    one fixed 24 h scenario at --scenario-seed, value.reward = -sum(magni_risk)),
  * the induced ranking, the gap between adjacent ranks, and the total spread,
  * (optionally) how those gaps compare to SENSOR NOISE: re-run each therapy on
    the SAME scenario under several sensor-noise seeds, get reward mean +/- std,
    and report a separation ratio gap / SE for each adjacent pair plus how often
    the ranking is stable across the noise draws.

Drop this in experiments/ and run from the repo root, e.g.:

    # benchmark readout (production Pi, 24 h, seed=1) for the first 8 patients
    python -m experiments.inspect_reward_separation --patients patients.csv --limit 8

    # one patient, quick (smoke Pi + 24 h), with a sensor-noise significance test
    python -m experiments.inspect_reward_separation --patients patients.csv \
        --patient synth0001_k2 --smoke --sensor-seeds 5
"""
from __future__ import annotations

import os
import sys
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "t1d_twin")) and _root not in sys.path:
    sys.path.insert(0, _root)

# One BLAS thread per process so the multiprocessing workers below don't
# oversubscribe the node (must be set before numpy/BLAS import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from experiments import exp_common as C
from experiments import patients as PT
from t1d_twin import value


def _rewards_one_seed(subject, hours, bolus_factors, basal_factors, scenario_seed, sensor_seed):
    """All therapies for one subject under ONE CGM sensor-noise seed.

    This is the parallel work unit. The meal scenario is fixed at
    ``scenario_seed`` (so scenario_seed == C.SEED reproduces the benchmark
    ground truth) and only the sensor seed varies, isolating measurement noise.
    """
    scenario, _ = C.weekly_scenario(hours, scenario_seed)
    controllers = subject.therapy_controllers(bolus_factors, basal_factors)
    recs = []
    for name, ctrl in controllers.items():
        ig = subject.run(ctrl, scenario, hours, sensor_seed=sensor_seed).bg()
        r = float(value.reward(ig))
        recs.append({"therapy": name, "sensor_seed": sensor_seed,
                     "reward": r, "mean_risk": -r / max(len(ig), 1)})
    return pd.DataFrame(recs)


def _compute_unit(payload):
    """Top-level worker (picklable): one (subject, sensor_seed) pair."""
    subject, hours, bolus_factors, basal_factors, scenario_seed, sensor_seed = payload
    return subject.name, _rewards_one_seed(subject, hours, bolus_factors,
                                           basal_factors, scenario_seed, sensor_seed)


def rewards_for_subject(subject, hours, bolus_factors, basal_factors,
                        scenario_seed, sensor_seeds):
    """All therapies x all sensor seeds for one subject (serial convenience)."""
    return pd.concat(
        [_rewards_one_seed(subject, hours, bolus_factors, basal_factors, scenario_seed, ss)
         for ss in sensor_seeds],
        ignore_index=True,
    )


def summarize_patient(df):
    """Per-therapy mean/std reward, ranked best->worst, with adjacent-gap stats."""
    n_seeds = df["sensor_seed"].nunique()
    g = df.groupby("therapy")["reward"]
    tab = pd.DataFrame({"reward_mean": g.mean(), "reward_std": g.std(ddof=0),
                        "mean_risk": df.groupby("therapy")["mean_risk"].mean()})
    tab = tab.sort_values("reward_mean", ascending=False)          # higher reward = better
    tab["rank"] = np.arange(1, len(tab) + 1)

    means = tab["reward_mean"].to_numpy()
    stds = tab["reward_std"].to_numpy()
    gap = np.full(len(tab), np.nan)
    sep = np.full(len(tab), np.nan)        # gap / standard error of the difference
    gap[:-1] = means[:-1] - means[1:]      # gap from this rank down to the next
    if n_seeds > 1:
        se = np.sqrt((stds[:-1] ** 2 + stds[1:] ** 2) / n_seeds)
        with np.errstate(divide="ignore", invalid="ignore"):
            sep[:-1] = np.where(se > 0, gap[:-1] / se, np.inf)
    tab["gap_to_next"] = gap
    tab["sep_ratio"] = sep
    return tab, n_seeds


def rank_stability(df):
    """Across sensor seeds: mean Spearman of each seed's ranking vs the pooled
    ranking, and fraction of seeds whose best therapy matches the pooled best."""
    seeds = sorted(df["sensor_seed"].unique())
    if len(seeds) < 2:
        return None, None
    pooled = df.groupby("therapy")["reward"].mean().sort_values(ascending=False)
    order = list(pooled.index)
    ref_rank = {t: i for i, t in enumerate(order)}
    rhos, best_match = [], 0
    for s in seeds:
        sub = df[df.sensor_seed == s].set_index("therapy")["reward"]
        this = sub.rank(ascending=False)
        ref = pd.Series({t: ref_rank[t] for t in this.index})
        rhos.append(this.corr(ref, method="spearman"))
        if sub.idxmax() == order[0]:
            best_match += 1
    return float(np.mean(rhos)), best_match / len(seeds)


def verdict(min_gap, spread, min_sep, n_seeds):
    frac = min_gap / spread if spread > 0 else 0.0
    msg = f"tightest adjacent gap = {min_gap:.1f} ({frac:.1%} of the {spread:.1f} spread)"
    if n_seeds > 1 and np.isfinite(min_sep):
        if min_sep >= 2.0:
            tag = "WELL SEPARATED (every adjacent pair > 2*SE apart)"
        elif min_sep >= 1.0:
            tag = "MARGINAL (an adjacent pair is only 1-2*SE apart)"
        else:
            tag = "NOT SEPARATED (an adjacent pair is within sensor noise)"
        msg += f"; min sep ratio = {min_sep:.2f}*SE -> {tag}"
    return msg


def main():
    ap = argparse.ArgumentParser()
    C.add_subject_args(ap)
    ap.add_argument("--limit", type=int, default=8, help="patients to scan if --patient not given")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="use smoke Pi + 24 h horizon (cheap)")
    ap.add_argument("--hours", type=float, default=None, help="override horizon [h]")
    ap.add_argument("--scenario-seed", type=int, default=C.SEED,
                    help="meal scenario seed; C.SEED reproduces the benchmark ground truth")
    ap.add_argument("--sensor-seeds", type=int, default=1,
                    help="number of CGM sensor-noise draws (>1 enables the significance test)")
    ap.add_argument("--out", default=None, help="write the long per-therapy table to CSV")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes; work units are (patient, sensor_seed) "
                         "pairs (default: SLURM_CPUS_PER_TASK or all cores; 1 = serial)")
    args = ap.parse_args()

    # --patients defaults to None via the shared add_subject_args; this script
    # needs the synthetic cohort, so fall back to patients.csv and fail clearly.
    if not args.patients:
        args.patients = "patients.csv"
    if not os.path.exists(args.patients):
        sys.exit(f"patients file not found: {args.patients!r}\n"
                 f"  pass --patients <path>, or run "
                 f"`python -m experiments.generate_patients --out patients.csv` first.")

    bolus_factors, basal_factors = C.factors_for(args.smoke)
    hours = args.hours if args.hours is not None else C.hours_for(args.smoke)
    sensor_seeds = list(range(args.scenario_seed, args.scenario_seed + max(args.sensor_seeds, 1)))

    subs = PT.load_subjects_csv(args.patients)
    if args.patient:
        subs = [s for s in subs if s.name == args.patient] or subs[:1]
    else:
        subs = subs[args.start:args.start + args.limit]

    n_ther = len(bolus_factors) + len(basal_factors)
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    # work units are (patient, sensor_seed) pairs, so cores scale with both axes
    payloads = [(s, hours, bolus_factors, basal_factors, args.scenario_seed, ss)
                for s in subs for ss in sensor_seeds]
    n_units = len(payloads)
    jobs = args.jobs if args.jobs else cpu
    jobs = max(1, min(jobs, n_units))
    print(f"|Pi| = {n_ther} therapies | horizon {hours:.0f} h | scenario seed {args.scenario_seed} "
          f"| {len(sensor_seeds)} sensor seed(s) | {len(subs)} patient(s)")
    print(f"~{n_ther * n_units} simglucose runs of {hours:.0f} h | "
          f"{n_units} (patient x seed) units across {jobs} worker(s)\n")

    # Every (patient, sensor_seed) unit is independent and fully seeded, so the
    # output is identical regardless of worker count or completion order.
    if jobs > 1 and n_units > 1:
        import multiprocessing as mp
        with mp.Pool(processes=jobs) as pool:
            results = pool.map(_compute_unit, payloads)
    else:
        results = [_compute_unit(p) for p in payloads]

    # regroup the per-seed pieces back into one table per patient
    from collections import defaultdict
    parts = defaultdict(list)
    for name, df in results:
        parts[name].append(df)
    by_name = {}
    for name, dfs in parts.items():
        full = pd.concat(dfs, ignore_index=True)
        full.insert(0, "patient", name)
        by_name[name] = full

    long_rows, roll = [], []
    for s in subs:
        df = by_name[s.name]
        long_rows.append(df)
        tab, n_seeds = summarize_patient(df)
        spread = tab["reward_mean"].max() - tab["reward_mean"].min()
        min_gap = np.nanmin(tab["gap_to_next"].to_numpy())
        min_sep = np.nanmin(tab["sep_ratio"].to_numpy()) if n_seeds > 1 else np.nan
        rho, best_frac = rank_stability(df)

        print(f"=== {s.name} ===  true best = {tab.index[0]}")
        show = tab.copy()
        show["reward_mean"] = show["reward_mean"].map(lambda x: f"{x:.1f}")
        show["reward_std"] = show["reward_std"].map(lambda x: f"{x:.1f}")
        show["mean_risk"] = show["mean_risk"].map(lambda x: f"{x:.2f}")
        show["gap_to_next"] = show["gap_to_next"].map(lambda x: "" if np.isnan(x) else f"{x:.1f}")
        show["sep_ratio"] = show["sep_ratio"].map(
            lambda x: "" if np.isnan(x) else ("inf" if np.isinf(x) else f"{x:.2f}"))
        cols = ["rank", "reward_mean", "reward_std", "mean_risk", "gap_to_next", "sep_ratio"]
        print(show[cols].to_string())
        print("  " + verdict(min_gap, spread, min_sep, n_seeds))
        if rho is not None:
            print(f"  rank stability across sensor seeds: mean spearman={rho:.3f}, "
                  f"top-1 unchanged in {best_frac:.0%} of draws")
        print()

        roll.append({"patient": s.name, "best": tab.index[0], "spread": spread,
                     "min_gap": min_gap, "min_gap_frac": min_gap / spread if spread else 0.0,
                     "min_sep_ratio": min_sep, "rank_spearman": rho})

    rolldf = pd.DataFrame(roll)
    if len(rolldf) > 1:
        print("=== across patients ===")
        print(f"  spread:        median {rolldf.spread.median():.1f}  "
              f"[min {rolldf.spread.min():.1f}, max {rolldf.spread.max():.1f}]")
        print(f"  min adj gap:   median {rolldf.min_gap.median():.1f}  "
              f"(median {rolldf.min_gap_frac.median():.1%} of spread)")
        tied = (rolldf.min_gap_frac < 0.02).sum()
        print(f"  patients with a near-tie (tightest gap < 2% of spread): {tied}/{len(rolldf)}")
        if rolldf.min_sep_ratio.notna().any():
            unsep = (rolldf.min_sep_ratio < 1.0).sum()
            print(f"  patients with an adjacent pair within sensor noise (sep<1): "
                  f"{unsep}/{len(rolldf)}")

    if args.out:
        pd.concat(long_rows, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
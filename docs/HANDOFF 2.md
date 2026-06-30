# Handoff — T1D Digital-Twin Decision-Transfer Benchmark

This document onboards a new agent to the project: what it is, how it's built, the
design decisions that must not be broken, and the exact state of the code. A
separate set of next-step instructions will follow this document.

---

## 1. What the project is

A benchmark for **digital twins** of a Type-1-diabetes patient. The setup:

- A **complex model** (`simglucose`, the UVA/Padova simulator) is treated as the
  ground-truth patient. **We assume we do not know its ODEs.** It only emits CGM
  (continuous glucose monitor) traces in response to insulin/meal inputs.
- A **simplified model** (`ReplayBG`, an 8-parameter ODE) is **deliberately
  different** from the complex model. It cannot reproduce the complex model
  exactly. This structural mismatch (written **Φ ∉ F**) is the entire point.
- A **digital twin is a parameter vector for the simplified model** — nothing
  else.
- A **twinning method is a map: CGM trace → simplified-model parameters.**
- Methods are judged by **DT² (decision transfer)**: take candidate therapies,
  rank them by reward when simulated on the twin, and compare that ranking to the
  *true* ranking obtained by running the same therapies on `simglucose`. A good
  method preserves the decision ranking even though the twin fits imperfectly.

The headline scientific phenomenon is a **dissociation**: a twin can have good
trajectory *fidelity* but poor *decision* quality, or vice versa. The benchmark
measures both layers and shows they come apart.

---

## 2. The core invariant — read this before changing anything

> Every twinning method maps **CGM → ReplayBG parameters**, and every method's
> machinery is confined to the **ReplayBG model family**. Methods differ only in
> *how they infer the parameters*, never in the model family.

This invariant is what makes the comparison fair and what keeps the Φ∉F premise
intact. The most likely way a new agent breaks the project is by violating it.
Concretely:

- A method must **never train on `simglucose` simulations.** Training on the
  ground-truth simulator is **simulator-family leakage**: the method would absorb
  the true dynamics that the other methods (confined to ReplayBG) can't see,
  which both breaks fairness and contaminates the Φ∉F claim. All methods train
  only on **ReplayBG forward-model simulations**; the only `simglucose` data any
  method may use is the single CGM trace it is twinning.
- This is exactly why the original neural twin was **removed**. It was a
  CGM→CGM state-space surrogate — a *different model family* with its own learned
  dynamics — not a map to ReplayBG parameters. It was replaced by an MLP that
  maps CGM→θ (see §5).
- The leakage-free way to use patient information is the **population prior**
  (§6): patients are *projected onto* the ReplayBG family (an 8-number
  bottleneck), so only ReplayBG-coordinate information crosses, and that same
  prior is shared by all three methods.

---

## 3. Repository layout

Three sibling packages at the repo root; **always run from the repo root** with
`python -m ...`.

```
t1d_twin/                 # core package — the simplified model + twinning methods
  replaybg_model.py       # the 8-param ReplayBG ODE: simulate(), steady_state()
  simglucose_adapter.py   # wraps simglucose (the ground-truth complex model)
  units.py scenarios.py policies.py sensor.py
  twin.py                 # Twin ABC: predict_ig, generate_cgm(_band), replay_run(_band)
  value.py                # reward + clinical metrics from a CGM series
  evaluate.py             # DT² scoring: decision metrics + fidelity metrics
  identify_mcmc.py        # MCMC twin (per-instance Bayesian)     -> MCMCTwin
  identify_sbi.py         # SBI twin (amortized posterior, NPE)   -> SBITwin
  identify_mlp.py         # MLP twin (amortized point estimate)   -> MLPTwin
  plotting.py experiment.py run_all.py
  nn_statespace.py        # RETIRED (old CGM->CGM twin) — not used; safe to delete

tests/                    # python -m tests.test_<name>; nn tests are retired

experiments/              # the benchmark suite (built this project)
  exp_common.py           # cadence/horizon constants, Subject resolution, save/load
  patients.py             # Subject + synthetic-patient factory + baseline controller
  generate_patients.py    # writes patients.csv (subset-averages of base adults)
  population.py            # leakage-free ReplayBG prior (fit + install)
  run_mcmc.py run_sbi.py run_mlp.py   # per-patient twin identification
  compute_results.py      # collect ground truth + score one patient's twins
  run_suite.py            # orchestrates fit->score over a slice of patients
  run_multipatient.py     # standalone ground-truth ranking sweep (optional)
  aggregate.py            # merge per-patient tables -> suite_summary.csv (+ figure)
  run_all_experiments.sh

patients.csv              # generated (1013 synthetic adults)
requirements.txt
run_experiments.slurm     # multi-node SLURM array job
results/  figures/  experiments/artifacts/   # auto-generated outputs
```

---

## 4. The two models

- **Complex (ground truth):** `simglucose` / UVA-Padova, ~13 states, ~40
  parameters (`vpatient_params.csv`). Used only to (a) generate each patient's
  baseline CGM for identification, and (b) compute the *true* therapy ranking in
  DT². Treated as a black box.
- **Simplified (the twin family):** `ReplayBG`, 8 parameters,
  `THETA_NAMES = ["ka2", "kd", "kempt", "kabs", "SG", "SI", "p2", "Gb"]`
  (SC-insulin rates, gastric emptying, glucose absorption, glucose effectiveness,
  insulin sensitivity, insulin-action rate, basal glucose). `simulate(theta,
  insulin, cho, Ib, dt)` is a batched RK4 integrator returning `(t, IG)`; it
  vectorizes over a batch of θ vectors.

The two share lineage but ReplayBG is a lossy reduction — there is no exact
parameter map between them; least-squares projection is as close as it gets
(this lossiness *is* Φ∉F).

---

## 5. The three twinning methods (the benchmark subjects)

All three output ReplayBG θ and run through the **same** ReplayBG twin-runner, so
replay/scoring is identical across methods.

| method | class | nature | trains on |
|---|---|---|---|
| **MCMC** | `MCMCTwin` | per-instance Bayesian posterior (emcee, vectorized walkers) | ReplayBG sims (likelihood) |
| **SBI** | `SBITwin` | amortized posterior (neural posterior estimation) | ReplayBG (θ, CGM) pairs |
| **MLP** | `MLPTwin` | amortized **point estimate** E[θ\|CGM] (regression) — the basic baseline | ReplayBG (θ, CGM) pairs |

- `MLPTwin` **subclasses `SBITwin`** with a one-row `theta_post` (a point
  estimate). Its prediction band is degenerate (sensor-noise only) by design — a
  point method carries no parameter uncertainty.
- The MLP and SBI draw training data from the **same** ReplayBG generator
  (`identify_sbi.generate_training_data`) and share the SBI prior box.
- Known property: the MSE-optimal MLP returns the *conditional mean* of θ, which
  under ReplayBG's non-identifiability can land in a low-density "compromise"
  region — so the MLP is expected to be the weakest method, and its `Gb` estimate
  is biased high. This is intended baseline behavior, not a bug.

---

## 6. The population prior (leakage-free)

`experiments/population.py` builds an informative, leakage-free prior:

1. Least-squares-fit ReplayBG to each of a cohort of patients' baseline CGM →
   one 8-vector θ per patient (`fit_replaybg`).
2. Aggregate into a distribution (median + per-parameter percentile range).
3. `install_population` repoints `POP_THETA` and the MCMC/SBI prior boxes at this
   distribution. The MLP inherits it through the SBI prior. **All three methods
   get the identical prior** → symmetric, no method-specific advantage.

Why it's leakage-free: fitting ReplayBG to a patient is a *projection* onto the
8-parameter family. It discards exactly the part of the patient's dynamics that
ReplayBG can't represent (the Φ∉F residual), so only ReplayBG-coordinate
(parametric) information crosses — never the complex model's *structure*. The
prior is derived from the **synthetic patients themselves** (`--patients
patients.csv --limit N`) so it covers the experiment population; `--exclude`
supports strict per-patient leave-one-out if needed (mild prior-level leakage
otherwise, shared across methods so it doesn't bias the comparison).

---

## 7. Evaluation — DT² (`t1d_twin/evaluate.py`)

Two layers, both written to each patient's `comparison_table.{csv,md}`:

- **Decision transfer (the research layer):** `spearman` (rank correlation
  between the twin's therapy ranking and the true `simglucose` ranking), `regret`
  (non-negative reward gap of the twin-chosen therapy vs the true best), `top_k`
  (top-k set agreement).
- **Fidelity (L3):** `l3_rmse` (CGM prediction RMSE on held-out therapies), plus
  bias/R² and prediction-band `coverage`/`calib_error`.

The dissociation is reported by sorting methods two ways — `by_fidelity`
(`l3_rmse`) vs `by_decision` (`spearman`) — which can disagree.

---

## 8. The pipeline and how a run flows

Per patient, `run_suite --start i --limit 1`:
1. `run_mcmc` / `run_sbi` / `run_mlp` → fit each twin, save to
   `experiments/artifacts/<patient>/{mcmc,sbi,mlp}_twin.*`.
2. `compute_results` → collect ground truth on `simglucose` for the therapy grid,
   score all three twins, write `results/<patient>/comparison_table.{csv,md}` and
   `figures/<patient>/*.png`.

`aggregate.py` merges all per-patient tables into `results/suite_summary.csv`
plus a cross-patient Spearman boxplot. (Each per-patient `run_suite` call also
writes a 1-row `suite_summary.csv`; these collide and are *not* the final answer —
always rebuild with `aggregate.py`.)

Key constants (`exp_common.py`): `SAMPLE_TIME=3` min (Dexcom), `DT=1` min,
`WEEK_HOURS=168` (production identification horizon), `SEED=1`, `SIGMA=10` mg/dL
sensor noise. `compute_results` uses `n_band=200` in production, `0` in `--smoke`.

---

## 9. Production configs (documented — keep as-is unless told otherwise)

```
MCMC : nwalkers=64, nburn=2000, nsample=6000, n_posterior=2000
SBI  : n_train=20000, max_num_epochs=400, n_posterior=2000,
       emb_out=16, hidden=64, transforms=8, train_batch=512, stop_after=30
MLP  : n_train=20000, hidden=128, n_epochs=300, lr=1e-3, batch_size=256
```

Each `run_*.py` has a `PROD` and a `SMOKE` dict; `--smoke` selects tiny configs
and a short horizon for plumbing tests.

---

## 10. How to run

```bash
# from the repo root, with the conda env active

# (1) one-time prep
python -m experiments.generate_patients --out patients.csv
python -m experiments.population --patients patients.csv --limit 100 --hours 12 \
       --out experiments/artifacts/population.npz

# (2) quick validation on a few patients (smoke OR a few real indices)
seq 0 3 | xargs -P 4 -I{} python -m experiments.run_suite \
    --patients patients.csv --population experiments/artifacts/population.npz \
    --start {} --limit 1

# (3) full multi-node sweep (SLURM array, one 256-patient chunk per node)
sbatch run_experiments.slurm            # default --array=0-3, CHUNK=256

# (4) aggregate once ALL array tasks finish
python -m experiments.aggregate
```

`run_experiments.slurm` pins BLAS threads to 1, runs 64 patients per node in
parallel (`xargs -P`), is resumable (skips patients whose
`comparison_table.csv` exists) and crash-tolerant. The prep MUST be run before
`sbatch` so array tasks don't race on it.

---

## 11. Work completed in the session that produced this handoff

- **Initial review + fixes** to `t1d_twin/`: import fix in `run_all.py`, removed a
  redundant simulate in `twin.py`, seeded RNGs in `identify_mcmc`/`identify_sbi`,
  fixed an SBI simulator test. All step tests pass; numerical anchors reproduce.
- **Built the entire `experiments/` suite** (everything in §3 under
  `experiments/`), including synthetic-patient generation (`patients.py`,
  `generate_patients.py` → 1013 adults that are subset-averages of the 10 base
  UVA-Padova adults; the `ExplicitBBController` reproduces the adult baseline).
- **Replaced the neural twin** (CGM→CGM state-space, a leakage risk) with the
  **MLP CGM→θ** point-estimate twin (`identify_mlp.py`); rewired all loaders,
  stages, and the suite from `nn` → `mlp`; retired `nn_statespace.py`.
- **Built the population prior** (`population.py`) with the leakage-free
  projection argument, wired `--population` through the fit scripts and suite.
- **Bug fix — band IndexError:** point-estimate twins (1-row `theta_post`) crashed
  in the n_band=200 scoring path (smoke used n_band=0 and never hit it). Fixed by
  tiling/resampling the IG ensemble up to `n_samples` in `SBITwin._ig_ensemble`
  and `MCMCTwin._ig_ensemble`.
- **Bug fix — SI prior railing:** all patients pinned `SI` at the prior cap
  (5e-4). Lifted the SI upper bound to **2e-3** in `population.py`,
  `identify_mcmc.py`, `identify_sbi.py`, and moved population derivation to the
  synthetic patients so the prior covers them.
- **Compute/runtime:** measured ~9.3 h MCMC / ~2.1 h SBI / ~1.6 h MLP per patient
  (~13 h total). Wrote the multi-node SLURM array (`run_experiments.slurm`) and
  `aggregate.py`; produced `requirements.txt`.

---

## 12. Known issues, caveats, and gotchas

- **The SI fix is not yet validated at scale.** Confirm on a few synthetic
  patients that `SI` no longer pins at a bound (now 2e-3) and that MCMC fit RMSE
  improves from the 37–87 mg/dL seen before. If `SI` now rails at 2e-3, widen
  further or increase the population `--limit`/margin.
- **Smoke does not exercise the band path** (`n_band=0`). Any change touching
  prediction bands, coverage, or `_ig_ensemble` must be tested with `n_band>0`
  (i.e. a non-smoke run), or it will pass smoke and fail in production — that's
  exactly how the IndexError slipped through.
- **`suite_summary.csv` collides** under parallelism; the per-patient
  `comparison_table.csv` files are the source of truth — always rebuild via
  `aggregate.py`.
- **Thread pinning is mandatory** when running many processes per node
  (`OMP_NUM_THREADS=1` etc.), or 64 processes oversubscribe the cores. The SLURM
  script sets this.
- **MLP `Gb` bias / weak fits are expected** (point estimate under
  non-identifiability), not a defect to "fix."
- **GPU would not help** the dominant cost (ODE integration is CPU/numpy); the
  only GPU-able parts are SBI/MLP neural training, which currently run on CPU.
  More CPU cores / nodes is the right scaling lever.
- **`nn_statespace.py` and `tests/test_step2_nn*` are retired** — not imported by
  the experiment path; delete or ignore.
- **Re-scoring vs refitting:** saved twin artifacts are valid across the bug
  fixes; to re-score without the expensive refit, call `compute_results`
  directly per patient (it only loads twins + collects ground truth). `run_suite`
  refits.
```

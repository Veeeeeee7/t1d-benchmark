# Implementation Plan — T1D Decision-Targeted Twinning Platform

This document specifies the remaining work as a sequence of small, independently
testable chunks. **Working rule for the coder:** implement one chunk, write and
run its test, confirm it passes, then move to the next. Do not start a chunk
before its dependencies' tests pass. Each test goes in `tests/` and is run from
the repo root with `python -m tests.test_<name>`.

---

## 0. Project context (read first)

**Goal.** Given a patient's CGM + insulin + carbohydrate data, identify a *digital
twin* (a personalized glucose-dynamics model), use the twin to rank candidate
control therapies in silico, and pick the best. The research question is which
*twinning method* best supports correct therapy ranking, measured by
**decision transfer** (does the twin rank therapies the way the real system
would), not by trajectory fidelity.

**Ground truth** is the `simglucose` simulator (UVA/Padova). **Twins** are
reduced ReplayBG-class models that provably cannot reproduce the ground truth
(Φ∉F) — this mismatch is intentional and is the whole point.

**Three twinning methods to compare:**
1. ReplayBG-style Bayesian identification by MCMC (mechanistic ODE).
2. Simulation-Based Inference / Neural Posterior Estimation (same ODE, amortized
   flow posterior, can jointly infer initial conditions).
3. Physiologically-constrained NN state-space model (hybrid neural-ODE,
   architected on the ReplayBG structure, trained on data).

**Repo layout (already in place):**
```
t1d_twin/
  __init__.py
  units.py                # simglucose <-> ReplayBG unit conversions
  scenarios.py            # meal scenarios
  policies.py             # controllers / candidate therapies
  simglucose_adapter.py   # run any controller on simglucose -> RunResult
  replaybg_model.py       # ReplayBG forward ODE model (the shared engine)
  plotting.py             # figures
tests/
  test_step1.py           # data generation (passing)
  test_step2_model.py     # forward model (passing)
requirements.txt          # numpy, scipy, pandas, matplotlib, simglucose,
                          # gymnasium, emcee, sbi, torch
```

**Test file convention** (every test file starts with this so it runs from
anywhere):
```python
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.dirname(_here)):
    if os.path.isdir(os.path.join(_cand, "t1d_twin")):
        sys.path.insert(0, _cand); break
```

**Key facts the coder must respect (do not re-derive):**
- simglucose `sample_time = 3 min`. CGM/BG in mg/dL. `info['meal']` is CHO rate in g/min. `Action(basal, bolus)` both in U/min. adult#001 `BW = 102.32 kg`.
- Conversions: `mU/kg/min = U/min * 1000 / BW`, `mg/kg/min = g/min * 1000 / BW`.
- ReplayBG inputs are insulin in mU/kg/min and CHO in mg/kg/min on a regular dt-minute grid.

**Existing public APIs (source of truth — read the files, do not reimplement):**
- `units`: `insulin_U_per_min_to_mU_per_kg_min(u, BW)`, `cho_g_per_min_to_mg_per_kg_min(g, BW)`, and inverses.
- `scenarios`: `single_meal_scenario(meal_time_h=1.0, meal_g=50.0, start_time=...)`.
- `policies`: `ModulatedBBController(bolus_factor=1.0, basal_factor=1.0, target=140.0)` (a simglucose `Controller`), `baseline_policy()`, `make_candidate_policies(bolus_factors=..., basal_factors=...) -> dict[str, Controller]`.
- `simglucose_adapter`: `run_policy(controller, scenario, hours, patient_name="adult#001", sensor_name="Dexcom", pump_name="Insulet", sensor_seed=1) -> RunResult`.
  - `RunResult.df` columns: `t_min, datetime, CGM, BG, CHO_g_min, basal_U_min, bolus_U_min, insulin_U_min, CHO_mg_kg_min, insulin_mU_kg_min, lbgi, hbgi, risk`.
  - `RunResult.BW`, `RunResult.sample_time (=3.0)`, `RunResult.cgm()`, `RunResult.bg()`, `RunResult.replaybg_inputs(dt=1.0) -> (t_grid, insulin_mU_kg_min, cho_mg_kg_min)` (sample-rate inputs held piecewise-constant on the dt grid).
- `replaybg_model`: `THETA_NAMES` (`["ka2","kd","kempt","kabs","SG","SI","p2","Gb"]`), `POP_THETA` (dict), `theta_to_array(theta)`, `rho(G, Gb)`, `steady_state(theta, Ib) -> (9,) or (B,9)`, `simulate(theta, insulin, cho, Ib, dt=1.0, x0=None, return_states=False) -> (t, ig[, states])` where `ig[...,m]` is IG at time `(m+1)*dt` and `theta` may be `(8,)` or `(B,8)` (batched), `sample_indices(n_samples, sample_time, dt=1.0)`. Fixed constants: `VI, KE, BETA, F, VG, ALPHA, GTH, R1, R2`.
- `plotting`: `plot_candidate_runs(...)`, `plot_forward_trajectory(...)`.

**Design decisions (fixed defaults — keep unless told otherwise):**
- Reward = negative summed Magni risk (primary); also report clinical metrics.
- Policy set Π = open-loop basal/bolus modulations (v1). Closed-loop is optional later.
- Method 3 (NN state-space) is architected on the **ReplayBG** ODE structure, so all three twins share one physiological scaffold.
- Twin CGM = twin IG passed through simglucose's CGM sensor model (so twin output is directly comparable to ground-truth CGM).
- Initial conditions: MCMC uses steady-state; SBI infers them; NN uses steady-state optimization.

---

## PHASE A — Finish the twinning methods (step 2)

### A1. Common `Twin` interface + sensor model
**Files:** `t1d_twin/twin.py`, `t1d_twin/sensor.py`
**Implement:**
- `sensor.py`: `add_cgm_noise(ig, seed=1, sensor_name="Dexcom") -> cgm` that runs simglucose's `CGMSensor` over the IG series (treat IG as the sensor's input glucose) to produce a realistically noisy CGM. Provide a `gaussian_noise(ig, sigma, seed)` fallback.
- `twin.py`: abstract base class `Twin` with:
  - `predict_ig(insulin, cho, Ib, dt=1.0, n_samples=None) -> np.ndarray` (median IG, or ensemble if the method has a posterior),
  - `generate_cgm(insulin, cho, Ib, sample_time, dt=1.0, seed=1) -> np.ndarray` (sample IG at observation times, add sensor noise),
  - `summary() -> dict`,
  - a `replay_run(run: RunResult) -> np.ndarray` convenience that takes a `RunResult`, pulls its inputs, and returns a predicted CGM aligned to the run's samples.
**Depends on:** existing modules.
**Test (`test_twin_interface.py`):** a trivial `ConstantThetaTwin(Twin)` backed by `replaybg_model.simulate` at `POP_THETA`. Assert `generate_cgm` returns the right length and physiological range; `add_cgm_noise` output has nonzero, bounded deviation from the clean IG (e.g., std in ~1–15 mg/dL).

### A2. MCMC twin
**File:** `t1d_twin/identify_mcmc.py` (implements `Twin`)
**Implement:**
- **Transform + priors:** sample the 7 positive params in log-space and `Gb` linearly; uniform priors inside bounds `ka2∈[1e-3,5e-2], kd∈[1e-3,8e-2], kempt∈[2e-2,6e-1], kabs∈[1e-3,8e-2], SG∈[1e-3,5e-2], SI∈[1e-5,5e-4], p2∈[1e-3,5e-2], Gb∈[80,200]`. `log_prior` = 0 inside, `-inf` outside.
- **Likelihood:** Gaussian on CGM residuals. Batched `simulate` over walkers, take `ig[:, sample_indices(n, sample_time, dt)]`, compute `-0.5*Σ((cgm-ig_obs)/σ)² - n*logσ`; fixed `σ≈10` mg/dL for v1; `-inf` if any sim is non-finite.
- **`log_prob = log_prior + log_likelihood`, vectorized;** use `emcee.EnsembleSampler(nwalkers, ndim=8, log_prob, vectorize=True)` (one batched `simulate` per step). 32–64 walkers initialized at `POP_THETA` (with `Gb`≈fasting median of the data) + small jitter inside the prior. Burn-in then sample; flatten; subsample 1000 posterior realizations.
- **`MCMCTwin(Twin)`** holding the posterior, `Ib`, `sample_time`, `dt`, `σ`, median θ.
- **`identify_twin_from_run(run, **kw) -> MCMCTwin`** pulling `cgm=run.cgm()`, `(_,insulin,cho)=run.replaybg_inputs(dt)`, `Ib=insulin[0]`, `sample_time=run.sample_time`.
**Depends on:** A1.
**Test (`test_step2_mcmc.py`):**
- *Round-trip recovery:* simulate IG from a known `θ_true` + therapy, add Gaussian noise → synthetic CGM; fit (short chain, e.g. 32 walkers × ~600 steps); assert `Gb` within ±10 and `SI` within ~2× of truth, and median-IG RMSE to the noiseless truth is a few mg/dL.
- *Acceptance ("create twin + generate CGM"):* fit to the step-1 simglucose baseline run, generate twin CGM under the same therapy, assert physiological range and RMSE to the data < ~30 mg/dL.
- **Note:** the model is a priori non-identifiable; test only the well-determined params (`Gb`, `SI`) and the predictive fit, not all 8. A short chain validates machinery, not convergence.

### A3. SBI twin — simulator, prior, training data
**File:** `t1d_twin/identify_sbi.py` (part 1)
**Implement:**
- A `sbi`-compatible prior over the 8 params (use the same bounds as A2; `sbi.utils.BoxUniform` in transformed or linear space).
- A `simulator(theta) -> observation` wrapping `replaybg_model.simulate` for a *fixed identification therapy* (the baseline run's inputs): returns the IG sampled at observation times (optionally + sensor noise) as the observation vector `y`.
- Training-data generation: draw N (e.g., 5000) θ from the prior, simulate, **rejection-sample** to keep only obs with all values in [40, 400] mg/dL.
**Depends on:** A1.
**Test (`test_step2_sbi_sim.py`):** prior samples lie in bounds; `simulator` returns the expected length; rejection sampling yields the requested count with all values in [40,400]; a batch of identical θ gives identical observations.

### A4. SBI twin — train NPE + Twin wrapper
**File:** `t1d_twin/identify_sbi.py` (part 2)
**Implement:**
- Train Neural Posterior Estimation with `sbi` (MAF density estimator) on the (θ, y) pairs from A3.
- `SBITwin(Twin)`: stores the trained posterior; `identify` = condition the posterior on an observed CGM and draw 1000 samples; then behaves like A1's `Twin` (predict/generate from posterior samples).
- `identify_twin_from_run(run, **kw) -> SBITwin`.
- **Optional extension (flag in code):** joint inference of initial conditions `θ̂=[θ, x0]∈R^17` per Hoang et al. — sample `x0` by simulating from steady state and taking a shifted window; pass `x0` into `simulate`. Keep behind a `infer_ic=False` default so v1 stays steady-state.
**Depends on:** A3.
**Test (`test_step2_sbi.py`):**
- *Amortized recovery:* on a held-out simulated trace, the posterior median recovers `Gb`/`SI` within tolerance and ~90%+ of truths fall in the 95% credible interval across several test traces (loose check on a small sample).
- *Acceptance:* create twin from the simglucose baseline run, generate CGM, assert physiological + RMSE < ~30 mg/dL.
- Keep N and training epochs small for test speed; mark the heavier full-accuracy run as non-test config.

### A5. NN state-space twin — architecture + differentiable integrator
**File:** `t1d_twin/nn_statespace.py` (part 1)
**Implement (PyTorch):**
- A neural state-space model mirroring the ReplayBG compartment structure: one small MLP per state-derivative (or grouped), inputs limited to the physiologically relevant states/inputs of that compartment (as in T1DSim_AI eq. 4, adapted to ReplayBG's 9 states).
- A differentiable forward integrator (Euler, `dt=1`) producing the IG trajectory from an initial state + input series, matching the `simulate` I/O convention (inputs on the dt grid, IG at `(m+1)*dt`).
**Depends on:** A1.
**Test (`test_step2_nn_arch.py`):** forward pass runs with correct shapes; gradients flow (a loss backprops without NaNs); the model can overfit a single short ReplayBG-generated trajectory to low RMSE after a few hundred steps (sanity that it can represent dynamics).

### A6. NN state-space twin — population training
**File:** `t1d_twin/nn_statespace.py` (part 2)
**Implement:**
- Generate a population training set of ReplayBG simulations (vary θ over the prior, vary meal/insulin scenarios) using `replaybg_model.simulate`.
- Train the population NN by truncated simulation-error minimization (short overlapping sequences) with a glucose-fit loss + a clinically-weighted penalty (heavier on hypo/hyper errors). Hold out a test split.
**Depends on:** A5.
**Test (`test_step2_nn_pop.py`):** on held-out ReplayBG sims, the trained population NN reproduces trajectories with low RMSE and clinically-equivalent TIR/TBR/TAR (within a few %). Keep dataset/epochs small for the test; provide a larger config separately.

### A7. NN state-space twin — individual residual + Twin wrapper
**File:** `t1d_twin/nn_statespace.py` (part 3) → `NNStateSpaceTwin(Twin)`
**Implement:**
- An individual-level residual NN added to the glucose-compartment derivative, trained by gradient descent on the patient's simglucose baseline run.
- Initial-state estimation by steady-state optimization (gradient descent to `ẋ(0)=0` given the model).
- `NNStateSpaceTwin(Twin)` exposing the standard `predict_ig` / `generate_cgm`; `identify_twin_from_run(run, **kw)`.
**Depends on:** A6.
**Test (`test_step2_nn.py`):** acceptance — create twin from the simglucose baseline run, generate CGM, assert physiological + RMSE to data < ~25 mg/dL (the NN twin should fit at least as well as MCMC here).

### A8 (optional). Conformance verification
**File:** `t1d_twin/conformance.py`
**Implement:** δ-monotonicity checks per sub-network against the ReplayBG partial-derivative signs (MILP via Gurobi if licensed; otherwise a sampling-based falsification fallback).
**Test:** report critical errors per compartment; assert the fallback runs without Gurobi. Mark Gurobi path as skipped when unlicensed.

---

## PHASE B — Evaluation (step 3)

### B1. Reward + clinical metrics
**File:** `t1d_twin/value.py`
**Implement:** from a glucose series (mg/dL) compute Magni risk, TIR (70–180), TBR (<70), TAR (>180), LBGI, HBGI, mean glucose. `reward(glucose) = -sum(magni_risk(glucose))` (default objective). All pure functions on arrays.
**Depends on:** none.
**Test (`test_value.py`):** hand-checked values — a constant 112 mg/dL series gives Magni risk ≈ 0 and reward ≈ 0 (max); a series with sustained lows gives high LBGI/TBR; a hyper series gives high HBGI/TAR; TIR+TBR+TAR ≈ 100%. Compare against literature anchor points.

### B2. Policy set Π + seen/unseen split
**File:** `t1d_twin/experiment.py`
**Implement:** assemble the candidate set from `make_candidate_policies`, possibly widening factors so over-dosing causes hypoglycemia (so the optimal therapy is *interior*, not "max bolus"). Tag a `seen` subset (baseline-adjacent therapies used for identification context) and an `unseen` subset (therapies far from baseline — the real generalization test). Return `{name: controller}` plus the partition.
**Depends on:** existing `policies`.
**Test (`test_experiment_policies.py`):** Π has the expected members; seen/unseen are disjoint and cover Π; running two members on simglucose yields distinct rewards (via B1).

### B3. Metrics + ground-truth ranking ("test metrics on simglucose first")
**File:** `t1d_twin/evaluate.py` (part 1)
**Implement:**
- `evaluate_policies_on_system(policies, run_fn) -> dict[name -> reward]` where `run_fn(controller)` returns a glucose series (for ground truth, `run_fn` runs `run_policy` on simglucose and returns CGM/BG).
- L1 metrics on two reward dicts: Spearman rank correlation, decision regret (true reward of the best-by-twin policy minus true best), top-k agreement.
- L2 outcome fidelity: predicted vs true clinical metrics across Π (bias, RMSE, R²).
- L3 trajectory metrics: RMSE / MARD between predicted and true CGM, and prediction-band calibration.
**Depends on:** B1, B2.
**Test (`test_step3_metrics.py`):** run Π on simglucose to get the **true** ranking, then validate the metric plumbing on simglucose-only: Spearman of the true ranking with itself = 1, regret of true-vs-true = 0, top-k agreement = 1; assert the true ranking is **non-trivial** (best policy is interior, not an endpoint). This gates the twin comparison.

### B4. Twin evaluation + head-to-head comparison ("then on the twins")
**File:** `t1d_twin/evaluate.py` (part 2)
**Implement:**
- `evaluate_twin(twin, policies, true_rewards, true_cgm_by_policy) -> dict` that replays Π on the twin (predict CGM per policy via `twin.replay_run` / `generate_cgm`), computes predicted rewards and the predicted ranking, and returns L1/L2/L3 vs ground truth.
- `run_experiment(twin_methods: dict[name -> identify_fn], identification_run, policies, ...) -> table` that identifies each twin from the identification run, evaluates each, and produces a comparison table (rows = methods, cols = Spearman, regret, top-k, L2 fidelity, L3 RMSE).
**Depends on:** B3, and at least A2 (MCMC) available.
**Test (`test_step3_twins.py`):** end-to-end with the MCMC twin only — pipeline runs, the comparison row is well-formed, all metrics in valid ranges, and the twin's ranking correlates positively with truth (loose `Spearman > 0`). Use a small Π and short MCMC chain for speed.

### B5. Full experiment runner + figures + reproducibility
**File:** `t1d_twin/run_all.py` (and `plotting.py` additions)
**Implement:** a config (seeds, scenario, Π, which methods) that runs the full three-method comparison, writes a results table (CSV/markdown) and figures: per-method ranking scatter (twin vs true reward), regret bar chart, and a decision-vs-fidelity plot (L1 ranking quality against L3 RMSE) illustrating the DT2 thesis that fidelity ≠ decision quality. Fix all seeds.
**Depends on:** B4, all of Phase A available.
**Test (`test_run_all_smoke.py`):** with a minimal config (MCMC only, few policies, short chain) `run_all` produces the expected output files and a table with the expected columns; no crashes; deterministic given the seed.

---

## PHASE C — Optional extensions (after A+B pass)

- **C1. Relax the steady-state assumption.** Add a non-stationary scenario (recent prior meal/insulin so steady-state ICs are wrong); show SBI's joint IC inference (A4 extension) degrades less than MCMC's fixed-IC fit. Test: SBI's L1/L3 beats MCMC's on the non-stationary scenario.
- **C2. Closed-loop Π (v2).** Add genuine closed-loop controllers (e.g., PID) behind the existing `Controller` interface; requires controller-in-the-loop replay on the twins. Test: a closed-loop controller runs on both simglucose and a twin through the same loop.
- **C3. Multiple subjects / robustness.** Repeat the comparison across several simglucose virtual patients and seeds; report distributions. Test: aggregation runs over ≥2 subjects and produces per-subject + pooled tables.

---

## Suggested order
A1 → A2 → B1 → B2 → B3 → B4 (you now have a working end-to-end comparison with one twin) → A3 → A4 → A5 → A6 → A7 (all three twins) → B5 (full comparison) → C as desired.

This order front-loads a complete, testable pipeline with the simplest twin (MCMC) before adding the heavier SBI and NN methods, so integration problems surface early.

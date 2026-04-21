# Day 1: Hovorka ODE Digital Twin Baseline

## What this does

Builds a minimum-viable mechanistic digital twin for Type 1 Diabetes and
measures both **reconstruction RMSE** (twin vs. real trajectory on held-out
time) and **counterfactual RMSE** (twin under perturbed inputs vs. ground
truth under the same perturbation).

The pipeline:

1. Generate a "real patient" via simglucose (`adolescent#001`, 48h, standard meal schedule).
2. Generate a counterfactual ground-truth (same patient, same meals, but all boluses +20%).
3. Fit Hovorka's four patient-specific parameters (Sf1, Sf2, Sf3, EGP0) to the first 24h of the real trace via Levenberg-Marquardt (lmfit).
4. Simulate the twin on hours 24-48 under the *real* inputs → **reconstruction RMSE**.
5. Simulate the twin on hours 0-48 under the *perturbed* inputs, compare to the counterfactual ground truth on hours 24-48 → **counterfactual RMSE**.

## Expected output

For `adolescent#001`:

```
Fit-window RMSE       :  14.51 mg/dL
Reconstruction RMSE   :  17.91 mg/dL
Counterfactual RMSE   :  12.60 mg/dL
CF/Rec ratio          :   0.70x
```

Reconstruction RMSE in the 15-25 mg/dL range and a CF/Rec ratio near 1.0 (anywhere from 0.5 to 2.0) indicate a healthy mechanistic baseline. This is the floor your fancier methods should beat on reconstruction, while keeping the CF/Rec ratio close to 1.

## Running

### 1. Environment

Python 3.10+ required (simglucose needs >=3.9 and some transitive deps want 3.10+).

```bash
python -m venv twin_env
source twin_env/bin/activate      # Linux/Mac
# or: twin_env\Scripts\activate   # Windows

pip install simglucose lmfit scipy pandas numpy matplotlib
```

### 2. Run

```bash
python day1_hovorka_twin.py
```

Takes ~30-60 seconds on a laptop. Most of that is the two simglucose runs;
the fit itself is ~10 seconds.

### 3. Outputs

All written to `./outputs/`:

- `patient_001_48h.csv` — ground-truth trace (real scenario)
- `patient_001_48h_cf.csv` — counterfactual ground truth (+20% boluses)
- `twin_results.csv` — per-step twin predictions (both scenarios)
- `day1_results.png` — two-panel diagnostic plot

Plus a printed summary of the four RMSE numbers.

## What each RMSE tells you

| Metric | What it measures | What a bad number means |
|--------|-----------------|-------------------------|
| Fit-window RMSE | How well the optimizer minimized the objective | If very high: optimizer failed or model is misspecified |
| Reconstruction RMSE | Honest held-out fidelity to the individual | If very high: twin won't represent this patient |
| Counterfactual RMSE | Twin's response to new controller inputs vs. ground truth | If very high: twin is not a valid substrate for controller eval |
| CF/Rec ratio | Stability of twin under perturbation | >3x: twin extrapolates badly, even if reconstruction looks OK |

The last row is the key insight: a twin with low reconstruction RMSE but 5x
higher counterfactual RMSE will make a researcher's controller look great on
the twin while performing terribly on the real patient. Your framework's job
is to expose this.

## Extending

To benchmark on other patients, change the `patient_name` argument in `generate_patient_trace(...)`. simglucose provides 30 virtual patients: `adolescent#001` through `#010`, `adult#001` through `#010`, `child#001` through `#010`.

To benchmark on different counterfactuals, change `bolus_scale` (tests dosing-multiplier robustness) or subclass `ScaledBolusController` to implement more interesting perturbations (timing shifts, basal changes, etc.).

To swap the twin method, replace `fit_twin` and `simulate_hovorka` with your new method. Everything else in the evaluation harness stays the same. This is the template for Day 2+ methods.

## Known limitations

- **Only 4 fitted parameters.** The twin is underparametrized. Real ReplayBG fits ~9 parameters including carb absorption rates. This is deliberate for a baseline — you want the mechanistic floor to be clearly beatable.
- **Steady-state init is numerical.** The warmup integration takes a few seconds per fit evaluation; a cheaper analytical init with EGP0 fitted may work but requires more algebra.
- **Piecewise-constant inputs.** simglucose reports inputs as step-averages over 3-min bins; the twin treats inputs as constant within each bin. Misalignment on sub-3-min scales is possible but negligible at CGM resolution.
- **No CGM sensor noise model.** simglucose's Dexcom sensor adds colored noise; the twin assumes perfect CGM. For real-data deployment, add a Kalman filter or sensor model.

## File manifest

- `day1_hovorka_twin.py` — single-file deliverable, ~550 lines
- `day1_results.png` — example output plot showing a successful run

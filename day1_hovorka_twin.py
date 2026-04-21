"""
Day 1: Mechanistic Hovorka ODE Digital Twin Baseline

This script builds a minimal-viable digital twin for Type 1 Diabetes using
the Hovorka glucose-insulin ODE model. It:

  1. Generates a "ground-truth patient" using simglucose (UVA/Padova simulator).
  2. Splits the trace into a fit window (first 24h) and an evaluation window (24-48h).
  3. Fits Hovorka's patient-specific insulin-sensitivity parameters (Sf1, Sf2, Sf3)
     to the fit-window CGM via Levenberg-Marquardt (lmfit).
  4. Measures RECONSTRUCTION RMSE: twin simulated with fitted params under
     the real insulin/carb inputs vs. the real CGM on the held-out eval window.
  5. Measures COUNTERFACTUAL RMSE: twin simulated with insulin boluses scaled
     by +20% vs. the ground-truth response (simglucose rerun with +20% boluses).

The twin's "reconstruction RMSE" tells you how faithful the twin is to the
individual. The "counterfactual RMSE" tells you whether the twin is valid
as a substrate for controller evaluation --- if it responds correctly to
perturbed inputs, a researcher's controller run on it will produce meaningful
TIR estimates.

Usage
-----
    python day1_hovorka_twin.py

Outputs
-------
    - patient_001_48h.csv          : ground-truth simglucose trace
    - patient_001_48h_cf.csv       : ground-truth with +20% boluses
    - twin_results.csv             : per-timestep twin predictions
    - day1_results.png             : diagnostic plot
    - printed summary of RMSE numbers
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import odeint
import lmfit

# simglucose imports
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.base import Controller, Action
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim


# ----------------------------------------------------------------------------
# 1. Ground-truth patient generation via simglucose
# ----------------------------------------------------------------------------

class ScaledBolusController(Controller):
    """Wraps BBController, multiplies the bolus portion by a scale factor.

    Used for generating counterfactual ground-truth: what would happen if the
    patient had taken +20% more insulin at each meal?
    """
    def __init__(self, bolus_scale=1.0):
        self.base = BBController()
        self.bolus_scale = bolus_scale

    def policy(self, observation, reward, done, **kwargs):
        action = self.base.policy(observation, reward, done, **kwargs)
        # Action is named tuple (basal, bolus); scale only bolus
        return Action(basal=action.basal, bolus=action.bolus * self.bolus_scale)

    def reset(self):
        self.base.reset()


def generate_patient_trace(patient_name='adolescent#001', duration_hours=48,
                           bolus_scale=1.0, out_path='patient_trace.csv',
                           seed=1):
    """Generate a simglucose CGM trace. Returns DataFrame with DatetimeIndex."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')

    # Standard 3-meal scenario: breakfast 45g at 7am, lunch 70g at noon,
    # dinner 80g at 6pm. Day 2 repeats the same pattern.
    meals = [(7, 45), (12, 70), (18, 80),
             (31, 45), (36, 70), (42, 80)]
    scenario = CustomScenario(start_time=start_time, scenario=meals)

    env = T1DSimEnv(patient, sensor, pump, scenario)
    ctrl = ScaledBolusController(bolus_scale=bolus_scale)

    sim_obj = SimObj(env, ctrl,
                     timedelta(hours=duration_hours),
                     animate=False, path='/tmp/sim_output')
    results = sim(sim_obj)

    # Add a minute-counter column for convenience
    results = results.copy()
    results['t_min'] = ((results.index - results.index[0]).total_seconds()
                        / 60.0)
    # Body weight is needed later for unit conversion
    results.attrs['BW'] = patient._params.BW
    results.to_csv(out_path)
    return results


# ----------------------------------------------------------------------------
# 2. Hovorka ODE model
# ----------------------------------------------------------------------------
#
# 10-state glucose-insulin model from Hovorka et al. (2004).
#   States:
#     S1, S2    : subcutaneous insulin (mU/kg)
#     I         : plasma insulin (mU/L)
#     X1, X2, X3: insulin action on glucose transport, disposal, endogenous
#                 production (per min)
#     Q1, Q2    : glucose in accessible, non-accessible compartments (mmol/kg)
#     C1, C2    : gut carbohydrate absorption (mmol/kg)
#   Exogenous inputs:
#     uI(t)     : SC insulin infusion rate (mU/kg/min)
#     ucarbs(t) : carb ingestion rate (mmol/kg/min)
#   Observable:
#     CGM = Q1 / VG * 18  (mg/dL, since VG is L/kg and 18 is mg/dL per mmol/L)

HOVORKA_POP = {
    # Population-level (literature) Hovorka parameters. The "Sf" parameters
    # are the most patient-specific and are what we fit.
    'tmax_I': 55.0,       # insulin SC-to-plasma time constant (min)
    'VI': 0.12,           # insulin distribution volume (L/kg)
    'ke': 0.138,          # insulin elimination rate (1/min)
    'ka1': 0.006,         # insulin action rate constants (1/min)
    'ka2': 0.06,
    'ka3': 0.03,
    'Sf1': 51.2e-4,       # <-- fit (insulin sensitivity for transport)
    'Sf2': 8.2e-4,        # <-- fit (insulin sensitivity for disposal)
    'Sf3': 520e-4,        # <-- fit (insulin sensitivity for EGP suppression)
    'k12': 0.066,         # transfer rate Q2->Q1 (1/min)
    'EGP0': 0.0161,       # endogenous glucose production at zero insulin
                          # (mmol/kg/min)
    'F01': 0.0097,        # non-insulin-dependent glucose uptake (mmol/kg/min)
    'VG': 0.16,           # glucose distribution volume (L/kg)
    'tmax_G': 40.0,       # gut absorption time constant (min)
    'Ag': 0.8,            # carb bioavailability
}


def hovorka_rhs(state, t, insulin_fn, carb_fn, p):
    """Right-hand side for Hovorka ODE. Returns list of 10 derivatives."""
    S1, S2, I, X1, X2, X3, Q1, Q2, C1, C2 = state
    uI = float(insulin_fn(t))      # mU/kg/min
    uC = float(carb_fn(t))         # mmol/kg/min

    # Insulin SC absorption
    dS1 = uI - S1 / p['tmax_I']
    dS2 = S1 / p['tmax_I'] - S2 / p['tmax_I']

    # Plasma insulin
    dI = S2 / (p['tmax_I'] * p['VI']) - p['ke'] * I

    # Insulin actions (three parallel 1st-order compartments)
    dX1 = -p['ka1'] * X1 + p['Sf1'] * p['ka1'] * I
    dX2 = -p['ka2'] * X2 + p['Sf2'] * p['ka2'] * I
    dX3 = -p['ka3'] * X3 + p['Sf3'] * p['ka3'] * I

    # Gut carb absorption (two-compartment)
    dC1 = uC - C1 / p['tmax_G']
    dC2 = C1 / p['tmax_G'] - C2 / p['tmax_G']
    UG = p['Ag'] * C2 / p['tmax_G']  # mmol/kg/min appearing in plasma

    # Glucose compartments
    dQ1 = (-X1 * Q1 - p['F01'] + p['k12'] * Q2
           + UG + p['EGP0'] * max(1.0 - X3, 0.0))
    dQ2 = X1 * Q1 - p['k12'] * Q2 - X2 * Q2

    return [dS1, dS2, dI, dX1, dX2, dX3, dQ1, dQ2, dC1, dC2]


def make_input_fn(times_min, values):
    """Piecewise-constant interpolator over t in minutes. Vectorizable via
    scalar calls in odeint."""
    times_arr = np.asarray(times_min)
    vals_arr = np.asarray(values)

    def fn(t):
        idx = np.searchsorted(times_arr, t, side='right') - 1
        idx = np.clip(idx, 0, len(vals_arr) - 1)
        return vals_arr[idx]
    return fn


def cgm_from_state(sol, VG):
    """Convert Q1 (mmol/kg) to CGM (mg/dL).
    CGM [mg/dL] = Q1 [mmol/kg] / VG [L/kg] * 18 [mg/dL per mmol/L].
    """
    Q1 = sol[:, 6]
    return Q1 / VG * 18.0


def simulate_hovorka(params, t_eval_min, insulin_fn, carb_fn, x0):
    """Integrate Hovorka ODE and return CGM (mg/dL) on t_eval_min grid.

    Uses EVENT-SEGMENTED integration: between each pair of consecutive
    t_eval points, insulin and carb inputs are treated as constants equal
    to the value at the LEFT endpoint. This correctly captures piecewise-
    constant inputs (meal spikes, bolus deltas) that adaptive ODE solvers
    would otherwise step over.
    """
    t_eval_min = np.asarray(t_eval_min)
    n = len(t_eval_min)
    sol = np.zeros((n, 10))
    sol[0] = x0
    x = np.asarray(x0, dtype=float)

    for i in range(n - 1):
        t0 = t_eval_min[i]
        t1 = t_eval_min[i + 1]
        # Constant inputs across [t0, t1) equal to values at t0.
        u_ins = float(insulin_fn(t0))
        u_cho = float(carb_fn(t0))
        const_ins = lambda t, v=u_ins: v
        const_cho = lambda t, v=u_cho: v

        try:
            seg = odeint(
                hovorka_rhs, x, [t0, t1],
                args=(const_ins, const_cho, params),
                rtol=1e-6, atol=1e-8, mxstep=5000,
            )
            x = seg[-1]
        except Exception:
            # Solver failure: freeze state (conservative) and keep going
            pass

        if not np.all(np.isfinite(x)):
            # Propagate last finite state
            x = sol[i].copy()

        # Clip glucose compartments to non-negative physical range
        x[6] = max(x[6], 1e-4)   # Q1 >= 0
        x[7] = max(x[7], 1e-4)   # Q2 >= 0
        sol[i + 1] = x

    return cgm_from_state(sol, params['VG']), sol


# ----------------------------------------------------------------------------
# 3. Input preparation and initial conditions
# ----------------------------------------------------------------------------

def build_inputs_from_trace(df, BW):
    """Build insulin_fn and carb_fn from a simglucose trace.

    simglucose columns:
      'insulin' : U/min (averaged over the 3-min step)
      'CHO'     : g/min (averaged over the 3-min step)
    Hovorka expects:
      uI    : mU/kg/min  -->  insulin * 1000 / BW
      uCHO  : mmol/kg/min -->  CHO / BW / 180.156  (180.156 g/mol glucose)
    """
    times_min = df['t_min'].values
    insulin_U_per_min = df['insulin'].values
    cho_g_per_min = df['CHO'].values

    # Unit conversion notes:
    #   Insulin: U/min -> mU/kg/min: multiply by 1000, divide by BW.
    #   Carbs:   g/min -> mmol/kg/min: divide by BW (kg) and by molar mass
    #            of glucose in g/mmol (= 0.180156 g/mmol, NOT 180.156 g/mol).
    uI = insulin_U_per_min * 1000.0 / BW              # mU/kg/min
    uC = cho_g_per_min / BW / 0.180156                # mmol/kg/min

    return make_input_fn(times_min, uI), make_input_fn(times_min, uC)


def steady_state_initial(params, basal_uI, Gb_mg_dL):
    """Build initial state by numerically equilibrating the ODE under
    constant basal insulin and no carbs.

    An analytical steady state is *not* achievable for Hovorka in general:
    the population EGP0/F01 and patient-specific Sf1-Sf3 rarely balance
    exactly at the observed fasting glucose. If we return an analytical
    guess that doesn't satisfy the ODE, the twin drifts away from Gb
    within the first hour of simulation even with no meals --- poisoning
    the entire trace.

    The fix: integrate forward 1000 min with constant basal insulin until
    the system settles into its natural fasting equilibrium. The final
    Q1 may differ from Gb_mg_dL (that mismatch reflects the genuine
    inconsistency between population parameters and the patient), but
    the ODE trajectory from this state will be drift-free.

    basal_uI: constant SC insulin infusion (mU/kg/min)
    Gb_mg_dL: observed fasting glucose (mg/dL) --- used only for seeding
    """
    def basal_fn(t):
        return basal_uI

    def zero_fn(t):
        return 0.0

    # Seed guess
    I_ss = basal_uI / (params['VI'] * params['ke'])
    x_guess = [
        basal_uI * params['tmax_I'],
        basal_uI * params['tmax_I'],
        I_ss,
        params['Sf1'] * I_ss,
        params['Sf2'] * I_ss,
        params['Sf3'] * I_ss,
        Gb_mg_dL * params['VG'] / 18.0,
        Gb_mg_dL * params['VG'] / 18.0 * 0.4,
        0.0,
        0.0,
    ]

    # Equilibrate. 1000 min is more than enough; insulin subsystems settle
    # in ~5*tmax_I ~= 275 min and glucose in ~1/k12 ~= 15 min once insulin
    # is steady.
    t_warmup = np.linspace(0, 1000, 1001)
    try:
        sol = odeint(
            hovorka_rhs, x_guess, t_warmup,
            args=(basal_fn, zero_fn, params),
            rtol=1e-8, atol=1e-10, mxstep=10000,
            full_output=False,
        )
        return list(sol[-1, :])
    except Exception:
        # If integration fails (e.g. during an early bad fit trial),
        # fall back to the analytical guess.
        return x_guess


# ----------------------------------------------------------------------------
# 4. Fitting via lmfit (Levenberg-Marquardt)
# ----------------------------------------------------------------------------

def fit_twin(df_fit, BW, verbose=True):
    """Fit patient-specific parameters (Sf1, Sf2, Sf3, EGP0) to CGM in the
    fit window.

    Note on fitting EGP0: the population value for endogenous glucose
    production (0.0161 mmol/kg/min) does not always balance insulin-driven
    uptake at the patient's observed basal insulin rate. If we fit only
    Sf1-Sf3, the system's natural fasting equilibrium may be physiologically
    implausible (e.g. negative glucose), which poisons the initial
    condition. Fitting EGP0 alongside the Sf parameters lets the model
    find a fasting glucose balance consistent with the observed basal.
    """
    insulin_fn, carb_fn = build_inputs_from_trace(df_fit, BW)

    # Initial condition from the first observed CGM (fasting-ish)
    G0 = df_fit['CGM'].iloc[0]
    # Basal insulin estimate: average insulin over first 30 minutes
    # (before any meals/boluses)
    early_mask = df_fit['t_min'] <= 30
    basal_U_per_min = float(df_fit.loc[early_mask, 'insulin'].mean())
    basal_uI = basal_U_per_min * 1000.0 / BW

    t_obs = df_fit['t_min'].values
    cgm_obs = df_fit['CGM'].values

    def residual(lm_params):
        p = dict(HOVORKA_POP)
        p['Sf1'] = lm_params['Sf1'].value
        p['Sf2'] = lm_params['Sf2'].value
        p['Sf3'] = lm_params['Sf3'].value
        p['EGP0'] = lm_params['EGP0'].value
        try:
            # Recompute x0 with current parameters so warmup uses the
            # right equilibrium.
            x0_local = steady_state_initial(p, basal_uI, G0)
            cgm_sim, _ = simulate_hovorka(
                p, t_obs, insulin_fn, carb_fn, x0_local
            )
        except Exception:
            return np.ones_like(cgm_obs) * 1e6
        res = cgm_sim - cgm_obs
        if not np.all(np.isfinite(res)):
            return np.ones_like(cgm_obs) * 1e6
        return res

    fit_params = lmfit.Parameters()
    fit_params.add('Sf1', value=HOVORKA_POP['Sf1'],
                   min=1e-5, max=1e-1)
    fit_params.add('Sf2', value=HOVORKA_POP['Sf2'],
                   min=1e-5, max=1e-1)
    fit_params.add('Sf3', value=HOVORKA_POP['Sf3'],
                   min=1e-3, max=1.0)
    fit_params.add('EGP0', value=HOVORKA_POP['EGP0'],
                   min=5e-3, max=5e-2)

    result = lmfit.minimize(residual, fit_params, method='leastsq',
                            max_nfev=300)

    if verbose:
        print("\n=== LMFIT RESULT ===")
        print(lmfit.fit_report(result))

    fitted = dict(HOVORKA_POP)
    for k in ['Sf1', 'Sf2', 'Sf3', 'EGP0']:
        fitted[k] = result.params[k].value

    # Final x0 with fitted parameters
    x0 = steady_state_initial(fitted, basal_uI, G0)
    return fitted, result, x0


# ----------------------------------------------------------------------------
# 5. Evaluation: reconstruction and counterfactual RMSE
# ----------------------------------------------------------------------------

def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_twin(fitted_params, df_full, df_eval, BW, x0_fit_start):
    """Run the fitted twin on the full trace (so the eval segment starts from
    the propagated state at t=fit_end, not a re-initialised steady state).
    Returns (cgm_twin_full, rmse_reconstruction).
    """
    insulin_fn, carb_fn = build_inputs_from_trace(df_full, BW)
    t_full = df_full['t_min'].values
    cgm_twin_full, _ = simulate_hovorka(
        fitted_params, t_full, insulin_fn, carb_fn, x0_fit_start
    )

    # Reconstruction RMSE only on the eval window
    mask = df_full['t_min'].isin(df_eval['t_min'])
    rmse_rec = rmse(cgm_twin_full[mask], df_full.loc[mask, 'CGM'].values)
    return cgm_twin_full, rmse_rec


def counterfactual_evaluate(fitted_params, df_full_cf, df_eval_cf,
                            BW, x0_fit_start):
    """Simulate the twin under counterfactual inputs (e.g. scaled boluses).
    Compare against the simglucose counterfactual ground-truth on eval window.
    """
    insulin_fn_cf, carb_fn_cf = build_inputs_from_trace(df_full_cf, BW)
    t_full = df_full_cf['t_min'].values
    cgm_twin_cf, _ = simulate_hovorka(
        fitted_params, t_full, insulin_fn_cf, carb_fn_cf, x0_fit_start
    )
    mask = df_full_cf['t_min'].isin(df_eval_cf['t_min'])
    rmse_cf = rmse(cgm_twin_cf[mask], df_full_cf.loc[mask, 'CGM'].values)
    return cgm_twin_cf, rmse_cf


# ----------------------------------------------------------------------------
# 6. Plotting
# ----------------------------------------------------------------------------

def plot_results(df_base, cgm_twin_base, df_cf, cgm_twin_cf,
                 fit_end_min, eval_end_min, rmse_rec, rmse_cf,
                 out_path='day1_results.png'):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: reconstruction
    ax = axes[0]
    ax.plot(df_base['t_min'] / 60, df_base['CGM'], 'k-',
            label='Ground truth (simglucose)', linewidth=1.5)
    ax.plot(df_base['t_min'] / 60, cgm_twin_base, 'r--',
            label='Twin (Hovorka, fitted)', linewidth=1.2)
    ax.axvline(fit_end_min / 60, color='gray', linestyle=':',
               label='Fit / Eval boundary')
    ax.set_ylabel('CGM (mg/dL)')
    ax.set_title(f'RECONSTRUCTION (original inputs)  '
                 f'--  Eval-window RMSE = {rmse_rec:.2f} mg/dL')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.axhspan(70, 180, color='green', alpha=0.05)

    # Panel 2: counterfactual
    ax = axes[1]
    ax.plot(df_cf['t_min'] / 60, df_cf['CGM'], 'k-',
            label='Ground truth (simglucose +20% bolus)', linewidth=1.5)
    ax.plot(df_cf['t_min'] / 60, cgm_twin_cf, 'b--',
            label='Twin (Hovorka, under +20% bolus)', linewidth=1.2)
    ax.axvline(fit_end_min / 60, color='gray', linestyle=':',
               label='Fit / Eval boundary')
    ax.set_ylabel('CGM (mg/dL)')
    ax.set_xlabel('Time (hours)')
    ax.set_title(f'COUNTERFACTUAL (+20% bolus)  '
                 f'--  Eval-window RMSE = {rmse_cf:.2f} mg/dL')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.axhspan(70, 180, color='green', alpha=0.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f'\nPlot saved to {out_path}')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)
    os.chdir('outputs')

    print('=' * 72)
    print('STEP 1: Generating ground-truth patient trace (simglucose, 48h)...')
    print('=' * 72)
    df_base = generate_patient_trace(
        patient_name='adolescent#001',
        duration_hours=48,
        bolus_scale=1.0,
        out_path='patient_001_48h.csv',
    )
    BW = df_base.attrs['BW']
    print(f'  Patient BW = {BW:.1f} kg')
    print(f'  Trace length: {len(df_base)} samples, dt = 3 min')
    print(f'  CGM range: {df_base["CGM"].min():.1f} - '
          f'{df_base["CGM"].max():.1f} mg/dL')

    print('\n' + '=' * 72)
    print('STEP 2: Generating counterfactual ground-truth (+20% boluses)...')
    print('=' * 72)
    df_cf = generate_patient_trace(
        patient_name='adolescent#001',
        duration_hours=48,
        bolus_scale=1.20,
        out_path='patient_001_48h_cf.csv',
    )
    print(f'  CGM range (CF): {df_cf["CGM"].min():.1f} - '
          f'{df_cf["CGM"].max():.1f} mg/dL')

    # Split: fit on [0, 24h), evaluate on [24h, 48h)
    FIT_END_MIN = 24 * 60
    EVAL_END_MIN = 48 * 60
    df_fit = df_base[df_base['t_min'] < FIT_END_MIN].copy()
    df_eval = df_base[(df_base['t_min'] >= FIT_END_MIN) &
                      (df_base['t_min'] < EVAL_END_MIN)].copy()
    df_eval_cf = df_cf[(df_cf['t_min'] >= FIT_END_MIN) &
                       (df_cf['t_min'] < EVAL_END_MIN)].copy()

    print(f'\n  Fit window : {len(df_fit)} samples (0-24h)')
    print(f'  Eval window: {len(df_eval)} samples (24-48h)')

    print('\n' + '=' * 72)
    print('STEP 3: Fitting Hovorka Sf1, Sf2, Sf3 via lmfit...')
    print('=' * 72)
    fitted_params, fit_result, x0 = fit_twin(df_fit, BW, verbose=True)
    print(f'\n  Fitted Sf1 = {fitted_params["Sf1"]:.4e}')
    print(f'  Fitted Sf2 = {fitted_params["Sf2"]:.4e}')
    print(f'  Fitted Sf3 = {fitted_params["Sf3"]:.4e}')

    # Fit-window RMSE for sanity check
    insulin_fn_fit, carb_fn_fit = build_inputs_from_trace(df_fit, BW)
    cgm_fit_sim, _ = simulate_hovorka(
        fitted_params, df_fit['t_min'].values,
        insulin_fn_fit, carb_fn_fit, x0
    )
    rmse_fit = rmse(cgm_fit_sim, df_fit['CGM'].values)
    print(f'\n  Fit-window RMSE (sanity check): {rmse_fit:.2f} mg/dL')

    print('\n' + '=' * 72)
    print('STEP 4: Reconstruction RMSE on held-out 24-48h window')
    print('        (twin with fitted params, real inputs vs real CGM)')
    print('=' * 72)
    cgm_twin_base, rmse_rec = evaluate_twin(
        fitted_params, df_base, df_eval, BW, x0
    )
    print(f'\n  RECONSTRUCTION RMSE = {rmse_rec:.2f} mg/dL')

    print('\n' + '=' * 72)
    print('STEP 5: Counterfactual RMSE on held-out 24-48h window')
    print('        (twin with +20% boluses vs simglucose with +20% boluses)')
    print('=' * 72)
    cgm_twin_cf, rmse_cf = counterfactual_evaluate(
        fitted_params, df_cf, df_eval_cf, BW, x0
    )
    print(f'\n  COUNTERFACTUAL RMSE = {rmse_cf:.2f} mg/dL')
    print(f'  Ratio (CF / Reconstruction) = '
          f'{rmse_cf / max(rmse_rec, 1e-6):.2f}x')
    print('  (A ratio near 1.0 indicates the twin degrades gracefully under')
    print('   perturbation. Large ratios mean the twin is unreliable for')
    print('   controller evaluation.)')

    print('\n' + '=' * 72)
    print('STEP 6: Saving results...')
    print('=' * 72)
    out_df = pd.DataFrame({
        't_min': df_base['t_min'].values,
        'cgm_true': df_base['CGM'].values,
        'cgm_twin': cgm_twin_base,
        'cgm_true_cf': df_cf['CGM'].values,
        'cgm_twin_cf': cgm_twin_cf,
    })
    out_df.to_csv('twin_results.csv', index=False)
    print('  Saved: twin_results.csv')

    plot_results(df_base, cgm_twin_base, df_cf, cgm_twin_cf,
                 FIT_END_MIN, EVAL_END_MIN, rmse_rec, rmse_cf,
                 out_path='day1_results.png')

    print('\n' + '=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'  Fit-window RMSE       : {rmse_fit:6.2f} mg/dL')
    print(f'  Reconstruction RMSE   : {rmse_rec:6.2f} mg/dL')
    print(f'  Counterfactual RMSE   : {rmse_cf:6.2f} mg/dL')
    print(f'  CF/Rec ratio          : {rmse_cf / max(rmse_rec, 1e-6):6.2f}x')
    print()
    print('  Interpretation:')
    print('  - Fit-window RMSE is artificially low (fitted here).')
    print('  - Reconstruction RMSE is the honest fidelity on unseen time.')
    print('  - Counterfactual RMSE tells you if the twin is valid for')
    print('    controller evaluation.')
    print()
    print('  Outputs in ./outputs/:')
    print('    patient_001_48h.csv     -- ground-truth trace')
    print('    patient_001_48h_cf.csv  -- counterfactual ground-truth')
    print('    twin_results.csv        -- per-step twin predictions')
    print('    day1_results.png        -- diagnostic plot')


if __name__ == '__main__':
    main()

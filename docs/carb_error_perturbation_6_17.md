# Carb-Counting-Error Perturbation — Implementation Details

**Status:** implemented · **Default:** ON for newly generated cohorts
**Scope:** `patients.py`, `generate_patients.py`, `exp_common.py` (+ new `dt2_fairness.py`)
**One-line summary:** each synthetic patient's *programmed* carb ratio is deterministically mis-set so that the unmodulated bolus (`x1.00`) is no longer optimal, turning a trivial "do nothing" benchmark into a real, individualized decision problem — without touching physiology or the population prior.

---

## 1. The problem this fixes (Issue C)

Each synthetic patient is a subset-average of well-tuned UVA/Padova adults. Crucially, `average_patient` averages **both** the physiology vector **and** the Quest therapy constants (`CR/CF/TDI`) over the *same* parents. Because each parent's carb ratio was tuned to its own physiology, the averaged `CR_true` is matched to the averaged physiology, so the baseline bolus `x1.00` is already near-optimal.

Consequences for the DT² benchmark:

- The decision/ranking task degenerates to "is baseline fine?" — answered correctly by recommending no change for everyone.
- Reward gaps near the optimum are sub-noise (~0.03 risk/sample), so Spearman/regret/top-k are noise-limited.
- The fidelity-vs-decision dissociation — the headline DT² claim — cannot appear, because there is no non-trivial decision to get right or wrong.

## 2. The fix, and why this framing

We deliberately break the match between each patient's physiology and the dose `x1.00` represents, by mis-setting the **programmed** carb ratio:

```
CR_prog = CR_true × m,    carb_bolus = meal / CR_prog = (meal / CR_true) / m
```

The carb→insulin mapping is scaled by `1/m`. This is exactly a persistent **carb-counting error**: programming `CR_prog = CR_true × m` is identical, for the carb term, to announcing `CHO_announced = CHO_true / m` each meal while keeping `CR_true` (since `meal/CR` is what enters the bolus). So `m` is the reciprocal of the patient's habitual carb-announcement ratio:

- `m > 1` → carbs systematically **under-announced** → under-dosed → patient runs high → optimal correction `f* > 1`.
- `m < 1` → carbs **over-announced** → over-dosed → patient runs low → optimal correction `f* < 1`.

Carb-counting error is one of the most-studied error sources in T1D management and is precisely ReplayBG's Scenario 2 (meal-CHO modulation). Framing the perturbation this way — rather than as an ad-hoc CR trick — puts it on documented, citable footing (see §9) while keeping the minimal one-line implementation.

### Why `f* ≈ m` (and the one caveat)

The candidate bolus factor `f` restores the matched dose when `f · (1/m) = 1`, i.e. `f* = m`. This places each patient's optimum on the candidate grid by construction.

**Caveat — it is approximate, not exact.** The full bolus is

```
B = meal/CR_prog + 1{g>150}·(g − target)/CF,   then × f
```

The correction term `(g−target)/CF` does **not** depend on CR, yet `f` scales the whole bolus. So `f = m` restores the carb term exactly but multiplies the correction term by `m` (it should stay ×1). The true optimum therefore deviates slightly from `m`, and **asymmetrically**: under-dosed patients (`m>1`) run high, trigger the `>150` correction more often, and land at `f*` a little **below** `m`; over-dosed patients rarely trigger it, so `f* ≈ m`. **Implication:** always locate each optimum from the *measured* grid rewards (`subject_ground_truth` already computes them), never assume it equals `m`. The grid is dense enough on both sides that optima stay interior even after this shift (§5).

## 3. Why the implementation is a one-liner here

Two facts about the existing pipeline make this clean:

1. **Every controller doses from `quest["CR"]`.** Both the identification baseline (`Subject.baseline_controller`) and the entire candidate grid (`Subject.therapy_controllers`) read `quest["CR"]`. Writing `CR_prog` into `quest["CR"]` automatically makes the identification window *and* the ground-truth grid consistent — the baseline is mis-dosed, and the grid is the corrective sweep around it. The mis-dosing leaves its signature (recurrent highs/lows) in the identification CGM, exactly as required.
2. **The twin never doses.** `Twin.replay_run` pulls the *actually delivered* insulin trace (`insulin_mU_kg_min`, via `RunResult.replaybg_inputs`) and replays it through inferred physiology. It does not recompute boluses from CR, so it needs no knowledge of `CR_prog`. The "twin's dosing math must match ground truth" concern from earlier drafts is therefore **moot** — consistency is automatic.

Net effect: **no changes** to the controllers, the twin, identification (`identify_*`), or evaluation (`evaluate.py`). Only patient generation and persistence change.

## 4. What changed, file by file

### `patients.py`
- **`carb_error_multiplier(name, choices=DEFAULT_CARB_ERROR_CHOICES)`** — deterministic per-patient draw of `m` from a stable SHA-256 hash of the name. (Python's built-in `hash()` is salted per process and is *not* reproducible across runs — SHA-256 is.)
- **`DEFAULT_CARB_ERROR_CHOICES = (0.75, 0.85, 1.25, 1.50)`** — balanced over-/under-dose set, away from `m≈1`, all interior to the grid (§5).
- **`apply_carb_error(subject, m=None)`** — returns a copy of the subject with `quest["CR"] = cr_true × m` and audit fields set. Perturbs relative to `cr_true`, so re-applying with the same `m` is a no-op rather than compounding.
- **`Subject`** gains `cr_true` and `dose_mult` fields (+ a `cr_prog` property). `quest["CR"]` always holds the *programmed* ratio used for dosing; `cr_true` keeps the matched ratio for analysis. Unperturbed subjects have `dose_mult == 1.0`, `cr_true == quest["CR"]`.
- **`write_patients_csv` / `load_subjects_csv`** persist and read two new columns, `CR_true` and `dose_mult`. **Back-compatible:** a CSV without them loads as an unperturbed cohort.

### `generate_patients.py`
- New flags `--carb-error` (default **ON**) / `--no-carb-error`.
- The generation loop calls `apply_carb_error(s)` per patient (m drawn from the name).
- Prints the over/under split and the per-`m` patient counts.

### `exp_common.py`
- Grid-rationale comment updated: `PROD_BOLUS = (0.7, 0.85, 1.0, 1.2, 1.3, 1.5, 1.6, 2.0)` is **unchanged** and documented to bracket the new optima (the multiplier set was chosen to fit this grid — see §5).

### `dt2_fairness.py` (new, additive)
- Cross-patient regret normalization (see §6). Nothing imports it yet; wire it in at aggregation time.

## 5. Design decisions (and the constraints they satisfy)

| Decision | Value | Why |
|---|---|---|
| Multiplier set | `{0.75, 0.85, 1.25, 1.50}` | Away from `m≈1` (avoids collapse to the trivial case); balanced over/under; magnitudes substantial (25–50% mis-dose) so reward gaps clear the noise floor. |
| Bracketing | every `f*` interior with grid neighbours on both sides | `0.75∈(0.70,0.85)`, `0.85` on a node with `0.70`/`1.00`, `1.25∈(1.20,1.30)`, `1.50` on a node with `1.30`/`1.60`. **This is why the grid did not need to change**, and it resolves the earlier `[0.6,1.6]`-vs-`[0.7,2.0]` edge bug (an optimum at `0.6` would have sat below the grid floor). Keep `m` and the grid in lock-step on any future change. |
| Determinism | SHA-256 of the patient name | Fixed, auditable cohort; reproducible across machines/runs. |
| Balance | ~50/50 over/under | Verified empirically: on the 1013-patient adult cohort, **512 over-dosed / 501 under-dosed**. |
| Prior validity | physiology untouched | Mechanism mis-sets a *dosing constant*, not a model parameter. The population prior is over physiology (`fit_replaybg` recovers physiology from `(insulin, cho, cgm)` regardless of the insulin profile), so it stays valid. |
| Crash safety | `m ∈ [0.75, 1.5]` | Keeps the mis-dose within a smooth, unimodal reward band — far enough from the severe-hypo regime that no baseline run terminates early into the worst-case padding in `run_on_patient`. **Verify** (§7). |

## 6. Cross-patient fairness

Two distinct fairness axes:

- **Within a patient (method-vs-method): unaffected.** Every twin method identifies from the same baseline run and is scored on the same ground-truth grid, so MCMC/SBI/MLP face an identical task regardless of `m`. This is the comparison the study is about, and the fix does not touch it.
- **Across patients: difficulty varies with `m`, and one aggregation can be biased.** Spearman and top-k are scale-free and aggregate fairly as-is. Raw `decision_regret`, however, is in absolute risk units, and a more mis-dosed patient spans a larger reward range — so averaging raw regret lets those patients dominate the cohort mean.

`dt2_fairness.py` provides:
- **`normalized_regret`** — regret as a fraction of the patient's own reward span, in `[0,1]`.
- **`baseline_gap_regret`** — regret as a fraction of the baseline-to-optimum gap ("of the improvement available over the mis-dosed baseline, how much did the twin forfeit?"); often the most interpretable cohort metric here.
- **`cohort_summary`** — scale-fair cohort means.

Integration point: `compute_results.score_subject` already has `true_rewards` in scope; compute `normalized_regret(true_rewards, pred_rewards)` there and aggregate that (not raw regret) in `run_suite`.

## 7. How to run, and the validation checklist

Regenerate the perturbed cohort:

```bash
python -m experiments.generate_patients                 # perturbed (default)
python -m experiments.generate_patients --no-carb-error # baseline-optimal (old behaviour)
```

The rest of the pipeline is unchanged (`run_phase*`, `compute_results`, `run_suite`) — it picks up `CR_prog` automatically via `quest["CR"]`.

Before trusting results, confirm:

1. **No baseline truncation.** For each subject, `subject_identification_run(...).meta["truncated"]` is `False` (no severe-hypo crash on the mis-set baseline). If any truncate, soften the over-dose end of `DEFAULT_CARB_ERROR_CHOICES` (e.g. drop `0.75`).
2. **Optima are interior, empirically.** For each subject, `argmax(true_rewards)` over the grid is **not** an endpoint (`bolus_x0.70` or `bolus_x2.00`). This checks the `f* ≈ m` approximation against the *measured* rewards, accounting for the CF-term shift in §2.
3. **`x1.00` is no longer optimal.** `bolus_x1.00` should rarely be the per-patient argmax — confirming the decision problem is now real.
4. **Reward gaps clear the noise floor.** Per-patient `reward_span` (best−worst) should be well above the ~0.03 risk/sample baseline-optimal scale.

## 8. What this does *not* change

- No controller / twin / identification / evaluation code changes.
- The candidate grid (`PROD_BOLUS`) is unchanged.
- Old `patients.csv` files still load (as unperturbed cohorts).
- The physiology prior and `derive_population` flow are unchanged.

## 9. References / citing

- **Perturbation precedent (in-project):** Cappon et al., *ReplayBG: A Digital Twin-Based Methodology to Identify a Personalized Model of Glucose-Insulin Dynamics in Type 1 Diabetes*, IEEE TBME 70(11), 2023 — Scenario 1 (bolus modulation) and especially Scenario 2 (meal-CHO / carb-counting modulation). The bolus formula `B = CHO/CR + (GC−GT)/CF` is their Eq. (8).
- **Cohort / simulator:** Dalla Man et al., *The UVA/PADOVA Type 1 Diabetes Simulator: New Features*, J Diabetes Sci Technol, 2014 — CR/CF definitions and virtual-subject generation.
- **Carb-counting error as a clinical error source:** insert your preferred clinical reference on carbohydrate-counting accuracy (e.g. Brazeau et al., *Carbohydrate counting accuracy and blood glucose variability in adults with type 1 diabetes*, Diabetes Res Clin Pract, 2013 — verify before citing). ReplayBG Scenario 2 is the modeling precedent.
- **Motivation (fidelity ≠ decision quality):** Lambert et al., *Objective Mismatch in Model-based Reinforcement Learning*, L4DC (PMLR vol. 120), 2020 — model prediction accuracy is not always correlated with downstream control performance; the general-ML name for the fidelity-vs-decision dissociation. Pair with the off-policy-evaluation-in-healthcare literature (e.g. Gottesman et al., *Nature Medicine*, 2019).

Suggested methods-section phrasing:

> Because each synthetic subject is a subset-average of well-tuned UVA/Padova adults, its carb ratio is matched to its physiology and the nominal bolus is near-optimal, collapsing the decision task. To recover a clinically meaningful, individualized decision we impose a persistent carb-counting error per subject: the programmed carb ratio is mis-set by a deterministic factor `m` (`CR_prog = CR_true·m`), equivalent to habitually mis-announcing carbohydrate by `1/m`, following the meal-modulation scenario of Cappon et al. (2023). The optimal corrective bolus multiplier is then `f* ≈ m`, set to lie strictly inside the candidate grid so each subject's optimum is interior and identifiable.

## 10. Open items to verify empirically

- Run the §7 checklist on the full cohort and record any truncations / edge optima.
- Confirm the CF-term shift (§2) does not move any under-dosed optimum onto a grid node shared with a neighbour (creating ties); if it does, nudge that `m` value.
- Decide cohort-level reporting: `normalized_regret` vs `baseline_gap_regret` (recommend reporting both alongside scale-free Spearman/top-k).

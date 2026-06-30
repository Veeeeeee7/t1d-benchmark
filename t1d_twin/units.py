"""Unit conversions between simglucose I/O and ReplayBG model inputs.

simglucose works in whole-body clinical units:
    insulin action (basal/bolus) : U/min
    meal CHO (info['meal'])      : g/min
    glucose (CGM/BG)             : mg/dL

The ReplayBG ODE consumes body-weight-normalised rates:
    exogenous insulin I(t) : mU/kg/min
    carbohydrate   CHO(t)  : mg/kg/min
    glucose                : mg/dL   (unchanged)

Conversions (1 U = 1000 mU, 1 g = 1000 mg):
    I_mU_per_kg_min  = insulin_U_per_min * 1000 / BW
    CHO_mg_per_kg_min = meal_g_per_min   * 1000 / BW

These are the single most error-prone seam in the whole pipeline, so they are
isolated here and exercised directly by the step-1 test (a no-input run must
stay flat, a meal must produce a plausible excursion once fed to ReplayBG).
"""
from __future__ import annotations

MG_PER_G = 1000.0   # mg per g
MU_PER_U = 1000.0   # mU per U


def insulin_U_per_min_to_mU_per_kg_min(u_per_min: float, BW: float) -> float:
    """simglucose insulin rate (U/min) -> ReplayBG I(t) (mU/kg/min)."""
    return u_per_min * MU_PER_U / BW


def cho_g_per_min_to_mg_per_kg_min(g_per_min: float, BW: float) -> float:
    """simglucose meal rate (g/min) -> ReplayBG CHO(t) (mg/kg/min)."""
    return g_per_min * MG_PER_G / BW


# Inverse conversions (useful when driving simglucose from ReplayBG-side values).
def mU_per_kg_min_to_insulin_U_per_min(mu_per_kg_min: float, BW: float) -> float:
    return mu_per_kg_min * BW / MU_PER_U


def mg_per_kg_min_to_cho_g_per_min(mg_per_kg_min: float, BW: float) -> float:
    return mg_per_kg_min * BW / MG_PER_G

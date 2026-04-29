"""
Second-Life EV Battery Degradation Simulation
================================================
Model 2: Schmalstieg et al. (2014) — NMC 18650 Holistic Aging Model
Application: Charging Station BESS (CS-BESS)

Also includes Model 1 (Baghdadi) for direct comparison.

Based on: "Second Life Applications for Degraded EV Batteries"
          by Elis Sandberg, Linköping University, 2023.

Model 2 structural features:
  ▸ Calendar aging ∝ t^0.75 (SEI growth: between diffusion & reaction limited)
  ▸ Cyclic aging  ∝ √Q_throughput (fatigue-based)
  ▸ Additive combination: Q_loss = Q_loss_cal + Q_loss_cyc
  ▸ Open-circuit voltage & Arrhenius temperature dependence

Key differences from Model 1 (Baghdadi):
  ▸ Power-law vs. exponential time dependence
  ▸ OCV voltage vs. SOC-direct for calendar aging
  ▸ Additive vs. multiplicative calendar/cyclic coupling

Reference:
  Schmalstieg, J., Käbitz, S., Ecker, M. & Sauer, D.U. (2014).
  "A holistic aging model for Li(NiMnCo)O2 based 18650 lithium-ion
  batteries." Journal of Power Sources, 257, 325-334.
"""

import numpy as np
import matplotlib.pyplot as plt

# NumPy 2.0 compatibility
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# =============================================================================
# 1. PHYSICAL CONSTANTS
# =============================================================================

R_GAS = 8.314  # Ideal gas constant [J/(mol·K)]

# =============================================================================
# 2. MODEL 2 PARAMETERS (Schmalstieg et al., 2014)
# =============================================================================
#
# Calendar aging:
#   Q_loss_cal(%) = α_cap(V,T) · t^0.75     (capacity loss)
#   R_inc_cal(%)  = α_res(V,T) · t^0.75     (resistance increase)
#
#   α_cap = max(p1·V + p2, 0) · exp(-Ea/(R·T))
#
#   The linear voltage term captures higher aging at higher OCV (higher SOC).
#   The Arrhenius exponential captures temperature acceleration.
#   The t^0.75 exponent is between pure diffusion (0.5) and reaction (1.0)
#   limited SEI growth — Schmalstieg's key empirical finding.
#
# Cyclic aging:
#   Q_loss_cyc(%) = β_cap(V_avg,DOD) · √Q_throughput    (capacity loss)
#   R_inc_cyc(%)  = β_res(V_avg,DOD) · √Q_throughput    (resistance increase)
#
#   β_cap = c1·(V_avg - V_ref)^2 + c2 + c3·DOD
#
#   Quadratic voltage term + DOD dependence from cycle aging tests.
#
# Total aging is additive:  Loss_total = Loss_calendar + Loss_cyclic
#
# Parameters calibrated to reproduce thesis-comparable results for
# the CS-BESS use case while preserving the Schmalstieg model structure.
# =============================================================================

# --- Calendar: Capacity loss ---
S_CAL_CAP_P1 = 1200.0      # Voltage slope [dimensionless]
S_CAL_CAP_P2 = -3600.0     # Voltage offset [dimensionless]
S_CAL_CAP_EA = 24500.0     # Activation energy [J/mol]

# --- Calendar: Resistance increase ---
S_CAL_RES_P1 = 800.0
S_CAL_RES_P2 = -2400.0
S_CAL_RES_EA = 22000.0

# --- Cyclic: Capacity loss ---
S_CYC_CAP_C1 = 0.40        # Voltage quadratic coefficient [%/Ah^0.5/V^2]
S_CYC_CAP_VREF = 3.667     # Voltage reference [V]
S_CYC_CAP_C2 = 0.008       # Constant term [%/Ah^0.5]
S_CYC_CAP_C3 = 0.050       # DOD coefficient [%/Ah^0.5]

# --- Cyclic: Resistance increase ---
S_CYC_RES_D1 = 0.035
S_CYC_RES_VREF = 3.725
S_CYC_RES_D2 = 0.001
S_CYC_RES_D3 = 0.012

# --- Time-law exponent ---
S_TIME_EXP = 0.75          # Schmalstieg's empirical t^0.75


# =============================================================================
# 3. MODEL 1 PARAMETERS (Baghdadi — included for comparison)
# =============================================================================

M1_CAP_A     = 0.0272
M1_CAP_ALPHA = 27.68
M1_CAP_BETA  = -61145.0
M1_CAP_EA    = 2824.0
M1_RES_A     = 0.040
M1_RES_ALPHA = 23.0
M1_RES_BETA  = -51000.0
M1_RES_EA    = 2200.0
M1_CYC_EA    = 49500.0
M1_CYC_C     = -24.06


# =============================================================================
# 4. CELL SPECIFICATION (thesis Appendix D)
# =============================================================================

CELL_CAPACITY_AH = 4.8
CELL_VOLTAGE_NOM = 3.7
CELL_WH          = CELL_CAPACITY_AH * CELL_VOLTAGE_NOM  # 17.76 Wh
CELL_WEIGHT_G    = 60


# =============================================================================
# 5. SOC → OPEN-CIRCUIT VOLTAGE MAPPING  (NMC cell)
# =============================================================================

def soc_to_ocv(soc):
    """
    Map State-of-Charge to Open-Circuit Voltage for an NMC cell.

    Uses a 5th-order polynomial fitted to typical NMC OCV data.
    Range: SOC 0→1 maps to approximately 3.0 V → 4.2 V.

    Parameters
    ----------
    soc : float or ndarray — State of charge [0–1]

    Returns
    -------
    ocv : float or ndarray — Open-circuit voltage [V]
    """
    soc = np.clip(soc, 0.0, 1.0)
    return (3.0 + 1.20 * soc - 0.42 * soc**2 +
            0.85 * soc**3 - 0.90 * soc**4 + 0.47 * soc**5)


# =============================================================================
# 6. MODEL 2 AGING FUNCTIONS
# =============================================================================

def m2_cal_alpha_cap(V, T):
    """
    Calendar capacity-loss rate coefficient.

    α_cap = max(p1·V + p2, 0) · exp(-Ea / (R·T))

    Returns rate in [% / day^0.75].
    """
    return max(S_CAL_CAP_P1 * V + S_CAL_CAP_P2, 0.0) * \
           np.exp(-S_CAL_CAP_EA / (R_GAS * T))


def m2_cal_alpha_res(V, T):
    """Calendar resistance-increase rate coefficient [% / day^0.75]."""
    return max(S_CAL_RES_P1 * V + S_CAL_RES_P2, 0.0) * \
           np.exp(-S_CAL_RES_EA / (R_GAS * T))


def m2_cal_alpha_cap_avg(T, soc_min, soc_max, n_pts=200):
    """Average α_cap over an SOC window via numerical integration."""
    delta = soc_max - soc_min
    if delta < 0.01:
        return m2_cal_alpha_cap(soc_to_ocv(0.5 * (soc_min + soc_max)), T)
    soc = np.linspace(soc_min, soc_max, n_pts)
    alphas = np.array([m2_cal_alpha_cap(soc_to_ocv(s), T) for s in soc])
    return _trapz(alphas, soc) / delta


def m2_cal_alpha_res_avg(T, soc_min, soc_max, n_pts=200):
    """Average α_res over an SOC window via numerical integration."""
    delta = soc_max - soc_min
    if delta < 0.01:
        return m2_cal_alpha_res(soc_to_ocv(0.5 * (soc_min + soc_max)), T)
    soc = np.linspace(soc_min, soc_max, n_pts)
    alphas = np.array([m2_cal_alpha_res(soc_to_ocv(s), T) for s in soc])
    return _trapz(alphas, soc) / delta


def m2_cyc_beta_cap(V_avg, dod):
    """
    Cyclic capacity-loss rate coefficient.

    β_cap = c1·(V_avg − V_ref)² + c2 + c3·DOD

    Returns rate in [% / Ah^0.5].
    """
    return (S_CYC_CAP_C1 * (V_avg - S_CYC_CAP_VREF)**2 +
            S_CYC_CAP_C2 + S_CYC_CAP_C3 * dod)


def m2_cyc_beta_res(V_avg, dod):
    """Cyclic resistance-increase rate coefficient [% / Ah^0.5]."""
    return (S_CYC_RES_D1 * (V_avg - S_CYC_RES_VREF)**2 +
            S_CYC_RES_D2 + S_CYC_RES_D3 * dod)


# =============================================================================
# 7. MODEL 1 AGING FUNCTIONS (for comparison)
# =============================================================================

def m1_cal_rate(T, SOC, mode="capacity"):
    """Baghdadi calendar aging rate at a single (T, SOC) point."""
    if mode == "capacity":
        A, a, b, Ea = M1_CAP_A, M1_CAP_ALPHA, M1_CAP_BETA, M1_CAP_EA
    else:
        A, a, b, Ea = M1_RES_A, M1_RES_ALPHA, M1_RES_BETA, M1_RES_EA
    return A * np.exp(a * SOC + (b * SOC - Ea) / (R_GAS * T))


def m1_cal_rate_avg(T, soc_min, soc_max, mode="capacity", n_pts=300):
    """Baghdadi calendar rate averaged over SOC window."""
    d = soc_max - soc_min
    if d < 0.01:
        return m1_cal_rate(T, 0.5 * (soc_min + soc_max), mode)
    s = np.linspace(soc_min, soc_max, n_pts)
    r = np.array([m1_cal_rate(T, x, mode) for x in s])
    return _trapz(r, s) / d


def m1_cyc_factor(T, I_avg):
    """Baghdadi cyclic amplification factor (≥1)."""
    if abs(I_avg) < 1e-8:
        return 1.0
    return np.exp(np.exp(M1_CYC_EA / (R_GAS * T) + M1_CYC_C) * abs(I_avg))


# =============================================================================
# 8. WEEKLY POWER DEMAND (thesis Fig. 4.1 / Sec. 4.2)
# =============================================================================

def generate_weekly_demand():
    """168-hour highway EV charging station power demand [kW]."""
    demand = np.zeros(168)
    hours = np.arange(24)

    def bell(c, w, a):
        return a * np.exp(-0.5 * ((hours - c) / w) ** 2)

    profiles = [
        80 + bell(8, 2, 380) + bell(13, 1.5, 280) + bell(17.5, 2, 520),
        80 + bell(8, 2, 340) + bell(13, 1.5, 240) + bell(17, 2, 450),
        80 + bell(8, 2, 280) + bell(13, 1.5, 200) + bell(17, 2, 370),
        80 + bell(8, 2, 360) + bell(13, 1.5, 270) + bell(17.5, 2, 490),
        80 + bell(8, 2, 300) + bell(13, 2, 330) + bell(16, 2.5, 540),
        60 + bell(11, 3, 220) + bell(16, 2, 170),
        50 + bell(12, 3, 170) + bell(16, 2, 130),
    ]
    for i, p in enumerate(profiles):
        demand[i * 24 : (i + 1) * 24] = p
    return demand


# =============================================================================
# 9. BATTERY SIZING (thesis eqs. 4.1–4.3)
# =============================================================================

def size_cs_battery(demand, grid_limit=500, dod=0.90, degradation_margin=0.875):
    """Size the CS-BESS to cover worst-day excess over the grid limit."""
    excess = np.maximum(demand - grid_limit, 0)
    daily_excess = [excess[d * 24 : (d + 1) * 24].sum() for d in range(7)]
    E_diff = max(daily_excess)
    E_nom = E_diff / dod
    E_initial = E_nom / degradation_margin
    n_cells = int(np.ceil(E_initial * 1000 / CELL_WH))
    return E_initial, n_cells


# =============================================================================
# 10. SIMULATION — MODEL 2 (Schmalstieg)
# =============================================================================

def simulate_model2(years=5):
    """
    Day-by-day aging simulation for CS-BESS using Model 2 (Schmalstieg).

    The loop:
      1. Determine if the day is a cycling day (excess > 5 kWh).
      2. On cycling days: compute SOC range → OCV range, DOD, temperature.
      3. On idle days: battery rests at ~85 % SOC, ambient temperature.
      4. Calendar increment:  Δ = α · [t_n^0.75 − t_{n−1}^0.75]
      5. Cyclic increment:   Δ = β · [√Q_n − √Q_{n−1}]
      6. Total loss is additive:  cap + cyc.
    """
    demand = generate_weekly_demand()
    battery_kwh, n_cells = size_cs_battery(demand)

    print("=" * 58)
    print("  CS-BESS Aging Simulation — Model 2 (Schmalstieg)")
    print("=" * 58)
    print(f"  Battery capacity : {battery_kwh:.0f} kWh")
    print(f"  Number of cells  : {n_cells:,}")
    print(f"  Total weight     : {n_cells * CELL_WEIGHT_G / 1000:.0f} kg")
    print(f"  Initial SOH      : 80%")
    print(f"  EOL SOH          : 70%")
    print()

    # --- Operating conditions ---
    T_IDLE      = 273.15 + 20    # 20 °C idle
    T_CYCLING   = 273.15 + 28    # 28 °C during operation
    SOC_IDLE    = 0.85
    CHARGE_KW   = 20
    INITIAL_SOH = 0.80
    EOL_SOH     = 0.70

    n_days = int(years * 365)

    # State arrays
    soh        = np.zeros(n_days + 1)
    res_factor = np.zeros(n_days + 1)
    soh[0]        = INITIAL_SOH
    res_factor[0] = 1.0

    # Cumulative losses [%]
    cum_cal_cap = 0.0
    cum_cal_res = 0.0
    cum_cyc_cap = 0.0
    cum_cyc_res = 0.0

    # Cumulative physical quantities
    cum_time_days = 0.0          # Calendar time [days]
    cum_throughput_Ah = 0.0      # Charge throughput per cell [Ah]

    # --- Day-by-day loop ---
    for day in range(n_days):
        dow = day % 7
        day_demand = demand[dow * 24 : (dow + 1) * 24]
        excess_kw  = np.maximum(day_demand - 500, 0)
        excess_kwh = excess_kw.sum()
        is_cycling  = excess_kwh > 5

        if is_cycling:
            T = T_CYCLING

            # Current effective capacity
            frac_remaining = max(1 - (cum_cal_cap + cum_cyc_cap) / 100, 0.5)
            cap_now = INITIAL_SOH * battery_kwh * frac_remaining

            dod     = min(excess_kwh / cap_now, 0.95)
            soc_max = 0.95
            soc_min = max(soc_max - dod, 0.05)
            soc_avg = 0.5 * (soc_min + soc_max)
            V_avg   = soc_to_ocv(soc_avg)

            # Charge throughput: discharge + recharge  [Ah per cell]
            dQ = 2 * excess_kwh * 1000 / (n_cells * CELL_VOLTAGE_NOM)

            # Rates
            alpha_cap = m2_cal_alpha_cap_avg(T, soc_min, soc_max)
            alpha_res = m2_cal_alpha_res_avg(T, soc_min, soc_max)
            beta_cap  = m2_cyc_beta_cap(V_avg, dod)
            beta_res  = m2_cyc_beta_res(V_avg, dod)

        else:
            T = T_IDLE
            soc_min = SOC_IDLE - 0.02
            soc_max = SOC_IDLE + 0.02
            dQ = 0.0

            alpha_cap = m2_cal_alpha_cap_avg(T, soc_min, soc_max)
            alpha_res = m2_cal_alpha_res_avg(T, soc_min, soc_max)
            beta_cap  = 0.0
            beta_res  = 0.0

        # ── Calendar increment: α · [t_n^p − t_{n-1}^p] ──
        cum_time_days += 1.0
        dt_pow = cum_time_days**S_TIME_EXP - (cum_time_days - 1)**S_TIME_EXP

        cum_cal_cap += alpha_cap * dt_pow
        cum_cal_res += alpha_res * dt_pow

        # ── Cyclic increment: β · [√Q_n − √Q_{n-1}] ──
        if dQ > 0:
            Q_prev = cum_throughput_Ah
            cum_throughput_Ah += dQ
            dQ_sqrt = np.sqrt(cum_throughput_Ah) - np.sqrt(Q_prev)
            cum_cyc_cap += beta_cap * dQ_sqrt
            cum_cyc_res += beta_res * dQ_sqrt

        # ── Total (additive) ──
        total_cap = cum_cal_cap + cum_cyc_cap
        total_res = cum_cal_res + cum_cyc_res

        soh[day + 1]        = INITIAL_SOH * max(1 - total_cap / 100, 0)
        res_factor[day + 1] = 1.0 + total_res / 100

    # --- Post-processing ---
    time_years = np.arange(n_days + 1) / 365.0
    cap_loss   = (1 - soh) * 100
    res_inc    = (res_factor - 1) * 100

    eol_mask = soh <= EOL_SOH
    rul = np.argmax(eol_mask) / 365.0 if np.any(eol_mask) else years

    total_aging = cum_cal_cap + cum_cyc_cap
    cal_pct = cum_cal_cap / total_aging * 100 if total_aging > 0 else 0
    cyc_pct = cum_cyc_cap / total_aging * 100 if total_aging > 0 else 0

    eol_day = min(int(rul * 365), n_days)
    res_at_eol = res_inc[eol_day]

    print(f"  Remaining Useful Life : {rul:.1f} years")
    print(f"  Final SOH at {years}yr    : {soh[-1]:.1%}")
    print(f"  Resistance increase   : {res_at_eol:.1f}% at EOL")
    print(f"  Calendar aging share  : {cal_pct:.0f}%")
    print(f"  Cyclic aging share    : {cyc_pct:.0f}%")
    print(f"  Total Ah throughput   : {cum_throughput_Ah:.0f} Ah/cell")

    return {
        "time": time_years, "soh": soh, "cap_loss": cap_loss,
        "res_inc": res_inc, "res_factor": res_factor, "rul": rul,
        "cal_pct": cal_pct, "cyc_pct": cyc_pct,
        "n_cells": n_cells, "battery_kwh": battery_kwh,
        "demand": demand,
    }


# =============================================================================
# 11. SIMULATION — MODEL 1 (Baghdadi, for comparison)
# =============================================================================

def simulate_model1(years=5):
    """Baghdadi simulation — condensed version for comparison overlay."""
    demand = generate_weekly_demand()
    battery_kwh, n_cells = size_cs_battery(demand)

    print("=" * 58)
    print("  CS-BESS Aging Simulation — Model 1 (Baghdadi)")
    print("=" * 58)
    print(f"  Battery capacity : {battery_kwh:.0f} kWh")
    print(f"  Number of cells  : {n_cells:,}")
    print()

    T_IDLE      = 273.15 + 20
    T_CYCLING   = 273.15 + 28
    SOC_IDLE    = 0.85
    CHARGE_KW   = 20
    INITIAL_SOH = 0.80

    n_days = int(years * 365)
    dt = 1.0 / 365.0

    soh        = np.zeros(n_days + 1)
    res_factor = np.zeros(n_days + 1)
    soh[0]        = INITIAL_SOH
    res_factor[0] = 1.0

    cum_cap = cum_res = 0.0
    cal_cont = cyc_cont = 0.0

    for day in range(n_days):
        dow = day % 7
        dd  = demand[dow * 24 : (dow + 1) * 24]
        excess_kw  = np.maximum(dd - 500, 0)
        excess_kwh = excess_kw.sum()
        is_cycling  = excess_kwh > 5

        if is_cycling:
            T = T_CYCLING
            cap_now = INITIAL_SOH * battery_kwh * np.exp(-cum_cap)
            dod     = min(excess_kwh / cap_now, 0.95)
            soc_max = 0.95
            soc_min = max(soc_max - dod, 0.05)
            dh = max(np.sum(excess_kw > 0), 1)
            pw = (excess_kwh / dh) * 1000
            I_dis = pw / (n_cells * CELL_VOLTAGE_NOM)
            I_chg = (CHARGE_KW * 1000) / (n_cells * CELL_VOLTAGE_NOM)
            ch = 24 - dh
            I_avg = (I_dis * dh + I_chg * max(ch, 0)) / 24.0
            k_cyc = m1_cyc_factor(T, I_avg)
        else:
            T = T_IDLE
            soc_min = SOC_IDLE - 0.02
            soc_max = SOC_IDLE + 0.02
            k_cyc = 1.0

        k_cal_cap = m1_cal_rate_avg(T, soc_min, soc_max, "capacity")
        k_cal_res = m1_cal_rate_avg(T, soc_min, soc_max, "resistance")
        k_tot_cap = k_cyc * k_cal_cap
        k_tot_res = k_cyc * k_cal_res

        cal_cont += k_cal_cap * dt
        cyc_cont += (k_tot_cap - k_cal_cap) * dt
        cum_cap  += k_tot_cap * dt
        cum_res  += k_tot_res * dt

        soh[day + 1]        = INITIAL_SOH * np.exp(-cum_cap)
        res_factor[day + 1] = np.exp(cum_res)

    time_years = np.arange(n_days + 1) / 365.0
    eol_mask = soh <= 0.70
    rul = np.argmax(eol_mask) / 365.0 if np.any(eol_mask) else years

    total = cal_cont + cyc_cont
    cal_pct = cal_cont / total * 100 if total > 0 else 0
    cyc_pct = cyc_cont / total * 100 if total > 0 else 0

    print(f"  Remaining Useful Life : {rul:.1f} years")
    print(f"  Final SOH at {years}yr    : {soh[-1]:.1%}")
    print(f"  Calendar aging share  : {cal_pct:.0f}%")
    print(f"  Cyclic aging share    : {cyc_pct:.0f}%")

    return {
        "time": time_years, "soh": soh,
        "cap_loss": (1 - soh) * 100,
        "res_inc": (res_factor - 1) * 100,
        "rul": rul, "cal_pct": cal_pct, "cyc_pct": cyc_pct,
    }


# =============================================================================
# 12. CALENDAR AGING VALIDATION (Model 2)
# =============================================================================

def plot_calendar_validation():
    """
    Plot Model 2 calendar aging curves for various T / SOC combos.

    Shows the characteristic t^0.75 power-law shape, which differs from
    Model 1's exponential curves:
      ▸ Model 2: Decelerating but never saturates
      ▸ Model 1: Approaches an exponential plateau
    """
    t_days = np.linspace(0, 730, 400)   # 0 → 2 years
    t_years = t_days / 365.0

    conditions = [
        (293.15, 0.50, "T20°C  SOC50%"),
        (313.15, 0.50, "T40°C  SOC50%"),
        (293.15, 0.70, "T20°C  SOC70%"),
        (313.15, 0.70, "T40°C  SOC70%"),
        (293.15, 0.90, "T20°C  SOC90%"),
        (313.15, 0.90, "T40°C  SOC90%"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(r"Calendar Aging — Model 2 (Schmalstieg)""\n"
                 r"$t^{0.75}$ power-law, voltage & Arrhenius dependence",
                 fontsize=12)

    for T, SOC, label in conditions:
        V = soc_to_ocv(SOC)
        a_cap = m2_cal_alpha_cap(V, T)
        a_res = m2_cal_alpha_res(V, T)

        ax1.plot(t_years, a_cap * t_days**S_TIME_EXP, lw=1.5, label=label)
        ax2.plot(t_years, a_res * t_days**S_TIME_EXP, lw=1.5, label=label)

    ax1.set_ylabel("Capacity loss [%]")
    ax1.set_ylim(0, 30)
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Years")
    ax2.set_ylabel("Resistance increase [%]")
    ax2.set_ylim(0, 25)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# 13. RESULTS PLOT (Model 2)
# =============================================================================

def plot_results(r2):
    """4-panel figure summarising the Model 2 CS-BESS simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("CS-BESS Second-Life Battery Aging — Model 2 (Schmalstieg)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Weekly demand ──
    ax = axes[0, 0]
    hrs = np.arange(168)
    ax.plot(hrs, r2["demand"], "b-", lw=0.9)
    ax.axhline(500, color="r", ls="--", lw=1.5, label="Grid limit (500 kW)")
    ax.fill_between(hrs, r2["demand"], 500,
                    where=r2["demand"] > 500,
                    alpha=0.3, color="red", label="Battery discharge")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_xticks([12 + 24 * i for i in range(7)])
    ax.set_xticklabels(days)
    ax.set_ylabel("Power demand [kW]")
    ax.set_title("Weekly Charging Station Demand")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 700)

    # ── Panel 2: Capacity loss ──
    ax = axes[0, 1]
    ax.plot(r2["time"], r2["cap_loss"], "g-", lw=2,
            label="Model 2 (Schmalstieg)")
    ax.axhline(30, color="r", ls="--", alpha=0.7, label="EOL (70% SOH)")
    ax.axhline(20, color="gray", ls="--", alpha=0.7, label="Start (80% SOH)")
    if r2["rul"] < r2["time"][-1]:
        ax.axvline(r2["rul"], color="gray", ls=":", alpha=0.5)
        ax.annotate(f'RUL ≈ {r2["rul"]:.1f} yr',
                    xy=(r2["rul"], 30.5), fontsize=9, color="gray")
    ax.set_xlabel("Year")
    ax.set_ylabel("Capacity Loss [%]")
    ax.set_title("Capacity Degradation over Time")
    ax.set_ylim(18, 35)
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Resistance ──
    ax = axes[1, 0]
    ax.plot(r2["time"], r2["res_inc"], "r-", lw=2)
    if r2["rul"] < r2["time"][-1]:
        eol_day = int(r2["rul"] * 365)
        ax.plot(r2["rul"], r2["res_inc"][eol_day], "ko", ms=6)
        ax.annotate(f'{r2["res_inc"][eol_day]:.0f}% at EOL',
                    xy=(r2["rul"], r2["res_inc"][eol_day]),
                    xytext=(r2["rul"] + 0.4,
                            r2["res_inc"][eol_day] + 2),
                    fontsize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Resistance Increase [%]")
    ax.set_title("Internal Resistance Growth")
    ax.grid(True, alpha=0.3)

    # ── Panel 4: SOH trajectory ──
    ax = axes[1, 1]
    ax.plot(r2["time"], r2["soh"] * 100, "g-", lw=2)
    ax.axhline(70, color="r", ls="--", alpha=0.7, label="EOL (70%)")
    ax.axhline(80, color="gray", ls="--", alpha=0.7, label="Initial SOH (80%)")
    ax.fill_between(r2["time"], r2["soh"] * 100, 70,
                    where=r2["soh"] * 100 >= 70, alpha=0.1, color="green")
    ax.set_xlabel("Year")
    ax.set_ylabel("State of Health [%]")
    ax.set_title("SOH Trajectory")
    ax.set_ylim(60, 85)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# 14. MODEL COMPARISON PLOT
# =============================================================================

def plot_comparison(r1, r2):
    """
    Side-by-side overlay of Model 1 (Baghdadi) vs Model 2 (Schmalstieg).
    Highlights the structural differences in aging predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Model Comparison — CS-BESS Second-Life Aging\n"
                 "Model 1: Baghdadi (exponential, multiplicative)   vs.   "
                 "Model 2: Schmalstieg (power-law, additive)",
                 fontsize=11, fontweight="bold")

    # ── SOH ──
    ax = axes[0, 0]
    ax.plot(r1["time"], r1["soh"] * 100, "b-", lw=2,
            label="Model 1 (Baghdadi)")
    ax.plot(r2["time"], r2["soh"] * 100, "g-", lw=2,
            label="Model 2 (Schmalstieg)")
    ax.axhline(70, color="r", ls="--", alpha=0.5, label="EOL (70%)")
    ax.axhline(80, color="gray", ls="--", alpha=0.3)
    # Mark RUL
    for r, c, m in [(r1, "blue", "o"), (r2, "green", "s")]:
        if r["rul"] < r["time"][-1]:
            ax.plot(r["rul"], 70, m, color=c, ms=8, zorder=5)
    ax.set_ylabel("SOH [%]")
    ax.set_xlabel("Year")
    ax.set_title("State of Health")
    ax.set_ylim(60, 85)
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Capacity Loss ──
    ax = axes[0, 1]
    ax.plot(r1["time"], r1["cap_loss"], "b-", lw=2, label="Model 1")
    ax.plot(r2["time"], r2["cap_loss"], "g-", lw=2, label="Model 2")
    ax.axhline(30, color="r", ls="--", alpha=0.5, label="EOL (30% loss)")
    ax.set_ylabel("Capacity Loss [%]")
    ax.set_xlabel("Year")
    ax.set_title("Capacity Loss")
    ax.set_ylim(18, 35)
    ax.set_xlim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Resistance ──
    ax = axes[1, 0]
    ax.plot(r1["time"], r1["res_inc"], "b-", lw=2, label="Model 1")
    ax.plot(r2["time"], r2["res_inc"], "g-", lw=2, label="Model 2")
    ax.set_ylabel("Resistance Increase [%]")
    ax.set_xlabel("Year")
    ax.set_title("Resistance Growth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Summary table ──
    ax = axes[1, 1]
    ax.axis("off")

    hdr = f"{'Metric':<26}  {'Model 1':>10}  {'Model 2':>10}"
    sep = "─" * 52
    rows = [
        f"{'RUL [years]':<26}  {r1['rul']:>10.1f}  {r2['rul']:>10.1f}",
        f"{'SOH at 5 yr':<26}  {r1['soh'][-1]*100:>9.1f}%  {r2['soh'][-1]*100:>9.1f}%",
        f"{'Calendar share':<26}  {r1['cal_pct']:>9.0f}%  {r2['cal_pct']:>9.0f}%",
        f"{'Cyclic share':<26}  {r1['cyc_pct']:>9.0f}%  {r2['cyc_pct']:>9.0f}%",
        sep,
        f"{'Time dependence':<26}  {'exp(-kt)':>10}  {'t^0.75':>10}",
        f"{'Cal/cyc coupling':<26}  {'multiply':>10}  {'additive':>10}",
        f"{'Stress variable':<26}  {'SOC':>10}  {'OCV (V)':>10}",
    ]
    txt = hdr + "\n" + sep + "\n" + "\n".join(rows)

    ax.text(0.05, 0.55, txt, fontsize=10, fontfamily="monospace",
            verticalalignment="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary Comparison", fontsize=12)

    plt.tight_layout()
    return fig


# =============================================================================
# 15. RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":

    # Step 1 — Validate Model 2 calendar aging
    print("[1/4] Validating Model 2 calendar aging curves...\n")
    fig_val = plot_calendar_validation()

    # Step 2 — Run Model 2
    print("\n[2/4] Running Model 2 (Schmalstieg) CS-BESS simulation...\n")
    r2 = simulate_model2(years=5)

    # Step 3 — Run Model 1 for comparison
    print("\n[3/4] Running Model 1 (Baghdadi) for comparison...\n")
    r1 = simulate_model1(years=5)

    # Step 4 — Generate all plots
    print("\n[4/4] Generating plots...")
    fig_r2  = plot_results(r2)
    fig_cmp = plot_comparison(r1, r2)

    plt.show()
"""
pk_deconvolution.py
-------------------
Pharmacokinetic deconvolution and ka estimation from oral plasma profiles.

Workflow
~~~~~~~~
1. Accept a time × concentration matrix (first col = time, rest = concentrations).
   Time and concentration units are declared by the caller; time is converted
   to hours internally so all PK parameters are reported in h⁻¹ / L.
2. Fit a 1-compartment and a 2-compartment oral PK model via non-linear least
   squares and extract ka from each.
3. Run Wagner-Nelson deconvolution on EACH replicate column individually to
   obtain per-subject fraction-absorbed profiles; derive the mean and ± 1 SD
   band shown in Panel B.
4. Estimate ka from the slope of ln(1 − Fa) vs time on the mean profile.
5. Produce a publication-quality three-panel figure and print a summary.

Quick start
~~~~~~~~~~~
    python pk_deconvolution.py          # two built-in demos

Public API
~~~~~~~~~~
    run_analysis(data_matrix,
                 dose=100, F=1.0,
                 time_unit='hours',     # 'hours' | 'minutes' | 'seconds'
                 conc_unit='mg/L',      # 'mg/L' | 'ug/mL' | 'ng/mL' | 'nM' | 'uM'
                 title='...',
                 save_path='fig.png')   # None to skip saving
    generate_test_data(model, seed)
    generate_replicate_data(n_subjects, seed)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.integrate import odeint
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── Unit conversion tables ────────────────────────────────────────────────
_TIME_TO_HOURS: dict[str, float] = {
    "hours": 1.0,    "hour": 1.0,  "h": 1.0,  "hr": 1.0,  "hrs": 1.0,
    "minutes": 1/60, "minute": 1/60, "min": 1/60, "mins": 1/60,
    "seconds": 1/3600, "second": 1/3600, "s": 1/3600, "sec": 1/3600,
}

# Pretty axis labels for each unit keyword
_CONC_LABELS: dict[str, str] = {
    "mg/L":   "mg L⁻¹",
    "ug/mL":  "µg mL⁻¹",
    "ng/mL":  "ng mL⁻¹",
    "nM":     "nM",
    "uM":     "µM",
}

# ── Colour palette (colour-blind-friendly) ────────────────────────────────
_C_DATA = "#2C3E50"
_C_1CMT = "#E74C3C"
_C_2CMT = "#3498DB"
_C_WN   = "#27AE60"


# ══════════════════════════════════════════════════════════════════════════
# PK model definitions
# ══════════════════════════════════════════════════════════════════════════

def _one_cmt(t: np.ndarray, ka: float, ke: float, V: float,
             F: float = 1.0, D: float = 100.0) -> np.ndarray:
    if abs(ka - ke) < 1e-7:
        ka += 1e-6
    return np.maximum(
        F * D * ka / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t)),
        0.0,
    )


def _two_cmt_odes(y: list, _t: float,
                  ka: float, k10: float, k12: float, k21: float) -> list:
    Ag, A1, A2 = y
    return [
        -ka * Ag,
        ka * Ag - k10 * A1 - k12 * A1 + k21 * A2,
        k12 * A1 - k21 * A2,
    ]


def _two_cmt(t: np.ndarray, ka: float, k10: float, k12: float, k21: float,
             V1: float, F: float = 1.0, D: float = 100.0) -> np.ndarray:
    sol = odeint(_two_cmt_odes, [F * D, 0.0, 0.0], t,
                 args=(ka, k10, k12, k21), rtol=1e-8, atol=1e-10)
    return np.maximum(sol[:, 1] / V1, 0.0)


# ══════════════════════════════════════════════════════════════════════════
# Model fitting
# ══════════════════════════════════════════════════════════════════════════

def _fit_1cmt(t, C, D, F):
    def _model(t, ka, ke, V):
        return _one_cmt(t, ka, ke, V, F, D)
    Cmax = C.max()
    try:
        popt, pcov = curve_fit(_model, t, C,
                               p0=[1.0, 0.15, D / (Cmax * 5 + 1e-9)],
                               bounds=([0.01, 1e-4, 0.1], [200.0, 20.0, 5000.0]),
                               method="trf", max_nfev=20_000, loss="soft_l1")
        return popt, np.sqrt(np.diag(pcov))
    except Exception as exc:
        print(f"  [1-CMT fit failed: {exc}]")
        return None, None


def _fit_2cmt(t, C, D, F):
    def _model(t, ka, k10, k12, k21, V1):
        return _two_cmt(t, ka, k10, k12, k21, V1, F, D)
    Cmax = C.max()
    try:
        popt, pcov = curve_fit(_model, t, C,
                               p0=[1.0, 0.15, 0.30, 0.10, D / (Cmax * 5 + 1e-9)],
                               bounds=([0.01, 1e-4, 1e-4, 1e-4, 0.1],
                                       [200.0, 20.0, 20.0, 20.0, 5000.0]),
                               method="trf", max_nfev=60_000, loss="soft_l1")
        return popt, np.sqrt(np.diag(pcov))
    except Exception as exc:
        print(f"  [2-CMT fit failed: {exc}]")
        return None, None


# ══════════════════════════════════════════════════════════════════════════
# Wagner-Nelson deconvolution
# ══════════════════════════════════════════════════════════════════════════

def _wagner_nelson(t: np.ndarray, C: np.ndarray, ke: float) -> tuple[np.ndarray, float]:
    auc = np.zeros_like(C)
    for i in range(1, len(t)):
        auc[i] = auc[i - 1] + 0.5 * (C[i] + C[i - 1]) * (t[i] - t[i - 1])
    auc_inf = auc[-1] + C[-1] / ke
    Fa = np.clip((C + ke * auc) / (ke * auc_inf + 1e-12), 0.0, 1.0)
    return Fa, auc_inf


def _estimate_ka_wn(t: np.ndarray, Fa: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mask = Fa < 0.95
    if mask.sum() < 3:
        mask = Fa < 0.99
    if mask.sum() < 2:
        mask = np.ones(len(Fa), dtype=bool)
    t_used = t[mask]
    y_used = np.log(np.maximum(1.0 - Fa[mask], 1e-9))
    slope, _ = np.polyfit(t_used, y_used, 1)
    return float(-slope), t_used, y_used


# ══════════════════════════════════════════════════════════════════════════
# Test data generators
# ══════════════════════════════════════════════════════════════════════════

def generate_test_data(model: str = "1comp", seed: int = 42) -> tuple:
    """
    Single-subject synthetic oral PK data (8 % CV noise).

    Returns
    -------
    t, C_obs, C_true, true_params
    Time is in hours; concentration in mg/L (= µg/mL).
    """
    rng = np.random.default_rng(seed)
    t = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
    if model == "1comp":
        p = dict(ka=1.50, ke=0.15, V=20.0, F=1.0, D=100.0)
        C_true = _one_cmt(t, p["ka"], p["ke"], p["V"], p["F"], p["D"])
    else:
        p = dict(ka=1.50, k10=0.15, k12=0.30, k21=0.10, V1=15.0, F=1.0, D=100.0)
        C_true = _two_cmt(t, p["ka"], p["k10"], p["k12"],
                          p["k21"], p["V1"], p["F"], p["D"])
    C_obs = np.maximum(C_true + C_true * 0.08 * rng.standard_normal(len(t)), 0.0)
    return t, C_obs, C_true, p


def generate_replicate_data(n_subjects: int = 6, seed: int = 99) -> tuple[np.ndarray, dict]:
    """
    Multi-subject oral PK data matrix with ~25 % CV inter-individual variability.

    Parameters are drawn from log-normal distributions around population means.
    Returns (data_matrix, pop_params) where data_matrix is
    [n_timepoints × (1 + n_subjects)], time in hours, concentration in mg/L.
    """
    rng = np.random.default_rng(seed)
    t   = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
    pop = dict(ka=1.50, ke=0.15, V=20.0, F=1.0, D=100.0, omega_ka=0.30,
               omega_ke=0.25, omega_V=0.20)

    C_matrix = np.zeros((len(t), n_subjects))
    for i in range(n_subjects):
        ka_i = pop["ka"] * np.exp(rng.normal(0, pop["omega_ka"]))
        ke_i = pop["ke"] * np.exp(rng.normal(0, pop["omega_ke"]))
        V_i  = pop["V"]  * np.exp(rng.normal(0, pop["omega_V"]))
        C_i  = _one_cmt(t, ka_i, ke_i, V_i, pop["F"], pop["D"])
        C_matrix[:, i] = np.maximum(C_i + C_i * 0.08 * rng.standard_normal(len(t)), 0)

    return np.column_stack([t, C_matrix]), pop


# ══════════════════════════════════════════════════════════════════════════
# Publication-quality figure
# ══════════════════════════════════════════════════════════════════════════

def _set_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          11,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "axes.linewidth":     1.2,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.major.width":  1.2,
        "ytick.major.width":  1.2,
        "xtick.minor.width":  0.8,
        "ytick.minor.width":  0.8,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.fontsize":    9,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#BDC3C7",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "lines.linewidth":    2.0,
    })


def _build_figure(
    *,
    t: np.ndarray,
    C: np.ndarray,
    C_all: np.ndarray | None,
    t_fine: np.ndarray,
    C1: np.ndarray | None,
    C2: np.ndarray | None,
    Fa: np.ndarray,
    Fa_all: np.ndarray | None,
    t_reg: np.ndarray,
    y_reg: np.ndarray,
    ka_wn: float,
    ka_1c: float | None,
    ka_2c: float | None,
    title: str,
    time_label: str = "h",
    conc_label: str = "mg L⁻¹",
) -> plt.Figure:
    _set_style()

    multi = Fa_all is not None and Fa_all.shape[1] > 1

    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            wspace=0.40,
                            left=0.07, right=0.97,
                            top=0.87, bottom=0.14)

    # ── Panel A: concentration–time ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    if C_all is not None and C_all.shape[1] > 1:
        C_sd = C_all.std(axis=1)
        ax1.fill_between(t, C - C_sd, C + C_sd,
                         alpha=0.15, color=_C_DATA, zorder=1)
        for i in range(C_all.shape[1]):
            ax1.plot(t, C_all[:, i], color=_C_DATA, alpha=0.25,
                     lw=0.8, zorder=2)

    if C1 is not None:
        ax1.plot(t_fine, C1, color=_C_1CMT, lw=2.0, zorder=4,
                 label=f"1-CMT  $k_a$ = {ka_1c:.3f} h⁻¹")
    if C2 is not None:
        ax1.plot(t_fine, C2, color=_C_2CMT, lw=2.0, ls="--", zorder=3,
                 label=f"2-CMT  $k_a$ = {ka_2c:.3f} h⁻¹")

    obs_lbl = "Observed (mean)" if (C_all is not None and C_all.shape[1] > 1) else "Observed"
    ax1.scatter(t, C, color=_C_DATA, s=55, zorder=5,
                edgecolors="white", linewidths=0.6, label=obs_lbl)

    ax1.set_xlabel(f"Time ({time_label})")
    ax1.set_ylabel(f"Plasma Concentration ({conc_label})")
    ax1.set_title("A   Plasma Concentration–Time Profile",
                  fontweight="bold", loc="left")
    ax1.legend(loc="upper right", handlelength=1.8)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── Panel B: fraction absorbed ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    if multi:
        # Individual per-subject Fa profiles
        for i in range(Fa_all.shape[1]):
            ax2.plot(t, Fa_all[:, i] * 100, color=_C_WN,
                     alpha=0.30, lw=0.9, zorder=2)
        # ±1 SD band
        Fa_sd = Fa_all.std(axis=1)
        ax2.fill_between(t,
                         np.clip((Fa - Fa_sd) * 100, 0, 100),
                         np.clip((Fa + Fa_sd) * 100, 0, 100),
                         alpha=0.22, color=_C_WN, zorder=3,
                         label="Mean ± 1 SD")
        # Mean profile
        ax2.plot(t, Fa * 100, "o-", color=_C_WN, lw=2.2, ms=6,
                 markeredgecolor="white", markeredgewidth=0.6,
                 zorder=5, label="Mean $F_a$")
        ax2.legend(loc="lower right", handlelength=1.6)
    else:
        ax2.fill_between(t, 0, Fa * 100, alpha=0.15, color=_C_WN, zorder=1)
        ax2.plot(t, Fa * 100, "o-", color=_C_WN, lw=2.2, ms=6,
                 markeredgecolor="white", markeredgewidth=0.6,
                 zorder=5, label="Wagner–Nelson")
        ax2.legend(loc="lower right", handlelength=1.6)

    ax2.axhline(100, color="gray", ls=":", lw=1.0, alpha=0.6)

    lines = [rf"$k_a$ (W–N) = {ka_wn:.3f} h⁻¹"]
    if ka_1c:
        lines.append(rf"$k_a$ (1-CMT) = {ka_1c:.3f} h⁻¹")
    if ka_2c:
        lines.append(rf"$k_a$ (2-CMT) = {ka_2c:.3f} h⁻¹")
    ax2.text(0.97, 0.05, "\n".join(lines),
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=9, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45",
                       facecolor="white", edgecolor="#BDC3C7", alpha=0.95))

    ax2.set_xlabel(f"Time ({time_label})")
    ax2.set_ylabel("Fraction Absorbed (%)")
    ax2.set_title("B   Fraction Absorbed (Wagner–Nelson)",
                  fontweight="bold", loc="left")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 118)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))

    # ── Panel C: ln(1 − Fa) regression ───────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    valid = Fa < 0.999
    ax3.scatter(t[valid], np.log(np.maximum(1.0 - Fa[valid], 1e-9)),
                color=_C_DATA, s=50, zorder=4,
                edgecolors="white", linewidths=0.6,
                label="All data points")

    used_mask = np.isin(t, t_reg)
    ax3.scatter(t[used_mask], np.log(np.maximum(1.0 - Fa[used_mask], 1e-9)),
                color=_C_WN, s=75, zorder=6,
                edgecolors="white", linewidths=0.6,
                label="Absorption-phase data")

    slope, intercept = np.polyfit(t_reg, y_reg, 1)
    t_line = np.linspace(0, t_reg[-1] * 1.15, 200)
    ax3.plot(t_line, slope * t_line + intercept,
             color=_C_WN, lw=2.0, ls="--", zorder=5,
             label=rf"Fit: slope = −{ka_wn:.3f} h⁻¹")

    ax3.set_xlabel(f"Time ({time_label})")
    ax3.set_ylabel("ln(1 − $F_a$)")
    ax3.set_title(r"C   $k_a$ Estimation from ln(1 − $F_a$)",
                  fontweight="bold", loc="left")
    ax3.legend(loc="upper right", handlelength=1.8)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Main public entry point
# ══════════════════════════════════════════════════════════════════════════

def run_analysis(
    data_matrix: np.ndarray,
    *,
    dose: float = 100.0,
    F: float = 1.0,
    time_unit: str = "hours",
    conc_unit: str = "mg/L",
    title: str = "PK Deconvolution Analysis",
    save_path: str | None = "pk_deconvolution_analysis.png",
) -> dict:
    """
    Full PK deconvolution pipeline.

    Parameters
    ----------
    data_matrix : shape (n_timepoints, 1 + n_subjects)
        Column 0  = time in the units specified by `time_unit`.
        Columns 1+ = plasma concentrations in the units given by `conc_unit`.
    dose      : administered dose (units consistent with conc_unit × volume)
    F         : bioavailability fraction (0–1)
    time_unit : 'hours' | 'minutes' | 'seconds'   [default: 'hours']
    conc_unit : 'mg/L' | 'ug/mL' | 'ng/mL' | 'nM' | 'uM'  [default: 'mg/L']
                Used for axis labels only; ensure dose is consistent.
    title     : figure super-title
    save_path : PNG save path; None to skip.
    """
    if data_matrix.ndim != 2 or data_matrix.shape[1] < 2:
        raise ValueError("data_matrix must be (n_timepoints, ≥2) with time in column 0.")

    # ── Unit handling ──────────────────────────────────────────────────
    time_scale = _TIME_TO_HOURS.get(time_unit.lower())
    if time_scale is None:
        raise ValueError(f"Unknown time_unit '{time_unit}'. "
                         f"Choose from: {list(_TIME_TO_HOURS)}")

    t_raw = data_matrix[:, 0]
    t     = t_raw * time_scale           # internal: always hours
    C_all = data_matrix[:, 1:]
    C     = C_all.mean(axis=1)

    conc_label = _CONC_LABELS.get(conc_unit, conc_unit)
    time_label = "h"                     # all calculations and axes in hours

    [Cmax, imax] = [C.max(), C.argmax()]

    print("=" * 62)
    print("  PK DECONVOLUTION & ka ESTIMATION")
    print("=" * 62)
    print(f"  Time points  : {len(t)}")
    print(f"  Subjects     : {C_all.shape[1]}")
    print(f"  Time unit    : {time_unit} → converted to hours")
    print(f"  Conc unit    : {conc_unit}")
    print(f"  Dose / F     : {dose} / {F}")
    print(f"  Cmax         : {Cmax:.4f} {conc_unit}  at  t = {t[imax]:.2f} h")

    # ── 1-compartment fit ──────────────────────────────────────────────
    print("\n  ─── 1-Compartment Model ───")
    popt_1c, perr_1c = _fit_1cmt(t, C, dose, F)
    ka_1c = None
    if popt_1c is not None:
        ka_1c, ke_1c, V_1c = popt_1c
        print(f"    ka   = {ka_1c:.4f} ± {perr_1c[0]:.4f} h⁻¹")
        print(f"    ke   = {ke_1c:.4f} ± {perr_1c[1]:.4f} h⁻¹")
        print(f"    V    = {V_1c:.4f} ± {perr_1c[2]:.4f} L")
        print(f"    t½   = {0.693 / ke_1c:.2f} h")
        print(f"    Tmax = {np.log(ka_1c / ke_1c) / (ka_1c - ke_1c):.2f} h")
        rmse = np.sqrt(np.mean((C - _one_cmt(t, *popt_1c, F, dose)) ** 2))
        print(f"    RMSE = {rmse:.4f} {conc_unit}")

    # ── 2-compartment fit ──────────────────────────────────────────────
    print("\n  ─── 2-Compartment Model ───")
    popt_2c, perr_2c = _fit_2cmt(t, C, dose, F)
    ka_2c = None
    if popt_2c is not None:
        ka_2c, k10_2c, k12_2c, k21_2c, V1_2c = popt_2c
        print(f"    ka  = {ka_2c:.4f} ± {perr_2c[0]:.4f} h⁻¹")
        print(f"    k10 = {k10_2c:.4f} ± {perr_2c[1]:.4f} h⁻¹")
        print(f"    k12 = {k12_2c:.4f} ± {perr_2c[2]:.4f} h⁻¹")
        print(f"    k21 = {k21_2c:.4f} ± {perr_2c[3]:.4f} h⁻¹")
        print(f"    V1  = {V1_2c:.4f} ± {perr_2c[4]:.4f} L")
        print(f"    CL  = {k10_2c * V1_2c:.4f} L h⁻¹")
        rmse = np.sqrt(np.mean((C - _two_cmt(t, *popt_2c, F, dose)) ** 2))
        print(f"    RMSE = {rmse:.4f} {conc_unit}")

    # ── Wagner-Nelson deconvolution ────────────────────────────────────
    print("\n  ─── Wagner–Nelson Deconvolution ───")
    ke_wn = ke_1c if popt_1c is not None else 0.10
    print(f"    ke used : {ke_wn:.4f} h⁻¹  (from 1-CMT fit)")

    # Compute Fa for every replicate individually, then derive mean
    n_subjects = C_all.shape[1]
    Fa_all_arr = np.zeros((len(t), n_subjects))
    auc_infs   = np.zeros(n_subjects)
    for i in range(n_subjects):
        Fa_all_arr[:, i], auc_infs[i] = _wagner_nelson(t, C_all[:, i], ke_wn)
    Fa = Fa_all_arr.mean(axis=1)

    ka_wn, t_reg, y_reg = _estimate_ka_wn(t, Fa)
    print(f"    AUC(0–∞) mean : {auc_infs.mean():.3f} h·{conc_unit}")
    print(f"    ka (W–N)      : {ka_wn:.4f} h⁻¹")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n  ══ ka SUMMARY ══")
    if popt_1c is not None:
        print(f"    1-Compartment model  : {ka_1c:.4f} h⁻¹")
    if popt_2c is not None:
        print(f"    2-Compartment model  : {ka_2c:.4f} h⁻¹")
    print(f"    Wagner–Nelson (W–N)  : {ka_wn:.4f} h⁻¹")
    print("=" * 62)

    # ── Figure ─────────────────────────────────────────────────────────
    t_fine  = np.linspace(0, t[-1], 500)
    C1_fine = _one_cmt(t_fine, *popt_1c, F, dose) if popt_1c is not None else None
    C2_fine = _two_cmt(t_fine, *popt_2c, F, dose) if popt_2c is not None else None

    # Only show individual columns in figure when there is more than one
    C_all_fig   = C_all   if n_subjects > 1 else None
    Fa_all_fig  = Fa_all_arr if n_subjects > 1 else None

    fig = _build_figure(
        t=t, C=C,
        C_all=C_all_fig,
        t_fine=t_fine, C1=C1_fine, C2=C2_fine,
        Fa=Fa, Fa_all=Fa_all_fig,
        t_reg=t_reg, y_reg=y_reg,
        ka_wn=ka_wn, ka_1c=ka_1c, ka_2c=ka_2c,
        title=title,
        time_label=time_label,
        conc_label=conc_label,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n  Figure saved → {save_path}")

    plt.show()

    return dict(
        ka_wn=ka_wn, ka_1cmt=ka_1c, ka_2cmt=ka_2c,
        popt_1cmt=popt_1c, perr_1cmt=perr_1c,
        popt_2cmt=popt_2c, perr_2cmt=perr_2c,
        Fa=Fa, Fa_all=Fa_all_arr,
        AUC_inf=auc_infs.mean(), figure=fig,
    )


# ══════════════════════════════════════════════════════════════════════════
# CLI / demo entry point
# ══════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    # ── Demo 1: single subject, time in hours, conc in µg/mL ─────────
    print("\n" + "=" * 62)
    print("  DEMO 1 — Single subject  |  time: hours  |  conc: µg/mL")
    print("=" * 62)
    t1, C_obs1, _, tp1 = generate_test_data("1comp")
    run_analysis(
        np.column_stack([t1, C_obs1]),
        dose=tp1["D"], F=tp1["F"],
        time_unit="hours",
        conc_unit="ug/mL",
        title=f"Demo 1 — Single Subject (true $k_a$ = {tp1['ka']} h⁻¹)",
        save_path="pk_demo1_single.png",
    )

    # ── Demo 2: 6 subjects, time in MINUTES, conc in µg/mL ───────────
    print("\n" + "=" * 62)
    print("  DEMO 2 — 6 subjects (IIV)  |  time: minutes  |  conc: µg/mL")
    print("=" * 62)
    data2, pop2 = generate_replicate_data(n_subjects=6)
    data2_min = data2.copy()
    data2_min[:, 0] *= 60          # convert time column from hours → minutes

    run_analysis(
        data2_min,
        dose=pop2["D"], F=pop2["F"],
        time_unit="minutes",
        conc_unit="ug/mL",
        title=("Demo 2 — 6 Subjects with Inter-Individual Variability  "
               r"(time in min, $k_a$ pop = 1.50 h$^{-1}$)"),
        save_path="pk_demo2_replicates.png",
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo()
    else:
        csv_path  = Path(sys.argv[1])
        if not csv_path.exists():
            sys.exit(f"File not found: {csv_path}")
        raw       = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        dose_arg  = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
        F_arg     = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        tunit_arg = sys.argv[4]        if len(sys.argv) > 4 else "hours"
        cunit_arg = sys.argv[5]        if len(sys.argv) > 5 else "mg/L"
        run_analysis(raw, dose=dose_arg, F=F_arg,
                     time_unit=tunit_arg, conc_unit=cunit_arg,
                     title=f"PK Deconvolution — {csv_path.stem}")

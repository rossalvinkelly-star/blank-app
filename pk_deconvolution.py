"""
pk_deconvolution.py
-------------------
Pharmacokinetic deconvolution and ka estimation from oral plasma profiles.

Workflow
~~~~~~~~
1. Accept a time × concentration matrix (first col = time, rest = concentrations).
2. Fit a 1-compartment and a 2-compartment oral PK model via non-linear least
   squares and extract ka from each.
3. Run Wagner-Nelson deconvolution to obtain the fraction-absorbed profile
   independent of a compartment assumption, then estimate ka from the
   terminal slope of ln(1 - Fa) vs time.
4. Produce a publication-quality three-panel figure (PK profile, Fa profile,
   ka regression) and print a numerical summary.

Quick start
~~~~~~~~~~~
    python pk_deconvolution.py          # runs built-in test-data demo
    python pk_deconvolution.py mydata.csv  # analyses a CSV file
                                           # (first col = time, remaining = C)

Public API
~~~~~~~~~~
    run_analysis(data_matrix, dose, F, title) -> dict of results + Figure
    generate_test_data(model, seed)            -> t, C_obs, C_true, true_params
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

# ── Colour palette (colour-blind-friendly) ────────────────────────────────
_C_DATA = "#2C3E50"   # near-black  – observed data
_C_1CMT = "#E74C3C"   # red         – 1-compartment fit
_C_2CMT = "#3498DB"   # blue        – 2-compartment fit
_C_WN   = "#27AE60"   # green       – Wagner-Nelson


# ══════════════════════════════════════════════════════════════════════════
# PK model definitions
# ══════════════════════════════════════════════════════════════════════════

def _one_cmt(t: np.ndarray, ka: float, ke: float, V: float,
             F: float = 1.0, D: float = 100.0) -> np.ndarray:
    """
    1-compartment oral model.
        C(t) = F·D·ka / [V·(ka−ke)] · (e^{−ke·t} − e^{−ka·t})
    A small offset prevents division-by-zero when ka ≈ ke.
    """
    if abs(ka - ke) < 1e-7:
        ka += 1e-6
    return np.maximum(
        F * D * ka / (V * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t)),
        0.0,
    )


def _two_cmt_odes(y: list[float], _t: float,
                  ka: float, k10: float, k12: float, k21: float) -> list[float]:
    """ODEs for 2-compartment oral model (amounts: gut, central, peripheral)."""
    Ag, A1, A2 = y
    return [
        -ka * Ag,
        ka * Ag - k10 * A1 - k12 * A1 + k21 * A2,
        k12 * A1 - k21 * A2,
    ]


def _two_cmt(t: np.ndarray, ka: float, k10: float, k12: float, k21: float,
             V1: float, F: float = 1.0, D: float = 100.0) -> np.ndarray:
    """2-compartment oral model solved via ODE integration."""
    sol = odeint(_two_cmt_odes, [F * D, 0.0, 0.0], t,
                 args=(ka, k10, k12, k21), rtol=1e-8, atol=1e-10)
    return np.maximum(sol[:, 1] / V1, 0.0)


# ══════════════════════════════════════════════════════════════════════════
# Model fitting
# ══════════════════════════════════════════════════════════════════════════

def _fit_1cmt(t: np.ndarray, C: np.ndarray,
              D: float, F: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (popt, perr) or (None, None) on failure.  params: ka, ke, V."""
    def _model(t, ka, ke, V):
        return _one_cmt(t, ka, ke, V, F, D)

    Cmax = C.max()
    p0     = [1.0,  0.15,  D / (Cmax * 5 + 1e-9)]
    bounds = ([0.01, 1e-4, 0.1], [200.0, 20.0, 5000.0])
    try:
        popt, pcov = curve_fit(_model, t, C, p0=p0, bounds=bounds,
                               method="trf", max_nfev=20_000,
                               loss="soft_l1")          # robust to outliers
        return popt, np.sqrt(np.diag(pcov))
    except Exception as exc:
        print(f"  [1-CMT fit failed: {exc}]")
        return None, None


def _fit_2cmt(t: np.ndarray, C: np.ndarray,
              D: float, F: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (popt, perr) or (None, None).  params: ka, k10, k12, k21, V1."""
    def _model(t, ka, k10, k12, k21, V1):
        return _two_cmt(t, ka, k10, k12, k21, V1, F, D)

    Cmax = C.max()
    p0     = [1.0,  0.15, 0.30, 0.10,  D / (Cmax * 5 + 1e-9)]
    bounds = ([0.01, 1e-4, 1e-4, 1e-4, 0.1],
              [200.0, 20.0, 20.0, 20.0, 5000.0])
    try:
        popt, pcov = curve_fit(_model, t, C, p0=p0, bounds=bounds,
                               method="trf", max_nfev=60_000,
                               loss="soft_l1")
        return popt, np.sqrt(np.diag(pcov))
    except Exception as exc:
        print(f"  [2-CMT fit failed: {exc}]")
        return None, None


# ══════════════════════════════════════════════════════════════════════════
# Wagner-Nelson deconvolution
# ══════════════════════════════════════════════════════════════════════════

def _wagner_nelson(t: np.ndarray, C: np.ndarray,
                   ke: float) -> tuple[np.ndarray, float]:
    """
    Wagner-Nelson method (1-compartment assumption).

        Fa(t) = [C(t) + ke · AUC(0,t)] / [ke · AUC(0,∞)]

    where AUC(0,∞) = AUC(0, t_last) + C(t_last)/ke.

    Returns
    -------
    Fa      : fraction absorbed at each time point  (0..1)
    AUC_inf : AUC from zero to infinity
    """
    # Cumulative trapezoidal AUC
    auc = np.zeros_like(C)
    for i in range(1, len(t)):
        auc[i] = auc[i - 1] + 0.5 * (C[i] + C[i - 1]) * (t[i] - t[i - 1])

    auc_inf = auc[-1] + C[-1] / ke
    Aa_t    = C + ke * auc          # amount absorbed by time t (per unit dose normalised)
    Aa_inf  = ke * auc_inf          # total amount absorbed

    Fa = np.clip(Aa_t / (Aa_inf + 1e-12), 0.0, 1.0)
    return Fa, auc_inf


def _estimate_ka_wn(t: np.ndarray, Fa: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate ka from Wagner-Nelson output.

    Assumes first-order absorption:
        ln(1 − Fa) = −ka · t   → slope = −ka

    Only the absorption phase (Fa < 0.95) is used to avoid the plateau
    region where ln(1 − Fa) becomes numerically unstable.

    Returns
    -------
    ka_est          : estimated absorption rate constant (h⁻¹)
    t_used, y_used  : time and ln(1−Fa) arrays used in the regression
    """
    mask = Fa < 0.95
    if mask.sum() < 3:
        mask = Fa < 0.99
    if mask.sum() < 2:
        # fallback – use all points
        mask = np.ones(len(Fa), dtype=bool)

    t_used = t[mask]
    y_used = np.log(np.maximum(1.0 - Fa[mask], 1e-9))

    slope, _ = np.polyfit(t_used, y_used, 1)
    return float(-slope), t_used, y_used


# ══════════════════════════════════════════════════════════════════════════
# Test data generator
# ══════════════════════════════════════════════════════════════════════════

def generate_test_data(
    model: str = "1comp",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate synthetic oral PK data with proportional noise (8 % CV).

    Parameters
    ----------
    model : '1comp' | '2comp'
    seed  : random seed for reproducibility

    Returns
    -------
    t          : time vector (h)
    C_obs      : observed (noisy) concentrations
    C_true     : noise-free concentrations
    true_params: dict of ground-truth parameter values
    """
    rng = np.random.default_rng(seed)
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                  4.0, 6.0, 8.0, 12.0, 16.0, 24.0])

    if model == "1comp":
        p = dict(ka=1.50, ke=0.15, V=20.0, F=1.0, D=100.0)
        C_true = _one_cmt(t, p["ka"], p["ke"], p["V"], p["F"], p["D"])
    else:
        p = dict(ka=1.50, k10=0.15, k12=0.30, k21=0.10, V1=15.0, F=1.0, D=100.0)
        C_true = _two_cmt(t, p["ka"], p["k10"], p["k12"],
                          p["k21"], p["V1"], p["F"], p["D"])

    noise  = C_true * 0.08 * rng.standard_normal(len(t))
    C_obs  = np.maximum(C_true + noise, 0.0)
    return t, C_obs, C_true, p


# ══════════════════════════════════════════════════════════════════════════
# Publication-quality figure
# ══════════════════════════════════════════════════════════════════════════

def _set_style() -> None:
    """Apply publication-ready rcParams."""
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
    t_reg: np.ndarray,
    y_reg: np.ndarray,
    ka_wn: float,
    ka_1c: float | None,
    ka_2c: float | None,
    title: str,
) -> plt.Figure:
    """Assemble and return the three-panel figure."""
    _set_style()

    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            wspace=0.40,
                            left=0.07, right=0.97,
                            top=0.87, bottom=0.14)

    # ── Panel A: concentration–time ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    if C_all is not None and C_all.shape[1] > 1:
        for i in range(C_all.shape[1]):
            ax1.plot(t, C_all[:, i], color=_C_DATA, alpha=0.25,
                     linewidth=0.8, zorder=1)

    ax1.scatter(t, C, color=_C_DATA, s=55, zorder=5,
                edgecolors="white", linewidths=0.6,
                label="Observed" + (" (mean)" if C_all is not None and C_all.shape[1] > 1 else ""))

    label_1c = f"1-CMT  $k_a$ = {ka_1c:.3f} h⁻¹" if ka_1c else None
    label_2c = f"2-CMT  $k_a$ = {ka_2c:.3f} h⁻¹" if ka_2c else None

    if C1 is not None:
        ax1.plot(t_fine, C1, color=_C_1CMT, lw=2.0,
                 label=label_1c, zorder=4)
    if C2 is not None:
        ax1.plot(t_fine, C2, color=_C_2CMT, lw=2.0,
                 ls="--", label=label_2c, zorder=3)

    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Plasma Concentration (mg L⁻¹)")
    ax1.set_title("A   Plasma Concentration–Time Profile",
                  fontweight="bold", loc="left")
    ax1.legend(loc="upper right", handlelength=1.8)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    # ── Panel B: fraction absorbed ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    ax2.fill_between(t, 0.0, Fa * 100, alpha=0.15, color=_C_WN, zorder=1)
    ax2.plot(t, Fa * 100, "o-", color=_C_WN, lw=2.0,
             ms=6, markeredgecolor="white", markeredgewidth=0.6,
             label="Wagner–Nelson", zorder=5)
    ax2.axhline(100, color="gray", ls=":", lw=1.0, alpha=0.6)

    # ka summary box
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

    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Fraction Absorbed (%)")
    ax2.set_title("B   Fraction Absorbed (Wagner–Nelson)",
                  fontweight="bold", loc="left")
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 118)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(MultipleLocator(10))

    # ── Panel C: ln(1 − Fa) regression ──────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    # all valid points in grey
    valid = Fa < 0.999
    ax3.scatter(t[valid], np.log(np.maximum(1.0 - Fa[valid], 1e-9)),
                color=_C_DATA, s=50, zorder=4,
                edgecolors="white", linewidths=0.6,
                label="All data points")

    # points used for regression highlighted
    used_mask = np.isin(t, t_reg)
    ax3.scatter(t[used_mask], np.log(np.maximum(1.0 - Fa[used_mask], 1e-9)),
                color=_C_WN, s=75, zorder=6,
                edgecolors="white", linewidths=0.6,
                label=f"Absorption-phase data")

    # regression line
    slope, intercept = np.polyfit(t_reg, y_reg, 1)
    t_line = np.linspace(0, t_reg[-1] * 1.15, 200)
    ax3.plot(t_line, slope * t_line + intercept,
             color=_C_WN, lw=2.0, ls="--", zorder=5,
             label=rf"Fit: slope = −{ka_wn:.3f} h⁻¹")

    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel("ln(1 − Fa)")
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
    title: str = "PK Deconvolution Analysis",
    save_path: str | None = "pk_deconvolution_analysis.png",
) -> dict:
    """
    Full PK deconvolution pipeline.

    Parameters
    ----------
    data_matrix : shape (n_timepoints, 1 + n_subjects)
        Column 0  = time (h).
        Columns 1+ = plasma concentrations (mg L⁻¹).
        If multiple concentration columns are supplied the mean is analysed.
    dose       : administered dose (mg)
    F          : bioavailability fraction (0–1)
    title      : figure super-title
    save_path  : file path for the saved PNG; pass None to skip saving.

    Returns
    -------
    dict with keys:
        ka_wn, ka_1cmt, ka_2cmt,
        popt_1cmt, perr_1cmt,
        popt_2cmt, perr_2cmt,
        Fa, AUC_inf, figure
    """
    if data_matrix.ndim != 2 or data_matrix.shape[1] < 2:
        raise ValueError("data_matrix must be (n_timepoints, ≥2) with time in column 0.")

    t = data_matrix[:, 0]
    C_all = data_matrix[:, 1:]
    C     = C_all.mean(axis=1)      # mean profile used for fitting

    print("=" * 62)
    print("  PK DECONVOLUTION & ka ESTIMATION")
    print("=" * 62)
    print(f"  Time points  : {len(t)}")
    print(f"  Subjects     : {C_all.shape[1]}")
    print(f"  Dose / F     : {dose} mg / {F}")
    print(f"  Cmax         : {C.max():.4f} mg L⁻¹  at  t = {t[np.argmax(C)]:.2f} h")

    # ── 1-compartment fit ──────────────────────────────────────────────
    print("\n  ─── 1-Compartment Model ───")
    popt_1c, perr_1c = _fit_1cmt(t, C, dose, F)
    ka_1c = None
    if popt_1c is not None:
        ka_1c, ke_1c, V_1c = popt_1c
        print(f"    ka  = {ka_1c:.4f} ± {perr_1c[0]:.4f} h⁻¹")
        print(f"    ke  = {ke_1c:.4f} ± {perr_1c[1]:.4f} h⁻¹")
        print(f"    V   = {V_1c:.4f} ± {perr_1c[2]:.4f} L")
        print(f"    t½  = {0.693 / ke_1c:.2f} h")
        print(f"    Tmax = {np.log(ka_1c / ke_1c) / (ka_1c - ke_1c):.2f} h")
        C_pred_1c = _one_cmt(t, *popt_1c, F, dose)
        rmse_1c = np.sqrt(np.mean((C - C_pred_1c) ** 2))
        print(f"    RMSE = {rmse_1c:.4f} mg L⁻¹")

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
        C_pred_2c = _two_cmt(t, *popt_2c, F, dose)
        rmse_2c = np.sqrt(np.mean((C - C_pred_2c) ** 2))
        print(f"    RMSE = {rmse_2c:.4f} mg L⁻¹")

    # ── Wagner-Nelson deconvolution ────────────────────────────────────
    print("\n  ─── Wagner–Nelson Deconvolution ───")
    ke_wn = ke_1c if popt_1c is not None else 0.10
    print(f"    ke used for W-N : {ke_wn:.4f} h⁻¹  (from 1-CMT fit)")
    Fa, auc_inf = _wagner_nelson(t, C, ke_wn)
    ka_wn, t_reg, y_reg = _estimate_ka_wn(t, Fa)
    print(f"    AUC(0–∞)        : {auc_inf:.3f} h·mg L⁻¹")
    print(f"    ka (W-N)        : {ka_wn:.4f} h⁻¹")

    # ── Summary table ──────────────────────────────────────────────────
    print("\n  ══ ka SUMMARY ══")
    if ka_1c is not None:
        print(f"    1-Compartment model  : {ka_1c:.4f} h⁻¹")
    if ka_2c is not None:
        print(f"    2-Compartment model  : {ka_2c:.4f} h⁻¹")
    print(f"    Wagner–Nelson (W-N)  : {ka_wn:.4f} h⁻¹")
    print("=" * 62)

    # ── Fine grid for smooth model curves ─────────────────────────────
    t_fine = np.linspace(0, t[-1], 500)
    C1_fine = _one_cmt(t_fine, *popt_1c, F, dose) if popt_1c is not None else None
    C2_fine = _two_cmt(t_fine, *popt_2c, F, dose) if popt_2c is not None else None

    # ── Build figure ───────────────────────────────────────────────────
    fig = _build_figure(
        t=t, C=C,
        C_all=C_all if C_all.shape[1] > 1 else None,
        t_fine=t_fine, C1=C1_fine, C2=C2_fine,
        Fa=Fa, t_reg=t_reg, y_reg=y_reg,
        ka_wn=ka_wn, ka_1c=ka_1c, ka_2c=ka_2c,
        title=title,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n  Figure saved → {save_path}")

    plt.show()

    return dict(
        ka_wn=ka_wn,
        ka_1cmt=ka_1c,
        ka_2cmt=ka_2c,
        popt_1cmt=popt_1c,
        perr_1cmt=perr_1c,
        popt_2cmt=popt_2c,
        perr_2cmt=perr_2c,
        Fa=Fa,
        AUC_inf=auc_inf,
        figure=fig,
    )


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """Run the built-in test-data demonstration."""
    print("Generating synthetic test data (1-compartment ground truth)…\n")
    t, C_obs, C_true, tp = generate_test_data(model="1comp")

    print("  True parameters:")
    for k, v in tp.items():
        print(f"    {k:4s} = {v}")
    print()

    data = np.column_stack([t, C_obs])
    run_analysis(
        data,
        dose=tp["D"],
        F=tp["F"],
        title="PK Deconvolution — Synthetic 1-CMT Test Data  "
              f"(true ka = {tp['ka']} h⁻¹)",
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _demo()
    else:
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            sys.exit(f"File not found: {csv_path}")
        raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        dose_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
        F_arg    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        run_analysis(
            raw,
            dose=dose_arg,
            F=F_arg,
            title=f"PK Deconvolution — {csv_path.stem}",
        )

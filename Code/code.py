from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

SCRIPT_DIR: Path = Path(__file__).parent


# region ------- Physical constants / parameters -------
LAMBDA_P: float = 532e-9  # Pump wavelength [m]
POLING_PERIOD: float = 8.5e-6  # Poling period [m]
M: int = 1  # Lattice order
G: float = M * 2 * np.pi / POLING_PERIOD  # Reciprocal lattice vector [rad/m]
T: float = 80.0  # Crystal temperature [C]
C0: float = 299_792_458.0  # Speed of light
N_SWEEP: int = 10000  # number of sweeps for Brents method

# This guarantees the idler never leaves the "valid" Sellmeier window.
LAMBDA_I_MAX: float = 7384e-9  # Idler wavelenght Sellmeier validity cutoff [m]
LAMBDA_S_MIN: float = LAMBDA_P * LAMBDA_I_MAX / (LAMBDA_I_MAX - LAMBDA_P) * 1.001
LAMBDA_S_MAX: float = 1500e-9

# endregion


# region ------- Sellmeier boundaries -------

# The scan below was used to derive LAMBDA_I_MAX = 7384e-9 m.
# Uncomment to re-run the validity check.


# def sellmeier_validity_check(scanspace: np.ndarray) -> np.ndarray:
#     """
#     Evaluate n² across a wavelength scanspace (in µm).
#     Used to identify the valid range of the Sellmeier equation.
#     """
#     x = scanspace
#     A, B, C, D, E, F = 4.54773, 0.0774167, 0.22025, -0.0226143, 2.39494, 7.45352
#     T_K = T + 273.15
#     bT = 4.23526e-8 * T_K**2
#     cT = -6.53227e-8 * T_K**2
#     C_eff = C + cT
#     return A + (B + bT) / (x**2 - C_eff**2) + E / (x**2 - F**2) + D * x**2


# # Scan Sellmeier over a wide range to find unphysical regions
# refractive_index_scanspace = np.linspace(0.01, 8.0, 100_000)  # µm

# vals = sellmeier_validity_check(refractive_index_scanspace)

# # n²=0 crossings: formula returns non-real index
# zero_indices = np.where(np.diff(np.sign(vals)))[0]
# crossings_zero_nm = refractive_index_scanspace[zero_indices] * 1e3

# # n²=1 crossings: n drops below vacuum
# one_indices = np.where(np.diff(np.sign(vals - 1)))[0]
# crossings_one_nm = refractive_index_scanspace[one_indices] * 1e3

# print(f"n²=0 crossings: {crossings_zero_nm} nm")
# print(f"n²=1 crossings: {crossings_one_nm} nm")
# print(
#     f"Valid window (n² > 1): {crossings_one_nm[1]:.0f} nm  -->  {crossings_one_nm[2]:.0f} nm"
# )


# # LAMBDA_S_MIN is derived from the validity limit of the Sellmeier equation (at which wavelengths does the Sellmeier eq. break down)
# # (n² > 1 up to ~7384 nm at 80°C).
# # Energy conservation links signal and idler:
# # λ_i = λ_p · λ_s / (λ_s − λ_p)  ≤  LAMBDA_I_MAX
# # ==> λ_s ≥ λ_p · LAMBDA_I_MAX / (LAMBDA_I_MAX − λ_p)

# endregion


# region ------- Core physics -------


def get_lambda_i(lambda_p: float, lambda_s: float) -> float:
    """
    Derive the idler wavelength from energy conservation (1/λ_p = 1/λ_s + 1/λ_i).
    λ_p < λ_s <= λ_i
    Args:
        lambda_p: Pump wavelength [m].
        lambda_s: Signal wavelength [m].

    Returns:
        Idler wavelength [m].
        Note: negative or very large values indicate an unphysical point
        (signal shorter than pump).
    """
    return (lambda_p * lambda_s) / (lambda_s - lambda_p)


def k_norm(wavelength: float, ref_index: float) -> float:
    """
    Wave-vector magnitude k = 2π * n(λ, T) / λ inside a medium.

    Args:
        wavelength: Free-space wavelength [m].
        ref_index:  Refractive index from Sellmeir eq. (dimensionless).

    Returns:
        Wave-vector magnitude [rad/m].
    """
    mag: float = (2 * np.pi * ref_index) / wavelength
    if mag < 0:
        raise ValueError("Negative wavevector")
    return mag


def sellmeier(wavelength_m: float, T: float) -> float:
    """
    Temperature-dependent Sellmeier equation for 1mol% MgO-doped nearly stoichiometric lithium tantalate.

    Args:
        wavelength_m: Free-space wavelength [m].
        T:            Crystal temperature [C].

    Returns:
        refractive index n_e.

    """

    lam = wavelength_m * 1e6
    A, B, C, D, E, F = 4.54773, 0.0774167, 0.22025, -0.0226143, 2.39494, 7.45352
    T_K = T + 273.15
    bT = 4.23526e-8 * T_K**2
    cT = -6.53227e-8 * T_K**2
    C_eff = C + cT
    n_sq = A + (B + bT) / (lam**2 - C_eff**2) + E / (lam**2 - F**2) + D * lam**2
    return np.sqrt(n_sq)


def wave_vectors(
    lambda_p: float, lambda_s: float, lambda_i: float, T: float
) -> tuple[float, float, float]:
    """
    Compute wave-vector magnitudes for pump, signal, and idler.

    Args:
        lambda_p: Pump wavelength [m].
        lambda_s: Signal wavelength [m].
        lambda_i: Idler wavelength [m].
        T:        Crystal temperature [°C].

    Returns:
        Tuple (kp, ks, ki) of wave-vector norms [rad/m].
    """
    return (
        k_norm(lambda_p, sellmeier(lambda_p, T)),
        k_norm(lambda_s, sellmeier(lambda_s, T)),
        k_norm(lambda_i, sellmeier(lambda_i, T)),
    )


# endregion


# region ------- Phase-matching residual functions f(...) = 0 -------


def phase_mismatch(theta_s: float, ks: float, ki: float, kp: float, G: float) -> float:
    """
    QPM residual for a given signal emission angle.

    The idler angle θ_i is derived from transverse momentum conservation:
        k_s * sin θ_s = k_i * sin θ_i ==>  θ_i = arcsin((k_s/k_i) * sin θ_s)
        (Since θ relative to optical axis and we assume that the pump photon travels perfectly along the optical axis)

    Phase matching is satisfied when the residual of the parallell momentum conservation including the reciprocal lattice vector correction term
    equals zero:
        k_s * cos θ_s + k_i * cos θ_i + G - k_p = 0

    Parameter
    ---------
    theta_s: float
        Signal emission angle w.r.t. optical axis [rad].
    ks: float
        Signal wave-vector magnitude [rad/m].
    ki: float
        Idler wave-vector magnitude [rad/m].
    kp: float
        Pump wave-vector magnitude [rad/m].
    G: float
        Reciprocal lattice vector of periodic poling [rad/m].

    Return
    ---------
        Phase-mismatch residual [rad/m]: float"""

    arg = (ks / ki) * np.sin(theta_s)
    if np.abs(arg) > 1:
        return np.nan
    # arg = np.clip(arg, -1, 1) # Use to stabilize numerical instabilities if needed
    theta_i = np.arcsin(arg)
    return ks * np.cos(theta_s) + ki * np.cos(theta_i) + G - kp


def collinear_pm(
    lambda_s: float,
    lambda_p: float = LAMBDA_P,
    T: float = T,
    G: float = G,
    m: int = 1 
) -> float:
    """
    Collinear QPM residual (θ_s = θ_i = 0): k_p - k_s - k_i - G = 0.

    Pass to a root-finder to locate the collinear phase-matching signal wavelength.

    Parameters
    ----------
    lambda_s: float
        Signal wavelength [m] (free variable for the root-finder).
    lambda_p: float
      Pump wavelength [m].
    T: float
        Crystal temperature [°C].
    G: float
        Reciprocal lattice vector [rad/m].

    Returns:
        Wave-vector mismatch [rad/m]; zero at the collinear phase-matching point.
    """
    lambda_i = get_lambda_i(lambda_p, lambda_s)
    kp, ks, ki = wave_vectors(lambda_p, lambda_s, lambda_i, T)
    return kp - ks - ki - m*G


# endregion


# region ------- Main sweep: solve non-collinear phase matching across signal wavelengths -------


def run_sweep(
    lambda_p: float = LAMBDA_P,
    T: float = T,
    G: float = G,
    lambda_s_min: float = LAMBDA_S_MIN,
    lambda_s_max: float = LAMBDA_S_MAX,
    n_sweep: int = N_SWEEP,
) -> list[dict]:
    """
    Sweep signal wavelengths and find the phase-matched emission angles.

    For each signal wavelength:
    1. Derives the idler wavelength from energy conservation.
    2. Validates both wavelengths against the Sellmeier transparency window.
    3. Computes wave vectors via the Sellmeier equation.
    4. Checks whether a root of phase_mismatch exists in [0, θ_max].
    5. Solves with Brent's method and records the result.

    Args:
        lambda_p:     Pump wavelength [m].
        T:            Crystal temperature [°C].
        G:            Reciprocal lattice vector [rad/m].
        lambda_s_min: Start of signal sweep [m].
        lambda_s_max: End of signal sweep [m].
        n_sweep:      Number of wavelength points.

    Returns:
        List of dicts with keys lambda_s, lambda_i, theta_s, theta_i, n_s, n_i
        for every wavelength where phase matching was found.
    """
    results = []
    for ls in np.linspace(lambda_s_min, lambda_s_max, n_sweep):
        li = get_lambda_i(lambda_p, ls)
        kp, ks, ki = wave_vectors(lambda_p, ls, li, T)
        theta_max = (
            np.arcsin(min(1.0, ki / ks)) if ks >= ki else np.pi / 2
        )  # θ_max: largest signal angle for which transverse PM (k_s sinθ_s = k_i sinθ_i) has a real solution
        bracket_high = (
            theta_max * 0.9999
        )  # Pull bracket just inside valid domain so brentq evaluates finite residuals at both endpoints

        # Evaluate endpoints
        f_low = phase_mismatch(0, ks, ki, kp, G)
        f_high = phase_mismatch(bracket_high, ks, ki, kp, G)
        if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
            continue
        try:
            theta_s_sol = brentq(phase_mismatch, 0, bracket_high, args=(ks, ki, kp, G))
            theta_i_sol = np.arcsin((ks / ki) * np.sin(theta_s_sol))
            results.append(
                {
                    "lambda_s": ls,
                    "lambda_i": li,
                    "theta_s": np.degrees(theta_s_sol),
                    "theta_i": np.degrees(theta_i_sol),
                    "n_s": sellmeier(ls, T),
                    "n_i": sellmeier(li, T),
                }
            )
        except ValueError:
            continue
    return results


# endregion


# region ------- Plots -------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

C_SIGNAL = "#1a4f9c"  # deep navy blue
C_IDLER = "#b22222"  # firebrick red
C_ANGULAR = "#5b2c8e"  # deep violet
C_WLCORR = "#b07800"  # dark amber
C_GREEN = "#1a6b3a"  # forest green
C_MARK = "#e8a800"  # gold for collinear markers


def param_box(ax, params, loc="upper right"):
    ha = "right" if "right" in loc else "left"
    x = 0.97 if "right" in loc else 0.03
    y = 0.97 if "upper" in loc else 0.03
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        params,
        transform=ax.transAxes,
        fontsize=9.5,
        va=va,
        ha=ha,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="0.70",
            linewidth=0.7,
            alpha=0.92,
        ),
    )


def plot_results(results: list[dict], ls_collinear: float) -> None:
    """
    Plot signal/idler emission angles, angular correlation, and wavelength
    correlation as separate publication-quality figures.

    Args:
        results:       Output from run_sweep (with collinear point appended).
        ls_collinear:  Collinear phase-matching signal wavelength [m].
    """
    ls_nm = np.array([r["lambda_s"] * 1e9 for r in results])
    li_nm = np.array([r["lambda_i"] * 1e9 for r in results])
    ts_deg = np.array([r["theta_s"] for r in results])
    ti_deg = np.array([r["theta_i"] for r in results])
    li_col_nm = get_lambda_i(LAMBDA_P, ls_collinear) * 1e9

    params = (
        f"$\lambda_p = {LAMBDA_P * 1e9:.0f}$ nm\n"
        f"$\Lambda = {POLING_PERIOD * 1e6:.1f}\;\mu$m\n"
        f"$T = {T:.0f}\,^\circ$C"
    )

    # ── Signal plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ls_nm, ts_deg, color=C_SIGNAL, lw=1.6, zorder=3)
    ax.scatter(
        ls_collinear * 1e9,
        0,
        color=C_MARK,
        edgecolors=C_SIGNAL,
        linewidths=0.9,
        s=55,
        zorder=5,
    )
    ax.annotate(
        rf"Collinear: $\lambda_s = {ls_collinear * 1e9:.1f}$ nm,  $\theta_s = 0$",
        xy=(ls_collinear * 1e9, 0),
        xytext=(ls_collinear * 1e9 - 160, 0.6),
        fontsize=9,
        color="0.2",
        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.7),
    )
    param_box(ax, params, "upper right")
    ax.set_xlabel(r"Signal wavelength $\lambda_s$ (nm)")
    ax.set_ylabel(r"Signal emission angle $\theta_s$ (°)")
    ax.set_title(r"QPM Signal Emission Angle — MgO:LiTaO$_3$")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_signal.png")

    # ── Idler plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(li_nm, ti_deg, color=C_IDLER, lw=1.6, zorder=3)
    ax.scatter(
        li_col_nm, 0, color=C_MARK, edgecolors=C_IDLER, linewidths=0.9, s=55, zorder=5
    )
    ax.annotate(
        rf"Collinear: $\lambda_i = {li_col_nm:.1f}$ nm,  $\theta_i = 0$",
        xy=(li_col_nm, 0),
        xytext=(li_col_nm + 350, 3.8),
        fontsize=9,
        color="0.2",
        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.7),
    )
    param_box(ax, params, "upper left")
    ax.set_xlabel(r"Idler wavelength $\lambda_i$ (nm)")
    ax.set_ylabel(r"Idler emission angle $\theta_i$ (°)")
    ax.set_title(r"QPM Idler Emission Angle — MgO:LiTaO$_3$")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_idler.png")

    # ── Angular correlation ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(ts_deg, ti_deg, color=C_ANGULAR, lw=1.6, zorder=3)
    ax.scatter(
        0,
        0,
        color=C_MARK,
        edgecolors=C_ANGULAR,
        linewidths=0.9,
        s=55,
        zorder=5,
        label=r"Collinear: $\theta_s = \theta_i = 0°$",
    )
    ax.scatter(
        ts_deg[0],
        ti_deg[0],
        s=45,
        facecolors="none",
        edgecolors=C_ANGULAR,
        linewidths=1.0,
        zorder=5,
    )
    ax.annotate(
        r"Sellmeier cutoff" + f"\n($\lambda_i = {LAMBDA_I_MAX * 1e9:.0f}$ nm)",
        xy=(ts_deg[0], ti_deg[0]),
        xytext=(ts_deg[0] + 0.5, ti_deg[0] + 4.5),
        fontsize=8.5,
        color="0.3",
        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.7),
    )
    param_box(ax, params, "upper right")
    ax.set_xlabel(r"Signal emission angle $\theta_s$ (°)")
    ax.set_ylabel(r"Idler emission angle $\theta_i$ (°)")
    ax.set_title(r"QPM Angular Correlation")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_angular_correlation.png")

    # ── Wavelength correlation ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(ls_nm, li_nm, color=C_WLCORR, lw=1.6, zorder=3)
    ax.scatter(
        ls_collinear * 1e9,
        li_col_nm,
        color=C_MARK,
        edgecolors=C_WLCORR,
        linewidths=0.9,
        s=55,
        zorder=5,
        label=f"Collinear: $\lambda_s={ls_collinear * 1e9:.1f}$ nm, $\lambda_i={li_col_nm:.1f}$ nm",
    )
    ax.set_xlabel(r"Signal wavelength $\lambda_s$ (nm)")
    ax.set_ylabel(r"Idler wavelength $\lambda_i$ (nm)")
    ax.set_title(r"QPM Wavelength Correlation")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.03,
        0.30,
        params,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="0.70",
            linewidth=0.7,
            alpha=0.92,
        ),
    )
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_wavelength_correlation.png")


def plot_refractive_index(
    lambda_min: float = LAMBDA_S_MIN,
    lambda_max: float = LAMBDA_I_MAX,
    T: float = T,
    n_points: int = 5000,
) -> None:
    """
    Plot the Sellmeier refractive index over the transparency window.

    Marks the pump wavelength and the degenerate wavelength as reference points.

    Args:
        lambda_min: Start of wavelength sweep [m].
        lambda_max: End of wavelength sweep [m].
        T:          Crystal temperature [°C].
        n_points:   Number of sample points.
    """
    lam_sweep = np.linspace(lambda_min, lambda_max, n_points)
    n_vals = np.array([sellmeier(lam, T) for lam in lam_sweep])
    n_pump = sellmeier(LAMBDA_P, T)
    lam_deg = 2 * LAMBDA_P
    n_deg = sellmeier(lam_deg, T)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        lam_sweep * 1e9,
        n_vals,
        color=C_GREEN,
        lw=1.6,
        zorder=3,
        label=r"$n_e(\lambda)$ — Sellmeier",
    )
    ax.axvline(LAMBDA_P * 1e9, color=C_SIGNAL, lw=0.8, ls=":", zorder=2)
    ax.axvline(lam_deg * 1e9, color=C_IDLER, lw=0.8, ls=":", zorder=2)
    ax.scatter(
        [LAMBDA_P * 1e9],
        [n_pump],
        color=C_SIGNAL,
        edgecolors="white",
        linewidths=1.0,
        s=60,
        zorder=5,
        label=f"$\lambda_p = {LAMBDA_P * 1e9:.0f}$ nm,  $n = {n_pump:.4f}$",
    )
    ax.scatter(
        [lam_deg * 1e9],
        [n_deg],
        color=C_IDLER,
        edgecolors="white",
        linewidths=1.0,
        s=60,
        zorder=5,
        label=f"$\lambda_{{\mathrm{{deg}}}} = {lam_deg * 1e9:.0f}$ nm,  $n = {n_deg:.4f}$",
    )
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax.set_ylabel(r"Refractive index $n_e(\lambda)$")
    ax.set_title(
        r"Sellmeier Refractive Index — 1 mol% MgO:LiTaO$_3$"
        + f"  ($T = {T:.0f}\,^\circ$C)"
    )
    ax.set_xlim(lam_sweep[0] * 1e9 - 50, lam_sweep[-1] * 1e9 + 50)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_refractive_index.png")


def plot_refractive_index_with_angles(
    lambda_min: float = LAMBDA_S_MIN,
    lambda_max: float = LAMBDA_I_MAX,
    T: float = T,
    n_points: int = 5000,
    results: list[dict] = None,
    ls_collinear: float = None,
) -> None:
    """
    Overlay signal and idler emission angles on the Sellmeier refractive
    index curve.  Refractive index uses the left y-axis; emission angles
    share the right y-axis.

    Args:
        lambda_min:    Start of wavelength sweep [m].
        lambda_max:    End of wavelength sweep [m].
        T:             Crystal temperature [°C].
        n_points:      Number of Sellmeier sample points.
        results:       Output from run_sweep (with collinear point appended).
        ls_collinear:  Collinear phase-matching signal wavelength [m].
    """
    # ── Sellmeier curve ──────────────────────────────────────────────────
    lam_sweep = np.linspace(lambda_min, lambda_max, n_points)
    n_vals = np.array([sellmeier(lam, T) for lam in lam_sweep])

    # ── Angle data from sweep ────────────────────────────────────────────
    ls_nm = np.array([r["lambda_s"] * 1e9 for r in results])
    li_nm = np.array([r["lambda_i"] * 1e9 for r in results])
    ts_deg = np.array([r["theta_s"] for r in results])
    ti_deg = np.array([r["theta_i"] for r in results])

    li_col_nm = get_lambda_i(LAMBDA_P, ls_collinear) * 1e9

    params = (
        f"$\\lambda_p = {LAMBDA_P * 1e9:.0f}$ nm\n"
        f"$\\Lambda = {POLING_PERIOD * 1e6:.1f}\\;\\mu$m\n"
        f"$T = {T:.0f}\\,^\\circ$C"
    )

    # ── Figure with twin y-axes ──────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax2 = ax1.twinx()

    # ── Left axis: Sellmeier ─────────────────────────────────────────────
    ln_n = ax1.plot(
        lam_sweep * 1e9,
        n_vals,
        color=C_GREEN,
        lw=1.6,
        zorder=3,
        label=r"$n_e(\lambda)$ — Sellmeier",
    )
    ax1.set_xlabel(r"Wavelength $\lambda$ (nm)")
    ax1.set_ylabel(r"Refractive index $n_e(\lambda)$", color=C_GREEN)
    ax1.tick_params(axis="y", labelcolor=C_GREEN)

    # ── Right axis: emission angles ──────────────────────────────────────
    ln_s = ax2.plot(
        ls_nm,
        ts_deg,
        color=C_SIGNAL,
        lw=1.6,
        zorder=4,
        label=r"Signal $\theta_s(\lambda_s)$",
    )
    ln_i = ax2.plot(
        li_nm,
        ti_deg,
        color=C_IDLER,
        lw=1.6,
        zorder=4,
        label=r"Idler $\theta_i(\lambda_i)$",
    )

    # Collinear markers on the angle axis
    ax2.scatter(
        ls_collinear * 1e9,
        0,
        color=C_MARK,
        edgecolors=C_SIGNAL,
        linewidths=0.9,
        s=55,
        zorder=6,
    )
    ax2.scatter(
        li_col_nm,
        0,
        color=C_MARK,
        edgecolors=C_IDLER,
        linewidths=0.9,
        s=55,
        zorder=6,
    )

    ax2.set_ylabel(r"Emission angle (°)", color="0.25")
    ax2.tick_params(axis="y", labelcolor="0.25")

    # Minor ticks on right axis (rcParams only sets the left by default)
    ax2.yaxis.set_minor_locator(plt.AutoLocator().__class__())
    from matplotlib.ticker import AutoMinorLocator

    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(
        axis="y", which="minor", right=True, width=0.5, length=2, direction="in"
    )
    ax2.tick_params(
        axis="y", which="major", right=True, width=0.8, length=4, direction="in"
    )

    # ── Pump wavelength reference line ───────────────────────────────────
    ax1.axvline(LAMBDA_P * 1e9, color=C_SIGNAL, lw=0.7, ls=":", zorder=2)
    ax1.axvline(li_col_nm, color=C_IDLER, lw=0.7, ls=":", zorder=2)

    # ── Legend (combine all line handles) ────────────────────────────────
    lines = ln_n + ln_s + ln_i
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", fontsize=9)

    # ── Parameter box ────────────────────────────────────────────────────
    param_box(ax1, params, loc="lower right")

    ax1.set_title(
        r"Sellmeier Index \& QPM Emission Angles — MgO:LiTaO$_3$"
        + f"  ($T = {T:.0f}\\,^\\circ$C)"
    )
    ax1.set_xlim(lam_sweep[0] * 1e9 - 50, lam_sweep[-1] * 1e9 + 50)
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_overlay.png")


# endregion


# region ------- Energy conservation verification -------


def verify_energy_conservation(
    lambda_p: float,
    lambda_s: float,
    lambda_i: float,
    tol: float = 1e-6,
    verbose: bool = True,
) -> tuple[bool, float]:
    """
    Verify that signal and idler wavelengths satisfy energy conservation
    w.r.t. the pump: 1/λ_p = 1/λ_s + 1/λ_i.

    Args:
        lambda_p: Pump wavelength [m].
        lambda_s: Signal wavelength [m].
        lambda_i: Idler wavelength [m].
        tol:      Fractional tolerance for pass/fail (default 1e-6).
        verbose:  If True, print a formatted report.

    Returns:
        Tuple (passed, fractional_residual).
    """
    inv_lp = 1.0 / lambda_p
    inv_ls = 1.0 / lambda_s
    inv_li = 1.0 / lambda_i

    residual_inv_m = abs(inv_lp - inv_ls - inv_li)
    fractional = residual_inv_m / inv_lp

    freq_p = C0 / lambda_p
    freq_s = C0 / lambda_s
    freq_i = C0 / lambda_i
    freq_residual_hz = abs(freq_p - freq_s - freq_i)

    passed = fractional < tol

    if verbose:
        status = "PASS" if passed else "FAIL"
        print("─" * 52)
        print(f"  Energy Conservation Check  [{status}]")
        print("─" * 52)
        print(f"  λ_p = {lambda_p * 1e9:10.4f} nm   f_p = {freq_p * 1e-12:10.4f} THz")
        print(f"  λ_s = {lambda_s * 1e9:10.4f} nm   f_s = {freq_s * 1e-12:10.4f} THz")
        print(f"  λ_i = {lambda_i * 1e9:10.4f} nm   f_i = {freq_i * 1e-12:10.4f} THz")
        print(f"  f_s + f_i     = {(freq_s + freq_i) * 1e-12:.4f} THz")
        print(f"  Residual |Δf| = {freq_residual_hz:.4e} Hz")
        print(f"  Fractional Δ  = {fractional:.4e}  (tol = {tol:.0e})")
        print("─" * 52)

    return passed, fractional


def verify_energy_conservation_sweep(
    results: list[dict],
    lambda_p: float = LAMBDA_P,
    tol: float = 1e-6,
) -> None:
    """
    Run energy conservation verification across all results from run_sweep()
    and print a summary.

    Args:
        results:  Output from run_sweep().
        lambda_p: Pump wavelength [m].
        tol:      Fractional tolerance for pass/fail (default 1e-6).
    """
    n_total = len(results)
    n_passed = 0
    max_frac = 0.0
    worst = None

    for r in results:
        passed, frac = verify_energy_conservation(
            lambda_p, r["lambda_s"], r["lambda_i"], tol=tol, verbose=False
        )
        if passed:
            n_passed += 1
        if frac > max_frac:
            max_frac = frac
            worst = r

    print("=" * 52)
    print("  Energy Conservation — Sweep Summary")
    print("=" * 52)
    print(f"  Points checked : {n_total}")
    print(f"  Passed         : {n_passed} / {n_total}")
    print(f"  Tolerance      : {tol:.0e} (fractional)")
    print(f"  Max Δ(1/λ) / (1/λ_p) : {max_frac:.4e}")
    if worst:
        print(
            f"  Worst point    : λ_s = {worst['lambda_s'] * 1e9:.2f} nm, "
            f"λ_i = {worst['lambda_i'] * 1e9:.2f} nm"
        )
    print("=" * 52)


# endregion


# region ------- Main, includes the collinear calculation -------


def main() -> None:
    """Run the QPM sweep, print a sanity-check table, and show the plots."""
    results = run_sweep()

    # Append the collinear point (θ_s = θ_i = 0) which the sweep misses
    # because the brentq bracket collapses to zero width there
    ls_collinear = brentq(
        collinear_pm, LAMBDA_S_MIN, LAMBDA_S_MAX, args=(LAMBDA_P, T, G)
    )
    li_collinear = get_lambda_i(LAMBDA_P, ls_collinear)
    kp, ks, ki = wave_vectors(LAMBDA_P, ls_collinear, li_collinear, T)
    results.append(
        {
            "lambda_s": ls_collinear,
            "lambda_i": li_collinear,
            "theta_s": 0.0,
            "theta_i": 0.0,
            "n_s": sellmeier(ls_collinear, T),
            "n_i": sellmeier(li_collinear, T),
        }
    )
    results.sort(key=lambda r: r["lambda_s"])  # keep list ordered by λ_s

    # Find the phase-matched pair at θ_s = 1°
    TARGET_THETA_S = 1.0  # degrees
    closest = min(results, key=lambda r: abs(r["theta_s"] - TARGET_THETA_S))
    print(
        f"\nAt θ_s = {TARGET_THETA_S}°:  "
        f"λ_s = {closest['lambda_s'] * 1e9:.2f} nm,  "
        f"λ_i = {closest['lambda_i'] * 1e9:.2f} nm,  "
        f"θ_i = {closest['theta_i']:.3f}°"
    )

    step = max(1, len(results) // 8)
    print(
        f"{'λ_s (nm)':>10}  {'λ_i (nm)':>10}  {'n_s':>7}  {'n_i':>7}  {'θ_s (°)':>9}  {'θ_i (°)':>9}"
    )
    print("─" * 64)
    for r in results[::step]:
        print(
            f"{r['lambda_s'] * 1e9:10.1f}  "
            f"{r['lambda_i'] * 1e9:10.1f}  "
            f"{r['n_s']:7.5f}  "
            f"{r['n_i']:7.5f}  "
            f"{r['theta_s']:9.3f}  "
            f"{r['theta_i']:9.3f}"
        )

    ls_lo, ls_hi = LAMBDA_S_MIN, LAMBDA_S_MAX
    ls_collinear = brentq(collinear_pm, ls_lo, ls_hi, args=(LAMBDA_P, T, G))
    li_collinear = get_lambda_i(LAMBDA_P, ls_collinear)
    print(
        f"\nCollinear phase-matching wavelength: "
        f"signal: {ls_collinear * 1e9:.2f} nm, "
        f"idler: {li_collinear * 1e9:.2f} nm"
    )
    plot_refractive_index_with_angles(results=results, ls_collinear=ls_collinear)
    # endregion


# endregion


if __name__ == "__main__":
    main()

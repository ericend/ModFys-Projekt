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
N_SWEEP: int = 1000  # number of sweeps for Brents method

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
    Wave-vector magnitude k = 2π * n / λ inside a medium.

    Args:
        wavelength: Free-space wavelength [m].
        ref_index:  Refractive index (dimensionless).

    Returns:
        Wave-vector magnitude [rad/m].
    """
    mag: float = (2 * np.pi * ref_index) / wavelength
    if mag < 0:
        raise ValueError("Invalid wavevector")
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
        k_s * sin θ_s = k_i * sin θ_i  ==>  θ_i = arcsin((k_s/k_i) * sin θ_s)

    Phase matching is satisfied when the residual equals zero:
        k_s * cos θ_s + k_i * cos θ_i + G - k_p = 0

    Args:
        theta_s: Signal emission angle w.r.t. optical axis [rad].
        ks:      Signal wave-vector magnitude [rad/m].
        ki:      Idler wave-vector magnitude [rad/m].
        kp:      Pump wave-vector magnitude [rad/m].
        G:       Reciprocal lattice vector of periodic poling [rad/m].

    Returns:
        Phase-mismatch residual [rad/m]"""

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
) -> float:
    """
    Collinear QPM residual (θ_s = θ_i = 0): k_p - k_s - k_i - G = 0.

    Pass to a root-finder to locate the collinear phase-matching signal wavelength.

    Args:
        lambda_s: Signal wavelength [m] (free variable for the root-finder).
        lambda_p: Pump wavelength [m].
        T:        Crystal temperature [°C].
        G:        Reciprocal lattice vector [rad/m].

    Returns:
        Wave-vector mismatch [rad/m]; zero at the collinear phase-matching point.
    """
    lambda_i = get_lambda_i(lambda_p, lambda_s)
    kp, ks, ki = wave_vectors(lambda_p, lambda_s, lambda_i, T)
    return kp - ks - ki - G


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
# [PLOT_START]

plt.style.use("seaborn-v0_8-whitegrid")

SMALL, MED, BIG = 11, 13, 14
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": MED,
        "axes.titlesize": BIG,
        "axes.labelsize": MED,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "figure.dpi": 300,
    }
)

BLUE = "#002fff"
ORANGE = "#c00000"


def plot_results(results: list[dict], ls_collinear: float) -> None:
    """
    Plot signal and idler emission angles vs wavelength as separate figures.

    Marks the collinear phase-matching point (θ = 0) on both plots.

    Args:
        results:       Output from run_sweep.
        ls_collinear:  Collinear phase-matching signal wavelength [m].
    """
    ls_nm = np.array([r["lambda_s"] * 1e9 for r in results])
    li_nm = np.array([r["lambda_i"] * 1e9 for r in results])
    ts_deg = np.array([r["theta_s"] for r in results])
    ti_deg = np.array([r["theta_i"] for r in results])
    li_col_nm = get_lambda_i(LAMBDA_P, ls_collinear) * 1e9

    params = (
        f"$\\lambda_p$ = {LAMBDA_P * 1e9:.0f} nm\n"
        f"$\\Lambda$ = {POLING_PERIOD * 1e6:.1f} µm\n"
        f"$T$ = {T:.0f} °C"
    )

    # ── Signal plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(ls_nm, ts_deg, color=BLUE, lw=1.8, zorder=3)
    # ax.fill_between(ls_nm, ts_deg, alpha=0.08, color=BLUE, zorder=2)
    ax.scatter(
        ls_collinear * 1e9,
        0,
        color=BLUE,
        edgecolors="white",
        linewidths=1.4,
        s=80,
        zorder=1,
        label=f"Collinear:  $\\lambda_s$ = {ls_collinear * 1e9:.1f} nm,  $\\theta_s$ = 0°",
    )
    ax.annotate(
        f"{ls_collinear * 1e9:.1f} nm",
        xy=(ls_collinear * 1e9, 0),
        xytext=(ls_collinear * 1e9 - 55, 0.55),
        fontsize=SMALL,
        color="black",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
    )
    ax.text(
        0.97,
        0.97,
        params,
        transform=ax.transAxes,
        fontsize=SMALL,
        color="0",
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.82",
            linewidth=0.8,
        ),
    )
    ax.set_xlabel("Signal wavelength  $\\lambda_s$  (nm)")
    ax.set_ylabel("Signal emission angle  $\\theta_s (°)$")
    ax.set_title("QPM Signal Emission Angle")
    ax.legend()
    ax.grid(alpha=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_signal.png", dpi=300, bbox_inches="tight")

    # plt.show()

    # ── Idler plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(li_nm, ti_deg, color=ORANGE, lw=1.8, zorder=3)
    # ax.fill_between(li_nm, ti_deg, alpha=0.08, color=ORANGE, zorder=2)
    ax.scatter(
        li_col_nm,
        0,
        color=ORANGE,
        edgecolors="white",
        linewidths=1.4,
        s=80,
        zorder=1,
        label=f"Collinear:  $\\lambda_i$ = {li_col_nm:.1f} nm,  $\\theta_i$ = 0°",
    )
    ax.annotate(
        f"{li_col_nm:.1f} nm",
        xy=(li_col_nm, 0),
        xytext=(li_col_nm + 200, 3.5),
        fontsize=SMALL,
        color="black",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0),
    )
    ax.text(
        0.03,
        0.97,
        params,
        transform=ax.transAxes,
        fontsize=SMALL,
        color="0",
        va="top",
        ha="left",  # ← was "right"
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.82",
            linewidth=0.8,
        ),
    )
    ax.set_xlabel("Idler wavelength  $\\lambda_i$  (nm)")
    ax.set_ylabel("Idler emission angle  $\\theta_i (°)$")
    ax.set_title("QPM Idler Emission Angle")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_idler.png", dpi=300, bbox_inches="tight")

    # plt.show()


def plot_refractive_index(
    lambda_min: float = LAMBDA_S_MIN,
    lambda_max: float = LAMBDA_I_MAX,
    T: float = T,
    n_points: int = 5000,
) -> None:
    """
    Plot the Sellmeier refractive index over a wavelength window.

    Marks the pump wavelength and the degenerate (collinear) signal/idler
    wavelength as reference points.

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

    GREEN = "#1a8c3f"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        lam_sweep * 1e9,
        n_vals,
        color=GREEN,
        lw=1.8,
        zorder=3,
        label="$n(\\lambda)$ — Sellmeier",
    )

    ax.axvline(LAMBDA_P * 1e9, color=BLUE, lw=0.9, ls=":", zorder=2)
    ax.axvline(lam_deg * 1e9, color=ORANGE, lw=0.9, ls=":", zorder=2)

    ax.scatter(
        [LAMBDA_P * 1e9],
        [n_pump],
        color=BLUE,
        edgecolors="white",
        linewidths=1.4,
        s=70,
        zorder=5,
        label=f"$\\lambda_p$ = {LAMBDA_P * 1e9:.0f} nm,  $n$ = {n_pump:.4f}",
    )
    ax.scatter(
        [lam_deg * 1e9],
        [n_deg],
        color=ORANGE,
        edgecolors="white",
        linewidths=1.4,
        s=70,
        zorder=5,
        label=f"$\\lambda_{{\\rm deg}}$ = {lam_deg * 1e9:.0f} nm,  $n$ = {n_deg:.4f}",
    )

    ax.text(
        LAMBDA_P * 1e9 + 60,
        n_pump - 0.04,
        f"{LAMBDA_P * 1e9:.0f} nm",
        fontsize=SMALL - 1,
        color=BLUE,
        va="top",
    )
    ax.text(
        lam_deg * 1e9 + 60,
        n_deg - 0.04,
        f"{lam_deg * 1e9:.0f} nm",
        fontsize=SMALL - 1,
        color=ORANGE,
        va="top",
    )

    ax.set_xlabel("Wavelength  $\\lambda$  (nm)")
    ax.set_ylabel("Refractive index  $n(\\lambda)$")
    ax.set_title(
        f"Sellmeier Refractive Index — 1 mol% MgO:LiTaO$_3$  ($T$ = {T:.0f} °C)"
    )
    ax.legend(loc="upper right", fontsize=SMALL - 1)
    ax.set_xlim(lam_sweep[0] * 1e9 - 50, lam_sweep[-1] * 1e9 + 50)
    ax.grid(alpha=0.4, linestyle="--")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "qpm_refractive_index.png", dpi=300, bbox_inches="tight")


# [PLOT_END]
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


# region ------- Main -------


def main() -> None:
    """Run the QPM sweep, print a sanity-check table, and show the plots."""
    results = run_sweep()

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

    plot_results(results, ls_collinear)
    # plt.show()

    # region ------ uncomment this to run a energy conservation check & plot the refractive index ------
    # print("\n--- Spot-check at collinear phase-matching point ---")
    # verify_energy_conservation(LAMBDA_P, ls_collinear, li_collinear)

    # print("\n--- Spot-check at θ_s = 1° ---")
    # verify_energy_conservation(LAMBDA_P, closest["lambda_s"], closest["lambda_i"])

    # print("\n--- Full sweep verification ---")
    # verify_energy_conservation_sweep(results)
    # plot_refractive_index()
    # endregion


# endregion


if __name__ == "__main__":
    main()

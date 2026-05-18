"""
Quasi-phase-matched (QPM) parametric down-conversion in MgO:LiNbO₃.

Sweeps over signal wavelengths and, for each, solves the non-collinear
phase-matching condition to find the signal and idler emission angles.
Also locates the degenerate collinear phase-matching wavelength.

Physical process
----------------
A pump photon (λ_p) spontaneously splits into a signal (λ_s) and idler (λ_i)
photon pair subject to:
  - Energy conservation : 1/λ_p = 1/λ_s + 1/λ_i
  - Momentum conservation: k_p = k_s * cos θ_s + k_i * cos θ_i  (longitudinal)
                           0   = k_s * sin θ_s - k_i * sin θ_i  (transverse)
The periodic poling provides a reciprocal lattice vector G = 2*pi/Λ which gives
the momentum condition: k_p = k_s + k_i + G (quasi-phase-matching).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Physical constants / device parameters
# ---------------------------------------------------------------------------

LAMBDA_P: float = 532e-9  # Pump wavelength [m]
POLING_PERIOD: float = 8.5e-6  # Poling period [m]
G: float = 2 * np.pi / POLING_PERIOD  # Reciprocal lattice vector [rad/m]
T: float = 80.0  # Crystal temperature [°C]

# Signal wavelength sweep range
LAMBDA_S_MIN: float = (
    LAMBDA_P * 1.05
)  # 5 % above pump to stay clear of the singularity when λ_s --> λ_p
LAMBDA_S_MAX: float = 1500e-9  # [m]
N_SWEEP: int = 1000


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------


def get_lambda_i(lambda_p: float, lambda_s: float) -> float:
    """
    Derive the idler wavelength from energy conservation (1/λ_p = 1/λ_s + 1/λ_i).

    Args:
        lambda_p: Pump wavelength [m].
        lambda_s: Signal wavelength [m].

    Returns:
        Idler wavelength [m].
        Note:
        Negative or very large values indicate an
        unphysical point (signal shorter than pump).
    """
    return (lambda_p * lambda_s) / (lambda_s - lambda_p)


def k_norm(wavelength: float, ref_index: float) -> float:
    """
    Wave-vector magnitude k = 2π·n / λ inside a medium.

    Args:
        wavelength: Free-space wavelength [m].
        ref_index:  Refractive index of the medium (dimensionless).

    Returns:
        Wave-vector magnitude [rad/m].
    """
    return (2 * np.pi * ref_index) / wavelength


def sellmeier(wavelength_m: float, T: float) -> float:
    """

    Temperature-dependent Sellmeier equation

    Args:
        wavelength_m: Free-space wavelength [m].
        T:            Crystal temperature [°C].

    Returns:
        refractive index n_e
    """
    lam = wavelength_m * 1e6  # Convert metres --> micrometres

    # Sellmeier coefficients ( Table 1)
    A, B, C, D, E, F = 4.54773, 0.0774167, 0.22025, -0.0226143, 2.39494, 7.45352

    # Temperature-dependent corrections to poles B and C
    T_K = T + 273.15  # Convert to Kelvin
    bT = 4.23526e-8 * T_K**2
    cT = -6.53227e-8 * T_K**2

    n_sq = A + (B + bT) / (lam**2 - (C + cT) ** 2) + E / (lam**2 - F**2) + D * lam**2
    return np.sqrt(n_sq)


def sellmeier_is_valid(wavelength_m: float, T: float) -> bool:
    """Return False if the Sellmeier equation produces an unphysical result."""
    lam = wavelength_m * 1e6
    A, B, C, D, E, F = 4.54773, 0.0774167, 0.22025, -0.0226143, 2.39494, 7.45352
    T_K = T + 273.15
    bT = 4.23526e-8 * T_K**2
    cT = -6.53227e-8 * T_K**2
    n_sq = A + (B + bT) / (lam**2 - (C + cT) ** 2) + E / (lam**2 - F**2) + D * lam**2
    return n_sq > 1  # physically meaningful refractive index


def wave_vectors(
    lambda_p: float, lambda_s: float, lambda_i: float, T: float
) -> tuple[float, float, float]:
    """
    Compute wave-vector magnitudes for pump, signal, and idler.

    Evaluates the Sellmeier equation at each wavelength, then converts to k.

    Args:
        lambda_p: Pump wavelength [m].
        lambda_s: Signal wavelength [m].
        lambda_i: Idler wavelength [m].
        T:        Crystal temperature [°C].

    Returns:
        Tuple (kp, ks, ki) of wave-vector magnitudes [rad/m].
    """
    n_p = sellmeier(lambda_p, T)
    n_s = sellmeier(lambda_s, T)
    n_i = sellmeier(lambda_i, T)

    kp = k_norm(lambda_p, n_p)
    ks = k_norm(lambda_s, n_s)
    ki = k_norm(lambda_i, n_i)

    return kp, ks, ki


# ---------------------------------------------------------------------------
# Phase-matching residual functions (roots passed to brentq)
# ---------------------------------------------------------------------------


def phase_mismatch(theta_s: float, ks: float, ki: float, kp: float, G: float) -> float:
    """
    QPM residual for a given signal emission angle.

    The idler angle θ_i is derived from momentum conservation:
        k_s * sin θ_s = k_i * sin θ_i  ==>  θ_i = arcsin((k_s/k_i) * sin θ_s)

    Phase matching is satisfied when the residual equals zero:
        k_s * cos θ_s + k_i * cos θ_i + G - k_p = 0

    Args:
        theta_s: Signal emission angle w.r.t. pump axis [rad].
        ks:      Signal wave-vector magnitude [rad/m].
        ki:      Idler wave-vector magnitude [rad/m].
        kp:      Pump wave-vector magnitude [rad/m].
        G:       Reciprocal lattice vector of periodic poling [rad/m].

    Returns:
        Phase-mismatch residual [rad/m], or ``nan`` if the implied idler
        angle is non-physical (|sin θ_i| > 1).
    """
    # momentum conservation -> idler angle
    arg = (ks / ki) * np.sin(theta_s)
    if np.abs(arg) > 1:
        # No real idler angle exists for this signal angle
        return np.nan
    theta_i = np.arcsin(arg)

    # momentum residual
    return ks * np.cos(theta_s) + ki * np.cos(theta_i) + G - kp


def collinear_pm(
    lambda_s: float,
    lambda_p: float = LAMBDA_P,
    T: float = T,
    G: float = G,
) -> float:
    """
    Collinear QPM residual: mismatch when all beams propagate along the pump axis.

    Assumes θ_s = θ_i = 0, so the condition reduces to:
        k_p - k_s - k_i - G = 0

    Pass this function to a root-finder to locate the collinear phase-matching
    signal wavelength.

    Args:
        lambda_s: Signal wavelength [m] (the free variable for the root-finder).
        lambda_p: Pump wavelength [m].
        T:        Crystal temperature [°C].
        G:        Reciprocal lattice vector [rad/m].

    Returns:
        Wave-vector mismatch [rad/m]; zero at the collinear phase-matching point.
    """
    lambda_i = get_lambda_i(lambda_p, lambda_s)
    kp, ks, ki = wave_vectors(lambda_p, lambda_s, lambda_i, T)
    return kp - ks - ki - G


# ---------------------------------------------------------------------------
# Main sweep: solve non-collinear phase matching across signal wavelengths
# ---------------------------------------------------------------------------


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

    For each signal wavelength the function:
    1. Derives the idler wavelength from energy conservation.
    2. Computes wave vectors via the Sellmeier equation. |k| = 2 * pi * n(w) / lambda
    3. Checks whether a root of ``phase_mismatch`` exists in [0, θ_max].
    4. Solves with Brent's method and records the result.

    Args:
        lambda_p:    Pump wavelength [m].
        T:           Crystal temperature [°C].
        G:           Reciprocal lattice vector [rad/m].
        lambda_s_min: Start of signal sweep [m].
        lambda_s_max: End of signal sweep [m].
        n_sweep:     Number of wavelength points.

    Returns:
        List of dicts with keys ``lambda_s``, ``lambda_i``, ``theta_s``,
        ``theta_i`` for every wavelength where phase matching was found.
    """
    lambda_s_sweep = np.linspace(lambda_s_min, lambda_s_max, n_sweep)
    results = []

    for ls in lambda_s_sweep:
        li = get_lambda_i(lambda_p, ls)

        # Skip unphysical idler wavelengths (negative or beyond mid-IR)
        if li <= 0 or li > 5e-6:
            continue

        kp, ks, ki = wave_vectors(lambda_p, ls, li, T)

        # Upper bound on θ_s: transverse conservation breaks down beyond
        # arcsin(k_i / k_s) when k_s > k_i (signal has shorter λ than idler)
        theta_max = np.arcsin(min(1.0, ki / ks)) if ks >= ki else np.pi / 2
        bracket_high = theta_max * 0.9999  # Stay just inside the valid range

        # Evaluate residual at bracket edges
        f_low = phase_mismatch(0, ks, ki, kp, G)
        f_high = phase_mismatch(bracket_high, ks, ki, kp, G)

        # Skip if either endpoint is non-physical or no sign change (no root)
        if np.isnan(f_low) or np.isnan(f_high):
            continue
        if f_low * f_high > 0:
            continue

        try:
            # Brent's method: guaranteed convergence given a sign change
            theta_s_sol = brentq(phase_mismatch, 0, bracket_high, args=(ks, ki, kp, G))
            # Recover idler angle from transverse momentum conservation
            theta_i_sol = np.arcsin((ks / ki) * np.sin(theta_s_sol))

            results.append(
                {
                    "lambda_s": ls,
                    "lambda_i": li,
                    "theta_s": np.degrees(theta_s_sol),
                    "theta_i": np.degrees(theta_i_sol),
                    "n_s": sellmeier(ls, T),  # ← add
                    "n_i": sellmeier(li, T),  # ← add
                }
            )
        except ValueError:
            # brentq can still fail on edge cases; skip gracefully
            continue

    return results


# [PLOT_START]


def plot_results(results: list[dict], ls_collinear: float) -> None:
    """
    Plot signal and idler emission angles vs wavelength.

    Marks the collinear phase-matching point (θ = 0) on the signal plot.

    Args:
        results:       Output from ``run_sweep``.
        ls_collinear:  Collinear phase-matching signal wavelength [m].
    """
    lambda_s_nm = [r["lambda_s"] * 1e9 for r in results]
    lambda_i_nm = [r["lambda_i"] * 1e9 for r in results]
    theta_s_deg = [r["theta_s"] for r in results]
    theta_i_deg = [r["theta_i"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Signal angle curve
    ax1.plot(lambda_s_nm, theta_s_deg)
    ax1.scatter(
        ls_collinear * 1e9,
        0,
        label=f"Collinear: {ls_collinear * 1e9:.1f} nm",
        zorder=5,
    )
    ax1.set_xlabel("λ_s (nm)")
    ax1.set_ylabel("θ_s (degrees)")
    ax1.set_title("Signal emission angle vs wavelength")
    ax1.legend()
    ax1.grid(True)

    # Idler angle curve
    ax2.plot(lambda_i_nm, theta_i_deg, color="orange")

    ax2.scatter(
        get_lambda_i(LAMBDA_P, ls_collinear) * 1e9,  # = 1515.57 nm
        0,
        label=f"Collinear: {get_lambda_i(LAMBDA_P, ls_collinear) * 1e9:.1f} nm",
        zorder=5,
    )
    ax2.legend()
    ax2.set_xlabel("λ_i (nm)")
    ax2.set_ylabel("θ_i (degrees)")
    ax2.set_title("Idler emission angle vs wavelength")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# [PLOT_END]


def main() -> None:
    """Run the QPM sweep, print a sanity-check table, and show the plots."""

    results = run_sweep()

    # Print ~8 evenly-spaced rows; step size adapts to however many solutions were found
    step = max(1, len(results) // 8)
    print(
        rf"{'λ_s (nm)':>10}  {'λ_i (nm)':>10}  {'n_s':>7}  {'n_i':>7}  {'θ_s (°)':>9}  {'θ_i (°)':>9}"
    )
    print("─" * 64)
    for r in results[::step]:
        print(
            rf"{r['lambda_s'] * 1e9:10.1f}  "
            rf"{r['lambda_i'] * 1e9:10.1f}  "
            rf"{r['n_s']:7.5f}  "
            rf"{r['n_i']:7.5f}  "
            rf"{r['theta_s']:9.3f}  "
            rf"{r['theta_i']:9.3f}"
        )

    # Find the collinear phase-matching wavelength (θ_s = θ_i = 0)
    ls_collinear = brentq(
        collinear_pm,
        LAMBDA_S_MIN,
        LAMBDA_S_MAX,
        args=(LAMBDA_P, T, G),
    )
    print(
        f"\nCollinear phase-matching wavelength: signal: {ls_collinear * 1e9:.2f} nm, idler: {get_lambda_i(LAMBDA_P, ls_collinear) * 1e9:.2f} nm"
    )

    plot_results(results, ls_collinear)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

lambda_p: float = 532e-9  # pump wavelength in meters
lambda_s_sweep = np.linspace(600e-9, 1500e-9, 1000)  # signal wavelengths in meters

T = 80.0  # temperature in Celsius
G: float = 2 * np.pi / 8.5e-6  # reciprocal lattice vector


def get_lambda_i(lambda_p: float, lambda_s: float) -> float:
    return (lambda_p * lambda_s) / (lambda_s - lambda_p)


def k_norm(wavelength: float, ref_index: float) -> float:
    return (2 * np.pi * ref_index) / wavelength


def sellmeier(wavelength_m: float, T: float) -> float:
    lam = wavelength_m * 1e6  # convert to micrometers
    A, B, C, D, E, F = 4.54773, 0.0774167, 0.22025, -0.0226143, 2.39494, 7.45352
    bT = 4.23526e-8 * (T + 273.15) ** 2
    cT = -6.53227e-8 * (T + 273.15) ** 2
    n_sq = A + (B + bT) / (lam**2 - (C + cT) ** 2) + E / (lam**2 - F**2) + D * lam**2
    return np.sqrt(n_sq)


def f(theta_s: float, ks: float, ki: float, kp: float, G: float) -> float:
    # check arcsin argument is valid
    arg = (ks / ki) * np.sin(theta_s)
    if np.abs(arg) > 1:
        return np.nan
    theta_i = np.arcsin(arg)
    return ks * np.cos(theta_s) + ki * np.cos(theta_i) + G - kp


results = []

for ls in lambda_s_sweep:
    li = get_lambda_i(lambda_p, ls)

    if li <= 0 or li > 10e-6:  # skip unphysical wavelengths
        continue

    n_p = sellmeier(lambda_p, T)
    n_s = sellmeier(ls, T)
    n_i = sellmeier(li, T)

    kp = k_norm(lambda_p, n_p)
    ks = k_norm(ls, n_s)
    ki = k_norm(li, n_i)

    # check that f changes sign over [0, pi/2] before trying to find root
    f_low = f(0, ks, ki, kp, G)
    theta_max = np.arcsin(min(1.0, ki / ks)) if ks >= ki else np.pi / 2
    bracket_high = theta_max * 0.9999
    f_high = f(bracket_high, ks, ki, kp, G)

    if np.isnan(f_low) or np.isnan(f_high):
        continue
    if f_low * f_high > 0:  # no sign change = no root in this bracket
        continue

    try:
        theta_s_sol = brentq(f, 0, bracket_high, args=(ks, ki, kp, G))
        theta_i_sol = np.arcsin((ks / ki) * np.sin(theta_s_sol))
        results.append(
            {
                "lambda_s": ls,
                "lambda_i": li,
                "theta_s": np.degrees(theta_s_sol),
                "theta_i": np.degrees(theta_i_sol),
            }
        )
    except ValueError:
        continue

# quick sanity check
for r in results[::100]:
    print(
        f"λ_s={r['lambda_s'] * 1e9:.1f} nm, λ_i={r['lambda_i'] * 1e9:.1f} nm, "
        f"θ_s={r['theta_s']:.3f}°, θ_i={r['theta_i']:.3f}°"
    )


lambda_s_nm = [r["lambda_s"] * 1e9 for r in results]
lambda_i_nm = [r["lambda_i"] * 1e9 for r in results]
theta_s_deg = [r["theta_s"] for r in results]
theta_i_deg = [r["theta_i"] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(lambda_s_nm, theta_s_deg)
ax1.set_xlabel("λ_s (nm)")
ax1.set_ylabel("θ_s (degrees)")
ax1.set_title("Signal emission angle vs wavelength")
ax1.grid(True)

ax2.plot(lambda_i_nm, theta_i_deg, color="orange")
ax2.set_xlabel("λ_i (nm)")
ax2.set_ylabel("θ_i (degrees)")
ax2.set_title("Idler emission angle vs wavelength")
ax2.grid(True)


def collinear_pm(ls):
    li = get_lambda_i(lambda_p, ls)

    n_p = sellmeier(lambda_p, T)
    n_s = sellmeier(ls, T)
    n_i = sellmeier(li, T)

    kp = k_norm(lambda_p, n_p)
    ks = k_norm(ls, n_s)
    ki = k_norm(li, n_i)

    return kp - ks - ki - G


ls_collinear = brentq(collinear_pm, 600e-9, 1500e-9)

ax1.scatter(ls_collinear * 1e9, 0, label="Collinear point")
ax1.legend()
plt.tight_layout()
plt.show()

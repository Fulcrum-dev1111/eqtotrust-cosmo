# analyze_wz.py
# Phase 5: Equation of State Analysis
# Computes w(z) for the scalar-field model and compares against LCDM (w=-1).
# Also finds the transition redshift z_t (deceleration -> acceleration).

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coherence_scalar_cosmo import integrate, V

samples = np.load("posterior_samples.npy")
labels = ["V0", "k", "Om", "C0", "dC0", "M"]
medians = np.median(samples, axis=0)

best_V0, best_k, best_Om, best_C0, best_dC0, best_M = medians
print("Using Phase 2 median parameters:")
for i, name in enumerate(labels):
    print(f"  {name:>6s} = {medians[i]:+.6f}")


def compute_wz_and_decel(V0, k, Om, C0, dC0, n_grid=8000):
    params = {"V0": V0, "k": k, "Om": Om, "Or": 0.0, "C0": C0, "dC0": dC0}

    a0 = 1e-4
    y0 = np.array([C0, dC0, 1.0], dtype=float)
    a_grid, y = integrate(a0, 1.0, y0, params, n=n_grid)

    C_arr = y[:, 0]
    Cp_arr = y[:, 1]
    E_arr = y[:, 2]
    z_grid = 1.0 / a_grid - 1.0

    rho_phi = 0.5 * (a_grid ** 2) * (Cp_arr ** 2) + V(C_arr, V0, k)
    p_phi = 0.5 * (a_grid ** 2) * (Cp_arr ** 2) - V(C_arr, V0, k)

    w_phi = np.where(np.abs(rho_phi) > 1e-30, p_phi / rho_phi, -1.0)

    rho_m = Om / a_grid ** 3
    rho_total = rho_m + rho_phi
    p_total = p_phi

    w_eff = np.where(np.abs(rho_total) > 1e-30, p_total / rho_total, -1.0)

    q = 0.5 * (1.0 + 3.0 * w_eff)

    return z_grid, w_phi, w_eff, q, E_arr


z_grid, w_phi, w_eff, q, E_arr = compute_wz_and_decel(
    best_V0, best_k, best_Om, best_C0, best_dC0
)

order = np.argsort(z_grid)
z_grid = z_grid[order]
w_phi = w_phi[order]
w_eff = w_eff[order]
q = q[order]
E_arr = E_arr[order]

mask = z_grid < 3.0
z_plot = z_grid[mask]
w_phi_plot = w_phi[mask]
w_eff_plot = w_eff[mask]
q_plot = q[mask]

sign_changes = np.where(np.diff(np.sign(q_plot)))[0]
if len(sign_changes) > 0:
    i = sign_changes[0]
    z_t = z_plot[i] + (z_plot[i + 1] - z_plot[i]) * (-q_plot[i]) / (q_plot[i + 1] - q_plot[i])
else:
    z_t = None

print("\n" + "=" * 70)
print("PHASE 5: EQUATION OF STATE ANALYSIS")
print("=" * 70)

z_today_idx = np.argmin(np.abs(z_plot))
print(f"\n  w_phi(z=0)  = {w_phi_plot[z_today_idx]:.6f}")
print(f"  w_eff(z=0)  = {w_eff_plot[z_today_idx]:.6f}")
print(f"  q(z=0)      = {q_plot[z_today_idx]:.6f}")

if z_t is not None:
    print(f"\n  Transition redshift z_t = {z_t:.4f}")
    print(f"  (Universe switched from deceleration to acceleration at z_t)")
else:
    print("\n  No transition redshift found in z < 3.0")

z_05_idx = np.argmin(np.abs(z_plot - 0.5))
z_10_idx = np.argmin(np.abs(z_plot - 1.0))
z_20_idx = np.argmin(np.abs(z_plot - 2.0))
print(f"\n  w_phi(z=0.5) = {w_phi_plot[z_05_idx]:.6f}")
print(f"  w_phi(z=1.0) = {w_phi_plot[z_10_idx]:.6f}")
print(f"  w_phi(z=2.0) = {w_phi_plot[z_20_idx]:.6f}")

delta_w_0 = w_phi_plot[z_today_idx] - (-1.0)
delta_w_05 = w_phi_plot[z_05_idx] - (-1.0)
delta_w_10 = w_phi_plot[z_10_idx] - (-1.0)
print(f"\n  Deviation from Lambda (w=-1):")
print(f"    delta_w(z=0)   = {delta_w_0:+.6f}")
print(f"    delta_w(z=0.5) = {delta_w_05:+.6f}")
print(f"    delta_w(z=1.0) = {delta_w_10:+.6f}")

z_t_lcdm = (2.0 * (1.0 - best_Om) / best_Om) ** (1.0 / 3.0) - 1.0
print(f"\n  LCDM transition redshift z_t = {z_t_lcdm:.4f} (for Om={best_Om:.4f})")

print("\n" + "=" * 70)
print("UNCERTAINTY BAND: sampling 200 posterior draws for w(z)")
print("=" * 70)

z_ref = np.linspace(0.001, 2.5, 300)
w_samples = []
zt_samples = []
n_draws = 200
indices = np.random.choice(samples.shape[0], size=n_draws, replace=False)
success = 0
for count, idx in enumerate(indices):
    V0_s, k_s, Om_s, C0_s, dC0_s, M_s = samples[idx]
    try:
        zg, wp, we, qq, _ = compute_wz_and_decel(V0_s, k_s, Om_s, C0_s, dC0_s)
        o = np.argsort(zg)
        zg, wp, qq = zg[o], wp[o], qq[o]
        wp_interp = np.interp(z_ref, zg, wp)
        if np.all(np.isfinite(wp_interp)):
            w_samples.append(wp_interp)
            sc = np.where(np.diff(np.sign(qq)))[0]
            if len(sc) > 0:
                ii = sc[0]
                zt_s = zg[ii] + (zg[ii + 1] - zg[ii]) * (-qq[ii]) / (qq[ii + 1] - qq[ii])
                zt_samples.append(zt_s)
            success += 1
    except Exception:
        pass
    if (count + 1) % 50 == 0:
        print(f"  processed {count + 1}/{n_draws} ({success} successful)")

w_samples = np.array(w_samples)
print(f"  {success} successful w(z) evaluations out of {n_draws} draws")

if len(zt_samples) > 0:
    zt_arr = np.array(zt_samples)
    print(f"\n  Transition redshift z_t = {np.median(zt_arr):.4f} "
          f"[{np.percentile(zt_arr, 16):.4f}, {np.percentile(zt_arr, 84):.4f}]")

w_median = np.median(w_samples, axis=0)
w_lo = np.percentile(w_samples, 16, axis=0)
w_hi = np.percentile(w_samples, 84, axis=0)
w_lo2 = np.percentile(w_samples, 2.5, axis=0)
w_hi2 = np.percentile(w_samples, 97.5, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

ax1 = axes[0]
ax1.fill_between(z_ref, w_lo2, w_hi2, alpha=0.15, color="steelblue", label="95% CI")
ax1.fill_between(z_ref, w_lo, w_hi, alpha=0.35, color="steelblue", label="68% CI")
ax1.plot(z_ref, w_median, color="navy", linewidth=2, label="Living TOE $w_\\phi(z)$ (median)")
ax1.axhline(y=-1.0, color="crimson", linestyle="--", linewidth=2, label="$\\Lambda$CDM ($w = -1$)")
ax1.axhline(y=-1.0 / 3.0, color="gray", linestyle=":", alpha=0.5, label="$w = -1/3$ (accel. boundary)")
if z_t is not None:
    ax1.axvline(x=z_t, color="green", linestyle="-.", alpha=0.7,
                label=f"$z_t = {z_t:.2f}$ (Living TOE)")
ax1.axvline(x=z_t_lcdm, color="orange", linestyle="-.", alpha=0.7,
            label=f"$z_t = {z_t_lcdm:.2f}$ ($\\Lambda$CDM)")
ax1.set_ylabel("$w_\\phi(z)$", fontsize=14)
ax1.set_title("Phase 5: Scalar-Field Equation of State vs $\\Lambda$CDM", fontsize=15)
ax1.legend(fontsize=10, loc="lower left")
ax1.set_ylim(-1.5, 0.5)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
delta_w_median = w_median - (-1.0)
delta_w_lo = w_lo - (-1.0)
delta_w_hi = w_hi - (-1.0)
ax2.fill_between(z_ref, delta_w_lo, delta_w_hi, alpha=0.35, color="steelblue")
ax2.plot(z_ref, delta_w_median, color="navy", linewidth=2)
ax2.axhline(y=0.0, color="crimson", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Redshift $z$", fontsize=14)
ax2.set_ylabel("$\\Delta w = w_\\phi - (-1)$", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("w_z_evolution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved w_z_evolution.png")
print("\nPhase 5 analysis complete.")

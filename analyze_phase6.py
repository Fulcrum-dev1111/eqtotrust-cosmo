# analyze_phase6.py
# Phase 6 full analysis:
# Step 2: Statistics comparison
# Step 3: Lambda limit (k->0)
# Step 4: Multi-observable predictions

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pantheon_likelihood_phase6 import (
    log_likelihood_phase6, log_likelihood_lcdm_planck,
    mu_model_phase6, PLANCK_OM_MEAN, PLANCK_OM_SIGMA,
    _integrate_scalar_constrained, _build_V0_table
)
from coherence_scalar_cosmo import V
from scipy.interpolate import interp1d

print("Pre-building V0 interpolation table...")
_build_V0_table(k_range=(0.01, 4.0), Om_range=(0.29, 0.34), nk=40, nOm=10, n=500)

data = pd.read_csv("pantheon_plus.csv")
dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}
n_data = len(dataset["z"])


def load_and_analyze():
    samples_s = np.load("posterior_samples_phase6_scalar.npy")
    samples_l = np.load("posterior_samples_phase6_lcdm.npy")

    print(f"Scalar samples: {samples_s.shape}")
    print(f"LCDM samples:   {samples_l.shape}")

    med_s = np.median(samples_s, axis=0)
    lo_s = np.percentile(samples_s, 16, axis=0)
    hi_s = np.percentile(samples_s, 84, axis=0)

    med_l = np.median(samples_l, axis=0)
    lo_l = np.percentile(samples_l, 16, axis=0)
    hi_l = np.percentile(samples_l, 84, axis=0)

    print("\n" + "=" * 70)
    print("STEP 2: CONSTRAINED FIT RESULTS")
    print("=" * 70)
    print("\nScalar field (k, Omega_m with Planck prior):")
    labels_s = ["k", "Omega_m"]
    for i, name in enumerate(labels_s):
        print(f"  {name:>10s} = {med_s[i]:.6f}  [{lo_s[i]:.6f}, {hi_s[i]:.6f}]")

    print("\nLCDM (Omega_m with Planck prior, M marginalized):")
    print(f"  {'Omega_m':>10s} = {med_l[0]:.6f}  [{lo_l[0]:.6f}, {hi_l[0]:.6f}]")

    print("\nScanning for max likelihood...")
    scan_s = np.random.choice(samples_s.shape[0], size=min(500, samples_s.shape[0]), replace=False)
    best_ll_s = log_likelihood_phase6(med_s, dataset)
    best_p_s = med_s.copy()
    for i in scan_s:
        ll = log_likelihood_phase6(samples_s[i], dataset)
        if ll > best_ll_s:
            best_ll_s = ll
            best_p_s = samples_s[i]

    scan_l = np.random.choice(samples_l.shape[0], size=min(1000, samples_l.shape[0]), replace=False)
    best_ll_l = log_likelihood_lcdm_planck(med_l, dataset)
    best_p_l = med_l.copy()
    for i in scan_l:
        ll = log_likelihood_lcdm_planck(samples_l[i], dataset)
        if ll > best_ll_l:
            best_ll_l = ll
            best_p_l = samples_l[i]

    k_s = 2
    k_l = 1
    chi2_s = -2.0 * best_ll_s
    chi2_l = -2.0 * best_ll_l
    aic_s = 2 * k_s + chi2_s
    aic_l = 2 * k_l + chi2_l
    delta_aic = aic_s - aic_l

    print(f"\n{'':>25s}  {'Scalar':>14s}  {'LCDM':>14s}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*14}")
    print(f"  {'Fitted params':>25s}  {'k, Om':>14s}  {'Om':>14s}")
    print(f"  {'AIC param count':>25s}  {k_s:>14d}  {k_l:>14d}")
    print(f"  {'Omega_m treatment':>25s}  {'Planck prior':>14s}  {'Planck prior':>14s}")
    print(f"  {'M treatment':>25s}  {'marginalized':>14s}  {'marginalized':>14s}")
    print(f"  {'-2*logL (eff chi2)':>25s}  {chi2_s:>14.4f}  {chi2_l:>14.4f}")
    print(f"  {'chi2/N_data':>25s}  {chi2_s/n_data:>14.6f}  {chi2_l/n_data:>14.6f}")
    print(f"  {'AIC':>25s}  {aic_s:>14.4f}  {aic_l:>14.4f}")
    print(f"\n  DELTA_AIC (Scalar - LCDM) = {delta_aic:.4f}")

    if abs(delta_aic) < 2:
        print("  >> STATISTICALLY EQUIVALENT")
    elif delta_aic < -2:
        print("  >> Scalar model preferred")
    else:
        print("  >> LCDM preferred")

    return samples_s, samples_l, best_p_s, best_p_l, delta_aic


def lambda_limit_test():
    print("\n" + "=" * 70)
    print("STEP 3: LAMBDA LIMIT ANALYSIS (k -> 0)")
    print("=" * 70)

    Om_test = PLANCK_OM_MEAN
    k_values = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]

    from lcdm_likelihood import mu_lcdm
    z_test = np.linspace(0.01, 2.0, 200)
    mu_lcdm_ref = mu_lcdm(z_test, Om_test, 0.0)

    print(f"\n  Testing convergence at Omega_m = {Om_test}")
    print(f"  {'k':>10s}  {'max|delta_mu|':>15s}  {'rms(delta_mu)':>15s}  {'converged?':>12s}")
    print(f"  {'-'*10}  {'-'*15}  {'-'*15}  {'-'*12}")

    residuals_by_k = {}
    for k_val in k_values:
        try:
            mu_scalar = mu_model_phase6(z_test, k_val, Om_test)
            offset = np.median(mu_scalar - mu_lcdm_ref)
            mu_scalar_shifted = mu_scalar - offset
            delta = mu_scalar_shifted - mu_lcdm_ref
            max_delta = np.max(np.abs(delta))
            rms_delta = np.sqrt(np.mean(delta ** 2))
            converged = "YES" if max_delta < 0.01 else "no"
            print(f"  {k_val:>10.4f}  {max_delta:>15.6f}  {rms_delta:>15.6f}  {converged:>12s}")
            residuals_by_k[k_val] = delta
        except Exception as e:
            print(f"  {k_val:>10.4f}  {'FAILED':>15s}  {str(e)[:30]}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0.25})

    ax1 = axes[0]
    ax1.plot(z_test, mu_lcdm_ref, "k--", linewidth=2, label="$\\Lambda$CDM")
    for k_val in [2.0, 1.0, 0.1, 0.01, 0.001]:
        if k_val in residuals_by_k:
            mu_s = mu_model_phase6(z_test, k_val, Om_test)
            offset = np.median(mu_s - mu_lcdm_ref)
            ax1.plot(z_test, mu_s - offset, label=f"$k = {k_val}$", alpha=0.8)
    ax1.set_ylabel("$\\mu(z)$ (shifted)", fontsize=13)
    ax1.set_title("$\\Lambda$ Limit: Scalar Field $\\to$ $\\Lambda$CDM as $k \\to 0$", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for k_val in [2.0, 1.0, 0.1, 0.01, 0.001]:
        if k_val in residuals_by_k:
            ax2.plot(z_test, residuals_by_k[k_val], label=f"$k = {k_val}$", alpha=0.8)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1)
    ax2.set_xlabel("Redshift $z$", fontsize=13)
    ax2.set_ylabel("$\\Delta\\mu$ (scalar $-$ $\\Lambda$CDM)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.savefig("lambda_limit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved lambda_limit.png")


def multi_observable_predictions(best_params_scalar):
    print("\n" + "=" * 70)
    print("STEP 4: MULTI-OBSERVABLE PREDICTIONS")
    print("=" * 70)

    k_best, Om_best = best_params_scalar

    from pantheon_likelihood_phase6 import _solve_V0_for_flatness
    C0, dC0 = 0.0, 0.0
    V0 = _solve_V0_for_flatness(k_best, Om_best)
    if V0 is None:
        V0 = max(1.0 - Om_best, 1e-30)
    print(f"\n  V0 (flatness-enforced) = {V0:.6f}  (naive: {1.0-Om_best:.6f})")

    a0 = 1e-4
    a_grid, C_arr_raw, Cp_arr_raw, E_arr_raw = _integrate_scalar_constrained(
        a0, 1.0, C0, dC0, V0, k_best, Om_best, n=8000
    )
    print(f"  E(a=1) = {E_arr_raw[-1]:.8f}")
    z_grid = 1.0 / a_grid - 1.0

    order = np.argsort(z_grid)
    z_grid = z_grid[order]
    E_arr = E_arr_raw[order]
    C_arr = C_arr_raw[order]
    Cp_arr = Cp_arr_raw[order]
    a_sorted = a_grid[order]

    rho_phi = 0.5 * (a_sorted ** 2) * (Cp_arr ** 2) + V(C_arr, V0, k_best)
    p_phi = 0.5 * (a_sorted ** 2) * (Cp_arr ** 2) - V(C_arr, V0, k_best)
    rho_m = Om_best / a_sorted ** 3
    rho_total = rho_m + rho_phi
    w_phi = np.where(np.abs(rho_phi) > 1e-30, p_phi / rho_phi, -1.0)
    w_eff = np.where(np.abs(rho_total) > 1e-30, p_phi / rho_total, -1.0)

    print("\n  A) Equation of state w(z):")
    for z_val in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
        idx = np.argmin(np.abs(z_grid - z_val))
        print(f"    w_phi(z={z_val:.1f}) = {w_phi[idx]:.6f}  (delta from -1: {w_phi[idx]+1:.6f})")

    q_arr = 0.5 * (1.0 + 3.0 * w_eff)
    sc = np.where(np.diff(np.sign(q_arr)))[0]
    if len(sc) > 0:
        ii = sc[0]
        z_t = z_grid[ii] + (z_grid[ii + 1] - z_grid[ii]) * (-q_arr[ii]) / (q_arr[ii + 1] - q_arr[ii])
        z_t_lcdm = (2.0 * (1.0 - Om_best) / Om_best) ** (1.0 / 3.0) - 1.0
        print(f"\n  B) Transition redshift:")
        print(f"    z_t (scalar)  = {z_t:.4f}")
        print(f"    z_t (LCDM)    = {z_t_lcdm:.4f}")
        print(f"    delta_z_t     = {z_t - z_t_lcdm:.4f}")
    else:
        print("\n  B) No transition redshift found")

    print("\n  C) BAO observable: D_V(z) / r_d ratios")
    print("     (D_V = [D_M^2 * c*z / H(z)]^{1/3})")

    E_interp = interp1d(z_grid, E_arr, bounds_error=False, fill_value="extrapolate")

    z_bao = [0.106, 0.15, 0.32, 0.57, 0.61, 1.52, 2.33]
    dz_fine = np.diff(z_grid)
    invE_fine = 1.0 / np.maximum(E_arr, 1e-12)
    Dc_fine = np.zeros_like(z_grid)
    Dc_fine[1:] = np.cumsum(dz_fine * 0.5 * (invE_fine[1:] + invE_fine[:-1]))
    Dc_interp = interp1d(z_grid, Dc_fine, bounds_error=False, fill_value="extrapolate")

    E_lcdm_fn = lambda z: np.sqrt(Om_best * (1 + z) ** 3 + (1 - Om_best))
    z_fine_lcdm = np.linspace(0, 1200.0, 50000)
    dz_l = np.diff(z_fine_lcdm)
    invE_l = 1.0 / E_lcdm_fn(z_fine_lcdm)
    Dc_l = np.zeros_like(z_fine_lcdm)
    Dc_l[1:] = np.cumsum(dz_l * 0.5 * (invE_l[1:] + invE_l[:-1]))
    Dc_l_interp = interp1d(z_fine_lcdm, Dc_l, bounds_error=False, fill_value="extrapolate")

    print(f"    {'z':>6s}  {'DV_scalar/DV_lcdm':>20s}  {'% deviation':>12s}")
    print(f"    {'-'*6}  {'-'*20}  {'-'*12}")
    for zb in z_bao:
        DM_s = Dc_interp(zb)
        E_s = E_interp(zb)
        DV_s = (DM_s ** 2 * zb / E_s) ** (1.0 / 3.0) if E_s > 0 else 0

        DM_l = Dc_l_interp(zb)
        E_l = E_lcdm_fn(zb)
        DV_l = (DM_l ** 2 * zb / E_l) ** (1.0 / 3.0)

        ratio = DV_s / DV_l if DV_l > 0 else 0
        pct = (ratio - 1.0) * 100
        print(f"    {zb:>6.3f}  {ratio:>20.6f}  {pct:>+11.4f}%")

    print("\n  D) CMB shift parameter R (note: radiation omitted, approximate):")
    z_star = 1089.92
    Dc_star_s = Dc_interp(z_star) if z_star <= z_grid[-1] else np.nan
    Dc_star_l = Dc_l_interp(z_star)
    R_s = np.sqrt(Om_best) * Dc_star_s if np.isfinite(Dc_star_s) else np.nan
    R_l = np.sqrt(Om_best) * Dc_star_l
    print(f"    R_scalar = {R_s:.4f}" if np.isfinite(R_s) else "    R_scalar = N/A (z* beyond grid)")
    print(f"    R_LCDM   = {R_l:.4f}")
    if np.isfinite(R_s):
        print(f"    delta_R  = {R_s - R_l:.4f} ({(R_s/R_l - 1)*100:+.4f}%)")
    print("    (Both omit radiation; fractional difference is meaningful but absolute R is approximate)")

    print("\n  E) Growth suppression factor (qualitative):")
    print("     In scalar-field models, the growth rate f*sigma8 is modified by")
    print("     the time-varying dark energy density. For w slightly > -1:")
    print("     - Dark energy clusters less than Lambda")
    print("     - Growth is slightly suppressed at late times")
    print("     - Predicted signature: f*sigma8(z<1) slightly below LCDM")
    delta_w_today = w_phi[np.argmin(np.abs(z_grid))] + 1.0
    print(f"     Current delta_w(z=0) = {delta_w_today:+.6f}")
    print(f"     Expected f*sigma8 suppression: ~{abs(delta_w_today)*0.3:.4f} (rough scaling)")


if __name__ == "__main__":
    samples_s, samples_l, best_p_s, best_p_l, delta_aic = load_and_analyze()
    lambda_limit_test()
    multi_observable_predictions(best_p_s)
    print("\n" + "=" * 70)
    print("PHASE 6 ANALYSIS COMPLETE")
    print("=" * 70)

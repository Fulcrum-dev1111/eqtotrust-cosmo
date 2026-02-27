import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from pantheon_likelihood import log_likelihood
from lcdm_likelihood import log_likelihood_lcdm

samples_scalar = np.load("posterior_samples.npy")
samples_lcdm = np.load("posterior_samples_lcdm.npy")
print(f"Scalar-field samples: {samples_scalar.shape[0]} x {samples_scalar.shape[1]}")
print(f"LCDM samples:         {samples_lcdm.shape[0]} x {samples_lcdm.shape[1]}")

data = pd.read_csv("pantheon_plus.csv")
dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}
n_data = len(dataset["z"])

labels_scalar = ["V0", "k", "Omega_m", "C0", "dC0", "M"]
labels_lcdm = ["Omega_m", "M"]

medians_scalar = np.median(samples_scalar, axis=0)
lo_scalar = np.percentile(samples_scalar, 16, axis=0)
hi_scalar = np.percentile(samples_scalar, 84, axis=0)

medians_lcdm = np.median(samples_lcdm, axis=0)
lo_lcdm = np.percentile(samples_lcdm, 16, axis=0)
hi_lcdm = np.percentile(samples_lcdm, 84, axis=0)

print("\n" + "=" * 70)
print("SCALAR-FIELD (Living TOE) MODEL — PARAMETERS (median ± 1σ)")
print("=" * 70)
for i, name in enumerate(labels_scalar):
    print(f"  {name:>10s} = {medians_scalar[i]:+.6f}  [{lo_scalar[i]:+.6f}, {hi_scalar[i]:+.6f}]")

print("\n" + "=" * 70)
print("ΛCDM BASELINE — PARAMETERS (median ± 1σ)")
print("=" * 70)
for i, name in enumerate(labels_lcdm):
    print(f"  {name:>10s} = {medians_lcdm[i]:+.6f}  [{lo_lcdm[i]:+.6f}, {hi_lcdm[i]:+.6f}]")

print("\nFinding best-fit (max likelihood) for scalar-field model...")
scan_size = 500
indices = np.random.choice(samples_scalar.shape[0], size=min(scan_size, samples_scalar.shape[0]), replace=False)
best_ll_scalar = log_likelihood(medians_scalar, dataset)
best_params_scalar = medians_scalar.copy()
for count, i in enumerate(indices):
    ll = log_likelihood(samples_scalar[i], dataset)
    if ll > best_ll_scalar:
        best_ll_scalar = ll
        best_params_scalar = samples_scalar[i]
    if (count + 1) % 100 == 0:
        print(f"  scanned {count + 1}/{scan_size}")

print("Finding best-fit (max likelihood) for LCDM model...")
scan_size_lcdm = 2000
indices_lcdm = np.random.choice(samples_lcdm.shape[0], size=min(scan_size_lcdm, samples_lcdm.shape[0]), replace=False)
best_ll_lcdm = log_likelihood_lcdm(medians_lcdm, dataset)
best_params_lcdm = medians_lcdm.copy()
for count, i in enumerate(indices_lcdm):
    ll = log_likelihood_lcdm(samples_lcdm[i], dataset)
    if ll > best_ll_lcdm:
        best_ll_lcdm = ll
        best_params_lcdm = samples_lcdm[i]
    if (count + 1) % 500 == 0:
        print(f"  scanned {count + 1}/{scan_size_lcdm}")

k_scalar = samples_scalar.shape[1]
k_lcdm = samples_lcdm.shape[1]
dof_scalar = n_data - k_scalar
dof_lcdm = n_data - k_lcdm

chi2_scalar = -2.0 * best_ll_scalar
chi2_lcdm = -2.0 * best_ll_lcdm

aic_scalar = 2 * k_scalar + chi2_scalar
aic_lcdm = 2 * k_lcdm + chi2_lcdm

delta_aic = aic_scalar - aic_lcdm

print("\n" + "=" * 70)
print("SCALAR-FIELD BEST-FIT POINT")
print("=" * 70)
for i, name in enumerate(labels_scalar):
    print(f"  {name:>10s} = {best_params_scalar[i]:+.6f}")

print("\n" + "=" * 70)
print("ΛCDM BEST-FIT POINT")
print("=" * 70)
for i, name in enumerate(labels_lcdm):
    print(f"  {name:>10s} = {best_params_lcdm[i]:+.6f}")

print("\n" + "=" * 70)
print("MODEL COMPARISON — FIT STATISTICS")
print("=" * 70)
print(f"  {'':>25s}  {'Scalar-Field':>14s}  {'ΛCDM':>14s}")
print(f"  {'-'*25}  {'-'*14}  {'-'*14}")
print(f"  {'N_data':>25s}  {n_data:>14d}  {n_data:>14d}")
print(f"  {'N_params':>25s}  {k_scalar:>14d}  {k_lcdm:>14d}")
print(f"  {'dof':>25s}  {dof_scalar:>14d}  {dof_lcdm:>14d}")
print(f"  {'chi2':>25s}  {chi2_scalar:>14.4f}  {chi2_lcdm:>14.4f}")
print(f"  {'chi2/dof':>25s}  {chi2_scalar/dof_scalar:>14.6f}  {chi2_lcdm/dof_lcdm:>14.6f}")
print(f"  {'log(L_max)':>25s}  {best_ll_scalar:>14.4f}  {best_ll_lcdm:>14.4f}")
print(f"  {'AIC':>25s}  {aic_scalar:>14.4f}  {aic_lcdm:>14.4f}")

print("\n" + "=" * 70)
print(f"  ΔAIC (Scalar − ΛCDM) = {delta_aic:.4f}")
print("=" * 70)
if delta_aic < -10:
    print("  >> Very strong preference for scalar-field model")
elif delta_aic < -2:
    print("  >> Moderate preference for scalar-field model")
elif delta_aic < 2:
    print("  >> Models are statistically indistinguishable")
elif delta_aic < 10:
    print("  >> Moderate preference for ΛCDM")
else:
    print("  >> Very strong preference for ΛCDM")

print("\nGenerating corner plot for scalar-field model...")
fig1 = corner.corner(
    samples_scalar,
    labels=labels_scalar,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
)
fig1.suptitle("Scalar-Field (Living TOE) Model", fontsize=14, y=1.02)
fig1.savefig("corner_plot_scalar.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved corner_plot_scalar.png")

print("Generating corner plot for LCDM model...")
fig2 = corner.corner(
    samples_lcdm,
    labels=labels_lcdm,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
)
fig2.suptitle("ΛCDM Baseline", fontsize=14, y=1.02)
fig2.savefig("corner_plot_lcdm.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved corner_plot_lcdm.png")

print("\nPhase 2 analysis complete.")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
from pantheon_likelihood import log_likelihood

samples = np.load("posterior_samples.npy")
print(f"Loaded {samples.shape[0]} posterior samples with {samples.shape[1]} parameters\n")

labels = ["V0", "k", "Omega_m", "C0", "dC0", "M"]

print("=" * 60)
print("BEST-FIT PARAMETERS (median +/- 1-sigma)")
print("=" * 60)
medians = np.median(samples, axis=0)
lo = np.percentile(samples, 16, axis=0)
hi = np.percentile(samples, 84, axis=0)
for i, name in enumerate(labels):
    print(f"  {name:>10s} = {medians[i]:+.6f}  ({lo[i]:+.6f}, {hi[i]:+.6f})")

print("\nGenerating corner plot...")
fig = corner.corner(
    samples,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
)
fig.savefig("corner_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Corner plot saved to corner_plot.png")

data = pd.read_csv("pantheon_plus.csv")
dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}

n_data = len(dataset["z"])
k_params = samples.shape[1]

print("\nComputing chi2 at median parameters...")
ll_median = log_likelihood(medians, dataset)

scan_size = 200
print(f"Scanning {scan_size} random posterior samples for max likelihood...")
scan_indices = np.random.choice(samples.shape[0], size=scan_size, replace=False)
best_ll = ll_median
best_params = medians.copy()
for count, i in enumerate(scan_indices):
    ll = log_likelihood(samples[i], dataset)
    if ll > best_ll:
        best_ll = ll
        best_params = samples[i]

chi2 = -2.0 * best_ll
chi2_dof = chi2 / (n_data - k_params)
aic = 2 * k_params + chi2

print()
print("=" * 60)
print("BEST-FIT POINT (maximum likelihood from posterior)")
print("=" * 60)
for i, name in enumerate(labels):
    print(f"  {name:>10s} = {best_params[i]:+.6f}")

print()
print("=" * 60)
print("MODEL FIT STATISTICS")
print("=" * 60)
print(f"  N_data       = {n_data}")
print(f"  N_params     = {k_params}")
print(f"  chi2         = {chi2:.4f}")
print(f"  chi2/dof     = {chi2_dof:.4f}")
print(f"  AIC          = {aic:.4f}")
print(f"  log(L_max)   = {best_ll:.4f}")

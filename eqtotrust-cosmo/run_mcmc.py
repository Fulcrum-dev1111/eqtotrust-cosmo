# run_mcmc.py
import numpy as np
import emcee
import pandas as pd
from pantheon_likelihood import log_likelihood

# Load Pantheon+ data
data = pd.read_csv("pantheon_plus.csv")

# Convert to structured dict
dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}

def log_prior(theta):
    V0, k, Om, C0, dC0, M = theta
    if 0.0 < Om < 1.0 and 0.0 < V0 < 5.0 and 0.0 < k < 5.0:
        return 0.0
    return -np.inf

def log_prob(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, dataset)

ndim = 6
nwalkers = 32

# Initial guess
initial = np.array([0.7, 1.549, 0.3, 0.0, 0.0, 0.0])
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(pos, 3000, progress=True)

samples = sampler.get_chain(discard=500, flat=True)
np.save("posterior_samples.npy", samples)

print("MCMC complete.")

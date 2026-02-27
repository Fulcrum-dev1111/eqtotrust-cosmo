# run_mcmc_lcdm.py
# LCDM baseline MCMC using full Pantheon+ covariance.
# Parameters: Omega_m, M (nuisance)

import numpy as np
import emcee
import pandas as pd
import os
from lcdm_likelihood import log_likelihood_lcdm

CHECKPOINT_FILE = "mcmc_lcdm_checkpoint.h5"
TOTAL_STEPS = 3000
CHECKPOINT_INTERVAL = 100

data = pd.read_csv("pantheon_plus.csv")

dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}

def log_prior_lcdm(theta):
    Om, M = theta
    if 0.01 < Om < 0.99 and -30.0 < M < 30.0:
        return 0.0
    return -np.inf

def log_prob_lcdm(theta):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(theta, dataset)

ndim = 2
nwalkers = 32

initial = np.array([0.3, 0.0])

backend = emcee.backends.HDFBackend(CHECKPOINT_FILE)

if os.path.exists(CHECKPOINT_FILE):
    completed = backend.iteration
    print(f"Resuming LCDM MCMC from checkpoint at step {completed}/{TOTAL_STEPS}")
    remaining = TOTAL_STEPS - completed
    if remaining <= 0:
        print("LCDM MCMC already complete.")
        samples = backend.get_chain(discard=500, flat=True)
        np.save("posterior_samples_lcdm.npy", samples)
        print(f"Saved {samples.shape[0]} LCDM posterior samples.")
        exit(0)
    pos = None
else:
    print("Starting fresh LCDM MCMC run...")
    backend.reset(nwalkers, ndim)
    pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
    remaining = TOTAL_STEPS

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_lcdm, backend=backend)

steps_done = TOTAL_STEPS - remaining
while steps_done < TOTAL_STEPS:
    batch = min(CHECKPOINT_INTERVAL, TOTAL_STEPS - steps_done)
    sampler.run_mcmc(pos, batch, progress=True)
    pos = None
    steps_done += batch
    print(f"Checkpoint saved: {steps_done}/{TOTAL_STEPS} steps complete")

samples = backend.get_chain(discard=500, flat=True)
np.save("posterior_samples_lcdm.npy", samples)

print(f"LCDM MCMC complete. Saved {samples.shape[0]} posterior samples.")

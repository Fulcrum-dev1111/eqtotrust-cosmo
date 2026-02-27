# run_mcmc.py
import numpy as np
import emcee
import pandas as pd
import os
from pantheon_likelihood import log_likelihood

CHECKPOINT_FILE = "mcmc_checkpoint.h5"
TOTAL_STEPS = 3000
CHECKPOINT_INTERVAL = 100

data = pd.read_csv("pantheon_plus.csv")

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

initial = np.array([0.7, 1.549, 0.3, 0.0, 0.0, 0.0])

backend = emcee.backends.HDFBackend(CHECKPOINT_FILE)

if os.path.exists(CHECKPOINT_FILE):
    completed = backend.iteration
    print(f"Resuming from checkpoint at step {completed}/{TOTAL_STEPS}")
    remaining = TOTAL_STEPS - completed
    if remaining <= 0:
        print("MCMC already complete.")
        samples = backend.get_chain(discard=500, flat=True)
        np.save("posterior_samples.npy", samples)
        print(f"Saved {samples.shape[0]} posterior samples.")
        exit(0)
    pos = None
else:
    print("Starting fresh MCMC run...")
    backend.reset(nwalkers, ndim)
    pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)
    remaining = TOTAL_STEPS

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)

steps_done = TOTAL_STEPS - remaining
while steps_done < TOTAL_STEPS:
    batch = min(CHECKPOINT_INTERVAL, TOTAL_STEPS - steps_done)
    sampler.run_mcmc(pos, batch, progress=True)
    pos = None
    steps_done += batch
    print(f"Checkpoint saved: {steps_done}/{TOTAL_STEPS} steps complete")

samples = backend.get_chain(discard=500, flat=True)
np.save("posterior_samples.npy", samples)

print(f"MCMC complete. Saved {samples.shape[0]} posterior samples.")

# run_mcmc_phase3.py
# Phase 3: Attractor-compressed scalar-field MCMC.
# Free parameters: k, Omega_m (2 physical params, same as LCDM).
# V0 derived from flatness, C0=dC0=0 (attractor), M analytically marginalized.

import numpy as np
import emcee
import pandas as pd
import os
import sys
from pantheon_likelihood_phase3 import log_likelihood_phase3

CHECKPOINT_FILE = "mcmc_phase3_checkpoint.h5"
TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CHECKPOINT_INTERVAL = 100

data = pd.read_csv("pantheon_plus.csv")

dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}

def log_prior_phase3(theta):
    k, Om = theta
    if 0.01 < k < 10.0 and 0.01 < Om < 0.99:
        return 0.0
    return -np.inf

def log_prob_phase3(theta):
    lp = log_prior_phase3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_phase3(theta, dataset)

ndim = 2
nwalkers = 32

initial = np.array([1.5, 0.3])

backend = emcee.backends.HDFBackend(CHECKPOINT_FILE)

if os.path.exists(CHECKPOINT_FILE):
    completed = backend.iteration
    print(f"Resuming Phase 3 MCMC from checkpoint at step {completed}/{TOTAL_STEPS}")
    remaining = TOTAL_STEPS - completed
    if remaining <= 0:
        print("Phase 3 MCMC already complete.")
        samples = backend.get_chain(discard=min(500, TOTAL_STEPS // 3), flat=True)
        np.save("posterior_samples_phase3.npy", samples)
        print(f"Saved {samples.shape[0]} Phase 3 posterior samples.")
        exit(0)
    pos = None
else:
    print(f"Starting fresh Phase 3 MCMC run ({TOTAL_STEPS} steps)...")
    backend.reset(nwalkers, ndim)
    pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)
    remaining = TOTAL_STEPS

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_phase3, backend=backend)

steps_done = TOTAL_STEPS - remaining
while steps_done < TOTAL_STEPS:
    batch = min(CHECKPOINT_INTERVAL, TOTAL_STEPS - steps_done)
    sampler.run_mcmc(pos, batch, progress=True)
    pos = None
    steps_done += batch
    print(f"Checkpoint saved: {steps_done}/{TOTAL_STEPS} steps complete")

discard = min(500, TOTAL_STEPS // 3)
samples = backend.get_chain(discard=discard, flat=True)
np.save("posterior_samples_phase3.npy", samples)

print(f"Phase 3 MCMC complete. Saved {samples.shape[0]} posterior samples.")

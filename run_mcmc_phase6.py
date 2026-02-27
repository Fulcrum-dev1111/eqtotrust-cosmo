# run_mcmc_phase6.py
# Phase 6: Constrained scalar-field MCMC.
# Samples k (scalar coupling) and Omega_m (with Planck prior).
# V0 derived from flatness, C0=dC0=0, M marginalized.
# Also runs LCDM baseline with identical Planck prior for fair comparison.

import numpy as np
import emcee
import pandas as pd
import os
import sys
from pantheon_likelihood_phase6 import log_likelihood_phase6, log_likelihood_lcdm_planck, _build_V0_table

TOTAL_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CHECKPOINT_INTERVAL = 100

data = pd.read_csv("pantheon_plus.csv")
dataset = {
    "z": data["z"].values,
    "mu": data["mu"].values,
    "sigma_mu": data["sigma_mu"].values
}

SCALAR_CHECKPOINT = "mcmc_phase6_scalar.h5"
LCDM_CHECKPOINT = "mcmc_phase6_lcdm.h5"


def run_chain(checkpoint_file, log_prob_fn, ndim, nwalkers, initial, label, total_steps):
    backend = emcee.backends.HDFBackend(checkpoint_file)

    completed = backend.iteration if os.path.exists(checkpoint_file) else 0
    if completed > 0:
        print(f"Resuming {label} from checkpoint at step {completed}/{total_steps}")
        remaining = total_steps - completed
        if remaining <= 0:
            print(f"{label} already complete.")
            discard = min(500, total_steps // 3)
            return backend.get_chain(discard=discard, flat=True)
        pos = None
    else:
        print(f"Starting fresh {label} ({total_steps} steps)...")
        backend.reset(nwalkers, ndim)
        pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
        remaining = total_steps

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, backend=backend)

    steps_done = total_steps - remaining
    while steps_done < total_steps:
        batch = min(CHECKPOINT_INTERVAL, total_steps - steps_done)
        sampler.run_mcmc(pos, batch, progress=True)
        pos = None
        steps_done += batch
        print(f"  {label} checkpoint: {steps_done}/{total_steps}")

    discard = min(500, total_steps // 3)
    return backend.get_chain(discard=discard, flat=True)


def log_prob_scalar(theta):
    k, Om = theta
    if not (0.001 < k < 4.0 and 0.1 < Om < 0.6):
        return -np.inf
    return log_likelihood_phase6(theta, dataset)


def log_prob_lcdm(theta):
    (Om,) = theta
    if not (0.1 < Om < 0.6):
        return -np.inf
    return log_likelihood_lcdm_planck(theta, dataset)


print("=" * 70)
print("PHASE 6: CONSTRAINED SCALAR FIELD + LCDM BASELINE")
print("=" * 70)

print("\n--- Building V0 flatness interpolation table ---")
import time
t0 = time.time()
_build_V0_table(k_range=(0.01, 4.0), Om_range=(0.28, 0.36), nk=60, nOm=30, n=2000)
print(f"Table built in {time.time()-t0:.1f}s")

print("\n--- Scalar field (k, Omega_m with Planck prior) ---")
samples_scalar = run_chain(
    SCALAR_CHECKPOINT, log_prob_scalar,
    ndim=2, nwalkers=32,
    initial=np.array([0.5, 0.315]),
    label="Scalar Phase6",
    total_steps=TOTAL_STEPS
)
np.save("posterior_samples_phase6_scalar.npy", samples_scalar)
print(f"Saved {samples_scalar.shape[0]} scalar samples")

print("\n--- LCDM (Omega_m with Planck prior, M marginalized) ---")
samples_lcdm = run_chain(
    LCDM_CHECKPOINT, log_prob_lcdm,
    ndim=1, nwalkers=32,
    initial=np.array([0.315]),
    label="LCDM Phase6",
    total_steps=TOTAL_STEPS
)
np.save("posterior_samples_phase6_lcdm.npy", samples_lcdm)
print(f"Saved {samples_lcdm.shape[0]} LCDM samples")

print("\nPhase 6 MCMC complete.")

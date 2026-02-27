# eqtotrust-cosmo

## Overview

A scalar-field cosmology model implemented in Python. Tests a potential of the form `V(C) = V_0 * e^(kC)` against the Pantheon+ Type Ia supernova dataset using MCMC parameter inference with full covariance.

## Project Structure

- `coherence_scalar_cosmo.py` — ODE integrator (4th-order Runge-Kutta) for the modified Friedmann + Klein-Gordon system
- `pantheon_likelihood.py` — Full-covariance Pantheon+ likelihood using Cholesky decomposition (no diagonal shortcut)
- `lcdm_likelihood.py` — Standard flat ΛCDM likelihood for baseline comparison (also uses full covariance)
- `ingest_pantheon.py` — Downloads and preprocesses the official Pantheon+SH0ES dataset and STAT+SYS covariance matrix from GitHub
- `run_mcmc.py` — Runs scalar-field MCMC (32 walkers, 3000 steps, checkpointing every 100 steps)
- `run_mcmc_lcdm.py` — Runs ΛCDM baseline MCMC (32 walkers, 3000 steps, checkpointing every 100 steps)
- `analyze_results.py` — Loads posterior samples, computes fit statistics (chi2, AIC), generates corner plot

## Data

Downloaded from the official Pantheon+SH0ES DataRelease repo:
- Data: `Pantheon+SH0ES.dat` → `pantheon_plus.csv` (1701 rows)
- Covariance: `Pantheon+SH0ES_STAT+SYS.cov` → `pantheon_plus_cov.txt` (1701×1701 matrix)

Row ordering is preserved from the official files to ensure perfect alignment between data and covariance matrix.

## Dependencies

Python 3.12 with: `numpy`, `scipy`, `pandas`, `emcee`, `matplotlib`, `tqdm`, `h5py`, `corner`

## Workflow

The "Start application" workflow runs sequentially:
1. `python ingest_pantheon.py` — fetches data + covariance (skips if already cached)
2. `python run_mcmc.py` — scalar-field MCMC → `posterior_samples.npy` (checkpoints to `mcmc_checkpoint.h5`)
3. `python run_mcmc_lcdm.py` — ΛCDM baseline MCMC → `posterior_samples_lcdm.npy` (checkpoints to `mcmc_lcdm_checkpoint.h5`)

Output type: console (no web server).

## Scalar-Field Model Parameters (6)

- `V0` — scalar potential amplitude
- `k` — exponential slope
- `Om` — matter density parameter
- `C0` — initial scalar field value
- `dC0` — initial scalar field derivative
- `M` — distance modulus nuisance parameter

## ΛCDM Baseline Parameters (2)

- `Om` — matter density parameter
- `M` — distance modulus nuisance parameter

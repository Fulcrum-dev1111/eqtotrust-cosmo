# eqtotrust-cosmo

## Overview

A scalar-field cosmology model implemented in Python. Tests a potential of the form `V(C) = V_0 * e^(kC)` against the Pantheon+ Type Ia supernova dataset using MCMC parameter inference.

## Project Structure

- `coherence_scalar_cosmo.py` — ODE integrator (4th-order Runge-Kutta) for the modified Friedmann + Klein-Gordon system
- `pantheon_likelihood.py` — Computes luminosity distances and Gaussian chi-squared log-likelihood
- `ingest_pantheon.py` — Downloads and preprocesses the official Pantheon+SH0ES dataset from GitHub
- `run_mcmc.py` — Runs MCMC sampling using `emcee` (32 walkers, 3000 steps, 500 burn-in); saves posterior to `posterior_samples.npy`

## Data

Downloaded from the official Pantheon+SH0ES DataRelease repo:
`https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat`

The ingestion script extracts `zHD`, `m_b_corr`, and `m_b_corr_err_DIAG` columns.

## Dependencies

Python 3.12 with: `numpy`, `scipy`, `pandas`, `emcee`, `matplotlib`

## Workflow

The "Start application" workflow runs:
1. `python ingest_pantheon.py` — fetches and cleans the supernova data
2. `python run_mcmc.py` — runs MCMC and saves `posterior_samples.npy`

Output type: console (no web server).

## Model Parameters

- `V0` — scalar potential amplitude
- `k` — exponential slope
- `Om` — matter density parameter
- `C0` — initial scalar field value
- `dC0` — initial scalar field derivative
- `M` — distance modulus nuisance parameter

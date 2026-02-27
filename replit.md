# eqtotrust-cosmo

## Overview

A scalar-field cosmology model testing V(C) = V₀e^{kC} against Pantheon+ Type Ia supernova data using MCMC parameter inference with full covariance.

## Project Structure

### Core Physics
- `coherence_scalar_cosmo.py` — ODE integrator (RK4) for modified Friedmann + Klein-Gordon with matter
- `pure_scalar_cosmo.py` — ODE integrator for pure scalar field (no matter, Phase 4)

### Likelihoods
- `pantheon_likelihood.py` — Phase 2: Full-covariance likelihood (6 free params)
- `lcdm_likelihood.py` — Standard flat ΛCDM baseline likelihood
- `pantheon_likelihood_phase3.py` — Phase 3: Attractor-compressed (k, Ω_m), M marginalized
- `pantheon_likelihood_phase4.py` — Phase 4: Pure scalar universe (k only), M marginalized

### MCMC Runners
- `run_mcmc.py` — Phase 2 scalar-field MCMC (6 params)
- `run_mcmc_lcdm.py` — ΛCDM baseline MCMC (2 params)
- `run_mcmc_phase3.py` — Phase 3 attractor-compressed MCMC (2 params)
- `run_mcmc_phase4.py` — Phase 4 pure scalar MCMC (1 param)

### Data & Analysis
- `ingest_pantheon.py` — Downloads Pantheon+SH0ES data + STAT+SYS covariance matrix
- `analyze_results.py` — Phase 1 analysis
- `analyze_phase2.py` — Phase 2 analysis (scalar vs ΛCDM comparison)

## Data Sources

- Data: `Pantheon+SH0ES.dat` → `pantheon_plus.csv` (1701 rows)
- Covariance: `Pantheon+SH0ES_STAT+SYS.cov` → `pantheon_plus_cov.txt` (1701×1701)

## Dependencies

Python 3.12: numpy, scipy, pandas, emcee, matplotlib, tqdm, h5py, corner

### Analysis
- `analyze_wz.py` — Phase 5: Equation of state w(z), transition redshift, ΛCDM comparison plot

## Phase Summary

- **Phase 1**: Diagonal-only likelihood, 6 free params
- **Phase 2**: Full covariance, 6 params vs 2-param ΛCDM (ΔAIC = +3.7 for scalar)
- **Phase 3**: Attractor compression (k, Ω_m), M marginalized (ΔAIC = +112.7)
- **Phase 4**: Pure scalar universe, k only, no Ω_m (ΔAIC = +106.1)
- **Phase 5**: Equation of state w(z) analysis — scalar field predicts w ≈ -0.999 with Δw ~ +0.0007 deviation from Λ

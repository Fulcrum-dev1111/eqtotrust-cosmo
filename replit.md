# eqtotrust-cosmo

## Overview

A scalar-field cosmology model testing V(C) = V₀e^{kC} against Pantheon+ Type Ia supernova data using MCMC parameter inference with full covariance.

## Project Structure

### Core Physics
- `coherence_scalar_cosmo.py` — ODE integrator (RK4) for modified Friedmann + Klein-Gordon with matter (uses relaxer for E)
- `pure_scalar_cosmo.py` — ODE integrator for pure scalar field (no matter, Phase 4)

### Likelihoods
- `pantheon_likelihood.py` — Phase 2: Full-covariance likelihood (6 free params)
- `lcdm_likelihood.py` — Standard flat ΛCDM baseline likelihood
- `pantheon_likelihood_phase3.py` — Phase 3: Attractor-compressed (k, Ω_m), M marginalized
- `pantheon_likelihood_phase4.py` — Phase 4: Pure scalar universe (k only), M marginalized
- `pantheon_likelihood_phase6.py` — Phase 6: Constrained scalar (k free, Ω_m Planck prior, V₀ from flatness, M marginalized). Uses algebraic Friedmann E(a) (no relaxer).

### MCMC Runners
- `run_mcmc.py` — Phase 2 scalar-field MCMC (6 params)
- `run_mcmc_lcdm.py` — ΛCDM baseline MCMC (2 params)
- `run_mcmc_phase3.py` — Phase 3 attractor-compressed MCMC (2 params)
- `run_mcmc_phase4.py` — Phase 4 pure scalar MCMC (1 param)
- `run_mcmc_phase6.py` — Phase 6 constrained MCMC (k + Ω_m with Planck prior)

### Analysis
- `analyze_results.py` — Phase 1 analysis
- `analyze_phase2.py` — Phase 2 analysis (scalar vs ΛCDM comparison)
- `analyze_wz.py` — Phase 5: Equation of state w(z), transition redshift, ΛCDM comparison plot
- `analyze_phase6.py` — Phase 6: Constrained fit statistics, Λ-limit convergence, multi-observable predictions

### Data & Ingestion
- `ingest_pantheon.py` — Downloads Pantheon+SH0ES data + STAT+SYS covariance matrix

## Data Sources

- Data: `Pantheon+SH0ES.dat` → `pantheon_plus.csv` (1701 rows)
- Covariance: `Pantheon+SH0ES_STAT+SYS.cov` → `pantheon_plus_cov.txt` (1701×1701)

## Dependencies

Python 3.12: numpy, scipy, pandas, emcee, matplotlib, tqdm, h5py, corner

## Phase Summary

- **Phase 1**: Diagonal-only likelihood, 6 free params
- **Phase 2**: Full covariance, 6 params vs 2-param ΛCDM (ΔAIC = +3.7 for scalar)
- **Phase 3**: Attractor compression (k, Ω_m), M marginalized (ΔAIC = +112.7)
- **Phase 4**: Pure scalar universe, k only, no Ω_m (ΔAIC = +106.1)
- **Phase 5**: Equation of state w(z) analysis — scalar field predicts w ≈ -0.999 with Δw ~ +0.0007 deviation from Λ
- **Phase 6**: Constrained scalar (Planck Ω_m prior, flatness via V₀ shooting, C₀=dC₀=0):
  - k = 1.92 [1.52, 2.20], Ω_m = 0.315 [0.308, 0.322]
  - ΔAIC = -6.76 (scalar preferred over ΛCDM)
  - Λ limit verified: k→0 reproduces ΛCDM to <0.01 mag for k<0.5
  - w(z=0) = -0.78 at best-fit k; BAO deviations 1-2.5%; CMB R deviation -0.96%
  - V₀ = 0.923 (solved via shooting to enforce E(a=1)=1)

## Key Technical Notes

- Phase 6 uses algebraic Friedmann constraint for E(a) and a V₀ shooting method (`_solve_V0_for_flatness`) to enforce E(a=1)=1. A 2D interpolation table is prebuilt for MCMC speed.
- M (absolute magnitude nuisance) is analytically marginalized in Phases 3, 4, 6.
- Planck 2018 prior: Ω_m = 0.315 ± 0.007 (Gaussian).
- AIC counting: scalar (k, Ω_m) = 2 params; ΛCDM (Ω_m) = 1 param. Both use same Planck prior and M marginalization.

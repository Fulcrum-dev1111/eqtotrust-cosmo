# eqtotrust-cosmo

## Purpose

This repository implements and tests a scalar-field cosmology model of the form:

V(C) = V_0 e^{kC}

under natural units M_{pl} = 1, using:
- Modified Friedmann equations
- Klein-Gordon evolution for scalar field C
- Runge-Kutta numerical integration
- Pantheon+ Type Ia supernova likelihood
- emcee MCMC parameter inference

This repo is intended for independent replication and adversarial testing.

---

## Model Definition

### Units
- Reduced Planck mass: M_{pl} = 1
- All quantities dimensionless
- V_0, k, C treated dimensionless

### Scalar Potential
V(C) = V_0 e^{kC}

### Free Parameters
- V_0
- k
- \Omega_m
- C_0
- dC_0
- M (distance modulus nuisance parameter)

---

## Numerical Implementation
- ODE system solved in scale factor a
- 4th-order Runge-Kutta integrator
- Friedmann constraint enforced dynamically
- Luminosity distance computed via comoving integral
- Distance modulus: \mu = 5 \log_{10}(D_L) + 25 + M

---

## Data Source
Pantheon+ Type Ia Supernova dataset
Official release: https://github.com/PantheonPlusSH0ES/DataRelease
The ingestion script pulls directly from the official data release.
No modified or custom supernova data is used.

---

## Statistical Framework
- Likelihood: Gaussian chi-squared on μ residuals
- Sampler: emcee EnsembleSampler
- Burn-in discarded
- Posterior chains saved to posterior_samples.npy

---

## Falsification Criteria
This model fails if:
1. It does not converge under MCMC sampling.
2. It produces unstable or non-physical expansion histories.
3. It performs worse than ΛCDM under ΔAIC or χ² per degree of freedom.
4. Independent replication cannot reproduce results.

---

## Success Criteria
Model considered competitive if:
\Delta AIC < -2
relative to ΛCDM baseline under identical dataset and likelihood.

---

## Independent Replication
To replicate:

```bash
git clone https://github.com/Fulcrum-dev1111/eqtotrust-cosmo
pip install -r requirements.txt
python run_mcmc.py
```

All numerical structures are transparent.
All parameters are exposed.
No hidden tuning.
No seeded bias.

---

## Scope Clarification
This repository does not claim:
- Proof of new physics
- Refutation of ΛCDM
- Replacement of standard cosmology

It provides:
- A formally specified scalar-field model
- A testable numerical implementation
- A transparent statistical comparison framework

---

## Statement of Intent
If the model survives adversarial review, the data speaks.
If it fails, the failure is informative.
Either outcome advances understanding.

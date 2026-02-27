# pantheon_likelihood_phase3.py
# Phase 3: Attractor-compressed scalar-field likelihood.
# Free physical parameters: k, Omega_m (same count as LCDM).
# V0 derived from flatness: V0 = (1 - Omega_m) / exp(k * C0)
# C0 = 0, dC0 = 0 (attractor/slow-roll initial conditions at a=1)
# Nuisance M is analytically marginalized out of the likelihood.

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

from coherence_scalar_cosmo import integrate
from pantheon_likelihood import load_covariance


def luminosity_distance(z, a_grid, E_grid):
    z_grid = 1.0 / a_grid - 1.0

    order = np.argsort(z_grid)
    z_grid = z_grid[order]
    E_grid = E_grid[order]

    dz = np.diff(z_grid)
    invE = 1.0 / np.maximum(E_grid, 1e-12)

    Dc = np.zeros_like(z_grid)
    Dc[1:] = np.cumsum(dz * 0.5 * (invE[1:] + invE[:-1]))

    Dc_interp = interp1d(z_grid, Dc, bounds_error=False, fill_value="extrapolate")
    Dc_z = Dc_interp(z)

    return (1.0 + z) * Dc_z


def mu_model_phase3(z, k, Om):
    C0 = 0.0
    dC0 = 0.0

    Omega_phi = 1.0 - Om
    V0 = max(Omega_phi / np.exp(k * C0), 1e-30)

    params = {
        "V0": V0,
        "k": k,
        "Om": Om,
        "Or": 0.0,
        "C0": C0,
        "dC0": dC0,
    }

    a0 = 1e-4
    y0 = np.array([C0, dC0, 1.0], dtype=float)

    a_grid, y = integrate(a0, 1.0, y0, params, n=4000)
    E_grid = y[:, 2]

    DL = luminosity_distance(z, a_grid, E_grid)

    mu_no_M = 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0
    return mu_no_M


_P3_COV_CACHE = {}

def _get_p3_cov(data, cov_path=None):
    key = id(data["mu"])
    if key not in _P3_COV_CACHE:
        cov = load_covariance(data, cov_path=cov_path)
        _P3_COV_CACHE[key] = {
            "cho": cho_factor(cov, lower=True, check_finite=False),
            "cov": cov,
        }
        print(f"[Phase3 likelihood] Loaded covariance matrix: {cov.shape[0]}x{cov.shape[1]}")
    return _P3_COV_CACHE[key]


def log_likelihood_phase3(theta, data, cov_path=None):
    k, Om = theta

    if not (0.01 < k < 10.0 and 0.0 < Om < 1.0):
        return -np.inf

    try:
        mu_pred = mu_model_phase3(data["z"], k, Om)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    cache = _get_p3_cov(data, cov_path=cov_path)
    c_fac = cache["cho"]

    delta = data["mu"] - mu_pred
    ones = np.ones(len(delta))

    Cinv_delta = cho_solve(c_fac, delta)
    Cinv_ones = cho_solve(c_fac, ones)

    A = ones @ Cinv_delta
    B = ones @ Cinv_ones
    C_val = delta @ Cinv_delta

    chi2_marg = C_val - (A * A) / B + np.log(B / (2.0 * np.pi))

    return -0.5 * chi2_marg

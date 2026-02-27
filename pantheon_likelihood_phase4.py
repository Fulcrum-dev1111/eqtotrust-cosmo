# pantheon_likelihood_phase4.py
# Phase 4: Pure Scalar Universe.
# NO standard matter (Omega_m = 0). NO dark matter. NO cosmological constant.
# The scalar field with V(C) = V0 * exp(k*C) drives the ENTIRE expansion.
#
# Free parameter: k (the scalar field coupling constant)
# V0 derived from flatness at a=1: V0 = 1.0 (E(a=1)=1 normalization)
# C0 = 0, dC0 = 0 (attractor conditions)
# Nuisance M is analytically marginalized.
#
# This is a 1-physical-parameter model vs LCDM's 1 (Omega_m) + nuisance M.

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

from pure_scalar_cosmo import integrate_pure
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


def mu_model_phase4(z, k):
    C0 = 0.0
    dC0 = 0.0
    V0 = 1.0

    params = {"V0": V0, "k": k}

    a0 = 1e-4
    y0 = np.array([C0, dC0, 1.0], dtype=float)

    a_grid, y = integrate_pure(a0, 1.0, y0, params, n=4000)
    E_grid = y[:, 2]

    DL = luminosity_distance(z, a_grid, E_grid)

    mu_no_M = 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0
    return mu_no_M


_P4_COV_CACHE = {}


def _get_p4_cov(data, cov_path=None):
    key = id(data["mu"])
    if key not in _P4_COV_CACHE:
        cov = load_covariance(data, cov_path=cov_path)
        _P4_COV_CACHE[key] = {
            "cho": cho_factor(cov, lower=True, check_finite=False),
            "cov": cov,
        }
        print(f"[Phase4 likelihood] Loaded covariance matrix: {cov.shape[0]}x{cov.shape[1]}")
    return _P4_COV_CACHE[key]


def log_likelihood_phase4(theta, data, cov_path=None):
    (k,) = theta

    if not (0.001 < k < 20.0):
        return -np.inf

    try:
        mu_pred = mu_model_phase4(data["z"], k)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    cache = _get_p4_cov(data, cov_path=cov_path)
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

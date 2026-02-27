# lcdm_likelihood.py
# Standard flat LCDM likelihood for Pantheon+ baseline comparison.

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
from pantheon_likelihood import load_covariance


def E_lcdm(z, Om):
    OL = 1.0 - Om
    return np.sqrt(Om * (1.0 + z)**3 + OL)


def mu_lcdm(z, Om, M):
    z_grid = np.linspace(0, np.max(z) * 1.05, 5000)
    E_grid = E_lcdm(z_grid, Om)

    dz = np.diff(z_grid)
    invE = 1.0 / E_grid

    Dc = np.zeros_like(z_grid)
    Dc[1:] = np.cumsum(dz * 0.5 * (invE[1:] + invE[:-1]))

    Dc_interp = interp1d(z_grid, Dc, bounds_error=False, fill_value="extrapolate")
    Dc_z = Dc_interp(z)

    DL = (1.0 + z) * Dc_z
    mu = 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0 + M
    return mu


_LCDM_COV_CACHE = {}

def _get_lcdm_cov(data, cov_path=None):
    key = id(data["mu"])
    if key not in _LCDM_COV_CACHE:
        cov = load_covariance(data, cov_path=cov_path)
        _LCDM_COV_CACHE[key] = cho_factor(cov, lower=True, check_finite=False)
        print(f"[LCDM likelihood] Loaded covariance matrix: {cov.shape[0]}x{cov.shape[1]}")
    return _LCDM_COV_CACHE[key]


def log_likelihood_lcdm(theta, data, cov_path=None):
    Om, M = theta

    if not (0.0 < Om < 1.0):
        return -np.inf

    try:
        mu_pred = mu_lcdm(data["z"], Om, M)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    residual = data["mu"] - mu_pred

    c_fac = _get_lcdm_cov(data, cov_path=cov_path)
    x = cho_solve(c_fac, residual)
    chi2 = float(residual @ x)

    return -0.5 * chi2

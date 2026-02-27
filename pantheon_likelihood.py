# pantheon_likelihood.py
# Full-covariance Pantheon+ likelihood (no diagonal shortcut).
# Dimensionless cosmology units (M_pl = 1) assumed by the solver.

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve

from coherence_scalar_cosmo import integrate


# ----------------------------
# Model: distances + mu(z)
# ----------------------------

def V(C, V0, k):
    return V0 * np.exp(k * C)


def luminosity_distance(z, a_grid, E_grid):
    """
    Dimensionless luminosity distance:
      D_L(z) = (1+z) * integral_0^z dz'/E(z')
    using the E(z) implied by the scalar-field integrator.
    """
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


def mu_model(z, params_cosmo, M_nuisance):
    """
    mu = 5 log10(D_L) + 25 + M
    (H0 absorbed into nuisance M, consistent with many SN-only fits)
    """
    a0 = params_cosmo.get("a0", 1e-4)

    y0 = np.array([params_cosmo["C0"], params_cosmo["dC0"], 1.0], dtype=float)

    a_grid, y = integrate(a0, 1.0, y0, params_cosmo, n=params_cosmo.get("n_grid", 4000))
    E_grid = y[:, 2]

    DL = luminosity_distance(z, a_grid, E_grid)

    mu = 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0 + M_nuisance
    return mu


# ----------------------------
# Covariance handling
# ----------------------------

def _infer_square_dim_from_flat(n_flat, n_data):
    if n_flat == n_data * n_data:
        return n_data
    root = int(np.sqrt(n_flat))
    if root * root != n_flat:
        raise ValueError(f"Covariance length {n_flat} is not a perfect square and "
                         f"does not match N^2 with N={n_data}.")
    return root


def load_covariance(data, cov_path=None):
    n = len(data["mu"])

    candidates = []
    if cov_path:
        candidates.append(cov_path)
    candidates += [
        "pantheon_plus_cov.csv",
        "pantheon_plus_cov.txt",
        "pantheon_plus_cov.cov",
        "pantheon_plus.cov",
        "covmat.txt",
        "covmat.cov",
    ]

    chosen = None
    for p in candidates:
        if p and os.path.exists(p):
            chosen = p
            break

    if chosen is None:
        if "sigma_mu" in data and data["sigma_mu"] is not None:
            s2 = np.asarray(data["sigma_mu"], dtype=float) ** 2
            return np.diag(s2)
        raise FileNotFoundError(
            "No covariance file found and sigma_mu not provided. "
            "Provide cov_path or include sigma_mu in dataset."
        )

    raw = np.loadtxt(chosen, delimiter="," if chosen.endswith(".csv") else None)
    raw = np.asarray(raw, dtype=float)

    if raw.ndim == 1:
        N = _infer_square_dim_from_flat(raw.size, n)
        cov = raw.reshape((N, N))
    elif raw.ndim == 2:
        cov = raw
    else:
        raise ValueError(f"Unexpected covariance array ndim={raw.ndim} from {chosen}.")

    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix from {chosen} is not square: {cov.shape}")

    if cov.shape[0] != n:
        raise ValueError(
            f"Covariance dimension {cov.shape[0]} does not match data length {n}. "
            "Make sure your pantheon_plus.csv rows correspond exactly to the cov ordering."
        )

    return cov


def chi2_full_cov(residual, cov):
    jitter = 0.0
    try:
        c_fac = cho_factor(cov, lower=True, check_finite=False)
    except Exception:
        jitter = 1e-12 * np.mean(np.diag(cov))
        c_fac = cho_factor(cov + jitter * np.eye(cov.shape[0]), lower=True, check_finite=False)

    x = cho_solve(c_fac, residual)
    return float(residual @ x)


_COV_CACHE = {}

def _get_cov(data, cov_path=None):
    key = id(data["mu"])
    if key not in _COV_CACHE:
        cov = load_covariance(data, cov_path=cov_path)
        _COV_CACHE[key] = cho_factor(cov, lower=True, check_finite=False)
        print(f"[likelihood] Loaded covariance matrix: {cov.shape[0]}x{cov.shape[1]}")
    return _COV_CACHE[key]


def log_likelihood(theta, data, cov_path=None):
    V0, k, Om, C0, dC0, M = theta

    if not (0.0 < Om < 1.0 and 0.0 < V0 < 5.0 and 0.0 < k < 5.0):
        return -np.inf

    params = {
        "V0": V0,
        "k": k,
        "Om": Om,
        "Or": 0.0,
        "C0": C0,
        "dC0": dC0,
    }

    try:
        mu_pred = mu_model(data["z"], params, M)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    residual = data["mu"] - mu_pred

    c_fac = _get_cov(data, cov_path=cov_path)
    x = cho_solve(c_fac, residual)
    chi2 = float(residual @ x)

    return -0.5 * chi2

# pantheon_likelihood.py
import numpy as np
from scipy.interpolate import interp1d
from coherence_scalar_cosmo import integrate

# Luminosity distance in dimensionless units
def luminosity_distance(z, a_grid, E_grid):
    # Convert scale factor grid to redshift grid
    z_grid = 1.0 / a_grid - 1.0
    z_grid = z_grid[::-1]
    E_grid = E_grid[::-1]

    # Compute comoving distance integral
    dz = np.diff(z_grid)
    invE = 1.0 / np.maximum(E_grid, 1e-12)
    Dc = np.zeros_like(z_grid)
    Dc[1:] = np.cumsum(dz * invE[1:])

    Dc_interp = interp1d(z_grid, Dc, bounds_error=False, fill_value="extrapolate")
    Dc_z = Dc_interp(z)

    return (1 + z) * Dc_z


def mu_model(z, params_cosmo, M_nuisance):
    a0 = 1e-4
    y0 = np.array([params_cosmo["C0"], params_cosmo["dC0"], 1.0])
    a_grid, y = integrate(a0, 1.0, y0, params_cosmo, n=4000)

    E_grid = y[:,2]

    DL = luminosity_distance(z, a_grid, E_grid)

    # Distance modulus (H0 absorbed into nuisance M)
    mu = 5.0 * np.log10(np.maximum(DL, 1e-12)) + 25.0 + M_nuisance
    return mu


def log_likelihood(theta, data):
    V0, k, Om, C0, dC0, M = theta

    if not (0.0 < Om < 1.0 and 0.0 < V0 < 5.0 and 0.0 < k < 5.0):
        return -np.inf

    params = {
        "V0": V0,
        "k": k,
        "Om": Om,
        "Or": 0.0,
        "C0": C0,
        "dC0": dC0
    }

    mu_pred = mu_model(data["z"], params, M)

    chi2 = np.sum(((data["mu"] - mu_pred) / data["sigma_mu"])**2)
    return -0.5 * chi2

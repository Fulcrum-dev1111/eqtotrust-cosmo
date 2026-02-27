import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

from coherence_scalar_cosmo import V, dVdC
from pantheon_likelihood import load_covariance

PLANCK_OM_MEAN = 0.315
PLANCK_OM_SIGMA = 0.007

_V0_CACHE = {}
_V0_INTERP = None


def _integrate_scalar_constrained(a0, a1, C0, dC0, V0, k, Om, n=4000):
    a = np.linspace(a0, a1, n)
    da = a[1] - a[0]

    C_arr = np.zeros(n)
    Cp_arr = np.zeros(n)
    E_arr = np.zeros(n)

    C_arr[0] = C0
    Cp_arr[0] = dC0

    kin0 = 0.5 * a0**2 * dC0**2
    E_arr[0] = np.sqrt(max(Om / a0**3 + kin0 + V(C0, V0, k), 1e-30))

    def get_E(ai, Ci, Cpi):
        kin = 0.5 * ai**2 * Cpi**2
        rho_total = Om / ai**3 + kin + V(Ci, V0, k)
        return np.sqrt(max(rho_total, 1e-30))

    def deriv(ai, Ci, Cpi):
        Ei = get_E(ai, Ci, Cpi)
        E2 = max(Ei**2, 1e-30)
        Cpp = -(3.0 / ai) * Cpi - (1.0 / (ai**2 * E2)) * dVdC(Ci, V0, k)
        return Cpi, Cpp

    for i in range(n - 1):
        ai = a[i]
        Ci, Cpi = C_arr[i], Cp_arr[i]

        dC1, dCp1 = deriv(ai, Ci, Cpi)
        dC2, dCp2 = deriv(ai + da / 2, Ci + da * dC1 / 2, Cpi + da * dCp1 / 2)
        dC3, dCp3 = deriv(ai + da / 2, Ci + da * dC2 / 2, Cpi + da * dCp2 / 2)
        dC4, dCp4 = deriv(ai + da, Ci + da * dC3, Cpi + da * dCp3)

        C_arr[i + 1] = Ci + (da / 6) * (dC1 + 2 * dC2 + 2 * dC3 + dC4)
        Cp_arr[i + 1] = Cpi + (da / 6) * (dCp1 + 2 * dCp2 + 2 * dCp3 + dCp4)
        E_arr[i + 1] = get_E(a[i + 1], C_arr[i + 1], Cp_arr[i + 1])

    return a, C_arr, Cp_arr, E_arr


def _E_at_today(V0, k, Om, n=4000):
    _, _, _, E_arr = _integrate_scalar_constrained(1e-4, 1.0, 0.0, 0.0, V0, k, Om, n=n)
    return E_arr[-1]


def _build_V0_table(k_range=(0.01, 4.0), Om_range=(0.29, 0.34), nk=40, nOm=10, n=500):
    global _V0_INTERP
    from scipy.interpolate import RegularGridInterpolator

    k_grid = np.linspace(k_range[0], k_range[1], nk)
    Om_grid = np.linspace(Om_range[0], Om_range[1], nOm)
    V0_table = np.full((nk, nOm), np.nan)

    for i, kv in enumerate(k_grid):
        for j, Omv in enumerate(Om_grid):
            V0_naive = max(1.0 - Omv, 1e-30)
            if abs(kv) < 0.01:
                V0_table[i, j] = V0_naive
                continue

            def residual(V0):
                return _E_at_today(V0, kv, Omv, n=n) - 1.0

            V0_lo = V0_naive
            V0_hi = V0_naive * 5.0
            found = False
            for _ in range(10):
                if residual(V0_hi) > 0:
                    found = True
                    break
                V0_hi *= 2.0

            if found:
                try:
                    V0_table[i, j] = brentq(residual, V0_lo, V0_hi, xtol=1e-6, rtol=1e-6)
                except ValueError:
                    pass

    _V0_INTERP = RegularGridInterpolator(
        (k_grid, Om_grid), V0_table,
        method="linear", bounds_error=False, fill_value=None
    )
    valid = np.sum(np.isfinite(V0_table))
    print(f"[Phase6] V0 table: {nk}x{nOm}, {valid}/{nk*nOm} valid ({100*valid/(nk*nOm):.1f}%)")
    return _V0_INTERP


def _solve_V0_for_flatness(k, Om, n=4000, tol=1e-8):
    global _V0_INTERP

    if _V0_INTERP is not None:
        try:
            V0_interp = float(_V0_INTERP(np.array([k, Om]))[0])
            if np.isfinite(V0_interp) and V0_interp > 0 and V0_interp < 100:
                return V0_interp
        except Exception:
            pass

    cache_key = (round(k, 10), round(Om, 10))
    if cache_key in _V0_CACHE:
        return _V0_CACHE[cache_key]

    V0_naive = max(1.0 - Om, 1e-30)

    if abs(k) < 0.01:
        _V0_CACHE[cache_key] = V0_naive
        return V0_naive

    def residual(V0):
        return _E_at_today(V0, k, Om, n=n) - 1.0

    E_naive = residual(V0_naive) + 1.0
    if abs(E_naive - 1.0) < tol:
        _V0_CACHE[cache_key] = V0_naive
        return V0_naive

    V0_lo = V0_naive
    V0_hi = V0_naive * 5.0
    for _ in range(10):
        if residual(V0_hi) > 0:
            break
        V0_hi *= 2.0
    else:
        _V0_CACHE[cache_key] = None
        return None

    try:
        V0_sol = brentq(residual, V0_lo, V0_hi, xtol=tol, rtol=tol)
    except ValueError:
        _V0_CACHE[cache_key] = None
        return None

    _V0_CACHE[cache_key] = V0_sol
    return V0_sol


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
    return (1.0 + z) * Dc_interp(z)


def mu_model_phase6(z, k, Om):
    C0 = 0.0
    dC0 = 0.0
    V0 = _solve_V0_for_flatness(k, Om)

    if V0 is None:
        return np.full_like(z, np.nan)

    a0 = 1e-4
    a_grid, C_arr, Cp_arr, E_grid = _integrate_scalar_constrained(
        a0, 1.0, C0, dC0, V0, k, Om, n=4000
    )

    DL = luminosity_distance(z, a_grid, E_grid)
    return 5.0 * np.log10(np.maximum(DL, 1e-30)) + 25.0


_P6_COV_CACHE = {}


def _get_p6_cov(data, cov_path=None):
    key = id(data["mu"])
    if key not in _P6_COV_CACHE:
        cov = load_covariance(data, cov_path=cov_path)
        _P6_COV_CACHE[key] = cho_factor(cov, lower=True, check_finite=False)
        print(f"[Phase6 likelihood] Loaded covariance matrix: {cov.shape[0]}x{cov.shape[1]}")
    return _P6_COV_CACHE[key]


def log_likelihood_phase6(theta, data, cov_path=None):
    k, Om = theta

    if not (0.0 < k < 20.0 and 0.0 < Om < 1.0):
        return -np.inf

    try:
        mu_pred = mu_model_phase6(data["z"], k, Om)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    c_fac = _get_p6_cov(data, cov_path=cov_path)

    delta = data["mu"] - mu_pred
    ones = np.ones(len(delta))

    Cinv_delta = cho_solve(c_fac, delta)
    Cinv_ones = cho_solve(c_fac, ones)

    A = ones @ Cinv_delta
    B = ones @ Cinv_ones
    C_val = delta @ Cinv_delta

    chi2_marg = C_val - (A * A) / B + np.log(B / (2.0 * np.pi))

    log_prior_Om = -0.5 * ((Om - PLANCK_OM_MEAN) / PLANCK_OM_SIGMA) ** 2

    return -0.5 * chi2_marg + log_prior_Om


def log_likelihood_lcdm_planck(theta, data, cov_path=None):
    (Om,) = theta

    if not (0.0 < Om < 1.0):
        return -np.inf

    from lcdm_likelihood import mu_lcdm
    try:
        mu_pred = mu_lcdm(data["z"], Om, 0.0)
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(mu_pred)):
        return -np.inf

    c_fac = _get_p6_cov(data, cov_path=cov_path)

    delta = data["mu"] - mu_pred
    ones = np.ones(len(delta))

    Cinv_delta = cho_solve(c_fac, delta)
    Cinv_ones = cho_solve(c_fac, ones)

    A = ones @ Cinv_delta
    B = ones @ Cinv_ones
    C_val = delta @ Cinv_delta

    chi2_marg = C_val - (A * A) / B + np.log(B / (2.0 * np.pi))

    log_prior_Om = -0.5 * ((Om - PLANCK_OM_MEAN) / PLANCK_OM_SIGMA) ** 2

    return -0.5 * chi2_marg + log_prior_Om

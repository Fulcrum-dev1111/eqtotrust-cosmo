# pure_scalar_cosmo.py
# Pure scalar-field cosmology: NO standard matter (Omega_m = 0).
# The entire expansion history is driven by the scalar field C
# with potential V(C) = V0 * exp(k * C).
# E = mc^2 * e^{kC} => mass itself is dynamically coupled to C.
# Friedmann: E^2 = rho_phi = 0.5 * a^2 * (dC/da)^2 + V(C)

import numpy as np


def V(C, V0, k):
    return V0 * np.exp(k * C)


def dVdC(C, V0, k):
    return k * V(C, V0, k)


def rhs_pure(a, y, params):
    C, Cp, E = y
    V0 = params["V0"]
    k = params["k"]

    kin = 0.5 * (a ** 2) * (Cp ** 2)
    rho_phi = kin + V(C, V0, k)

    E2 = max(rho_phi, 1e-12)
    E_new = np.sqrt(E2)

    Cpp = -(3.0 / a) * Cp - (1.0 / (a ** 2 * E2)) * dVdC(C, V0, k)

    return np.array([Cp, Cpp, (E_new - E) * 10.0])


def integrate_pure(a0, a1, y0, params, n=4000):
    a = np.linspace(a0, a1, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    da = a[1] - a[0]

    for i in range(n - 1):
        k1 = rhs_pure(a[i], y[i], params)
        k2 = rhs_pure(a[i] + da / 2, y[i] + da * k1 / 2, params)
        k3 = rhs_pure(a[i] + da / 2, y[i] + da * k2 / 2, params)
        k4 = rhs_pure(a[i] + da, y[i] + da * k3, params)
        y[i + 1] = y[i] + (da / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return a, y

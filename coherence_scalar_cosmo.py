# coherence_scalar_cosmo.py
import numpy as np

# Natural units convention: C is dimensionless (M_{pl}=1)
# V_0 is dimensionless in these units.

def V(C, V0, k):
    return V0 * np.exp(k * C)

def dVdC(C, V0, k):
    return k * V(C, V0, k)

def rhs(a, y, params):
    """
    ODE system in terms of scale factor a:
    y = [C, dC/da, E(a)] where E(a)=H(a)/H0 (dimensionless)
    """
    C, Cp, E = y
    V0, k, Om, Or = params["V0"], params["k"], params["Om"], params.get("Or", 0.0)

    # Matter/radiation densities (dimensionless)
    rho_m = Om / a**3
    rho_r = Or / a**4

    # Scalar "kinetic" term in a-variable
    kin = 0.5 * (a**2) * (Cp**2)

    rho_C = kin + V(C, V0, k)

    # Friedmann (dimensionless): E^2 = rho_m + rho_r + rho_C
    E2 = max(rho_m + rho_r + rho_C, 1e-12)
    E_new = np.sqrt(E2)

    # KG in a-variable:
    Cpp = -(3.0/a) * Cp - (1.0 / (a**2 * E2)) * dVdC(C, V0, k)

    return np.array([Cp, Cpp, (E_new - E) * 10.0])  # relaxer toward constraint

def integrate(a0, a1, y0, params, n=2000):
    a = np.linspace(a0, a1, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    da = a[1] - a[0]

    for i in range(n-1):
        k1 = rhs(a[i], y[i], params)
        k2 = rhs(a[i] + da/2, y[i] + da*k1/2, params)
        k3 = rhs(a[i] + da/2, y[i] + da*k2/2, params)
        k4 = rhs(a[i] + da, y[i] + da*k3, params)
        y[i+1] = y[i] + (da/6)*(k1 + 2*k2 + 2*k3 + k4)

    return a, y

if __name__ == "__main__":
    params = {"V0": 0.7, "k": 1.549, "Om": 0.3, "Or": 0.0}
    # initial conditions at a0
    a0 = 1e-3
    y0 = np.array([0.0, 0.0, 1.0])  # C, dC/da, E
    a, y = integrate(a0, 1.0, y0, params)
    print("Final E(a=1):", y[-1,2])

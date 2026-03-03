# dark_energy_living_toe.py
# Living TOE Scalar Field Module for CAMB / Cobaya
# Implements V(C) = V0 * exp(k*C)

import numpy as np
from camb.dark_energy import DarkEnergyFluid

class LivingToeQuintessence(DarkEnergyFluid):
    """
    Custom Dark Energy class for the Living TOE model.
    Overrides standard Lambda to implement dynamical scalar field.
    """
    def __init__(self, k_coupling=1.92):
        super().__init__()
        self.k = k_coupling
        # We assume initial conditions are set by the attractor
        self.has_perturbations = True 

    def get_w(self, a):
        """
        Returns the equation of state w(a).
        In a full implementation, this integrates the Klein-Gordon equation.
        For a fast Cobaya interface, we can use the analytic approximation 
        derived from Phase 6, or dynamically integrate it.

        Using the Phase 6 analytic scaling:
        w(a) = -1 + (k^2 / 3) * (Omega_phi(a))
        """
        # Approximating Omega_phi(a) from flatness and scaling
        # Omega_m0 = 0.315 (Planck prior)
        Omega_m0 = 0.315 
        Omega_phi0 = 1.0 - Omega_m0

        # Approximate evolution of Omega_phi
        # (This is a simplified fast-compute version. Full version requires ODE solve)
        Omega_phi_a = Omega_phi0 / (Omega_phi0 + Omega_m0 * (a**-3))

        w = -1.0 + (self.k**2 / 3.0) * Omega_phi_a

        # Quintessence cannot cross the phantom divide (w < -1)
        # And should not exceed w=1
        return np.clip(w, -1.0, 1.0)

    def get_cs2(self, a):
        """
        Speed of sound squared for a canonical scalar field is strictly 1.
        This prevents catastrophic structure collapse at S8.
        """
        return 1.0

# In your Cobaya YAML configuration, you will point the theory code to this class:
# theory:
#   camb:
#     extra_args:
#       dark_energy_model: 'LivingToeQuintessence'

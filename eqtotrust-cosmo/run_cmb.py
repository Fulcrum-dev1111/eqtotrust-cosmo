import numpy as np
from cobaya.run import run
from camb.dark_energy import DarkEnergyFluid
import camb

class LivingToeQuintessence(DarkEnergyFluid):
    def __init__(self):
        super().__init__()
        self.k = 1.92
        self.has_perturbations = True 

    def get_w(self, a):
        Omega_m0 = 0.315 
        Omega_phi0 = 1.0 - Omega_m0
        Omega_phi_a = Omega_phi0 / (Omega_phi0 + Omega_m0 * (a**-3))
        w = -1.0 + (self.k**2 / 3.0) * Omega_phi_a
        return np.clip(w, -1.0, 1.0)

    def get_cs2(self, a):
        return 1.0

# THE HACK: Inject the class directly into CAMB's hidden internal registry
camb.baseconfig.F2003Class._class_names["LivingToeQuintessence"] = LivingToeQuintessence

info = {
    "force": True,
    "theory": {
        "camb": {
            "extra_args": {
                "dark_energy_model": "LivingToeQuintessence"
            }
        }
    },
    "likelihood": {
        "planck_2018_highl_plik.TTTEEE": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_lowl.TT": None
    },
    "params": {
        "logA": {"prior": {"min": 1.61, "max": 3.91}, "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001}, "proposal": 0.001, "drop": True},
        "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
        "ns": {"prior": {"min": 0.8, "max": 1.2}, "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004}, "proposal": 0.002},
        "H0": {"prior": {"min": 20, "max": 100}, "ref": {"dist": "norm", "loc": 67, "scale": 2}, "proposal": 2},
        "ombh2": {"prior": {"min": 0.005, "max": 0.1}, "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001}, "proposal": 0.0001},
        "omch2": {"prior": {"min": 0.001, "max": 0.99}, "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001}, "proposal": 0.0005},
        "tau": {"prior": {"min": 0.01, "max": 0.8}, "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006}, "proposal": 0.003}
    },
    "sampler": {
        "mcmc": {
            "drag": True,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "covmat": "auto",
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.2
        }
    },
    "output": "chains/living_toe_cmb",
    "packages_path": "./cobaya_packages"
}

print("Initializing Living TOE CAMB Integration...")
updated_info, sampler = run(info)
print("Simulation Complete.")

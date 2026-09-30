import numpy as np
from universe.generator import softened_v_circ

class Diagnostics:
    @staticmethod
    def center_of_mass(pos, mass):
        total_mass = np.sum(mass)
        if total_mass == 0: return np.zeros(2)
        return np.sum(pos * mass[:, None], axis=0) / total_mass

    @staticmethod
    def total_momentum(vel, mass):
        return np.sum(vel * mass[:, None], axis=0)
        
    @staticmethod
    def mass_fractions(mass, types):
        unique_types = np.unique(types)
        total_mass = np.sum(mass)
        if total_mass == 0: return {}
        return {t: np.sum(mass[types == t]) / total_mass for t in unique_types}
        
    @staticmethod
    def bound_fraction(pos, vel, mass, eps=1.0, G=1.0):
        # A simple approximation of bound fraction
        # Compute potential energy approximation
        N = len(pos)
        if N == 0: return 0.0
        
        # We can't do O(N^2) for large N easily here, so we approximate with a central potential 
        # around the center of mass. This is just for fast validation metrics.
        com = Diagnostics.center_of_mass(pos, mass)
        r = np.linalg.norm(pos - com, axis=1)
        M_total = np.sum(mass)
        
        # Approximate potential: phi ~ -G * M(<r) / r
        phi = -G * M_total / (r + eps + 1e-9)
        kinetic = 0.5 * np.sum(vel**2, axis=1)
        energy = kinetic + phi
        
        bound_count = np.sum(energy < 0)
        return bound_count / N

    @staticmethod
    def validate_preset_initialization(preset_def, pos, vel, mass, types):
        results = {}
        results['is_finite'] = bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(mass)))
        results['total_mass'] = float(np.sum(mass))
        results['com'] = Diagnostics.center_of_mass(pos, mass).tolist()
        results['momentum'] = Diagnostics.total_momentum(vel, mass).tolist()
        results['mass_fractions'] = {int(k): float(v) for k, v in Diagnostics.mass_fractions(mass, types).items()}
        results['approx_bound_fraction'] = float(Diagnostics.bound_fraction(pos, vel, mass))
        return results

import numpy as np
from universe.spatial import SpatialModel
from universe.kinematics import KinematicModel

class PlummerSpatial(SpatialModel):
    gravity_classification = "exact_spherical"
    def __init__(self, a, total_mass, center=(0.0, 0.0)):
        self.a = float(a)
        self.total_mass = float(total_mass)
        self.center = np.array(center, dtype=np.float64)
        
    def sample_positions(self, N, rng):
        # r = a / sqrt(u^(-2/3) - 1)
        u = rng.uniform(0.001, 0.999, N)
        r = self.a / np.sqrt(u**(-2.0/3.0) - 1.0)
        theta = rng.uniform(0, 2*np.pi, N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center
        return pos, r
        
    def enclosed_mass(self, r):
        return self.total_mass * (r**3 / (r**2 + self.a**2)**1.5)
        
    def radial_force(self, r, G, eps):
        M_enc = self.enclosed_mass(r)
        return G * M_enc * r / (r**2 + eps**2)**1.5

class IsotropicKinematics(KinematicModel):
    kinematic_classification = "heuristic_isotropic"
    def __init__(self, drift=(0.0, 0.0)):
        self.drift = np.array(drift, dtype=np.float64)
        
    def solve(self, pos, r, all_components, unit_system, rng):
        N = len(pos)
        if N == 0:
            return np.zeros((0, 2))
            
        total_radial_force = np.zeros(N)
        for c in all_components:
            if c.spatial_model is not None:
                total_radial_force += c.spatial_model.radial_force(r, unit_system.G, unit_system.eps)
                
        # Simple isotropic dispersion approximation
        v_circ = np.sqrt(np.maximum(0.0, total_radial_force * r))
        sigma = v_circ / np.sqrt(2.0)
        
        vx = rng.normal(0, sigma, N)
        vy = rng.normal(0, sigma, N)
        return np.column_stack((vx, vy)) + self.drift

class KeplerianSpatial(SpatialModel):
    def __init__(self, a_list, total_mass=0.0):
        self.a_list = np.array(a_list, dtype=np.float64)
        self.total_mass = float(total_mass)
        
    def sample_positions(self, N, rng):
        # We assume N == len(a_list)
        # Random true anomalies
        theta = rng.uniform(0, 2*np.pi, N)
        r = self.a_list
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        return pos, r
        
    def enclosed_mass(self, r):
        return np.zeros_like(r)
        
    def radial_force(self, r, G, eps):
        return np.zeros_like(r)

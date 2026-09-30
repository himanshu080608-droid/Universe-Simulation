import numpy as np
from universe.generator import softened_v_circ

class KinematicModel:
    def solve(self, pos, r, all_components, unit_system, rng):
        raise NotImplementedError

class DiskKinematics(KinematicModel):
    def __init__(self, drift=(0.0, 0.0), velocity_dispersion=0.05):
        self.drift = np.array(drift, dtype=np.float64)
        self.velocity_dispersion = velocity_dispersion

    def solve(self, pos, r, all_components, unit_system, rng):
        N = len(pos)
        if N == 0:
            return np.zeros((0, 2))
        
        total_radial_force = np.zeros(N)
        for c in all_components:
            if c.spatial_model is not None:
                total_radial_force += c.spatial_model.radial_force(r, unit_system.G, unit_system.eps)
                
        v_circ = np.sqrt(np.maximum(0.0, total_radial_force * r))
        
        angles = np.arctan2(pos[:, 1], pos[:, 0]) + np.pi/2
        vel_x = v_circ * np.cos(angles) + rng.normal(0, v_circ * self.velocity_dispersion)
        vel_y = v_circ * np.sin(angles) + rng.normal(0, v_circ * self.velocity_dispersion)
        vel = np.column_stack((vel_x, vel_y)) + self.drift
        return vel

class NFWKinematics(KinematicModel):
    def __init__(self, drift=(0.0, 0.0)):
        self.drift = np.array(drift, dtype=np.float64)
        
    def solve(self, pos, r, all_components, unit_system, rng):
        N = len(pos)
        if N == 0:
            return np.zeros((0, 2))
            
        total_radial_force = np.zeros(N)
        M_enc_total = np.zeros(N)
        for c in all_components:
            if c.spatial_model is not None:
                total_radial_force += c.spatial_model.radial_force(r, unit_system.G, unit_system.eps)
                M_enc_total += c.spatial_model.enclosed_mass(r)
                
        # NFW particles are pressure-supported (dispersion only)
        v_circ = np.sqrt(np.maximum(0.0, total_radial_force * r))
        sigma = v_circ / np.sqrt(2.0)
        
        vx = rng.normal(0, sigma, N)
        vy = rng.normal(0, sigma, N)
        
        # Clamp to escape velocity to keep particles bound
        v_esc = np.sqrt(2.0 * unit_system.G * M_enc_total / (r + 1e-9))
        speeds = np.sqrt(vx**2 + vy**2)
        clamp = np.minimum(1.0, 0.95 * v_esc / (speeds + 1e-9))
        
        vx *= clamp
        vy *= clamp
        
        return np.column_stack((vx, vy)) + self.drift

class PointMassKinematics(KinematicModel):
    def __init__(self, drift=(0.0, 0.0)):
        self.drift = np.array(drift, dtype=np.float64)
        
    def solve(self, pos, r, all_components, unit_system, rng):
        N = len(pos)
        return np.zeros((N, 2)) + self.drift

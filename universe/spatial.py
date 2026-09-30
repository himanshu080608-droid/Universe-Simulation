import numpy as np
from universe.generator import softened_v_circ

class SpatialModel:
    gravity_classification = "unknown"
    def sample_positions(self, N, rng):
        raise NotImplementedError
    def enclosed_mass(self, r):
        raise NotImplementedError
    def radial_force(self, r, G, eps):
        raise NotImplementedError

class ExponentialDiskSpatial(SpatialModel):
    gravity_classification = "spherical_approximation"
    def __init__(self, r_scale, total_mass, center=(0.0, 0.0)):
        self.r_scale = r_scale
        self.total_mass = total_mass
        self.center = np.array(center, dtype=np.float64)
        
    def _exp_disk_cdf(self, x):
        return 1.0 - (1.0 + x) * np.exp(-x)
        
    def sample_positions(self, N, rng):
        x_min = 0.1625
        x_max = 4.605
        F0 = self._exp_disk_cdf(x_min)
        F1 = self._exp_disk_cdf(x_max)
        
        u_rand = rng.uniform(0.0, 1.0, N)
        target_F = F0 + u_rand * (F1 - F0)
        
        x = x_min + u_rand * (x_max - x_min)
        for _ in range(10):
            fx = 1.0 - (1.0 + x) * np.exp(-x) - target_F
            dfx = x * np.exp(-x)
            x = x - fx / dfx
            
        r = x * self.r_scale
        th = rng.uniform(0, 2 * np.pi, N)
        
        n_arms = 2.0
        pitch_angle = 12.0 * (np.pi / 180.0)
        spiral_phase = n_arms * th - (n_arms / np.tan(pitch_angle)) * (r / self.r_scale)
        amplitude = 0.8
        th -= (amplitude / n_arms) * np.sin(spiral_phase)
        
        pos = np.column_stack((r * np.cos(th), r * np.sin(th))) + self.center
        return pos, r
        
    def enclosed_mass(self, r):
        x_min = 0.1625
        x_max = 4.605
        F0 = self._exp_disk_cdf(x_min)
        F1 = self._exp_disk_cdf(x_max)
        frac = (self._exp_disk_cdf(r / self.r_scale) - F0) / (F1 - F0)
        return self.total_mass * np.clip(frac, 0.0, 1.0)
        
    def radial_force(self, r, G, eps):
        # Controlled approximation: spherical-equivalent enclosed mass force.
        # This is not an exact 2D disk solution, but an approximation used for initialization.
        M_enc = self.enclosed_mass(r)
        return G * M_enc * r / (r**2 + eps**2)**1.5

class PointMassSpatial(SpatialModel):
    gravity_classification = "exact_point_mass"
    def __init__(self, total_mass, center=(0.0, 0.0)):
        self.total_mass = total_mass
        self.center = np.array(center, dtype=np.float64)
        
    def sample_positions(self, N, rng):
        pos = np.zeros((N, 2)) + self.center
        r = np.zeros(N)
        return pos, r
        
    def enclosed_mass(self, r):
        return np.full_like(r, self.total_mass)
        
    def radial_force(self, r, G, eps):
        return G * self.total_mass * r / (r**2 + eps**2)**1.5

class NFWSpatial(SpatialModel):
    gravity_classification = "exact_spherical"
    def __init__(self, R_s, C, total_mass, center=(0.0, 0.0)):
        self.R_s = R_s
        self.C = C
        self.total_mass = total_mass
        self.center = np.array(center, dtype=np.float64)
        
    def sample_positions(self, N, rng):
        from universe.nfw_calibration import get_calibration
        eps = 1.0
        R_cdf, cdf = get_calibration(self.R_s, self.C, eps)
        _, unique_indices = np.unique(cdf[::-1], return_index=True)
        unique_indices = np.sort(len(cdf) - 1 - unique_indices)
        
        cdf_clean = cdf[unique_indices]
        R_clean = R_cdf[unique_indices]
        
        u = rng.uniform(0.5/N, 1.0 - 0.5/N, N)
        r = np.interp(u, cdf_clean, R_clean)
        r = np.clip(r, 0.0, R_cdf[-1])
        rng.shuffle(r)
        
        theta = rng.uniform(0, 2 * np.pi, N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center
        return pos, r
        
    def enclosed_mass(self, r):
        def nfw_mass_frac(x):
            return np.log(1.0 + x) - x / (1.0 + x)
        m_vir = nfw_mass_frac(self.C)
        return self.total_mass * (nfw_mass_frac(r / self.R_s) / m_vir)

    def radial_force(self, r, G, eps):
        # Spherical NFW radial force
        M_enc = self.enclosed_mass(r)
        return G * M_enc * r / (r**2 + eps**2)**1.5

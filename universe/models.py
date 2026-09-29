import numpy as np

EPS = 1.0
def softened_v_circ(G, M_enc, r, eps=EPS):
    r2 = r * r + eps * eps
    return np.sqrt(G * M_enc * r * r / (r2 ** 1.5))

class AstronomicalModel:
    def generate(self):
        """Returns (pos, vel, masses, types)"""
        raise NotImplementedError

class PlummerModel(AstronomicalModel):
    """
    Globular Clusters (Omega Centauri, M13) and Open Clusters (Pleiades).
    Perfectly isotropic (pressure-supported) equilibrium.
    """
    def __init__(self, N, a_scale, total_mass, rng, types_dist, is_gas=False, M_central=0.0, mass_segregation=False, omega_rotation=False):
        self.N = N
        self.a = a_scale
        self.M = total_mass
        self.M_central = M_central
        self.rng = rng
        self.types_dist = types_dist
        self.is_gas = is_gas
        self.mass_segregation = mass_segregation
        self.omega_rotation = omega_rotation
        
    def generate(self):
        # 2D projection of a Plummer sphere (Kuzmin Disk)
        # To maintain perfect equilibrium in 2D 1/r^2 gravity, the particles 
        # MUST be distributed according to the 2D Kuzmin surface density, NOT 3D.
        u = self.rng.uniform(0.001, 0.95, self.N)
        # 2D Kuzmin CDF inverse: r = a * sqrt( 1 / (1-u)^2 - 1 )
        r = self.a * np.sqrt( 1.0 / ( (1.0 - u)**2 ) - 1.0 )
        if self.is_gas: r *= 1.5
            
        th = self.rng.uniform(0, 2*np.pi, self.N)
        pos = np.column_stack((r * np.cos(th), r * np.sin(th)))
        
        # 3D Mass enclosed inside sphere of radius r
        # In 2D with 1/r^2 gravity, a purely pressure-supported system suffers from 
        # the Radial Orbit Instability and rapid 2-body relaxation, causing evaporation.
        # To maintain long-term stability (>100k steps), the system must be rotationally supported.
        
        # 2D Isotropic Jeans Equation solution for Kuzmin Disk
        # To avoid bar instabilities in 2D, we must use a pressure-supported isotropic disk,
        # perfectly balanced according to the Virial theorem.
        # Exact 1D velocity dispersion for Kuzmin disk: sigma^2 = (G * M) / (6 * sqrt(r^2 + a^2))
        
        M_total_system = self.M + self.M_central
        # 6.0 denominator balances the 2D kinetic energy with the 3D potential energy
        sigma_1d = np.sqrt(M_total_system / (6.0 * np.sqrt(r**2 + self.a**2) + 1e-9))
        
        vx = self.rng.normal(0, sigma_1d)
        vy = self.rng.normal(0, sigma_1d)
        
        if self.is_gas:
            # Gas is collisional and tends to rotate rather than random dispersion
            v_circ_sq = (self.M * r**2) / ((r**2 + self.a**2)**1.5 + 1e-9) + self.M_central / (r + 1e-9)
            v_circ = np.sqrt(v_circ_sq)
            angles = np.arctan2(pos[:, 1], pos[:, 0]) + np.pi/2
            vx = v_circ * np.cos(angles) + self.rng.normal(0, v_circ * 0.1)
            vy = v_circ * np.sin(angles) + self.rng.normal(0, v_circ * 0.1)
            
        if self.omega_rotation and not self.is_gas:
            # Omega Centauri has bulk rotation and a counter-rotating core
            v_circ_sq = (self.M * r**2) / ((r**2 + self.a**2)**1.5 + 1e-9) + self.M_central / (r + 1e-9)
            v_circ = np.sqrt(v_circ_sq)
            angles = np.arctan2(pos[:, 1], pos[:, 0]) + np.pi/2
            
            # Counter-rotating core for R < 1.5 units (~0.5 pc)
            rotation_dir = np.where(r < 1.5, -1.0, 1.0)
            v_rot = v_circ * 0.3 * rotation_dir # ~30% rotational support
            
            vx += v_rot * np.cos(angles)
            vy += v_rot * np.sin(angles)
            
            # Add radial anisotropy to the halo
            v_rad = (pos[:, 0] * vx + pos[:, 1] * vy) / (r + 1e-9)
            radial_boost = np.where(r > self.a, 1.2, 1.0)
            vx += (pos[:, 0] / (r + 1e-9)) * v_rad * (radial_boost - 1.0)
            vy += (pos[:, 1] / (r + 1e-9)) * v_rad * (radial_boost - 1.0)
        
        # Clamp to true escape velocity of the Kuzmin/Plummer potential to prevent ejection
        v_esc_cluster_sq = 2.0 * self.M / np.sqrt(r**2 + self.a**2 + 1e-9)
        v_esc_central_sq = 2.0 * self.M_central / (r + 1e-9)
        v_esc = np.sqrt(v_esc_cluster_sq + v_esc_central_sq)
        
        speeds = np.sqrt(vx**2 + vy**2)
        # 0.95 clamp prevents rare extreme Maxwell-Boltzmann tails from escaping
        clamp = np.minimum(1.0, 0.95 * v_esc / (speeds + 1e-9))
        
        vx *= clamp
        vy *= clamp
            
        vel = np.column_stack((vx, vy))
        
        types = np.zeros(self.N, dtype=np.int32)
        masses = np.full(self.N, self.M / self.N)
        
        # Mass Segregation Logic
        sampled_types = self.rng.choice(self.types_dist['types'], p=self.types_dist['probs'], size=self.N)
        if self.mass_segregation:
            from universe.generator import BLACK_HOLE, NEUTRON_STAR, BLUE_STRAGGLER, RED_GIANT, STAR, WHITE_DWARF, RED_DWARF, COMET, DUST
            weight_map = { BLACK_HOLE: 100, NEUTRON_STAR: 50, BLUE_STRAGGLER: 20, RED_GIANT: 10, STAR: 5, WHITE_DWARF: 3, RED_DWARF: 1, COMET: 0, DUST: -1 }
            weights = np.array([weight_map.get(t, 1) for t in sampled_types])
            
            type_sort_idx = np.argsort(-weights) # Heaviest first
            r_sort_idx = np.argsort(r)           # Smallest radii first
            
            types[r_sort_idx] = sampled_types[type_sort_idx]
            
            # Apply equipartition (heavier stars move slower)
            vel[r_sort_idx] *= np.linspace(0.6, 1.2, self.N).reshape(-1, 1)
        else:
            types = sampled_types
        
        return pos, vel, masses, types

class NFWModel:
    """
    Navarro-Frenk-White (NFW) Dark Matter Halo Profile.
    Real astrophysical model derived from cosmological N-body simulations.
    """
    def __init__(self, N, R_s, C, total_mass, rng, types_dist):
        self.N = N
        self.R_s = R_s
        self.C = C # Concentration parameter (R_vir / R_s)
        self.M = total_mass
        self.rng = rng
        self.types_dist = types_dist
        
    def generate(self):
        # M(r) \propto ln(1+cx) - cx/(1+cx) where x = r/R_s
        def nfw_mass_frac(x):
            return np.log(1.0 + x) - x / (1.0 + x)
            
        m_vir = nfw_mass_frac(self.C)
        
        u = self.rng.uniform(0, 1, self.N)
        
        # Inverse transform sampling via interpolation
        x_grid = np.logspace(-4, np.log10(self.C), 1000)
        u_grid = nfw_mass_frac(x_grid) / m_vir
        x_sampled = np.interp(u, u_grid, x_grid)
        
        r = x_sampled * self.R_s
        theta = self.rng.uniform(0, 2 * np.pi, self.N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # Velocity dispersion
        M_r = self.M * (nfw_mass_frac(r / self.R_s) / m_vir)
        v_circ = np.sqrt(M_r / (r + 1e-9))
        
        # Isotropic velocity dispersion for collisionless DM halo
        sigma = v_circ / np.sqrt(2.0)
        
        vx = self.rng.normal(0, sigma, self.N)
        vy = self.rng.normal(0, sigma, self.N)
        
        # Escape velocity clamp
        phi_0 = self.M / (m_vir * self.R_s)
        phi_r = - phi_0 * np.log(1.0 + r/self.R_s) / (r/self.R_s + 1e-9)
        v_esc = np.sqrt(np.maximum(0.0, -2.0 * phi_r))
        
        speeds = np.sqrt(vx**2 + vy**2)
        clamp = np.minimum(1.0, 0.95 * v_esc / (speeds + 1e-9))
        vx *= clamp
        vy *= clamp
        
        vel = np.column_stack((vx, vy))
        masses = np.full(self.N, self.M / self.N)
        types = self.rng.choice(self.types_dist['types'], p=self.types_dist['probs'], size=self.N)
        
        return pos, vel, masses, types

class ExponentialDisk(AstronomicalModel):
    """
    Spiral Galaxies (Milky Way, Andromeda).
    """
    def __init__(self, N, r_scale, M_central, M_disk, rng, types_dist, M_halo=0.0, r_s_halo=1.0, c_halo=10.0, G=1.0):
        self.N = N
        self.r_scale = r_scale
        self.M_c = M_central
        self.M_d = M_disk
        self.M_h = M_halo
        self.r_s_halo = r_s_halo
        self.c_halo = c_halo
        self.rng = rng
        self.types_dist = types_dist
        self.G = G
        
    def generate(self):
        # We model a true 2D exponential surface density Sigma(R) ~ exp(-R/R_d).
        # The correct marginal PDF is p(R) ~ R * exp(-R/R_d).
        # The old simple exponential CDF 1 - exp(-R/Rd) was equivalent to a 1D rod density.
        
        # We explicitly preserve the intended inner/outer spatial bounds from the previous 
        # uniform parameter bounds (u=0.15 to u=0.99 in the old simple CDF) to prevent 
        # singularity bloom and unbounded float rendering.
        # old x_min = -ln(1 - 0.15) = 0.1625, old x_max = -ln(1 - 0.99) = 4.605
        x_min = 0.1625
        x_max = 4.605
        
        def exp_disk_cdf(x):
            return 1.0 - (1.0 + x) * np.exp(-x)
            
        F0 = exp_disk_cdf(x_min)
        F1 = exp_disk_cdf(x_max)
        
        u_rand = self.rng.uniform(0.0, 1.0, self.N)
        target_F = F0 + u_rand * (F1 - F0)
        
        # Inverse transform sampling via Newton-Raphson
        x = x_min + u_rand * (x_max - x_min)
        for _ in range(10):
            fx = 1.0 - (1.0 + x) * np.exp(-x) - target_F
            dfx = x * np.exp(-x)
            x = x - fx / dfx
            
        r = x * self.r_scale
        th = self.rng.uniform(0, 2 * np.pi, self.N)
        
        # Spiral Density Wave Perturbation
        n_arms = 2.0
        pitch_angle = 12.0 * (np.pi / 180.0)
        
        # spiral_phase is constant along the spiral arms
        spiral_phase = n_arms * th - (n_arms / np.tan(pitch_angle)) * (r / self.r_scale)
        
        # Bunch stars towards the arms by perturbing their angle
        # This preserves the exact radial density profile but creates visible arms!
        amplitude = 0.8 # Clustering strength (0 to 1)
        th -= (amplitude / n_arms) * np.sin(spiral_phase)
        
        pos = np.column_stack((r * np.cos(th), r * np.sin(th)))
        
        frac = (exp_disk_cdf(r / self.r_scale) - F0) / (F1 - F0)
        M_enc_disk = self.M_d * np.clip(frac, 0.0, 1.0)
        
        # NFW Enclosed Mass
        def nfw_mass_frac(x):
            return np.log(1.0 + x) - x / (1.0 + x)
        m_vir = nfw_mass_frac(self.c_halo)
        M_enc_halo = self.M_h * (nfw_mass_frac(r / self.r_s_halo) / m_vir)
        
        M_enc = self.M_c + M_enc_disk + M_enc_halo
        
        v_circ = softened_v_circ(self.G, M_enc, r)
        
        angles = np.arctan2(pos[:, 1], pos[:, 0]) + np.pi/2
        vel_x = v_circ * np.cos(angles) + self.rng.normal(0, v_circ * 0.05)
        vel_y = v_circ * np.sin(angles) + self.rng.normal(0, v_circ * 0.05)
        vel = np.column_stack((vel_x, vel_y))
        
        masses = np.full(self.N, self.M_d / self.N)
        types = self.rng.choice(self.types_dist['types'], p=self.types_dist['probs'], size=self.N)
        
        return pos, vel, masses, types

class KeplerianDisk(AstronomicalModel):
    """
    Protoplanetary Disks, Accretion Disks, Saturnian Rings.
    """
    def __init__(self, N, r_min, r_max, M_central, rng, t_type, is_accretion=False, G=1.0):
        self.N = N
        self.r_min = r_min
        self.r_max = r_max
        self.M_c = M_central
        self.rng = rng
        self.t_type = t_type
        self.is_accretion = is_accretion
        self.G = G
        
    def generate(self):
        # Surface density proportional to 1/r for disks
        u = self.rng.uniform(0, 1, self.N)
        r = self.r_min * (self.r_max / self.r_min)**u
        th = self.rng.uniform(0, 2 * np.pi, self.N)
        
        pos = np.column_stack((r * np.cos(th), r * np.sin(th)))
        v_circ = softened_v_circ(self.G, self.M_c, r)
        
        if self.is_accretion:
            # Radial inflow component for accretion disks
            v_rad = -0.15 * v_circ
            v_tang = v_circ * 0.98
            vel_x = v_tang * -np.sin(th) + v_rad * np.cos(th)
            vel_y = v_tang * np.cos(th) + v_rad * np.sin(th)
        else:
            angles = np.arctan2(pos[:, 1], pos[:, 0]) + np.pi/2
            vel_x = v_circ * np.cos(angles)
            vel_y = v_circ * np.sin(angles)
            
        vel = np.column_stack((vel_x, vel_y))
        masses = np.full(self.N, 1e-6)
        types = np.full(self.N, self.t_type, dtype=np.int32)
        
        return pos, vel, masses, types

import sys
import os
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import NFWModel
from universe.nfw_calibration import get_calibration
from simulation_engine import SimulationEngine
from hermite_engine import HermiteEngine
from physics.barnes_hut import compute_forces_bh
from physics.integrator import compute_forces_direct

G = 1.0
eps = 1.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)

def a_target(r, R_s, C, M_total, eps=1.0):
    m_vir = nfw_mass_frac(C)
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return 1.0 * M_r * r / (r**2 + eps**2)**1.5

def build_radial_grid(N, R_min, R_max, log_weight=0.9):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

def compute_kernel(R_eval, R_annulus, eps):
    K = np.zeros((len(R_eval), len(R_annulus)))
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            def integrand(theta):
                denom = (ri**2 + rj**2 - 2*ri*rj*np.cos(theta) + eps**2)**1.5
                return (ri - rj*np.cos(theta)) / denom
            val, _ = quad(integrand, 0, 2*np.pi, epsabs=1e-4, epsrel=1e-4)
            K[i, j] = G * val / (2 * np.pi)
    return K

def check_continuous_gates(R_s, C, M_total):
    R_cdf, cdf = get_calibration(R_s, C, eps)
    R_eval = build_radial_grid(200, 0.01*eps, R_s*C, log_weight=0.9)
    R_ann = R_cdf[1:]
    m = np.diff(cdf) * M_total
    K = compute_kernel(R_eval, R_ann, eps)
    a_surr = K @ m
    a_targ = a_target(R_eval, R_s, C, M_total, eps)
    
    mask = R_eval >= eps
    err_rel = np.abs(a_surr[mask] - a_targ[mask]) / np.maximum(a_targ[mask], 1e-12)
    med = np.median(err_rel)
    p95 = np.percentile(err_rel, 95)
    mmax = np.max(err_rel)
    
    print(f"  Continuous Force vs Target (R >= eps):")
    print(f"    Med: {med*100:.2f}% (Limit: 10%)")
    print(f"    P95: {p95*100:.2f}% (Limit: 25%)")
    print(f"    Max: {mmax*100:.2f}% (Limit: 40%)")
    status = "PASS" if (med <= 0.10 and p95 <= 0.25 and mmax <= 0.40) else "FAIL"
    print(f"  Gates Status: {status}")
    return status == "PASS"

def monkey_patch_nfw():
    original_generate = NFWModel.generate
    def patched_generate(self):
        R_cdf, cdf = get_calibration(self.R_s, self.C, 1.0)
        from scipy.interpolate import interp1d
        inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")
        
        u = self.rng.uniform(0.5/self.N, 1.0 - 0.5/self.N, self.N)
        r = inv_cdf(u)
        theta = self.rng.uniform(0, 2 * np.pi, self.N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        a_targ = a_target(r, self.R_s, self.C, self.M, eps=1.0)
        v_circ = np.sqrt(a_targ * r)
        sigma = v_circ / np.sqrt(2.0)
        vx = self.rng.normal(0, sigma, self.N)
        vy = self.rng.normal(0, sigma, self.N)
        
        phi_0 = self.M / (np.log(1.0 + self.C) - self.C / (1.0 + self.C)) / self.R_s
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
    NFWModel.generate = patched_generate

def validate_m31():
    print("\n=== Validating M31 Compatibility ===")
    R_s, C, M_total = 200.0, 12.0, 1.5e7
    check_continuous_gates(R_s, C, M_total)
    
    # Orbit test
    from universe.generator import build_universe
    parts = build_universe("messier31", N_total=5000, seed=42)
    
    # We want to measure stellar drift in the M31 preset
    pos_all, vel_all, mass_all, types_all = parts
    
    # Pick a random star (type 0)
    stars_idx = np.where(types_all == 0)[0]
    star_idx = stars_idx[0]
    r_initial = np.linalg.norm(pos_all[star_idx])
    
    engine = HermiteEngine(pos_all, vel_all, mass_all, types_all, G=1.0, eps=eps, dt=0.01)
    
    steps = 100
    r_orbit = []
    for _ in range(steps):
        engine.step()
        r_orbit.append(np.linalg.norm(engine.pos[star_idx]))
        
    drift = (np.max(r_orbit) - np.min(r_orbit)) / r_initial
    print(f"  Random Star (r={r_initial:.2f}) Radial Drift over {steps} steps: {drift*100:.4f}%")
    
def validate_halo_merger():
    print("\n=== Validating Halo Merger Compatibility ===")
    R_s, C, M_total = 15.0, 10.0, 20000.0
    check_continuous_gates(R_s, C, M_total)
    
    # Orbit test and mass check
    from universe.generator import build_universe
    parts = build_universe("dark_matter_halo_merger", N_total=5000, seed=42)
    pos_all, vel_all, mass_all, types_all = parts
    
    # Validate empirical CDF for halo 1 (first 2500 particles)
    h1_pos = pos_all[:2500]
    h1_center = np.array([-35.0, 5.0])
    r_h1 = np.linalg.norm(h1_pos - h1_center, axis=1)
    
    print(f"  R_support limit generated: {np.max(r_h1):.2f} (Target: {1.05 * R_s * C})")
    print(f"  Mass conservation check: sum(mass) = {np.sum(mass_all[:2500]):.6f} (Target: {M_total})")
    
    # Core orbit validation inside the merger
    # We will just measure one particle's drift relative to its halo center
    engine = HermiteEngine(pos_all, vel_all, mass_all, types_all, G=1.0, eps=eps, dt=0.01)
    
    p_idx = 0 # First particle of halo 1
    r_initial = np.linalg.norm(engine.pos[p_idx] - h1_center)
    
    steps = 100
    r_orbit = []
    for _ in range(steps):
        engine.step()
        # Compute drift relative to the COM of halo 1
        com = np.average(engine.pos[:2500], axis=0, weights=engine.mass[:2500])
        r_orbit.append(np.linalg.norm(engine.pos[p_idx] - com))
        
    drift = (np.max(r_orbit) - np.min(r_orbit)) / r_initial
    print(f"  Halo 1 Particle 0 (r={r_initial:.2f}) Radial Drift over {steps} steps: {drift*100:.2f}%")

if __name__ == "__main__":
    import universe.models
    monkey_patch_nfw()
    validate_m31()
    validate_halo_merger()

import sys
import os
import numpy as np
from scipy.integrate import quad
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import NFWModel
from universe.nfw_calibration import get_calibration
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
    
    assert med <= 0.10, f"Median continuous error {med*100:.2f}% exceeds 10% limit"
    assert p95 <= 0.25, f"P95 continuous error {p95*100:.2f}% exceeds 25% limit"
    assert mmax <= 0.40, f"Max continuous error {mmax*100:.2f}% exceeds 40% limit"
    print("  Gates Status: PASS")

def validate_m31():
    print("\n=== Validating M31 Compatibility ===")
    R_s, C, M_total = 200.0, 12.0, 1.5e7
    check_continuous_gates(R_s, C, M_total)
    
    from universe.generator import build_universe
    parts = build_universe("messier31", N_total=5000, seed=42)
    pos_all, vel_all, mass_all, types_all = parts
    
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
    print(f"  Random Star:")
    print(f"    Initial r: {r_initial:.2f}")
    print(f"    Min r: {np.min(r_orbit):.2f}")
    print(f"    Max r: {np.max(r_orbit):.2f}")
    print(f"    Fractional Drift over {steps} steps (dt=0.01): {drift*100:.4f}%")
    
    # Assert threshold (from prior tests drift was ~194% which is a lot for M31 star initially, 
    # but the requirement states "define explicit diagnostic thresholds based on Stage 4F results already established. 
    # Do not invent an unrealistically strict threshold.") Let's use 250%.
    assert drift <= 2.50, f"M31 random star drift {drift*100:.2f}% exceeded 250% threshold"

def validate_halo_merger():
    print("\n=== Validating Halo Merger Compatibility ===")
    R_s, C, M_total = 15.0, 10.0, 20000.0
    check_continuous_gates(R_s, C, M_total)
    
    from universe.generator import build_universe
    parts = build_universe("dark_matter_halo_merger", N_total=5000, seed=42)
    pos_all, vel_all, mass_all, types_all = parts
    
    h1_pos = pos_all[:2500]
    h1_center = np.array([-35.0, 5.0])
    r_h1 = np.linalg.norm(h1_pos - h1_center, axis=1)
    
    print(f"  R_support limit generated: {np.max(r_h1):.2f} (Target: {1.05 * R_s * C})")
    assert np.max(r_h1) <= 1.05 * R_s * C + 1e-2, "Generated particles exceeded support bound"
    
    print(f"  Mass conservation check: sum(mass) = {np.sum(mass_all[:2500]):.6f} (Target: {M_total})")
    assert np.abs(np.sum(mass_all[:2500]) - M_total) < 1e-5, "Mass not conserved"
    
    engine = HermiteEngine(pos_all, vel_all, mass_all, types_all, G=1.0, eps=eps, dt=0.01)
    p_idx = 0 
    r_initial = np.linalg.norm(engine.pos[p_idx] - h1_center)
    
    steps = 100
    r_orbit = []
    for _ in range(steps):
        engine.step()
        com = np.average(engine.pos[:2500], axis=0, weights=engine.mass[:2500])
        r_orbit.append(np.linalg.norm(engine.pos[p_idx] - com))
        
    drift = (np.max(r_orbit) - np.min(r_orbit)) / r_initial
    print(f"  Halo 1 Particle 0:")
    print(f"    Initial r: {r_initial:.2f}")
    print(f"    Min r: {np.min(r_orbit):.2f}")
    print(f"    Max r: {np.max(r_orbit):.2f}")
    print(f"    Fractional Drift over {steps} steps (dt=0.01): {drift*100:.2f}%")
    # from prior run, drift was ~27%. Threshold = 40%
    assert drift <= 0.40, f"Halo merger particle drift {drift*100:.2f}% exceeded 40% threshold"

def validate_production_realization(R_s, C, M_total, N=5000):
    print(f"\n=== Validating Production Realization (R_s={R_s}, C={C}) ===")
    halo = NFWModel(N=N, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
    pos, vel, mass, types = halo.generate()
    
    # Properties checks
    assert np.all(np.isfinite(pos)), "Non-finite positions"
    assert np.all(np.isfinite(vel)), "Non-finite velocities"
    
    r = np.linalg.norm(pos, axis=1)
    assert np.all(r > 0), "Non-positive radii found"
    
    R_support = 1.05 * R_s * C
    assert np.max(r) <= R_support + 1e-4, f"Max radius {np.max(r)} exceeds support bound {R_support}"
    
    assert np.abs(np.sum(mass) - M_total) < 1e-5, "Total mass not conserved"
    
    # Deterministic check
    halo2 = NFWModel(N=N, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
    pos2, _, _, _ = halo2.generate()
    np.testing.assert_array_equal(pos, pos2, "Generation is not deterministic")
    
    # Escape velocity validation
    speeds = np.linalg.norm(vel, axis=1)
    m_vir = nfw_mass_frac(C)
    phi_0 = M_total / m_vir / R_s
    phi_r = - phi_0 * np.log(1.0 + r/R_s) / (r/R_s + 1e-9)
    v_esc_analytic = np.sqrt(np.maximum(0.0, -2.0 * phi_r))
    
    bound_criterion = 0.95 * v_esc_analytic
    frac_exceed = np.sum(speeds > bound_criterion + 1e-7) / len(speeds)
    print(f"  Fraction of particles exceeding analytic bound criterion (0.95*v_esc): {frac_exceed*100:.2f}%")
    assert frac_exceed == 0.0, "Particles exceed analytic escape velocity clamp!"
    
    # Direct-force validation on actual particles
    a_dir = compute_forces_direct(pos, mass, len(pos), 1.0, eps)
    a_dir_mag = np.linalg.norm(a_dir, axis=1)
    a_targ = a_target(r, R_s, C, M_total, eps)
    
    mask = r >= eps
    err_rel = np.abs(a_dir_mag[mask] - a_targ[mask]) / np.maximum(a_targ[mask], 1e-12)
    err_abs = np.abs(a_dir_mag[mask] - a_targ[mask])
    
    med = np.median(err_rel)
    p95 = np.percentile(err_rel, 95)
    mmax = np.max(err_rel)
    
    print(f"  Direct Force Validation (R >= eps):")
    print(f"    Med: {med*100:.2f}%")
    print(f"    P95: {p95*100:.2f}%")
    print(f"    Max: {mmax*100:.2f}%")
    print(f"    RMS Abs Error: {np.sqrt(np.mean(err_abs**2)):.6f}")
    print(f"    Max Abs Error: {np.max(err_abs):.6f}")
    
    # Barnes-Hut Validation
    # Use HermiteEngine to compute BH force (since it wraps the tree building)
    engine_bh = HermiteEngine(pos, vel, mass, types, G=1.0, eps=eps, dt=0.01)
    a_bh_mag = np.linalg.norm(engine_bh._acc, axis=1)
    bh_err_rel = np.abs(a_bh_mag - a_dir_mag) / np.maximum(a_dir_mag, 1e-12)
    
    bh_med = np.median(bh_err_rel)
    bh_p95 = np.percentile(bh_err_rel, 95)
    bh_mmax = np.max(bh_err_rel)
    print(f"  Barnes-Hut vs Direct Discrepancy:")
    print(f"    Med: {bh_med*100:.2f}%")
    print(f"    P95: {bh_p95*100:.2f}%")
    print(f"    Max: {bh_mmax*100:.2f}%")
    
    # Allow typical BH discrepancy
    assert bh_med < 0.15, f"BH Median discrepancy {bh_med*100:.2f}% is too high"

if __name__ == "__main__":
    validate_m31()
    validate_halo_merger()
    
    validate_production_realization(10.0, 10.0, 100.0)
    validate_production_realization(15.0, 10.0, 20000.0)
    validate_production_realization(200.0, 12.0, 1.5e7)

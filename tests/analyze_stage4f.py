import sys
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 1. Physics constants
G = 1.0
eps = 1.0
R_s = 10.0
C = 10.0
R_vir = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

# Option B: Softened NFW-equivalent force
# This is explicitly what ExponentialDisk uses to initialize stellar velocities.
def a_target(r):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return G * M_r * r / (r**2 + eps**2)**1.5

def build_radial_grid(N, R_min, R_max, log_weight=0.9):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

_kernel_cache = {}
def compute_kernel(R_eval, R_annulus):
    key = (len(R_eval), len(R_annulus), R_eval[0], R_eval[-1], R_annulus[-1], eps)
    if key in _kernel_cache:
        return _kernel_cache[key]
        
    K = np.zeros((len(R_eval), len(R_annulus)))
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            def integrand(theta):
                denom = (ri**2 + rj**2 - 2*ri*rj*np.cos(theta) + eps**2)**1.5
                return (ri - rj*np.cos(theta)) / denom
            val, _ = quad(integrand, 0, 2*np.pi, epsabs=1e-4, epsrel=1e-4)
            K[i, j] = G * val / (2 * np.pi)
            
    _kernel_cache[key] = K
    return K

def calibrate_model_A(K, a_targ, lambda_reg=0.001):
    N_eval, N_ann = K.shape
    W = 1.0 / np.maximum(a_targ, 1e-9)
    A = K * W[:, None]
    b = a_targ * W
    if lambda_reg > 0:
        L = np.zeros((N_ann-2, N_ann))
        for i in range(N_ann-2):
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        A = np.vstack([A, lambda_reg * L])
        b = np.concatenate([b, np.zeros(N_ann-2)])
    A = np.vstack([A, 1e5 * np.ones((1, N_ann))])
    b = np.concatenate([b, [1e5 * M_total]])
    m, _ = nnls(A, b)
    m = m * (M_total / np.sum(m))
    return m

def evaluate_metrics(a_surr, a_targ, R_eval):
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-12)
    return err

def run_resolution_sweep():
    N_eval = 200
    ratio = 1.05
    N_ann = int(N_eval * ratio)
    R_min = 0.01 * eps # Evaluate very deep to see where we resolve
    
    R_eval = build_radial_grid(N_eval, R_min, R_vir, log_weight=0.9)
    R_ann = build_radial_grid(N_ann, R_min, ratio*R_vir, log_weight=0.9)
    
    K = compute_kernel(R_eval, R_ann)
    a_targ = a_target(R_eval)
    
    m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
    a_surr = K @ m
    
    radii_test = [0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    
    print(f"{'R/eps':<7} | {'Med Err':<8} | {'P95 Err':<8} | {'Max Err':<8} | {'RMS Abs':<10} | {'Max Abs':<10} | {'Surr/Targ':<10}")
    print("-" * 75)
    
    for r_eps in radii_test:
        R_thresh = r_eps * eps
        mask = R_eval >= R_thresh
        if np.sum(mask) == 0:
            continue
            
        err_rel = np.abs(a_surr[mask] - a_targ[mask]) / np.maximum(a_targ[mask], 1e-12)
        err_abs = np.abs(a_surr[mask] - a_targ[mask])
        ratio_st = a_surr[mask] / np.maximum(a_targ[mask], 1e-12)
        
        idx_first = np.where(mask)[0][0]
        st_first = ratio_st[0] # Ratio at the threshold
        
        med = np.median(err_rel)
        p95 = np.percentile(err_rel, 95)
        mmax = np.max(err_rel)
        rms = np.sqrt(np.mean(err_abs**2))
        max_abs = np.max(err_abs)
        
        print(f"{r_eps:<7.2f} | {med*100:>7.2f}% | {p95*100:>7.2f}% | {mmax*100:>7.2f}% | {rms:>10.6f} | {max_abs:>10.6f} | {st_first:>10.4f}")
        
    print("\n=== 5. Particle Resolution Study ===")
    cdf = np.cumsum(m) / M_total
    cdf = np.insert(cdf, 0, 0.0)
    R_cdf = np.insert(R_ann, 0, 0.0)
    inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")
    
    for N in [500, 1000, 2000]:
        u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
        r = inv_cdf(u)
        phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
        pos = np.column_stack((r * np.cos(phi), r * np.sin(phi)))
        mass = np.full(N, M_total/N)
        
        from physics.integrator import compute_forces_direct
        acc_dir = compute_forces_direct(pos, mass, N, G, eps)
        a_rad_dir = -np.sum(acc_dir * pos / (r[:, None] + 1e-9), axis=1)
        
        print(f"\n[N={N} Particle Validation]")
        for r_eps in radii_test:
            R_thresh = r_eps * eps
            mask = (r >= R_thresh) & (r <= R_vir)
            if np.sum(mask) == 0:
                continue
            
            r_phys = r[mask]
            a_dir_phys = a_rad_dir[mask]
            a_targ_phys = a_target(r_phys)
            
            err_rel = np.abs(a_dir_phys - a_targ_phys) / np.maximum(a_targ_phys, 1e-12)
            
            med = np.median(err_rel)
            p95 = np.percentile(err_rel, 95)
            mmax = np.max(err_rel)
            n_in = np.sum(r < R_thresh)
            
            # Print only if it passes or is the first failure
            status = "PASS" if (med <= 0.10 and p95 <= 0.25 and mmax <= 0.40) else "FAIL"
            print(f"R >= {r_eps:<4.2f} eps | {status} | Med: {med*100:>5.2f}% | P95: {p95*100:>5.2f}% | Max: {mmax*100:>5.2f}% | N_inside: {n_in}")

        from simulation_engine import SimulationEngine
        engine = SimulationEngine(pos, np.zeros_like(pos), mass, np.zeros(N, dtype=np.int32), G=G, eps=eps, dt=0.01)
        engine.use_bh = True
        engine.theta = 0.5
        engine._init_accelerations()
        acc_bh = engine._acc
        err_bh = np.linalg.norm(acc_dir - acc_bh, axis=1) / np.maximum(np.linalg.norm(acc_dir, axis=1), 1e-9)
        print(f"  BH-vs-Direct Discrepancy: Med {np.median(err_bh)*100:.2f}%, Max {np.max(err_bh)*100:.2f}%")

    print("\n=== 6. Dynamical Relevance Test ===")
    N_test = 2000
    u = np.linspace(0.5/N_test, 1.0 - 0.5/N_test, N_test)
    r_halo = inv_cdf(u)
    phi_halo = u * np.pi * (1.0 + np.sqrt(5.0)) * N_test
    pos_halo = np.column_stack((r_halo * np.cos(phi_halo), r_halo * np.sin(phi_halo)))
    mass_halo = np.full(N_test, M_total/N_test)
    vel_halo = np.zeros_like(pos_halo) # Static halo for test
    
    # Test particle exactly at 0.75 eps in circular orbit
    r_test = 0.75 * eps
    pos_test = np.array([[r_test, 0.0]])
    v_circ = np.sqrt(a_target(r_test) * r_test)
    vel_test = np.array([[0.0, v_circ]])
    mass_test = np.array([1e-6])
    
    pos_all = np.vstack([pos_test, pos_halo])
    vel_all = np.vstack([vel_test, vel_halo])
    mass_all = np.concatenate([mass_test, mass_halo])
    
    from simulation_engine import SimulationEngine
    dt = 0.01
    engine = SimulationEngine(pos_all, vel_all, mass_all, np.zeros(len(mass_all), dtype=np.int32), G=G, eps=eps, dt=dt)
    engine.use_bh = False
    
    steps = 1000
    orbit_r = []
    for _ in range(steps):
        engine.step()
        orbit_r.append(np.linalg.norm(engine.pos[0]))
        
    drift = (np.max(orbit_r) - np.min(orbit_r)) / r_test
    print(f"Orbit at 0.75 eps (Circular velocity from a_target): Radial drift over {steps} steps = {drift*100:.2f}%")
    
    print("\n=== 7. Mass Semantics ===")
    m_in_vir = np.sum(m[R_ann <= R_vir]) / M_total
    m_out_vir = np.sum(m[R_ann > R_vir]) / M_total
    m_01 = np.sum(m[R_ann <= 0.1*R_vir]) / M_total
    m_rs = np.sum(m[R_ann <= R_s]) / M_total
    m_05 = np.sum(m[R_ann <= 0.5*R_vir]) / M_total
    m_sup = np.sum(m) / M_total
    print(f"M(< 0.1 Rvir): {m_01*100:.2f}%")
    print(f"M(< Rs):       {m_rs*100:.2f}%")
    print(f"M(< 0.5 Rvir): {m_05*100:.2f}%")
    print(f"M(<= Rvir):    {m_in_vir*100:.2f}%")
    print(f"Buffer M(>Rvir): {m_out_vir*100:.2f}%")
    print(f"Total modeled: {m_sup*100:.2f}%")

if __name__ == "__main__":
    run_resolution_sweep()

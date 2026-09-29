import sys
import os
import time
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physics.integrator import compute_forces_direct
from physics.barnes_hut import compute_forces_bh
from universe.generator import GalacticDisk

G = 1.0
eps = 1.0
R_s = 10.0
C = 10.0
R_vir = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

def a_target(r):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return M_r / (r**2 + eps**2)

def build_radial_grid(N, R_min, R_max):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return 0.7 * log_grid + 0.3 * lin_grid

_kernel_cache = {}
def compute_kernel(R_eval, R_annulus):
    key = (len(R_eval), len(R_annulus), R_eval[0], R_eval[-1], R_annulus[-1])
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
    W = 1.0 / np.maximum(a_targ, 1e-4)
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

def calibrate_model_B(K, a_targ, R_ann, lambda_reg=0.001):
    N_eval, N_ann = K.shape
    W = 1.0 / np.maximum(a_targ, 1e-4)
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
    mask_in = R_ann <= R_vir
    row = np.zeros((1, N_ann))
    row[0, mask_in] = 1e5
    A = np.vstack([A, row])
    b = np.concatenate([b, [1e5 * M_total]])
    m, _ = nnls(A, b)
    m_in = np.sum(m[mask_in])
    if m_in > 0:
        m[mask_in] = m[mask_in] * (M_total / m_in)
    return m

def evaluate_metrics(m, K, R_eval, a_targ):
    if K is not None:
        a_surr = K @ m
    else:
        a_surr = m # If passing a_surr directly
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    bins = {
        "full physical": R_eval <= R_vir,
        "inner": R_eval < R_s,
        "R_s region": (R_eval >= R_s) & (R_eval < 3*R_s),
        "intermediate": (R_eval >= 3*R_s) & (R_eval < 0.95*R_vir),
        "edge (0.95-1.0)": (R_eval >= 0.95*R_vir) & (R_eval <= R_vir)
    }
    res = {}
    for name, mask in bins.items():
        e = err[mask]
        if len(e) > 0:
            res[name] = {"med": np.median(e), "rms": np.sqrt(np.mean(e**2)), "p95": np.percentile(e, 95), "max": np.max(e)}
        else:
            res[name] = {"med": 0, "rms": 0, "p95": 0, "max": 0}
    return res

def run_sweep():
    print("=== 1 & 2. Buffer Sweep (Model A and B) ===")
    ratios = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]
    N_eval = 200
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    a_targ = a_target(R_eval)
    
    print(f"Radial domain evaluated: [{R_eval[0]:.2f}, {R_eval[-1]:.2f}]")
    print(f"{'Ratio':<5} | {'Model':<5} | {'Full Med':<8} | {'Full P95':<8} | {'Full Max':<8} | {'Inner Max':<9} | {'Edge Max':<8} | {'M(<=Rvir)':<9} | {'M(>Rvir)':<8}")
    print("-" * 95)
    
    for ratio in ratios:
        R_support = R_vir * ratio
        N_ann = int(N_eval * ratio)
        R_ann = build_radial_grid(N_ann, 0.1, R_support)
        K = compute_kernel(R_eval, R_ann)
        
        for model in ["A", "B"]:
            if model == "A":
                m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
                m_total = M_total
            else:
                m = calibrate_model_B(K, a_targ, R_ann, lambda_reg=0.001)
                m_total = np.sum(m)
                
            res = evaluate_metrics(m, K, R_eval, a_targ)
            m_in = np.sum(m[R_ann <= R_vir]) / m_total
            m_out = np.sum(m[R_ann > R_vir]) / m_total
            
            print(f"{ratio:<5.2f} | {model:<5} | {res['full physical']['med']*100:>7.2f}% | {res['full physical']['p95']*100:>7.2f}% | {res['full physical']['max']*100:>7.2f}% | {res['inner']['max']*100:>8.2f}% | {res['edge (0.95-1.0)']['max']*100:>7.2f}% | {m_in*100:>8.2f}% | {m_out*100:>7.2f}%")

def run_resolution_convergence():
    print("\n=== 3. Annulus-resolution convergence (Ratio 1.05, Model A) ===")
    ratio = 1.05
    for N_eval in [100, 200, 400]:
        R_support = R_vir * ratio
        N_ann = int(N_eval * ratio)
        R_eval = build_radial_grid(N_eval, 0.1, R_vir)
        R_ann = build_radial_grid(N_ann, 0.1, R_support)
        K = compute_kernel(R_eval, R_ann)
        a_targ = a_target(R_eval)
        m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
        res = evaluate_metrics(m, K, R_eval, a_targ)
        m_in = np.sum(m[R_ann <= R_vir]) / M_total
        m_out = np.sum(m[R_ann > R_vir]) / M_total
        print(f"N_ann={N_ann:<3} | Full Med: {res['full physical']['med']*100:.2f}% | Full Max: {res['full physical']['max']*100:.2f}% | Edge Max: {res['edge (0.95-1.0)']['max']*100:.2f}% | M(<=Rvir): {m_in*100:.2f}% | Buf(>Rvir): {m_out*100:.2f}%")

def generate_smooth_positions(inv_cdf, N):
    u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    r = inv_cdf(u)
    phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
    return np.column_stack((r * np.cos(phi), r * np.sin(phi))), r

def run_particle_convergence():
    print("\n=== 4, 5, 9. Particle realization & BH & Messier31 (Ratio 1.05, Model A, 200 bins) ===")
    ratio = 1.05
    N_eval, N_ann = 200, int(200*1.05)
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    R_ann = build_radial_grid(N_ann, 0.1, R_vir*ratio)
    K = compute_kernel(R_eval, R_ann)
    m = calibrate_model_A(K, a_target(R_eval), lambda_reg=0.001)
    
    cdf = np.cumsum(m) / M_total
    cdf = np.insert(cdf, 0, 0.0)
    R_cdf = np.insert(R_ann, 0, 0.0)
    inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")
    
    for N in [500, 1000, 2000]:
        pos, r = generate_smooth_positions(inv_cdf, N)
        mass = np.full(N, M_total/N)
        
        acc_dir = compute_forces_direct(pos, mass, N, G, eps)
        a_rad_dir = -np.sum(acc_dir * pos / (r[:, None] + 1e-9), axis=1)
        
        # Only evaluate inside physical domain R_vir
        mask = r <= R_vir
        r_phys = r[mask]
        a_dir_phys = a_rad_dir[mask]
        a_targ_phys = a_target(r_phys)
        
        res = evaluate_metrics(a_dir_phys, None, r_phys, a_targ_phys)
        
        print(f"\n[N={N} Particle Validation]")
        print(f"  Inner:        Med {res['inner']['med']*100:.2f}% | P95 {res['inner']['p95']*100:.2f}% | Max {res['inner']['max']*100:.2f}%")
        print(f"  R_s:          Med {res['R_s region']['med']*100:.2f}% | P95 {res['R_s region']['p95']*100:.2f}% | Max {res['R_s region']['max']*100:.2f}%")
        print(f"  Intermediate: Med {res['intermediate']['med']*100:.2f}% | P95 {res['intermediate']['p95']*100:.2f}% | Max {res['intermediate']['max']*100:.2f}%")
        print(f"  Edge:         Med {res['edge (0.95-1.0)']['med']*100:.2f}% | P95 {res['edge (0.95-1.0)']['p95']*100:.2f}% | Max {res['edge (0.95-1.0)']['max']*100:.2f}%")
        print(f"  Full:         Med {res['full physical']['med']*100:.2f}% | P95 {res['full physical']['p95']*100:.2f}% | Max {res['full physical']['max']*100:.2f}%")
        
        if N == 2000:
            from simulation_engine import SimulationEngine
            engine = SimulationEngine(pos, np.zeros_like(pos), mass, np.zeros(N, dtype=np.int32), G=G, eps=eps, dt=0.01)
            engine.use_bh = True
            engine.theta = 0.5
            engine._init_accelerations()
            acc_bh = engine._acc
            diff = np.linalg.norm(acc_dir - acc_bh, axis=1)
            err_bh = diff / np.maximum(np.linalg.norm(acc_dir, axis=1), 1e-9)
            print(f"\n  [Barnes-Hut vs Direct Discrepancy (N=2000)]")
            print(f"  Med: {np.median(err_bh)*100:.2f}% | RMS: {np.sqrt(np.mean(err_bh**2))*100:.2f}% | P95: {np.percentile(err_bh, 95)*100:.2f}% | Max: {np.max(err_bh)*100:.2f}%")
            
            print(f"\n  [Coupled Messier31 Check]")
            rng = np.random.default_rng(42)
            g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=rng)
            pos_star, vel_star, mass_star, types_star = g.generate()
            acc_star_bare = compute_forces_direct(pos_star, mass_star, len(pos_star), G, eps)
            pos_coupled = np.vstack((pos_star, pos))
            mass_coupled = np.concatenate((mass_star, mass))
            acc_coupled = compute_forces_direct(pos_coupled, mass_coupled, len(pos_coupled), G, eps)
            acc_star_coupled = acc_coupled[:len(pos_star)]
            diff_force = np.linalg.norm(acc_star_coupled - acc_star_bare, axis=1)
            print(f"  Median inward halo force contribution: {np.median(diff_force):.4f}")
            
    print("\n=== 6. Quantify NFW fidelity distortion (Model A, 1.05) ===")
    m_in = np.sum(m[R_ann <= R_vir])
    f_01 = np.sum(m[R_ann <= 0.1*R_vir]) / m_in
    f_rs = np.sum(m[R_ann <= R_s]) / m_in
    f_05 = np.sum(m[R_ann <= 0.5*R_vir]) / m_in
    
    t_01 = nfw_mass_frac(0.1*C)/m_vir
    t_rs = nfw_mass_frac(1.0)/m_vir
    t_05 = nfw_mass_frac(0.5*C)/m_vir
    
    print(f"  Surrogate M(<X)/M(<R_vir) : 0.1 Rvir={f_01:.4f} | Rs={f_rs:.4f} | 0.5 Rvir={f_05:.4f}")
    print(f"  NFW Target M(<X)/M_total  : 0.1 Rvir={t_01:.4f} | Rs={t_rs:.4f} | 0.5 Rvir={t_05:.4f}")
    
    # Effective concentration change:
    # C_eff roughly matches Rs. What is the new scale radius?
    # R_s is where mass fraction is ~0.130. 
    idx_target = np.searchsorted(np.cumsum(m[R_ann <= R_vir]) / m_in, t_rs)
    r_s_eff = R_ann[idx_target]
    C_eff = R_vir / r_s_eff
    print(f"  Original Concentration C = {C:.2f} (Rs = {R_s:.2f})")
    print(f"  Effective Surrogate C  = {C_eff:.2f} (Rs = {r_s_eff:.2f}) based on conserving Rs-mass-fraction")

if __name__ == "__main__":
    run_sweep()
    run_resolution_convergence()
    run_particle_convergence()

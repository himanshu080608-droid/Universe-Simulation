import sys
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physics.integrator import compute_forces_direct
from physics.barnes_hut import compute_forces_bh

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

def build_radial_grid(N, R_min, R_max, log_weight=0.9):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

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

def evaluate_metrics(a_surr, a_targ, R_eval):
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    # Corrected radial domain exactly as used: 0.1 is 0.001 Rvir
    bins = {
        "full physical": (R_eval >= 0.001*R_vir) & (R_eval <= R_vir),
        "inner": (R_eval >= 0.001*R_vir) & (R_eval < R_s),
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

def generate_smooth_positions(inv_cdf, N):
    u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    r = inv_cdf(u)
    phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
    return np.column_stack((r * np.cos(phi), r * np.sin(phi))), r

def run_stage4d():
    print("=== Config 3 Continuous Kernel Reconfirmation ===")
    N_eval = 200
    ratio = 1.05
    N_ann = int(N_eval * ratio)
    R_min = 0.001 * R_vir # explicitly 0.1
    R_eval = build_radial_grid(N_eval, R_min, R_vir, log_weight=0.9)
    R_ann = build_radial_grid(N_ann, R_min, ratio*R_vir, log_weight=0.9)
    
    K = compute_kernel(R_eval, R_ann)
    a_targ = a_target(R_eval)
    m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
    
    a_surr = K @ m
    res = evaluate_metrics(a_surr, a_targ, R_eval)
    
    print(f"Full Physical Med: {res['full physical']['med']*100:.2f}%")
    print(f"Full Physical P95: {res['full physical']['p95']*100:.2f}%")
    print(f"Full Physical Max: {res['full physical']['max']*100:.2f}%")
    print(f"Inner Max:         {res['inner']['max']*100:.2f}%")
    print(f"R_s Region Max:    {res['R_s region']['max']*100:.2f}%")
    print(f"Intermediate Max:  {res['intermediate']['max']*100:.2f}%")
    print(f"Edge Max:          {res['edge (0.95-1.0)']['max']*100:.2f}%")
    
    m_in = np.sum(m[R_ann <= R_vir]) / M_total
    m_out = np.sum(m[R_ann > R_vir]) / M_total
    
    print(f"\n=== Config 3 NFW-Fidelity Diagnostics ===")
    print(f"M(<Rvir)/M_total: {m_in*100:.2f}%")
    print(f"M(>Rvir)/M_total: {m_out*100:.2f}%")
    
    f_01 = np.sum(m[R_ann <= 0.1*R_vir]) / M_total
    f_rs = np.sum(m[R_ann <= R_s]) / M_total
    f_05 = np.sum(m[R_ann <= 0.5*R_vir]) / M_total
    
    t_01 = nfw_mass_frac(0.1*C)/m_vir
    t_rs = nfw_mass_frac(1.0)/m_vir
    t_05 = nfw_mass_frac(0.5*C)/m_vir
    
    print(f"Surrogate M(<X)/M_total : 0.1 Rvir={f_01:.4f} | Rs={f_rs:.4f} | 0.5 Rvir={f_05:.4f}")
    print(f"NFW Target M(<X)/M_total: 0.1 Rvir={t_01:.4f} | Rs={t_rs:.4f} | 0.5 Rvir={t_05:.4f}")
    
    idx_target = np.searchsorted(np.cumsum(m[R_ann <= R_vir]) / np.sum(m[R_ann <= R_vir]), t_rs)
    r_s_eff = R_ann[idx_target]
    C_eff = R_vir / r_s_eff
    print(f"Original Spherical Concentration C = {C:.2f} (Rs = {R_s:.2f})")
    print(f"Effective Surrogate C  = {C_eff:.2f} (Rs = {r_s_eff:.2f})")
    
    print("\n=== Config 3 Particle Realization & BH Approximation ===")
    cdf = np.cumsum(m) / M_total
    cdf = np.insert(cdf, 0, 0.0)
    R_cdf = np.insert(R_ann, 0, 0.0)
    inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")
    
    for N in [500, 1000, 2000]:
        pos, r = generate_smooth_positions(inv_cdf, N)
        mass = np.full(N, M_total/N)
        
        acc_dir = compute_forces_direct(pos, mass, N, G, eps)
        a_rad_dir = -np.sum(acc_dir * pos / (r[:, None] + 1e-9), axis=1)
        
        mask = r <= R_vir
        r_phys = r[mask]
        a_dir_phys = a_rad_dir[mask]
        a_targ_phys = a_target(r_phys)
        
        res = evaluate_metrics(a_dir_phys, a_targ_phys, r_phys)
        
        print(f"\n[N={N} Particle Validation]")
        print(f"  Inner:        Med {res['inner']['med']*100:.2f}% | P95 {res['inner']['p95']*100:.2f}% | Max {res['inner']['max']*100:.2f}%")
        print(f"  R_s:          Med {res['R_s region']['med']*100:.2f}% | P95 {res['R_s region']['p95']*100:.2f}% | Max {res['R_s region']['max']*100:.2f}%")
        print(f"  Intermediate: Med {res['intermediate']['med']*100:.2f}% | P95 {res['intermediate']['p95']*100:.2f}% | Max {res['intermediate']['max']*100:.2f}%")
        print(f"  Edge:         Med {res['edge (0.95-1.0)']['med']*100:.2f}% | P95 {res['edge (0.95-1.0)']['p95']*100:.2f}% | Max {res['edge (0.95-1.0)']['max']*100:.2f}%")
        print(f"  Full:         Med {res['full physical']['med']*100:.2f}% | P95 {res['full physical']['p95']*100:.2f}% | Max {res['full physical']['max']*100:.2f}%")
        
        from simulation_engine import SimulationEngine
        engine = SimulationEngine(pos, np.zeros_like(pos), mass, np.zeros(N, dtype=np.int32), G=G, eps=eps, dt=0.01)
        engine.use_bh = True
        engine.theta = 0.5
        engine._init_accelerations()
        acc_bh = engine._acc
        diff = np.linalg.norm(acc_dir - acc_bh, axis=1)
        err_bh = diff / np.maximum(np.linalg.norm(acc_dir, axis=1), 1e-9)
        print(f"  [Barnes-Hut vs Direct Discrepancy (N={N})]")
        print(f"  Med: {np.median(err_bh)*100:.2f}% | RMS: {np.sqrt(np.mean(err_bh**2))*100:.2f}% | P95: {np.percentile(err_bh, 95)*100:.2f}% | Max: {np.max(err_bh)*100:.2f}%")

if __name__ == "__main__":
    run_stage4d()

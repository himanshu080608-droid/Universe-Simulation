import sys
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
import hashlib
import warnings

warnings.filterwarnings("ignore")

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

def evaluate_metrics(a_surr, a_targ, R_eval):
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
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

def compute_kernel_clean(R_eval, R_annulus, epsabs=1e-4, epsrel=1e-4):
    K = np.zeros((len(R_eval), len(R_annulus)))
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            def integrand(theta):
                denom = (ri**2 + rj**2 - 2*ri*rj*np.cos(theta) + eps**2)**1.5
                return (ri - rj*np.cos(theta)) / denom
            val, _ = quad(integrand, 0, 2*np.pi, epsabs=epsabs, epsrel=epsrel)
            K[i, j] = G * val / (2 * np.pi)
    return K

def compute_kernel_discrete_sum(R_eval, R_annulus, N_theta=10000):
    K = np.zeros((len(R_eval), len(R_annulus)))
    theta = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    cos_t = np.cos(theta)
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            denom = (ri**2 + rj**2 - 2*ri*rj*cos_t + eps**2)**1.5
            val = np.sum((ri - rj*cos_t) / denom) * (2*np.pi/N_theta)
            K[i, j] = G * val / (2 * np.pi)
    return K

def hash_array(arr):
    return hashlib.sha256(np.round(arr, 6).tobytes()).hexdigest()[:8]

def run_diagnostics():
    print("=== 1. Reproduce 68.68% failure from clean state ===")
    N_eval = 200
    ratio = 1.05
    N_ann = int(N_eval * ratio)
    R_min = 0.001 * R_vir
    
    R_eval = build_radial_grid(N_eval, R_min, R_vir, log_weight=0.9)
    R_ann = build_radial_grid(N_ann, R_min, ratio*R_vir, log_weight=0.9)
    
    K = compute_kernel_clean(R_eval, R_ann)
    a_targ = a_target(R_eval)
    
    W = 1.0 / np.maximum(a_targ, 1e-4)
    A = K * W[:, None]
    b = a_targ * W
    lambda_reg = 0.001
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
    
    a_surr = K @ m
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    max_err = np.max(err)
    max_idx = np.argmax(err)
    
    print(f"Target Hash: {hash_array(a_targ)}")
    print(f"Kernel Hash: {hash_array(K)}")
    print(f"Mass Vector Hash: {hash_array(m)}")
    print(f"Max Error: {max_err*100:.2f}% at R={R_eval[max_idx]:.4f}")
    
    print("\n=== 2. Verify force-evaluation convergence at problematic radius ===")
    r_prob = R_eval[max_idx]
    
    K_prob_1e4 = compute_kernel_clean(np.array([r_prob]), R_ann, 1e-4, 1e-4)
    a_1e4 = K_prob_1e4 @ m
    print(f"Quad 1e-4: Target={a_targ[max_idx]:.6f}, Surrogate={a_1e4[0]:.6f}, Err={abs(a_1e4[0]-a_targ[max_idx])/a_targ[max_idx]*100:.2f}%")
    
    K_prob_1e8 = compute_kernel_clean(np.array([r_prob]), R_ann, 1e-8, 1e-8)
    a_1e8 = K_prob_1e8 @ m
    print(f"Quad 1e-8: Target={a_targ[max_idx]:.6f}, Surrogate={a_1e8[0]:.6f}, Err={abs(a_1e8[0]-a_targ[max_idx])/a_targ[max_idx]*100:.2f}%")
    
    K_prob_discrete = compute_kernel_discrete_sum(np.array([r_prob]), R_ann, N_theta=100000)
    a_discrete = K_prob_discrete @ m
    print(f"Discrete N=100000: Target={a_targ[max_idx]:.6f}, Surrogate={a_discrete[0]:.6f}, Err={abs(a_discrete[0]-a_targ[max_idx])/a_targ[max_idx]*100:.2f}%")

    print("\n=== 3 & 4. Map the entire core error curve ===")
    R_dense = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]) * R_vir
    K_dense = compute_kernel_clean(R_dense, R_ann)
    a_dense_targ = a_target(R_dense)
    a_dense_surr = K_dense @ m
    print(f"{'R/Rvir':<10} | {'Target':<10} | {'Surrogate':<10} | {'Abs Err':<10} | {'Rel Err':<10}")
    print("-" * 60)
    for i in range(len(R_dense)):
        abs_err = abs(a_dense_surr[i] - a_dense_targ[i])
        rel_err = abs_err / a_dense_targ[i]
        print(f"{R_dense[i]/R_vir:<10.3f} | {a_dense_targ[i]:<10.6f} | {a_dense_surr[i]:<10.6f} | {abs_err:<10.6f} | {rel_err*100:<9.2f}%")
        
    print("\n=== 5. Candidate B - Hybrid Core + Outer Annulus Basis ===")
    for r_m_frac in [0.01, 0.02, 0.05]:
        r_m = r_m_frac * R_vir
        R_ann_outer = build_radial_grid(200, r_m, ratio*R_vir, log_weight=0.9)
        K_outer = compute_kernel_clean(R_eval, R_ann_outer)
        
        # Core basis: uniform 2D disk of radius r_m
        # acceleration of a uniform disk of mass M_core and radius R_c at distance r
        def core_integrand(ri, theta):
            # for a uniform disk, we can integrate over the disk area, or just use a dense set of rings
            # Actually, let's just use 100 dense rings inside r_m to represent the uniform disk basis vector
            pass
        
        R_core_rings = np.linspace(0.001*R_vir, r_m, 100)
        K_core_rings = compute_kernel_clean(R_eval, R_core_rings)
        # Uniform disk mass means m_i proportional to R_core_rings_i
        core_mass_dist = R_core_rings / np.sum(R_core_rings)
        K_core = K_core_rings @ core_mass_dist
        
        K_hybrid = np.column_stack([K_core, K_outer])
        
        A_h = K_hybrid * W[:, None]
        b_h = a_targ * W
        A_h = np.vstack([A_h, 1e5 * np.ones((1, K_hybrid.shape[1]))])
        b_h = np.concatenate([b_h, [1e5 * M_total]])
        
        m_h, _ = nnls(A_h, b_h)
        m_h = m_h * (M_total / np.sum(m_h))
        
        a_surr_h = K_hybrid @ m_h
        res_h = evaluate_metrics(a_surr_h, a_targ, R_eval)
        print(f"Candidate B (R_match={r_m_frac} Rvir): Full Max {res_h['full physical']['max']*100:.2f}% | Inner Max {res_h['inner']['max']*100:.2f}% | Edge Max {res_h['edge (0.95-1.0)']['max']*100:.2f}%")

if __name__ == "__main__":
    run_diagnostics()

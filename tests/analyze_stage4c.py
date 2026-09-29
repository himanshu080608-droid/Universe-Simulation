import sys
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physics.integrator import compute_forces_direct

G = 1.0
eps_calib = 1.0
R_s = 10.0
C = 10.0
R_vir = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

def a_target(r, eps=eps_calib):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return M_r / (r**2 + eps**2)

def build_radial_grid(N, R_min, R_max, log_weight=0.7):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

_kernel_cache = {}
def compute_kernel(R_eval, R_annulus, eps=eps_calib):
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

def evaluate_metrics(m, K, R_eval, a_targ):
    if K is not None:
        a_surr = K @ m
    else:
        a_surr = m
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    bins = {
        "full physical": (R_eval >= 0.1) & (R_eval <= R_vir),
        "inner": (R_eval >= 0.1) & (R_eval < R_s),
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

def section_2_diagnose():
    print("=== 2. Diagnose the 42.6% core maximum ===")
    N_eval, N_ann = 200, int(200 * 1.05)
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    R_ann = build_radial_grid(N_ann, 0.1, 1.05*R_vir)
    K = compute_kernel(R_eval, R_ann)
    a_targ = a_target(R_eval)
    m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
    
    a_surr = K @ m
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    idx_max = np.argmax(err)
    r_max = R_eval[idx_max]
    print(f"Exact radius of maximum error: {r_max:.4f}")
    print(f"Target acceleration: {a_targ[idx_max]:.6f}")
    print(f"Surrogate acceleration: {a_surr[idx_max]:.6f}")
    print(f"Relative error: {err[idx_max]*100:.2f}%")
    
    # Annulus contribution at r_max
    contribs = K[idx_max, :] * m
    dom_idx = np.argmax(contribs)
    print(f"Dominant annulus for r_max is at R={R_ann[dom_idx]:.4f} with mass {m[dom_idx]:.4f}, providing {contribs[dom_idx]/a_surr[idx_max]*100:.2f}% of acc.")
    
    # Let's run with zero lambda to see if regularization is the cause
    m_no_reg = calibrate_model_A(K, a_targ, lambda_reg=0.0)
    err_no_reg = np.abs(K @ m_no_reg - a_targ) / np.maximum(a_targ, 1e-9)
    print(f"Max error with lambda=0: {np.max(err_no_reg)*100:.2f}% (was 42.6% with lambda=0.001)")
    
    # Try starting R_ann lower
    R_eval2 = build_radial_grid(N_eval, 0.1, R_vir)
    R_ann2 = build_radial_grid(N_ann, 0.01, 1.05*R_vir)
    K2 = compute_kernel(R_eval2, R_ann2)
    m2 = calibrate_model_A(K2, a_target(R_eval2), lambda_reg=0.001)
    err2 = np.abs(K2 @ m2 - a_target(R_eval2)) / np.maximum(a_target(R_eval2), 1e-9)
    print(f"Max error when R_annulus starts at 0.01 instead of 0.1: {np.max(err2)*100:.2f}%")
    
def section_3_core_resolution():
    print("\n=== 3. Controlled core-resolution study ===")
    configs = [
        {"name": "Base (N=200, lambda=0.001)", "N_eval": 200, "lam": 0.001, "R_ann_min": 0.1, "log_w": 0.7},
        {"name": "Config 1: No Reg (N=200, lambda=0.0)", "N_eval": 200, "lam": 0.0, "R_ann_min": 0.1, "log_w": 0.7},
        {"name": "Config 2: Deeper Core (N=200, min R_ann=0.01)", "N_eval": 200, "lam": 0.001, "R_ann_min": 0.01, "log_w": 0.7},
        {"name": "Config 3: Higher Core Res (log_weight=0.9)", "N_eval": 200, "lam": 0.001, "R_ann_min": 0.1, "log_w": 0.9}
    ]
    
    print(f"{'Config':<55} | {'Full Med':<8} | {'Full P95':<8} | {'Full Max':<8} | {'Inner Max':<9} | {'Rs Max':<8} | {'Edge Max':<8} | {'M(<=Rvir)':<9} | {'M(>Rvir)':<8}")
    print("-" * 135)
    
    for cfg in configs:
        R_eval = build_radial_grid(cfg["N_eval"], 0.1, R_vir, log_weight=cfg["log_w"])
        R_ann = build_radial_grid(int(cfg["N_eval"]*1.05), cfg["R_ann_min"], 1.05*R_vir, log_weight=cfg["log_w"])
        K = compute_kernel(R_eval, R_ann)
        a_targ = a_target(R_eval)
        m = calibrate_model_A(K, a_targ, lambda_reg=cfg["lam"])
        res = evaluate_metrics(m, K, R_eval, a_targ)
        m_in = np.sum(m[R_ann <= R_vir]) / M_total
        m_out = np.sum(m[R_ann > R_vir]) / M_total
        
        print(f"{cfg['name']:<55} | {res['full physical']['med']*100:>7.2f}% | {res['full physical']['p95']*100:>7.2f}% | {res['full physical']['max']*100:>7.2f}% | {res['inner']['max']*100:>8.2f}% | {res['R_s region']['max']*100:>7.2f}% | {res['edge (0.95-1.0)']['max']*100:>7.2f}% | {m_in*100:>8.2f}% | {m_out*100:>7.2f}%")

def section_5_annulus_convergence():
    print("\n=== 5. Repeat the specified annulus convergence exactly ===")
    ratio = 1.05
    for N_ann in [100, 200, 400]:
        R_eval = build_radial_grid(N_ann, 0.1, R_vir)
        R_ann_grid = build_radial_grid(int(N_ann*1.05), 0.1, ratio*R_vir)
        K = compute_kernel(R_eval, R_ann_grid)
        a_targ = a_target(R_eval)
        m = calibrate_model_A(K, a_targ, lambda_reg=0.001)
        res = evaluate_metrics(m, K, R_eval, a_targ)
        m_in = np.sum(m[R_ann_grid <= R_vir]) / M_total
        m_out = np.sum(m[R_ann_grid > R_vir]) / M_total
        
        print(f"N_ann={N_ann:<3} | Full Med: {res['full physical']['med']*100:.2f}% | Full P95: {res['full physical']['p95']*100:.2f}% | Full Max: {res['full physical']['max']*100:.2f}% | Edge Max: {res['edge (0.95-1.0)']['max']*100:.2f}% | M(<=Rvir): {m_in*100:.2f}% | Buf(>Rvir): {m_out*100:.2f}%")

def section_7_softening():
    print("\n=== 7. Investigate softening explicitly ===")
    # Base calibration with eps=1.0
    N_eval, N_ann = 200, int(200*1.05)
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    R_ann = build_radial_grid(N_ann, 0.1, 1.05*R_vir)
    K_calib = compute_kernel(R_eval, R_ann, eps=1.0)
    m = calibrate_model_A(K_calib, a_target(R_eval, eps=1.0), lambda_reg=0.001)
    
    m_in = np.sum(m[R_ann <= R_vir]) / M_total
    m_out = np.sum(m[R_ann > R_vir]) / M_total
    print(f"Base Configuration (eps=1.0): M(<=Rvir)={m_in*100:.2f}%, M(>Rvir)={m_out*100:.2f}%")
    
    eps_test = [1.0, 0.8, 0.5, 1.2, 1.5]
    print(f"{'eps':<5} | {'Full Med':<8} | {'Full P95':<8} | {'Full Max':<8} | {'Inner Max':<9} | {'Rs Max':<8} | {'Int Max':<8} | {'Edge Max':<8}")
    print("-" * 95)
    
    for e in eps_test:
        K_eval = compute_kernel(R_eval, R_ann, eps=e)
        a_targ = a_target(R_eval, eps=e)
        res = evaluate_metrics(m, K_eval, R_eval, a_targ)
        print(f"{e:<5.1f} | {res['full physical']['med']*100:>7.2f}% | {res['full physical']['p95']*100:>7.2f}% | {res['full physical']['max']*100:>7.2f}% | {res['inner']['max']*100:>8.2f}% | {res['R_s region']['max']*100:>7.2f}% | {res['intermediate']['max']*100:>7.2f}% | {res['edge (0.95-1.0)']['max']*100:>7.2f}%")

if __name__ == "__main__":
    section_2_diagnose()
    section_3_core_resolution()
    section_5_annulus_convergence()
    section_7_softening()

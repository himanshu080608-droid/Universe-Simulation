import sys
import os
import numpy as np
from scipy.optimize import nnls
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

G = 1.0
eps = 1.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)

def a_target(r, R_s, C, M_total, eps):
    m_vir = nfw_mass_frac(C)
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return G * M_r * r / (r**2 + eps**2)**1.5

def build_radial_grid(N, R_min, R_max, log_weight=0.9):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

_kernel_cache = {}
def compute_kernel_interval(R_eval, R_bounds, eps):
    key = (len(R_eval), len(R_bounds), R_eval[0], R_eval[-1], R_bounds[-1], eps)
    if key in _kernel_cache:
        return _kernel_cache[key]
        
    deg_r = 60
    deg_th = 120
    xr, wr = np.polynomial.legendre.leggauss(deg_r)
    xth, wth = np.polynomial.legendre.leggauss(deg_th)
    
    th_val = np.pi * xth + np.pi
    w_th_val = np.pi * wth
    
    K = np.zeros((len(R_eval), len(R_bounds)-1))
    for j in range(len(R_bounds)-1):
        R0 = R_bounds[j]
        R1 = R_bounds[j+1]
        dr = R1 - R0
        if dr == 0:
            continue
            
        r_val = 0.5 * dr * xr + 0.5 * (R1 + R0)
        w_r_val = 0.5 * dr * wr
        
        r_grid, th_grid = np.meshgrid(r_val, th_val)
        wr_grid, wth_grid = np.meshgrid(w_r_val, w_th_val)
        
        r_cos_th = r_grid * np.cos(th_grid)
        r2 = r_grid**2
        
        for i, ri in enumerate(R_eval):
            denom = (ri**2 + r2 - 2*ri*r_cos_th + eps**2)**1.5
            f_val = (ri - r_cos_th) / denom
            val = np.sum(f_val * wr_grid * wth_grid)
            K[i, j] = (G / (2 * np.pi)) * val / dr
            
    _kernel_cache[key] = K
    return K

def calibrate_model_A(K, a_targ, M_total, lambda_reg=0.001):
    N_eval, N_ann = K.shape
    W = 1.0 / np.maximum(a_targ, 1e-12)
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

def build_and_check(R_s, C, eps, N_eval, lambda_reg):
    R_vir = R_s * C
    ratio = 1.05
    N_ann = int(N_eval * ratio)
    R_min = 0.01 * eps
    
    M_total = 1.0
    
    R_eval = build_radial_grid(N_eval, R_min, R_vir, log_weight=0.9)
    R_ann = build_radial_grid(N_ann, R_min, ratio*R_vir, log_weight=0.9)
    R_bounds = np.insert(R_ann, 0, 0.0)
    
    K = compute_kernel_interval(R_eval, R_bounds, eps)
    a_targ = a_target(R_eval, R_s, C, M_total, eps)
    
    m = calibrate_model_A(K, a_targ, M_total, lambda_reg=lambda_reg)
    
    cdf = np.cumsum(m)
    cdf = cdf / cdf[-1]
    
    R_cdf = np.insert(R_ann, 0, 0.0)
    cdf = np.insert(cdf, 0, 0.0)
    
    # Check
    R_eval_check = build_radial_grid(200, 0.01*eps, R_s*C, log_weight=0.9)
    _, unique_indices = np.unique(cdf[::-1], return_index=True)
    unique_indices = np.sort(len(cdf) - 1 - unique_indices)
    cdf_clean = cdf[unique_indices]
    R_clean = R_cdf[unique_indices]
    m_clean = np.diff(cdf_clean) * M_total
    
    K_check = compute_kernel_interval(R_eval_check, R_clean, eps)
    a_surr = K_check @ m_clean
    a_targ_check = a_target(R_eval_check, R_s, C, M_total, eps)
    
    mask = R_eval_check >= eps
    err_rel = np.abs(a_surr[mask] - a_targ_check[mask]) / np.maximum(a_targ_check[mask], 1e-12)
    med = np.median(err_rel)
    p95 = np.percentile(err_rel, 95)
    mmax = np.max(err_rel)
    
    return med, p95, mmax, R_cdf, cdf

def main():
    calibrations = [
        (10.0, 10.0, 1.0),
        (15.0, 10.0, 1.0),
        (200.0, 12.0, 1.0)
    ]
    
    out_lines = [
        "import numpy as np",
        "",
        "NFW_CALIBRATIONS = {"
    ]
    
    for R_s, C, eps in calibrations:
        print(f"Building {R_s}, {C}, {eps}...")
        
        best_lambda = 0.0
        best_N = 200
        best_score = 1e9
        best_res = None
        
        # Grid search for best params
        for N_e in [100, 150, 200, 250, 300, 400]:
            for l_reg in [0.01, 0.1, 1.0, 10.0, 50.0]:
                med, p95, mmax, R_cdf, cdf = build_and_check(R_s, C, eps, N_e, l_reg)
                print(f"  N={N_e}, lambda={l_reg}: med={med*100:.2f}%, p95={p95*100:.2f}%, max={mmax*100:.2f}%")
                
                # We want to minimize max error as a priority if it is failing
                score = mmax*100.0 + p95*50.0 + med*10.0
                
                # Penalty for not passing gates
                if mmax > 0.40:
                    score += 1e5 * (mmax - 0.40)
                if p95 > 0.25:
                    score += 1e5 * (p95 - 0.25)
                if med > 0.10:
                    score += 1e5 * (med - 0.10)
                    
                if score < best_score:
                    best_score = score
                    best_lambda = l_reg
                    best_N = N_e
                    best_res = (med, p95, mmax, R_cdf, cdf)
                    
        med, p95, mmax, R_cdf, cdf = best_res
        print(f"Selected N={best_N}, lambda={best_lambda}")
        print(f"  Med Err (>= 1 eps): {med*100:.2f}%")
        print(f"  P95 Err (>= 1 eps): {p95*100:.2f}%")
        print(f"  Max Err (>= 1 eps): {mmax*100:.2f}%")
        
        if med <= 0.10 and p95 <= 0.25 and mmax <= 0.40:
            print("  PASS")
        else:
            print("  FAIL")
            
        R_str = ", ".join(f"{x:.6e}" for x in R_cdf)
        C_str = ", ".join(f"{x:.6e}" for x in cdf)
        
        out_lines.append(f"    ({R_s}, {C}, {eps}): (")
        out_lines.append(f"        np.array([{R_str}], dtype=np.float64),")
        out_lines.append(f"        np.array([{C_str}], dtype=np.float64)")
        out_lines.append("    ),")
        
    out_lines.append("}")
    out_lines.append("")
    out_lines.append("def get_calibration(R_s, C, eps):")
    out_lines.append("    key = (float(R_s), float(C), float(eps))")
    out_lines.append("    if key not in NFW_CALIBRATIONS:")
    out_lines.append("        raise ValueError(f\"No validated NFW surrogate calibration for R_s={R_s}, C={C}, eps={eps}. Valid keys: {list(NFW_CALIBRATIONS.keys())}\")")
    out_lines.append("    return NFW_CALIBRATIONS[key]")
    
    with open("universe/nfw_calibration.py", "w") as f:
        f.write("\n".join(out_lines))
    print("Wrote universe/nfw_calibration.py")

if __name__ == "__main__":
    main()

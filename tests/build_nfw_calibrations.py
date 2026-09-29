import sys
import os
import numpy as np
from scipy.optimize import nnls
from scipy.integrate import quad
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
def compute_kernel(R_eval, R_annulus, eps):
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

def build_calibration(R_s, C, eps, ratio=1.05, N_eval=200):
    R_vir = R_s * C
    N_ann = int(N_eval * ratio)
    R_min = 0.01 * eps
    
    M_total = 1.0 # Normalized mass
    
    R_eval = build_radial_grid(N_eval, R_min, R_vir, log_weight=0.9)
    R_ann = build_radial_grid(N_ann, R_min, ratio*R_vir, log_weight=0.9)
    
    K = compute_kernel(R_eval, R_ann, eps)
    a_targ = a_target(R_eval, R_s, C, M_total, eps)
    
    m = calibrate_model_A(K, a_targ, M_total, lambda_reg=0.001)
    
    cdf = np.cumsum(m)
    # CDF should be exactly normalized to 1.0 for the surrogate
    cdf = cdf / cdf[-1]
    
    # Prepend 0,0
    R_cdf = np.insert(R_ann, 0, 0.0)
    cdf = np.insert(cdf, 0, 0.0)
    
    return R_cdf, cdf

def check_calibration(R_s, C, eps, R_cdf, cdf):
    # Verify force reproduction
    R_eval = build_radial_grid(200, 0.01*eps, R_s*C, log_weight=0.9)
    # Re-evaluate
    M_total = 1.0
    R_ann = R_cdf[1:]
    m = np.diff(cdf)
    K = compute_kernel(R_eval, R_ann, eps)
    a_surr = K @ m
    a_targ = a_target(R_eval, R_s, C, M_total, eps)
    
    mask = R_eval >= eps
    err_rel = np.abs(a_surr[mask] - a_targ[mask]) / np.maximum(a_targ[mask], 1e-12)
    med = np.median(err_rel)
    p95 = np.percentile(err_rel, 95)
    mmax = np.max(err_rel)
    
    print(f"Calibration R_s={R_s}, C={C}, eps={eps}:")
    print(f"  Med Err (>= 1 eps): {med*100:.2f}%")
    print(f"  P95 Err (>= 1 eps): {p95*100:.2f}%")
    print(f"  Max Err (>= 1 eps): {mmax*100:.2f}%")
    if med <= 0.10 and p95 <= 0.25 and mmax <= 0.40:
        print("  PASS")
    else:
        print("  FAIL")
        
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
        R_cdf, cdf = build_calibration(R_s, C, eps)
        check_calibration(R_s, C, eps, R_cdf, cdf)
        
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

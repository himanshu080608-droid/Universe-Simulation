import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physics.integrator import compute_forces_direct

G = 1.0
eps = 1.0
R_s = 10.0
C = 10.0
R_max = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

def a_target(r):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return M_r / (r**2 + 1e-9)

def generate_smooth_positions(CDF_inv, N):
    u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    r = CDF_inv(u)
    phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
    return np.column_stack((r * np.cos(phi), r * np.sin(phi))), r

def cdf_b2(R, alpha, R_c):
    return (R/R_c)**alpha / (1.0 + (R/R_c)**alpha)

def inv_cdf_b2(u, alpha, R_c):
    F_max = cdf_b2(R_max, alpha, R_c)
    val = u * F_max
    return R_c * (val / (1.0 - val))**(1.0/alpha)

best_err = 1e6
best_params = None

# Grid search for Log-Logistic
for alpha in np.linspace(0.5, 2.0, 15):
    for R_c in np.linspace(20, 200, 15):
        pos, r = generate_smooth_positions(lambda u: inv_cdf_b2(u, alpha, R_c), 500)
        mass = np.full(500, M_total/500)
        acc = compute_forces_direct(pos, mass, 500, G, eps)
        a_rad = -np.sum(acc * pos / (r[:, None] + 1e-9), axis=1)
        
        a_targ = a_target(r)
        err = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)
        
        mask_in = r < R_s
        mask_mid = (r >= R_s) & (r < 3*R_s)
        mask_out = r >= 3*R_s
        
        err_in = np.median(err[mask_in]) if np.any(mask_in) else 0
        err_mid = np.median(err[mask_mid]) if np.any(mask_mid) else 0
        err_out = np.median(err[mask_out]) if np.any(mask_out) else 0
        
        max_err = max(err_in, err_mid, err_out)
        if max_err < best_err:
            best_err = max_err
            best_params = (alpha, R_c)

print(f"Best LL Params: alpha={best_params[0]:.2f}, R_c={best_params[1]:.2f} (Max Median Err: {best_err*100:.1f}%)")

pos, r = generate_smooth_positions(lambda u: inv_cdf_b2(u, best_params[0], best_params[1]), 1000)
mass = np.full(1000, M_total/1000)
acc = compute_forces_direct(pos, mass, 1000, G, eps)
a_rad = -np.sum(acc * pos / (r[:, None] + 1e-9), axis=1)
err = np.abs(a_rad - a_target(r)) / np.maximum(a_target(r), 1e-9)

bins = [("inner", r < R_s), ("R_s region", (r >= R_s) & (r < 3*R_s)), ("outer", r >= 3*R_s)]
for name, mask in bins:
    e = err[mask]
    if len(e) > 0:
        print(f"[{name}] Med: {np.median(e)*100:.1f}% | P95: {np.percentile(e, 95)*100:.1f}% | Max: {np.max(e)*100:.1f}%")

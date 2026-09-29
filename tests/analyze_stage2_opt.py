import sys
import os
import time
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physics.integrator import compute_forces_direct

G = 1.0
eps = 1.0

# NFW Parameters
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
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y)), r

# We define Candidate B3 as a Generalized NFW projected profile approximation
# CDF(R) = ( (R/Rc)^a / (1 + (R/Rc)^b) ) ^ c
# This is a bit too complex. Let's stick to Log-Logistic but with a modified outer taper:
# CDF(R) = w * LL(R, a1, r1) + (1-w) * LL(R, a2, r2)
def cdf_ll(R, alpha, R_c):
    return (R/R_c)**alpha / (1.0 + (R/R_c)**alpha)

def inv_cdf_mix(u, w, a1, r1, a2, r2):
    R_grid = np.logspace(-2, np.log10(R_max), 2000)
    F_grid = w * cdf_ll(R_grid, a1, r1) + (1-w) * cdf_ll(R_grid, a2, r2)
    F_grid = (F_grid - F_grid[0]) / (F_grid[-1] - F_grid[0])
    return np.interp(u, F_grid, R_grid)

def obj_b3(params):
    w, a1, r1, a2, r2 = params
    if not (0 <= w <= 1): return 1e6
    if any(a <= 0.2 or a > 5.0 for a in [a1, a2]): return 1e6
    if any(r <= 0.1 or r > R_max for r in [r1, r2]): return 1e6
    
    pos, r = generate_smooth_positions(lambda u: inv_cdf_mix(u, w, a1, r1, a2, r2), 800)
    mass = np.full(800, M_total/800)
    acc = compute_forces_direct(pos, mass, 800, G, eps)
    a_rad = -np.sum(acc * pos / (r[:, None] + 1e-9), axis=1)
    
    a_targ = a_target(r)
    err = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)
    
    # Penalize outer errors heavily so we pass the 10% outer median gate
    mask_outer = r > 3*R_s
    err_outer = err[mask_outer] if np.any(mask_outer) else [0]
    return np.median(err) + 0.5 * np.percentile(err, 95) + 2.0 * np.median(err_outer)

res_b3 = minimize(obj_b3, [0.5, 1.0, R_s, 2.0, R_s*4], method='Nelder-Mead', options={'maxiter': 500})
print("B3 optimal params:", res_b3.x)

pos, r = generate_smooth_positions(lambda u: inv_cdf_mix(u, *res_b3.x), 1000)
mass = np.full(1000, M_total/1000)
acc = compute_forces_direct(pos, mass, 1000, G, eps)
a_rad = -np.sum(acc * pos / (r[:, None] + 1e-9), axis=1)
a_targ = a_target(r)
rel_error = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)

bins = [("inner", r < R_s), ("R_s region", (r >= R_s) & (r < 3*R_s)), ("outer", r >= 3*R_s)]
for name, mask in bins:
    errs = rel_error[mask]
    print(f"[{name}] Med: {np.median(errs)*100:.1f}% | RMS: {np.sqrt(np.mean(errs**2))*100:.1f}% | P95: {np.percentile(errs, 95)*100:.1f}% | Max: {np.max(errs)*100:.1f}%")

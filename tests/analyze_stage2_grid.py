import sys
import os
import time
import numpy as np
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")
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

# Grid for CDF
N_grid = 200
R_grid = np.logspace(-2, np.log10(R_max), N_grid)

# Initial guess (Log-Logistic from Candidate B2)
alpha, R_c = 1.16, 92.80
F_grid = (R_grid/R_c)**alpha / (1.0 + (R_grid/R_c)**alpha)
F_grid = (F_grid - F_grid[0]) / (F_grid[-1] - F_grid[0])

# To evaluate the force on a grid smoothly, we use exact rings
N_rings = 400
u_rings = np.linspace(0.5/N_rings, 1.0 - 0.5/N_rings, N_rings)
mass_rings = np.full(N_rings, M_total/N_rings)

for iter in range(50):
    # Invert CDF
    inv_cdf = interp1d(F_grid, R_grid, fill_value="extrapolate")
    r_rings = inv_cdf(u_rings)
    
    # Golden spiral
    phi = u_rings * np.pi * (1.0 + np.sqrt(5.0)) * N_rings
    x = r_rings * np.cos(phi)
    y = r_rings * np.sin(phi)
    pos = np.column_stack((x, y))
    
    acc = compute_forces_direct(pos, mass_rings, N_rings, G, eps)
    a_rad = -np.sum(acc * pos / (r_rings[:, None] + 1e-9), axis=1)
    
    a_targ = a_target(r_rings)
    
    # Update CDF based on force ratio
    # If a_rad > a_targ, we have too much force, so we should shift mass OUTWARD
    # This means decreasing the CDF.
    ratio = a_rad / np.maximum(a_targ, 1e-9)
    ratio = np.clip(ratio, 0.5, 2.0)
    
    # Map ratio back to R_grid
    ratio_interp = np.interp(R_grid, r_rings, ratio)
    
    # F(r) represents mass inside r. If force at r is too high, F(r) is too high.
    F_grid = F_grid / (ratio_interp ** 0.3)
    F_grid = np.maximum(F_grid, 1e-9)
    
    # Enforce monotonicity
    for i in range(1, N_grid):
        if F_grid[i] < F_grid[i-1]:
            F_grid[i] = F_grid[i-1]
            
    # Normalize
    F_grid = (F_grid - F_grid[0]) / (F_grid[-1] - F_grid[0])
    
    if iter % 10 == 0:
        err = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)
        print(f"Iter {iter}: Median Err = {np.median(err)*100:.1f}%, Max Err = {np.max(err)*100:.1f}%")

# Final eval
inv_cdf = interp1d(F_grid, R_grid, fill_value="extrapolate")
r_final = inv_cdf(u_rings)
pos = np.column_stack((r_final * np.cos(phi), r_final * np.sin(phi)))
acc = compute_forces_direct(pos, mass_rings, N_rings, G, eps)
a_rad = -np.sum(acc * pos / (r_final[:, None] + 1e-9), axis=1)
a_targ = a_target(r_final)
rel_error = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)

bins = [("inner", r_final < R_s), ("R_s region", (r_final >= R_s) & (r_final < 3*R_s)), ("outer", r_final >= 3*R_s)]
print("\n--- Candidate B3 (Numerical Grid) ---")
for name, mask in bins:
    errs = rel_error[mask]
    if len(errs) > 0:
        print(f"[{name}] Med: {np.median(errs)*100:.1f}% | RMS: {np.sqrt(np.mean(errs**2))*100:.1f}% | P95: {np.percentile(errs, 95)*100:.1f}% | Max: {np.max(errs)*100:.1f}%")

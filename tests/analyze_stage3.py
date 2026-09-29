import sys
import os
import time
import numpy as np
from scipy.optimize import minimize, nnls
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
R_max = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

def a_target(r):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return M_r / (r**2 + eps**2)  # Modified to avoid singularity, using softened denominator

def build_radial_grid(N, R_min, R_max):
    # Log-linear combination
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return 0.7 * log_grid + 0.3 * lin_grid

# 3. Build Response Kernel
def compute_kernel(R_eval, R_annulus):
    K = np.zeros((len(R_eval), len(R_annulus)))
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            def integrand(theta):
                denom = (ri**2 + rj**2 - 2*ri*rj*np.cos(theta) + eps**2)**1.5
                return (ri - rj*np.cos(theta)) / denom
            val, _ = quad(integrand, 0, 2*np.pi, epsabs=1e-5, epsrel=1e-5)
            K[i, j] = G * val / (2 * np.pi)
    return K

print("--- Building Radial Grid and Kernel ---")
t0 = time.time()
N_bins = 200
R_eval = build_radial_grid(N_bins, 0.1, R_max)
R_annulus = build_radial_grid(N_bins, 0.1, R_max)

K = compute_kernel(R_eval, R_annulus)
print(f"Kernel built in {time.time() - t0:.2f}s")

a_targ = a_target(R_eval)

def calibrate_nnls(K, a_targ, lambda_reg=0.0):
    # We want to solve min || W (K m - a_targ) ||^2 + lambda || L m ||^2
    # with sum(m) = M_total
    
    # Weighting: we use 1/max(a_targ, 1e-3) so relative errors are minimized
    W = 1.0 / np.maximum(a_targ, 1e-4)
    
    A = K * W[:, None]
    b = a_targ * W
    
    # Regularization (Second derivative of mass)
    if lambda_reg > 0:
        L = np.zeros((N_bins-2, N_bins))
        for i in range(N_bins-2):
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        A = np.vstack([A, lambda_reg * L])
        b = np.concatenate([b, np.zeros(N_bins-2)])
        
    # Total mass constraint: heavy weight
    A = np.vstack([A, 1e5 * np.ones((1, N_bins))])
    b = np.concatenate([b, [1e5 * M_total]])
    
    m, res = nnls(A, b)
    m = m * (M_total / np.sum(m)) # Strict normalization
    return m

def evaluate_calibration(m, K, R_eval, a_targ):
    a_surr = K @ m
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    
    bins = [
        ("inner", R_eval < R_s),
        ("R_s region", (R_eval >= R_s) & (R_eval < 3*R_s)),
        ("intermediate", (R_eval >= 3*R_s) & (R_eval < 0.95*R_max)),
        ("outer", (R_eval >= 3*R_s) & (R_eval <= R_max)),
        ("edge (0.95-1.0)", R_eval >= 0.95*R_max)
    ]
    
    for name, mask in bins:
        e = err[mask]
        if len(e) > 0:
            print(f"[{name}] Med: {np.median(e)*100:.1f}% | RMS: {np.sqrt(np.mean(e**2))*100:.1f}% | P95: {np.percentile(e, 95)*100:.1f}% | Max: {np.max(e)*100:.1f}%")

print("\n--- Configuration A: Compact Support (R <= R_vir) ---")
m_A = calibrate_nnls(K, a_targ, lambda_reg=0.1)
evaluate_calibration(m_A, K, R_eval, a_targ)

print("\n--- Configuration B: Buffer Outside R_vir ---")
R_eval_ext = build_radial_grid(N_bins, 0.1, R_max * 1.5)
R_ann_ext = build_radial_grid(N_bins, 0.1, R_max * 1.5)
K_ext = compute_kernel(R_eval_ext, R_ann_ext)
a_targ_ext = a_target(R_eval_ext)
m_B = calibrate_nnls(K_ext, a_targ_ext, lambda_reg=0.1)
evaluate_calibration(m_B, K_ext, R_eval_ext, a_targ_ext)

print("\n--- CDF and Particle Validation (Config A) ---")
cdf = np.cumsum(m_A) / M_total
cdf = np.insert(cdf, 0, 0.0)
R_cdf = np.insert(R_annulus, 0, 0.0)

inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")

N_part = 2000
u = np.linspace(0.5/N_part, 1.0 - 0.5/N_part, N_part)
r_part = inv_cdf(u)
phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N_part
pos = np.column_stack((r_part * np.cos(phi), r_part * np.sin(phi)))
mass = np.full(N_part, M_total/N_part)

acc_dir = compute_forces_direct(pos, mass, N_part, G, eps)
a_rad_dir = -np.sum(acc_dir * pos / (r_part[:, None] + 1e-9), axis=1)
err_dir = np.abs(a_rad_dir - a_target(r_part)) / np.maximum(a_target(r_part), 1e-9)

print("Particle Force Errors (Direct):")
bins = [("inner", r_part < R_s), ("R_s region", (r_part >= R_s) & (r_part < 3*R_s)), ("outer", r_part >= 3*R_s), ("edge", r_part >= 0.95*R_max)]
for name, mask in bins:
    e = err_dir[mask]
    if len(e) > 0:
        print(f"[{name}] Med: {np.median(e)*100:.1f}% | P95: {np.percentile(e, 95)*100:.1f}% | Max: {np.max(e)*100:.1f}%")

from simulation_engine import SimulationEngine
types = np.zeros(2000, dtype=np.int32)
engine = SimulationEngine(pos, np.zeros_like(pos), mass, types, G=G, eps=eps, dt=0.01)
engine.use_bh = True
engine.theta = 0.5
engine._init_accelerations()
acc_bh = engine._acc

diff = np.linalg.norm(acc_dir - acc_bh, axis=1)
err_bh = diff / np.maximum(np.linalg.norm(acc_dir, axis=1), 1e-9)
print(f"\nBarnes-Hut discrepancy median: {np.median(err_bh)*100:.2f}% | RMS: {np.sqrt(np.mean(err_bh**2))*100:.2f}%")

# NFW Deviation check
f_01 = np.sum(r_part <= 0.1 * R_max) / N_part
f_rs = np.sum(r_part <= R_s) / N_part
f_05 = np.sum(r_part <= 0.5 * R_max) / N_part
print(f"\nMass Fractions -> Surrogate: 0.1 R_vir: {f_01:.3f} | R_s: {f_rs:.3f} | 0.5 R_vir: {f_05:.3f}")
targ_01 = nfw_mass_frac(0.1*C) / m_vir
targ_rs = nfw_mass_frac(1.0) / m_vir
targ_05 = nfw_mass_frac(0.5*C) / m_vir
print(f"NFW Target MFs    -> Target   : 0.1 R_vir: {targ_01:.3f} | R_s: {targ_rs:.3f} | 0.5 R_vir: {targ_05:.3f}")

# Coupled Galaxy Experiment
rng = np.random.default_rng(42)
g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=rng)
pos_star, vel_star, mass_star, types_star = g.generate()

acc_star_bare = compute_forces_direct(pos_star, mass_star, len(pos_star), G, eps)

pos_coupled = np.vstack((pos_star, pos))
mass_coupled = np.concatenate((mass_star, mass))
acc_coupled = compute_forces_direct(pos_coupled, mass_coupled, len(pos_coupled), G, eps)

acc_star_coupled = acc_coupled[:len(pos_star)]
diff_force = np.linalg.norm(acc_star_coupled - acc_star_bare, axis=1)
print(f"\nCoupled Stellar Radial Force Addition (median): {np.median(diff_force):.4f}")

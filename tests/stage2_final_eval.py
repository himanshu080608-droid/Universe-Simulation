import sys
import os
import time
import numpy as np

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
    return M_r / (r**2 + 1e-9)

def generate_smooth_positions(CDF_inv, N):
    u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    r = CDF_inv(u)
    phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
    return np.column_stack((r * np.cos(phi), r * np.sin(phi))), r

# 1. Candidate B1: Pseudo-isothermal
def inv_b1(u):
    R_c = 18.66
    return R_c * np.sqrt( np.exp(u * np.log(1.0 + (R_max/R_c)**2)) - 1.0 )

# 2. Candidate B2: Log-Logistic
def inv_b2(u):
    alpha, R_c = 1.16, 92.80
    F_max = (R_max/R_c)**alpha / (1.0 + (R_max/R_c)**alpha)
    val = u * F_max
    return R_c * (val / (1.0 - val))**(1.0/alpha)

# 3. Candidate B3: Log-Logistic tuned for median tradeoff
def inv_b3(u):
    alpha, R_c = 1.04, 187.14
    F_max = (R_max/R_c)**alpha / (1.0 + (R_max/R_c)**alpha)
    val = u * F_max
    return R_c * (val / (1.0 - val))**(1.0/alpha)

candidates = {
    "Candidate B1 (Pseudo-isothermal)": inv_b1,
    "Candidate B2 (Log-Logistic)": inv_b2,
    "Candidate B3 (LL tuned)": inv_b3
}

def eval_candidate(name, inv_func):
    print(f"\\n=== {name} ===")
    t0 = time.time()
    pos, r = generate_smooth_positions(inv_func, 2000)
    mass = np.full(2000, M_total/2000)
    t_init = time.time() - t0
    print(f"Initialization time: {t_init*1000:.2f} ms")
    
    # Boundary fidelity
    print(f"Boundary: max R = {np.max(r):.2f} (Target R_vir = {R_max:.2f})")
    
    # Concentration Behavior (Mass Fractions)
    f_01 = np.sum(r <= 0.1 * R_max) / 2000
    f_05 = np.sum(r <= 0.5 * R_max) / 2000
    f_rs = np.sum(r <= R_s) / 2000
    f_vir = np.sum(r <= R_max) / 2000
    print(f"Mass Fractions: 0.1 R_vir: {f_01:.3f} | 0.5 R_vir: {f_05:.3f} | R_s: {f_rs:.3f} | R_vir: {f_vir:.3f}")
    targ_01 = nfw_mass_frac(0.1*C) / m_vir
    targ_05 = nfw_mass_frac(0.5*C) / m_vir
    targ_rs = nfw_mass_frac(1.0) / m_vir
    print(f"NFW Target MFs: 0.1 R_vir: {targ_01:.3f} | 0.5 R_vir: {targ_05:.3f} | R_s: {targ_rs:.3f} | R_vir: 1.000")
    
    # Force Fidelity (Direct)
    acc = compute_forces_direct(pos, mass, 2000, G, eps)
    a_rad = -np.sum(acc * pos / (r[:, None] + 1e-9), axis=1)
    a_targ = a_target(r)
    err = np.abs(a_rad - a_targ) / np.maximum(a_targ, 1e-9)
    
    bins = [("inner", r < R_s), ("R_s region", (r >= R_s) & (r < 3*R_s)), ("outer", r >= 3*R_s)]
    for reg, mask in bins:
        e = err[mask]
        print(f"[{reg}] Med: {np.median(e)*100:.1f}% | RMS: {np.sqrt(np.mean(e**2))*100:.1f}% | P95: {np.percentile(e, 95)*100:.1f}% | Max: {np.max(e)*100:.1f}%")

    # Barnes-Hut Discrepancy
    from simulation_engine import SimulationEngine
    types = np.zeros(2000, dtype=np.int32)
    engine = SimulationEngine(pos, np.zeros_like(pos), mass, types, G=G, eps=eps, dt=0.01)
    engine.use_bh = True
    engine.theta = 0.5
    engine._init_accelerations()
    acc_bh = engine._acc
    
    diff = np.linalg.norm(acc - acc_bh, axis=1)
    err_bh = diff / np.maximum(np.linalg.norm(acc, axis=1), 1e-9)
    print(f"Barnes-Hut solver discrepancy median: {np.median(err_bh)*100:.2f}%")

for n, f in candidates.items():
    eval_candidate(n, f)

# Coupled Galaxy Test
print("\n=== Coupled Galaxy Test (messier31 configuration) ===")
# Generate stellar disk
rng = np.random.default_rng(42)
g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=rng)
pos_star, vel_star, mass_star, types_star = g.generate()

# Compute bare stellar force
acc_star_bare = compute_forces_direct(pos_star, mass_star, len(pos_star), G, eps)

# Add B2 Halo
pos_halo, _ = generate_smooth_positions(inv_b2, 2000)
mass_halo = np.full(2000, M_total/2000)

pos_coupled = np.vstack((pos_star, pos_halo))
mass_coupled = np.concatenate((mass_star, mass_halo))
acc_coupled = compute_forces_direct(pos_coupled, mass_coupled, len(pos_coupled), G, eps)

# Compare stellar forces
acc_star_coupled = acc_coupled[:len(pos_star)]
diff_force = np.linalg.norm(acc_star_coupled - acc_star_bare, axis=1)
print(f"Coupled stellar radial-force addition median: {np.median(diff_force):.4f}")
print(f"Stellar force field entirely changed by halo (as expected physically).")

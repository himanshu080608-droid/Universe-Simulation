import sys
import os
import time
import numpy as np
from scipy.optimize import nnls
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
R_vir = R_s * C
M_total = 1000.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)
m_vir = nfw_mass_frac(C)

def a_target(r):
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return M_r / (r**2 + eps**2)

def build_radial_grid(N, R_min, R_max):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return 0.7 * log_grid + 0.3 * lin_grid

# Cache for kernel
_kernel_cache = {}
def compute_kernel(R_eval, R_annulus):
    key = (len(R_eval), len(R_annulus), R_eval[-1], R_annulus[-1])
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

def calibrate_model_A(K, a_targ, lambda_reg=0.1):
    """Model A: Redistributed mass. sum(all_mass) = M_total"""
    N_eval, N_ann = K.shape
    W = 1.0 / np.maximum(a_targ, 1e-4)
    A = K * W[:, None]
    b = a_targ * W
    
    # Regularization
    if lambda_reg > 0:
        L = np.zeros((N_ann-2, N_ann))
        for i in range(N_ann-2):
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        A = np.vstack([A, lambda_reg * L])
        b = np.concatenate([b, np.zeros(N_ann-2)])
        
    # Total mass constraint
    A = np.vstack([A, 1e5 * np.ones((1, N_ann))])
    b = np.concatenate([b, [1e5 * M_total]])
    
    m, _ = nnls(A, b)
    m = m * (M_total / np.sum(m))
    return m

def calibrate_model_B(K, a_targ, R_ann, lambda_reg=0.1):
    """Model B: Force-only buffer. sum(mass inside R_vir) = M_total. Buffer mass is free."""
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
        
    # Physical mass constraint
    mask_in = R_ann <= R_vir
    row = np.zeros((1, N_ann))
    row[0, mask_in] = 1e5
    A = np.vstack([A, row])
    b = np.concatenate([b, [1e5 * M_total]])
    
    m, _ = nnls(A, b)
    # Strictly enforce physical mass (buffer is free)
    m_in = np.sum(m[mask_in])
    if m_in > 0:
        m[mask_in] = m[mask_in] * (M_total / m_in)
    return m

def evaluate_metrics(m, K, R_eval, a_targ, print_results=False):
    a_surr = K @ m
    err = np.abs(a_surr - a_targ) / np.maximum(a_targ, 1e-9)
    
    bins = {
        "inner": R_eval < R_s,
        "R_s region": (R_eval >= R_s) & (R_eval < 3*R_s),
        "intermediate": (R_eval >= 3*R_s) & (R_eval < 0.95*R_vir),
        "edge (0.95-1.0)": (R_eval >= 0.95*R_vir) & (R_eval <= R_vir),
        "full physical": R_eval <= R_vir
    }
    
    res = {}
    for name, mask in bins.items():
        e = err[mask]
        if len(e) > 0:
            res[name] = {
                "med": np.median(e),
                "rms": np.sqrt(np.mean(e**2)),
                "p95": np.percentile(e, 95),
                "max": np.max(e)
            }
        else:
            res[name] = {"med": 0, "rms": 0, "p95": 0, "max": 0}
            
    if print_results:
        for name in ["full physical", "inner", "R_s region", "intermediate", "edge (0.95-1.0)"]:
            e = res[name]
            print(f"  [{name}] Med: {e['med']*100:.1f}% | RMS: {e['rms']*100:.1f}% | P95: {e['p95']*100:.1f}% | Max: {e['max']*100:.1f}%")
            
    return res

def run_buffer_sweep():
    ratios = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]
    N_eval = 150
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    a_targ = a_target(R_eval)
    
    print("=== Buffer Sweep (Model A vs Model B) ===")
    print(f"{'R_supp/R_vir':<12} | {'Model':<5} | {'Full Med':<8} | {'Edge P95':<8} | {'Edge Max':<8} | {'Buf Mass':<8} | Gate")
    print("-" * 75)
    
    best_ratio_A = None
    best_ratio_B = None
    
    for ratio in ratios:
        R_support = R_vir * ratio
        N_ann = int(150 * ratio)
        R_ann = build_radial_grid(N_ann, 0.1, R_support)
        
        K = compute_kernel(R_eval, R_ann)
        
        # Model A
        m_A = calibrate_model_A(K, a_targ, lambda_reg=0.001)
        res_A = evaluate_metrics(m_A, K, R_eval, a_targ)
        buf_mass_A = np.sum(m_A[R_ann > R_vir]) / M_total
        
        gate_A = "PASS" if (res_A["full physical"]["med"] <= 0.10 and 
                            res_A["full physical"]["p95"] <= 0.25 and 
                            res_A["full physical"]["max"] <= 0.40) else "FAIL"
                            
        print(f"{ratio:<12.2f} | {'A':<5} | {res_A['full physical']['med']*100:>7.1f}% | {res_A['full physical']['max']*100:>7.1f}% | {res_A['inner']['max']*100:>7.1f}% | {buf_mass_A*100:>7.1f}% | {gate_A}")
        if gate_A == "PASS" and best_ratio_A is None:
            best_ratio_A = ratio
            
        # Model B
        m_B = calibrate_model_B(K, a_targ, R_ann, lambda_reg=0.001)
        res_B = evaluate_metrics(m_B, K, R_eval, a_targ)
        buf_mass_B = np.sum(m_B[R_ann > R_vir]) / M_total
        
        gate_B = "PASS" if (res_B["full physical"]["med"] <= 0.10 and 
                            res_B["full physical"]["p95"] <= 0.25 and 
                            res_B["full physical"]["max"] <= 0.40) else "FAIL"
                            
        print(f"{ratio:<12.2f} | {'B':<5} | {res_B['full physical']['med']*100:>7.1f}% | {res_B['full physical']['max']*100:>7.1f}% | {res_B['inner']['max']*100:>7.1f}% | {buf_mass_B*100:>7.1f}% | {gate_B}")
        if gate_B == "PASS" and best_ratio_B is None:
            best_ratio_B = ratio
            
    return best_ratio_A, best_ratio_B

def generate_smooth_positions(inv_cdf, N):
    u = np.linspace(0.5/N, 1.0 - 0.5/N, N)
    r = inv_cdf(u)
    phi = u * np.pi * (1.0 + np.sqrt(5.0)) * N
    return np.column_stack((r * np.cos(phi), r * np.sin(phi))), r

def run_particle_validation(ratio, model_type="B"):
    print(f"\n=== Resolution & Particle Validation (Ratio {ratio}, Model {model_type}) ===")
    
    R_support = R_vir * ratio
    N_eval = 200
    N_ann = int(200 * ratio)
    R_eval = build_radial_grid(N_eval, 0.1, R_vir)
    R_ann = build_radial_grid(N_ann, 0.1, R_support)
    
    K = compute_kernel(R_eval, R_ann)
    a_targ = a_target(R_eval)
    
    if model_type == "A":
        m = calibrate_model_A(K, a_targ, lambda_reg=0.05)
    else:
        m = calibrate_model_B(K, a_targ, R_ann, lambda_reg=0.05)
        
    print("\n[Continuous Kernel Metrics]")
    evaluate_metrics(m, K, R_eval, a_targ, print_results=True)
    
    cdf = np.cumsum(m) / np.sum(m)
    cdf = np.insert(cdf, 0, 0.0)
    R_cdf = np.insert(R_ann, 0, 0.0)
    inv_cdf = interp1d(cdf, R_cdf, kind='linear', fill_value="extrapolate")
    
    # Check outside boundary force
    R_out = np.linspace(R_vir, R_support * 1.1, 50)
    K_out = compute_kernel(R_out, R_ann)
    a_out = K_out @ m
    # Just checking for NaNs or explosions
    print(f"\n[Buffer Zone Force (R_vir to 1.1*R_support)]")
    print(f"  Monotonic drop: {np.all(np.diff(a_out) <= 0)}")
    print(f"  Force at R_vir: {a_out[0]:.4f} | Force at 1.1*R_supp: {a_out[-1]:.4f}")
    
    for N in [500, 1000, 2000]:
        print(f"\n[Particle Validation N={N}]")
        pos, r = generate_smooth_positions(inv_cdf, N)
        total_m = np.sum(m)
        mass = np.full(N, total_m / N)
        
        acc_dir = compute_forces_direct(pos, mass, N, G, eps)
        a_rad_dir = -np.sum(acc_dir * pos / (r[:, None] + 1e-9), axis=1)
        
        # We only evaluate physical domain metrics
        mask = r <= R_vir
        r_phys = r[mask]
        a_dir_phys = a_rad_dir[mask]
        a_targ_phys = a_target(r_phys)
        
        # evaluate_metrics(m, None, r_phys, a_targ_phys, print_results=False) # Removed dummy call
        
        err_dir = np.abs(a_dir_phys - a_targ_phys) / np.maximum(a_targ_phys, 1e-9)
        bins = {
            "inner": r_phys < R_s,
            "R_s region": (r_phys >= R_s) & (r_phys < 3*R_s),
            "edge (0.95-1.0)": (r_phys >= 0.95*R_vir) & (r_phys <= R_vir),
            "full physical": r_phys <= R_vir
        }
        for name, mask_bin in bins.items():
            e = err_dir[mask_bin]
            if len(e)>0:
                print(f"  [{name}] Med: {np.median(e)*100:.1f}% | P95: {np.percentile(e, 95)*100:.1f}% | Max: {np.max(e)*100:.1f}%")
                
        # Barnes-Hut check at N=2000
        if N == 2000:
            from simulation_engine import SimulationEngine
            engine = SimulationEngine(pos, np.zeros_like(pos), mass, np.zeros(N, dtype=np.int32), G=G, eps=eps, dt=0.01)
            engine.use_bh = True
            engine.theta = 0.5
            engine._init_accelerations()
            acc_bh = engine._acc
            diff = np.linalg.norm(acc_dir - acc_bh, axis=1)
            err_bh = diff / np.maximum(np.linalg.norm(acc_dir, axis=1), 1e-9)
            print(f"  Barnes-Hut discrepancy median: {np.median(err_bh)*100:.2f}% | P95: {np.percentile(err_bh, 95)*100:.2f}%")
            
            # Coupled galaxy test
            print(f"\n[Coupled messier31 test]")
            rng = np.random.default_rng(42)
            g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=rng)
            pos_star, vel_star, mass_star, types_star = g.generate()
            acc_star_bare = compute_forces_direct(pos_star, mass_star, len(pos_star), G, eps)
            
            pos_coupled = np.vstack((pos_star, pos))
            mass_coupled = np.concatenate((mass_star, mass))
            acc_coupled = compute_forces_direct(pos_coupled, mass_coupled, len(pos_coupled), G, eps)
            
            acc_star_coupled = acc_coupled[:len(pos_star)]
            diff_force = np.linalg.norm(acc_star_coupled - acc_star_bare, axis=1)
            print(f"  Coupled stellar radial force median addition: {np.median(diff_force):.4f}")
            
    # Physical Mass fractions
    f_01 = np.sum(m[R_ann <= 0.1*R_vir]) / np.sum(m[R_ann <= R_vir])
    f_05 = np.sum(m[R_ann <= 0.5*R_vir]) / np.sum(m[R_ann <= R_vir])
    f_rs = np.sum(m[R_ann <= R_s]) / np.sum(m[R_ann <= R_vir])
    print(f"\n[Physical NFW Deviation (Inside R_vir)]")
    print(f"  Surrogate (relative to M(<=R_vir)): 0.1 R_vir: {f_01:.3f} | R_s: {f_rs:.3f} | 0.5 R_vir: {f_05:.3f}")
    print(f"  Target NFW                        : 0.1 R_vir: {nfw_mass_frac(0.1*C)/m_vir:.3f} | R_s: {nfw_mass_frac(1.0)/m_vir:.3f} | 0.5 R_vir: {nfw_mass_frac(0.5*C)/m_vir:.3f}")

if __name__ == "__main__":
    best_A, best_B = run_buffer_sweep()
    
    if best_A is None and best_B is None:
        print("\nNo buffer passed the hard gates.")
    else:
        # Use B if available, else A
        if best_B is not None:
            print(f"\nSelecting Model B with R_support/R_vir = {best_B}")
            run_particle_validation(best_B, "B")
        else:
            print(f"\nSelecting Model A with R_support/R_vir = {best_A}")
            run_particle_validation(best_A, "A")

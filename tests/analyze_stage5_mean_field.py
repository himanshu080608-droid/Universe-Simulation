import sys
import os
import numpy as np
from scipy.integrate import quad
from numba import njit, prange
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import NFWModel
from universe.nfw_calibration import get_calibration
from hermite_engine import HermiteEngine
from physics.integrator import compute_forces_direct

G = 1.0
eps = 1.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)

def a_target(r, R_s, C, M_total, eps=1.0):
    m_vir = nfw_mass_frac(C)
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return 1.0 * M_r * r / (r**2 + eps**2)**1.5

def build_radial_grid(N, R_min, R_max, log_weight=0.9):
    u = np.linspace(0, 1, N)
    log_grid = R_min * (R_max/R_min)**u
    lin_grid = R_min + u * (R_max - R_min)
    return log_weight * log_grid + (1.0 - log_weight) * lin_grid

def compute_kernel(R_eval, R_annulus, eps):
    K = np.zeros((len(R_eval), len(R_annulus)))
    for i, ri in enumerate(R_eval):
        for j, rj in enumerate(R_annulus):
            def integrand(theta):
                denom = (ri**2 + rj**2 - 2*ri*rj*np.cos(theta) + eps**2)**1.5
                return (ri - rj*np.cos(theta)) / denom
            val, _ = quad(integrand, 0, 2*np.pi, epsabs=1e-4, epsrel=1e-4)
            K[i, j] = G * val / (2 * np.pi)
    return K

@njit(parallel=True)
def compute_probe_forces(pos, mass, probes, eps):
    N = len(pos)
    M = len(probes)
    acc = np.zeros((M, 2), dtype=np.float64)
    eps2 = eps * eps
    for i in prange(M):
        px = probes[i, 0]
        py = probes[i, 1]
        ax = 0.0
        ay = 0.0
        for j in range(N):
            dx = pos[j, 0] - px
            dy = pos[j, 1] - py
            r2 = dx*dx + dy*dy + eps2
            f = G * mass[j] / (r2 * np.sqrt(r2))
            ax += f * dx
            ay += f * dy
        acc[i, 0] = ax
        acc[i, 1] = ay
    return acc

def test_continuous(R_s, C, M_total):
    print(f"\n--- Continuous Calibration Test (R_s={R_s}, C={C}) ---")
    R_cdf, cdf = get_calibration(R_s, C, eps)
    R_eval = build_radial_grid(200, 0.01*eps, R_s*C, log_weight=0.9)
    R_ann = R_cdf[1:]
    m = np.diff(cdf) * M_total
    K = compute_kernel(R_eval, R_ann, eps)
    a_surr = K @ m
    a_targ = a_target(R_eval, R_s, C, M_total, eps)
    
    mask = R_eval >= eps
    err_rel = np.abs(a_surr[mask] - a_targ[mask]) / np.maximum(a_targ[mask], 1e-12)
    med = np.median(err_rel)
    p95 = np.percentile(err_rel, 95)
    mmax = np.max(err_rel)
    
    print(f"Continuous Gates: Med {med*100:.2f}% (<=10%), P95 {p95*100:.2f}% (<=25%), Max {mmax*100:.2f}% (<=40%)")
    assert med <= 0.10 and p95 <= 0.25 and mmax <= 0.40, "Continuous force gate failed!"
    print("STATUS: PASS")
    return R_eval, a_surr, a_targ

def evaluate_mean_field(R_s, C, M_total, N_particles, N_azim=256):
    print(f"\n=== Evaluating Mean Field (R_s={R_s}, C={C}, N={N_particles}) ===")
    halo = NFWModel(N=N_particles, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
    pos, vel, mass, types = halo.generate()
    
    # Define radial sweep
    R_vir = R_s * C
    radii_list = sorted(list(set([
        eps, 2*eps, 5*eps, 0.01*R_vir, 0.1*R_vir, R_s, 2*R_s, 0.5*R_vir, R_vir
    ])))
    
    # Dense log grid + explicit radii
    log_radii = np.geomspace(eps, R_vir, 50)
    eval_radii = np.sort(np.unique(np.concatenate([radii_list, log_radii])))
    
    probes = []
    r_probes = []
    theta_grid = np.linspace(0, 2*np.pi, N_azim, endpoint=False)
    for R in eval_radii:
        for th in theta_grid:
            probes.append([R * np.cos(th), R * np.sin(th)])
            r_probes.append(R)
            
    probes = np.array(probes)
    r_probes = np.array(r_probes)
    
    # Compute direct forces
    print("Computing independent probe direct forces...")
    t0 = time.time()
    acc_dir = compute_probe_forces(pos, mass, probes, eps)
    print(f"Done in {time.time()-t0:.2f}s")
    
    # Compute BH forces by combining pos and probes
    print("Computing BH forces on independent probes...")
    t0 = time.time()
    pos_all = np.vstack([pos, probes])
    mass_all = np.concatenate([mass, np.zeros(len(probes))])
    vel_all = np.zeros_like(pos_all)
    types_all = np.zeros(len(pos_all), dtype=np.int32)
    engine_bh = HermiteEngine(pos_all, vel_all, mass_all, types_all, G=G, eps=eps, dt=0.01)
    acc_bh = engine_bh._acc[len(pos):]
    print(f"Done in {time.time()-t0:.2f}s")
    
    # Project to inward radial
    dir_rad = -probes / (r_probes[:, None] + 1e-12)
    a_rad_dir = np.sum(acc_dir * dir_rad, axis=1)
    
    # Compare
    a_targ_all = a_target(r_probes, R_s, C, M_total, eps)
    
    # Group by radius
    num_R = len(eval_radii)
    a_rad_dir_grouped = a_rad_dir.reshape((num_R, N_azim))
    acc_bh_grouped = acc_bh.reshape((num_R, N_azim, 2))
    acc_dir_grouped = acc_dir.reshape((num_R, N_azim, 2))
    a_targ_grouped = a_targ_all.reshape((num_R, N_azim))[:, 0]
    
    mean_field = np.mean(a_rad_dir_grouped, axis=1)
    median_field = np.median(a_rad_dir_grouped, axis=1)
    std_field = np.std(a_rad_dir_grouped, axis=1)
    min_field = np.min(a_rad_dir_grouped, axis=1)
    max_field = np.max(a_rad_dir_grouped, axis=1)
    p5_field = np.percentile(a_rad_dir_grouped, 5, axis=1)
    p95_field = np.percentile(a_rad_dir_grouped, 95, axis=1)
    
    # Surrogate forces vs Target (Continuous)
    R_cdf, cdf = get_calibration(R_s, C, eps)
    R_ann = R_cdf[1:]
    m_ann = np.diff(cdf) * M_total
    K_eval = compute_kernel(eval_radii, R_ann, eps)
    a_surr_cont = K_eval @ m_ann
    
    # 1. MEAN FIELD VS TARGET (Surrogate Bias)
    err_mean_rel = np.abs(mean_field - a_targ_grouped) / np.maximum(a_targ_grouped, 1e-12)
    err_mean_abs = np.abs(mean_field - a_targ_grouped)
    
    # Use resolved domain
    mask = eval_radii >= eps
    med_err = np.median(err_mean_rel[mask])
    p95_err = np.percentile(err_mean_rel[mask], 95)
    max_err = np.max(err_mean_rel[mask])
    rms_err = np.sqrt(np.mean(err_mean_abs[mask]**2))
    max_abs = np.max(err_mean_abs[mask])
    
    print("\n--- 1. ANGULAR MEAN-FIELD VS TARGET (R >= eps) ---")
    print(f"  Median Rel Error: {med_err*100:.2f}% (Limit: 10%)")
    print(f"  P95 Rel Error:    {p95_err*100:.2f}% (Limit: 25%)")
    print(f"  Max Rel Error:    {max_err*100:.2f}% (Limit: 40%)")
    print(f"  RMS Abs Error:    {rms_err:.6f}")
    print(f"  Max Abs Error:    {max_abs:.6f}")
    
    status = "PASS" if (med_err <= 0.10 and p95_err <= 0.25 and max_err <= 0.40) else "FAIL - genuine discrete mean-field mismatch"
    print(f"  MEAN FIELD STATUS: {status}")
    
    # 2. FINITE-N PARTICLE NOISE
    print("\n--- 2. FINITE-N PARTICLE NOISE AT SPECIFIC RADII ---")
    print(f" {'R':>10} | {'Mean/Targ':>12} | {'Noise Sigma / Targ':>20} | {'P95-P5 / Targ':>15}")
    for R_val in radii_list:
        idx = np.argmin(np.abs(eval_radii - R_val))
        atarg = np.maximum(a_targ_grouped[idx], 1e-12)
        ratio = mean_field[idx] / atarg
        noise_sigma = std_field[idx] / atarg
        p95_p5 = (p95_field[idx] - p5_field[idx]) / atarg
        print(f" {R_val:10.2f} | {ratio:12.4f} | {noise_sigma*100:19.2f}% | {p95_p5*100:14.2f}%")
        
    # 3. BARNES-HUT VALIDATION ON INDEPENDENT PROBES
    bh_err_mag = np.linalg.norm(acc_bh - acc_dir, axis=1)
    dir_mag = np.linalg.norm(acc_dir, axis=1)
    bh_rel = bh_err_mag / np.maximum(dir_mag, 1e-12)
    
    bh_med = np.median(bh_rel)
    bh_p95 = np.percentile(bh_rel, 95)
    bh_max = np.max(bh_rel)
    bh_rms = np.sqrt(np.mean(bh_err_mag**2))
    
    max_idx = np.argmax(bh_rel)
    max_r = r_probes[max_idx]
    
    print("\n--- 3. BARNES-HUT VALIDATION ---")
    print(f"  Median Rel Discrepancy: {bh_med*100:.2f}%")
    print(f"  P95 Rel Discrepancy:    {bh_p95*100:.2f}%")
    print(f"  Max Rel Discrepancy:    {bh_max*100:.2f}% (at R={max_r:.2f})")
    print(f"  RMS Abs Discrepancy:    {bh_rms:.6f}")
    
    return med_err, p95_err, max_err, std_field / np.maximum(a_targ_grouped, 1e-12)

def run_N_scaling():
    print("\n\n" + "="*50)
    print("=== N SCALING STUDY (R_s=15, C=10) ===")
    print("="*50)
    
    results = {}
    for N_particles in [500, 1000, 2500, 5000]:
        med, p95, mmax, noise_rel = evaluate_mean_field(15.0, 10.0, 20000.0, N_particles, N_azim=256)
        # Average noise across radii
        mean_noise = np.mean(noise_rel)
        results[N_particles] = {
            'med': med, 'p95': p95, 'max': mmax, 'noise': mean_noise
        }
        
    print("\n--- N SCALING SUMMARY ---")
    print(f"{'N':>6} | {'Mean-Field Med Error':>22} | {'Avg Angular Noise':>20} | {'Expected Poisson ~1/sqrt(N)':>30}")
    ref_noise = results[500]['noise'] * np.sqrt(500)
    for N in [500, 1000, 2500, 5000]:
        med_err = results[N]['med'] * 100
        noise = results[N]['noise'] * 100
        expected = (ref_noise / np.sqrt(N)) * 100
        print(f"{N:6d} | {med_err:21.2f}% | {noise:19.2f}% | {expected:29.2f}%")

if __name__ == "__main__":
    # 1. Continuous sanity check
    test_continuous(10.0, 10.0, 100.0)
    test_continuous(15.0, 10.0, 20000.0)
    test_continuous(200.0, 12.0, 1.5e7)
    
    # 2. Production Mean-Field Checks
    evaluate_mean_field(10.0, 10.0, 100.0, 2500)
    evaluate_mean_field(15.0, 10.0, 20000.0, 2500)
    evaluate_mean_field(200.0, 12.0, 1.5e7, 2500)
    
    # 3. N Scaling
    run_N_scaling()

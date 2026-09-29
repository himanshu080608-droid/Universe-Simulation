import sys
import os
import numpy as np
from scipy.integrate import dblquad, quad
from numba import njit, prange
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import NFWModel
from universe.nfw_calibration import get_calibration

G = 1.0
eps = 1.0

def nfw_mass_frac(x):
    return np.log(1.0 + x) - x / (1.0 + x)

def a_target(r, R_s, C, M_total, eps=1.0):
    m_vir = nfw_mass_frac(C)
    M_r = M_total * (nfw_mass_frac(r / R_s) / m_vir)
    return 1.0 * M_r * r / (r**2 + eps**2)**1.5

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

def compute_kernel_piecewise(R_eval, R_clean, m_clean, eps):
    a_piecewise = np.zeros(len(R_eval))
    for i, ri in enumerate(R_eval):
        force_sum = 0.0
        for j in range(len(R_clean)-1):
            R0 = R_clean[j]
            R1 = R_clean[j+1]
            dm = m_clean[j]
            if dm == 0 or R1 == R0:
                continue
                
            def integrand(theta, r):
                denom = (ri**2 + r**2 - 2*ri*r*np.cos(theta) + eps**2)**1.5
                return (ri - r*np.cos(theta)) / denom
                
            val, _ = dblquad(integrand, R0, R1, lambda r: 0, lambda r: 2*np.pi, epsabs=1e-4, epsrel=1e-4)
            force_sum += (dm / (R1 - R0)) * (G * val / (2 * np.pi))
            
        a_piecewise[i] = force_sum
    return a_piecewise

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

def run_analysis(R_s, C, M_total, do_mc=False, mc_N_list=None):
    print(f"\n{'='*70}\n=== SAMPLER CONSISTENCY: R_s={R_s}, C={C} ===\n{'='*70}")
    
    R_vir = R_s * C
    radii = [1*eps, 2*eps, 5*eps, 10*eps, 20*eps, R_s, 0.5*R_vir]
    
    if R_s == 200 and C == 12:
        radii.extend([1, 2, 5, 10, 20, 50, 100, 200])
        
    radii = np.sort(np.unique(np.array(radii)))
    radii = radii[radii <= R_vir]
    
    # 2. Field A — ring calibration
    R_cdf, cdf = get_calibration(R_s, C, eps)
    R_ann = R_cdf[1:]
    m_ann = np.diff(cdf) * M_total
    K = compute_kernel(radii, R_ann, eps)
    a_ring = K @ m_ann
    
    # 3. Field B — production-sampler limiting field
    _, unique_indices = np.unique(cdf[::-1], return_index=True)
    unique_indices = np.sort(len(cdf) - 1 - unique_indices)
    cdf_clean = cdf[unique_indices]
    R_clean = R_cdf[unique_indices]
    m_clean = np.diff(cdf_clean) * M_total
    a_piecewise = compute_kernel_piecewise(radii, R_clean, m_clean, eps)
    
    # 4. Field C — analytic target
    a_target_vals = a_target(radii, R_s, C, M_total, eps)
    
    # 5. Compare A, B, C
    E_AvsC = np.abs(a_ring - a_target_vals) / a_target_vals
    E_BvsC = np.abs(a_piecewise - a_target_vals) / a_target_vals
    E_BvsA = np.abs(a_piecewise - a_ring) / a_ring
    
    print("\n--- 5. Comparison: Field A vs B vs C ---")
    print("A vs C = current calibration error")
    print("B vs C = production-sampler limiting error")
    print("B vs A = representation mismatch")
    print(f"{'Metric':>15} | {'A vs C':>15} | {'B vs C':>15} | {'B vs A':>15}")
    print(f"{'Median':>15} | {np.median(E_AvsC)*100:14.2f}% | {np.median(E_BvsC)*100:14.2f}% | {np.median(E_BvsA)*100:14.2f}%")
    print(f"{'P95':>15} | {np.percentile(E_AvsC, 95)*100:14.2f}% | {np.percentile(E_BvsC, 95)*100:14.2f}% | {np.percentile(E_BvsA, 95)*100:14.2f}%")
    print(f"{'Max':>15} | {np.max(E_AvsC)*100:14.2f}% | {np.max(E_BvsC)*100:14.2f}% | {np.max(E_BvsA)*100:14.2f}%")
    
    # 6. M31-specific table
    if R_s == 200 and C == 12:
        print("\n--- 6. M31-specific table (R_s=200, C=12) ---")
        print(f"{'R':>5} | {'A / C':>10} | {'B / C':>10} | {'B / A':>10}")
        for r_val in [1, 2, 5, 10, 20, 50, 100, 200]:
            if r_val in radii:
                j = np.where(radii == r_val)[0][0]
                print(f"{r_val:5.1f} | {a_ring[j]/a_target_vals[j]:10.2f} | {a_piecewise[j]/a_target_vals[j]:10.2f} | {a_piecewise[j]/a_ring[j]:10.2f}")
                
    # 7 & 8. Monte-Carlo convergence test & exact-ring control
    if do_mc and mc_N_list is not None:
        print("\n--- 7 & 8. Monte-Carlo Convergence & Control ---")
        N_azim = 256
        for N_eval in mc_N_list:
            print(f"\nEvaluating N={N_eval} over 32 seeds...")
            seeds = np.arange(42, 42+32)
            
            ang_means_prod = np.zeros((len(seeds), len(radii)))
            ang_means_ctrl = np.zeros((len(seeds), len(radii)))
            
            for idx_s, s in enumerate(seeds):
                rng = np.random.default_rng(s)
                
                # Production sampler
                halo = NFWModel(N=N_eval, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(s), types_dist={'types': [14], 'probs': [1.0]})
                pos_prod, vel_prod, mass_prod, _ = halo.generate()
                
                # Control sampler (exact-ring)
                m_j = np.diff(cdf) * M_total
                p_j = m_j / np.sum(m_j)
                ring_indices = rng.choice(len(R_ann), size=N_eval, p=p_j)
                r_ctrl = R_ann[ring_indices]
                th_ctrl = rng.uniform(0, 2*np.pi, size=N_eval)
                pos_ctrl = np.column_stack([r_ctrl * np.cos(th_ctrl), r_ctrl * np.sin(th_ctrl)])
                mass_ctrl = np.full(N_eval, M_total / N_eval)
                
                probes = []
                r_probes = []
                theta_grid = np.linspace(0, 2*np.pi, N_azim, endpoint=False)
                for R in radii:
                    for th in theta_grid:
                        probes.append([R * np.cos(th), R * np.sin(th)])
                        r_probes.append(R)
                probes = np.array(probes)
                r_probes = np.array(r_probes)
                
                acc_prod = compute_probe_forces(pos_prod, mass_prod, probes, eps)
                acc_ctrl = compute_probe_forces(pos_ctrl, mass_ctrl, probes, eps)
                
                dir_rad = -probes / (r_probes[:, None] + 1e-12)
                
                a_rad_prod = np.sum(acc_prod * dir_rad, axis=1)
                a_rad_ctrl = np.sum(acc_ctrl * dir_rad, axis=1)
                
                ang_means_prod[idx_s, :] = np.mean(a_rad_prod.reshape((len(radii), N_azim)), axis=1)
                ang_means_ctrl[idx_s, :] = np.mean(a_rad_ctrl.reshape((len(radii), N_azim)), axis=1)
                
            ens_prod = np.mean(ang_means_prod, axis=0)
            ens_ctrl = np.mean(ang_means_ctrl, axis=0)
            
            print(f"Results for N={N_eval}:")
            print(f"{'R':>6} | {'Prod vs A':>11} | {'Prod vs B':>11} | {'Prod vs C':>11} || {'Ctrl vs A':>11} | {'Ctrl vs B':>11} | {'Ctrl vs C':>11}")
            for j, R in enumerate(radii):
                if R in [1, 2, 5, 10, 20, 50, 100, 200] or R == R_s:
                    err_prod_A = abs(ens_prod[j] - a_ring[j]) / a_ring[j]
                    err_prod_B = abs(ens_prod[j] - a_piecewise[j]) / a_piecewise[j]
                    err_prod_C = abs(ens_prod[j] - a_target_vals[j]) / a_target_vals[j]
                    
                    err_ctrl_A = abs(ens_ctrl[j] - a_ring[j]) / a_ring[j]
                    err_ctrl_B = abs(ens_ctrl[j] - a_piecewise[j]) / a_piecewise[j]
                    err_ctrl_C = abs(ens_ctrl[j] - a_target_vals[j]) / a_target_vals[j]
                    
                    print(f"{R:6.1f} | {err_prod_A*100:10.2f}% | {err_prod_B*100:10.2f}% | {err_prod_C*100:10.2f}% || {err_ctrl_A*100:10.2f}% | {err_ctrl_B*100:10.2f}% | {err_ctrl_C*100:10.2f}%")
                    
    return E_BvsA

def classification(results):
    print("\n" + "="*70)
    print("=== FINAL CLASSIFICATION ===")
    print("="*70)
    
    max_b_vs_a = max([np.max(r) for r in results])
    
    if max_b_vs_a > 0.10:
        classification = "CALIBRATION/SAMPLER REPRESENTATION MISMATCH CONFIRMED"
        reason = "Field B (production sampler limiting field) fundamentally diverges from Field A (ring calibration field). The current sampling implies a piecewise-uniform measure, while the calibration assumes a purely discrete annular measure."
    else:
        classification = "INCONCLUSIVE"
        reason = "The representation mismatch is negligible. Some other bias must exist."
        
    print(f"Classification: {classification}")
    print(f"Reason: {reason}")
    print("\nRecommendations based on confirmed mismatch:")
    if classification == "CALIBRATION/SAMPLER REPRESENTATION MISMATCH CONFIRMED":
        print("DO NOT APPLY FIX IN THIS DIAGNOSTIC COMMIT.")
        print("Next step options: ")
        print(" A) Retain sampler, rebuild calibration using interval-integrated kernels so representations match.")
        print(" B) Retain calibration, change sampler to exactly reproduce discrete rings (as done in exact-ring control).")

if __name__ == "__main__":
    configs = [
        (10.0, 10.0, 100.0, False, None),
        (15.0, 10.0, 20000.0, True, [2500, 10000, 50000]),
        (200.0, 12.0, 1.5e7, True, [2500, 10000, 50000])
    ]
    
    all_b_vs_a = []
    for R_s, C, M_total, do_mc, mc_N_list in configs:
        b_vs_a = run_analysis(R_s, C, M_total, do_mc, mc_N_list)
        all_b_vs_a.append(b_vs_a)
        
    classification(all_b_vs_a)

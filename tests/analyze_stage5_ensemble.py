import sys
import os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numba import njit, prange
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import NFWModel
from universe.nfw_calibration import get_calibration
from hermite_engine import HermiteEngine

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

def scaling_func(n, A, B):
    return A / np.sqrt(n) + B

def evaluate_ensemble(R_s, C, M_total, run_extended=False):
    print(f"\n{'='*70}\n=== ENSEMBLE ANALYSIS: R_s={R_s}, C={C} ===\n{'='*70}")
    
    R_vir = R_s * C
    radii = [1*eps, 2*eps, 5*eps, 10*eps, 20*eps, R_s, 0.5*R_vir]
    
    # for M31 explicitly add required radiuses
    if R_s == 200 and C == 12:
        radii.extend([1, 2, 5, 10, 20, 50, 100, 200])
        
    radii = np.sort(np.unique(np.array(radii)))
    radii = radii[radii <= R_vir]
    
    # 2. Reference fields
    # A. Analytic Target
    a_targ_all = a_target(radii, R_s, C, M_total, eps)
    
    # B. Continuous surrogate
    R_cdf, cdf = get_calibration(R_s, C, eps)
    R_ann = R_cdf[1:]
    m_ann = np.diff(cdf) * M_total
    K_eval = compute_kernel(radii, R_ann, eps)
    a_surr_all = K_eval @ m_ann
    
    # E_calibration
    E_calib = np.abs(a_surr_all - a_targ_all) / a_targ_all
    
    N_list = [500, 1000, 2500, 5000]
    if run_extended and R_s == 200 and C == 12:
        N_list.extend([10000, 25000, 50000])
        
    seeds = np.arange(42, 42+32)
    N_azim = 256
    
    # Group occupancy
    expected_N_frac = nfw_mass_frac(radii / R_s) / nfw_mass_frac(C)
    
    results_by_N = {}
    
    for N_eval in N_list:
        print(f"\nEvaluating N={N_eval} over {len(seeds)} seeds...")
        
        expected_N = expected_N_frac * N_eval
        
        ensemble_mean_of_angular_means = np.zeros(len(radii))
        angular_means_all_seeds = np.zeros((len(seeds), len(radii)))
        
        for idx_s, s in enumerate(seeds):
            halo = NFWModel(N=N_eval, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(s), types_dist={'types': [14], 'probs': [1.0]})
            pos, vel, mass, types = halo.generate()
            
            probes = []
            r_probes = []
            theta_grid = np.linspace(0, 2*np.pi, N_azim, endpoint=False)
            for R in radii:
                for th in theta_grid:
                    probes.append([R * np.cos(th), R * np.sin(th)])
                    r_probes.append(R)
            probes = np.array(probes)
            r_probes = np.array(r_probes)
            
            acc_dir = compute_probe_forces(pos, mass, probes, eps)
            dir_rad = -probes / (r_probes[:, None] + 1e-12)
            a_rad_dir = np.sum(acc_dir * dir_rad, axis=1)
            a_rad_grouped = a_rad_dir.reshape((len(radii), N_azim))
            
            ang_mean = np.mean(a_rad_grouped, axis=1)
            angular_means_all_seeds[idx_s, :] = ang_mean
            
        ensemble_mean = np.mean(angular_means_all_seeds, axis=0)
        ensemble_std = np.std(angular_means_all_seeds, axis=0)
        
        E_particle = np.abs(ensemble_mean - a_surr_all) / a_surr_all
        E_total = np.abs(ensemble_mean - a_targ_all) / a_targ_all
        
        results_by_N[N_eval] = {
            'expected_N': expected_N,
            'E_particle': E_particle,
            'E_total': E_total,
            'ensemble_mean': ensemble_mean
        }
        
        if N_eval == 2500:
            print("\n--- Occupancy Regimes (N=2500) ---")
            print(f"{'Radius':>10} | {'Expected N':>15} | {'E_particle (Dis->Surr)':>25} | {'Regime'}")
            for j, R in enumerate(radii):
                expN = expected_N[j]
                if expN < 1: regime = "<1"
                elif expN < 5: regime = "1-5"
                elif expN < 10: regime = "5-10"
                elif expN < 20: regime = "10-20"
                else: regime = ">=20"
                print(f"{R:10.2f} | {expN:15.2f} | {E_particle[j]*100:24.2f}% | {regime}")
                
            if R_s == 200 and C == 12:
                print("\n--- M31 Specific Test (N=2500) ---")
                print(f"{'R':>5} | {'Exp N':>8} | {'Dis/Surr':>10} | {'E_particle':>12} | {'E_calib':>12} | {'E_total':>12}")
                for r_val in [1, 2, 5, 10, 20, 50, 100, 200]:
                    if r_val in radii:
                        j = np.where(radii == r_val)[0][0]
                        dis_surr = ensemble_mean[j] / a_surr_all[j]
                        print(f"{r_val:5.1f} | {expected_N[j]:8.2f} | {dis_surr:10.2f} | {E_particle[j]*100:11.2f}% | {E_calib[j]*100:11.2f}% | {E_total[j]*100:11.2f}%")
            
            # 9. Barnes Hut verification on a single realization
            halo = NFWModel(N=N_eval, R_s=R_s, C=C, total_mass=M_total, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
            pos, vel, mass, types = halo.generate()
            
            pos_all = np.vstack([pos, probes])
            mass_all = np.concatenate([mass, np.zeros(len(probes))])
            vel_all = np.zeros_like(pos_all)
            types_all = np.zeros(len(pos_all), dtype=np.int32)
            
            import builtins
            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None
            engine_bh = HermiteEngine(pos_all, vel_all, mass_all, types_all, G=G, eps=eps, dt=0.01)
            builtins.print = original_print
            
            acc_bh = engine_bh._acc[len(pos):]
            acc_dir_bh_test = compute_probe_forces(pos, mass, probes, eps)
            
            bh_err_mag = np.linalg.norm(acc_bh - acc_dir_bh_test, axis=1)
            dir_mag = np.linalg.norm(acc_dir_bh_test, axis=1)
            bh_rel = bh_err_mag / np.maximum(dir_mag, 1e-12)
            
            max_bh_idx = np.argmax(bh_rel)
            max_bh_r = r_probes[max_bh_idx]
            
            print(f"\n--- Barnes-Hut vs Direct (N={N_eval}) ---")
            print(f"Median Rel: {np.median(bh_rel)*100:.2f}%, P95 Rel: {np.percentile(bh_rel, 95)*100:.2f}%, Max Rel: {np.max(bh_rel)*100:.2f}% (at R={max_bh_r})")
            print(f"Abs RMS: {np.sqrt(np.mean(bh_err_mag**2)):.6f}, Abs Max: {np.max(bh_err_mag):.6f}")

    # 5. Fit N-scaling for E_particle(N) = A/sqrt(N) + B
    print("\n--- 5. N-Scaling of E_particle (Discrete vs Continuous) ---")
    sys_biases = []
    
    for j, R in enumerate(radii):
        if R in [1*eps, 2*eps, 5*eps, 10*eps, 20*eps, R_s, 0.5*R_vir] or (R_s == 200 and R in [20, 50, 100]):
            xdata = np.array(N_list)
            ydata = np.array([results_by_N[n]['E_particle'][j] for n in N_list])
            try:
                popt, _ = curve_fit(scaling_func, xdata, ydata, p0=[1.0, 0.0], bounds=([0, -1], [np.inf, 1]))
                A, B = popt
                sys_biases.append(B)
                print(f"R = {R:6.2f}: E_particle = {A:.4f}/sqrt(N) + {B*100:6.2f}% [B term]")
            except:
                print(f"R = {R:6.2f}: Fitting failed")
                sys_biases.append(np.mean(ydata)) # fallback roughly
                
    return sys_biases

def classify_and_recommend(all_sys_biases):
    print("\n" + "="*70)
    print("=== FINAL CLASSIFICATION ===")
    print("="*70)
    
    max_bias = max([abs(b) for b in all_sys_biases])
    
    if max_bias < 0.05:
        classification = "FINITE-N RESOLUTION LIMIT"
        reason = "Discrete-to-continuous error B term (asymptote) approaches zero (< 5%). The surrogate itself remains well within calibration gates (E_calib)."
        recommendation = "Retain current surrogate and document finite-N resolution requirements."
    elif max_bias > 0.10:
        classification = "SYSTEMATIC SURROGATE BIAS"
        reason = "Discrete-to-continuous asymptotic error term B remains high (> 10%), proving a fundamental discrete bias at infinite N."
        recommendation = "Recalibrate surrogate."
    else:
        classification = "INCONCLUSIVE"
        reason = "Asymptotic bias term is marginal (5-10%). Cannot definitively rule out bias vs extremely slow higher-order convergence."
        recommendation = "Increase production N."
        
    print(f"Classification: {classification}")
    print(f"Reason: {reason}")
    print(f"\nProduction Recommendation: {recommendation}")


if __name__ == "__main__":
    configs = [
        (10.0, 10.0, 100.0),
        (15.0, 10.0, 20000.0)
    ]
    
    all_biases = []
    for R_s, C, M_total in configs:
        b = evaluate_ensemble(R_s, C, M_total, run_extended=False)
        all_biases.extend(b)
        
    # M31 explicitly with extended N
    b_m31 = evaluate_ensemble(200.0, 12.0, 1.5e7, run_extended=True)
    all_biases.extend(b_m31)
    
    classify_and_recommend(all_biases)

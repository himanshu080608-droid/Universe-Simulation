import sys
import os
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.models import NFWModel
from physics.integrator import compute_forces_direct

def analyze_nfw():
    print("--- A. Radial Distribution ---")
    seed = 42
    rng = np.random.default_rng(seed)
    
    R_s = 10.0
    C = 10.0
    M = 1000.0
    N = 10000
    
    halo = NFWModel(N=N, R_s=R_s, C=C, total_mass=M, rng=rng, types_dist={'types': [14], 'probs': [1.0]})
    pos, vel, mass, types = halo.generate()
    
    r = np.linalg.norm(pos, axis=1)
    
    # fractions
    frac_inner = np.sum(r < 0.5 * R_s) / N
    frac_rs = np.sum((r >= 0.5 * R_s) & (r < 2.0 * R_s)) / N
    frac_outer = np.sum(r >= 8.0 * R_s) / N
    
    print(f"Fraction near center (r < 0.5 R_s): {frac_inner:.4f}")
    print(f"Fraction near R_s (0.5 R_s <= r < 2 R_s): {frac_rs:.4f}")
    print(f"Fraction near boundary (r >= 8 R_s): {frac_outer:.4f}")
    print(f"Maximum radius: {np.max(r):.4f} (Expected R_vir = {C * R_s})")
    
    # CDF discrepancy
    def nfw_mass_frac(x):
        return np.log(1.0 + x) - x / (1.0 + x)
    m_vir = nfw_mass_frac(C)
    
    r_sorted = np.sort(r)
    cdf_empirical = np.arange(1, N + 1) / N
    cdf_target = nfw_mass_frac(r_sorted / R_s) / m_vir
    
    discrepancy = np.abs(cdf_empirical - cdf_target)
    max_discrepancy = np.max(discrepancy)
    rms_discrepancy = np.sqrt(np.mean(discrepancy**2))
    
    print(f"Max CDF Discrepancy: {max_discrepancy:.4f}")
    print(f"RMS CDF Discrepancy: {rms_discrepancy:.4f}")
    
    print("\n--- B. Force Measurements ---")
    N_force = 500
    halo_force = NFWModel(N=N_force, R_s=R_s, C=C, total_mass=M, rng=rng, types_dist={'types': [14], 'probs': [1.0]})
    pos_f, vel_f, mass_f, types_f = halo_force.generate()
    
    # Engine direct force
    # compute_forces_direct(pos, mass, G, eps)
    acc_engine = compute_forces_direct(pos_f, mass_f, N_force, 1.0, 1.0)
    r_f = np.linalg.norm(pos_f, axis=1)
    r_hat = pos_f / r_f[:, None]
    a_rad_engine = -np.sum(acc_engine * r_hat, axis=1) # positive means inward
    
    # Target force from the code's assumption: a = v_circ^2 / r = G M_r / r^2
    M_r = M * (nfw_mass_frac(r_f / R_s) / m_vir)
    a_rad_model = M_r / (r_f**2 + 1e-9)
    
    rel_error = np.abs(a_rad_model - a_rad_engine) / np.maximum(np.abs(a_rad_engine), 1e-9)
    
    bins = [
        ("inner", r_f < 1.0 * R_s),
        ("R_s region", (r_f >= 1.0 * R_s) & (r_f < 3.0 * R_s)),
        ("outer", r_f >= 3.0 * R_s)
    ]
    
    for name, mask in bins:
        errs = rel_error[mask]
        if len(errs) == 0: continue
        med = np.median(errs)
        rms = np.sqrt(np.mean(errs**2))
        p95 = np.percentile(errs, 95)
        mx = np.max(errs)
        print(f"[{name}] Med: {med:.4f} | RMS: {rms:.4f} | P95: {p95:.4f} | Max: {mx:.4f}")

    print("\n--- C. Velocity Measurements ---")
    r_v = r_f
    v_vec = vel_f
    t_hat = np.column_stack((-r_hat[:, 1], r_hat[:, 0]))
    
    v_rad = np.sum(v_vec * r_hat, axis=1)
    v_tan = np.sum(v_vec * t_hat, axis=1)
    
    v_circ_target = np.sqrt(M_r / (r_v + 1e-9))
    
    for name, mask in bins:
        vr = v_rad[mask]
        vt = v_tan[mask]
        vc = v_circ_target[mask]
        if len(vr) == 0: continue
        
        mean_vr = np.mean(vr)
        mean_vt = np.mean(vt)
        sig_r = np.std(vr)
        sig_t = np.std(vt)
        
        ratio_r = np.median(sig_r / vc) if np.median(vc) > 0 else 0
        ratio_t = np.median(sig_t / vc) if np.median(vc) > 0 else 0
        
        print(f"[{name}] Mean Vr: {mean_vr:.4f} | Mean Vt: {mean_vt:.4f}")
        print(f"         sigma_R: {sig_r:.4f} | sigma_phi: {sig_t:.4f}")
        print(f"         sig_R/v_c: {ratio_r:.4f} | sig_phi/v_c: {ratio_t:.4f}")

if __name__ == "__main__":
    analyze_nfw()

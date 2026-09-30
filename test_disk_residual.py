import numpy as np
from scipy.special import i0, i1, k0, k1

def disk_residual():
    y_vals = np.linspace(0.1, 5.0, 100) / 2.0
    r_vals = y_vals * 2.0  # r in units of R_d
    
    # Exact Force (G=1, M=1, R_d=1) -> Sigma_0 = 1 / (2*pi)
    # v_c^2 = 4*pi * (1 / 2*pi) * y^2 * [ I0K0 - I1K1 ]
    # v_c^2 = 2 * y^2 * [ I0K0 - I1K1 ]
    I0, I1, K0, K1 = i0(y_vals), i1(y_vals), k0(y_vals), k1(y_vals)
    vc2_exact = 2.0 * y_vals**2 * (I0*K0 - I1*K1)
    F_exact = vc2_exact / r_vals
    
    # Approx Force = M_enc(r) / r^2
    M_enc = 1.0 - (1.0 + r_vals) * np.exp(-r_vals)
    F_approx = M_enc / r_vals**2
    
    rel_error = np.abs(F_approx - F_exact) / F_exact
    
    print(f"Max rel error: {np.max(rel_error):.4f} at r/R_d = {r_vals[np.argmax(rel_error)]:.2f}")
    print(f"Median rel error: {np.median(rel_error):.4f}")
    
if __name__ == "__main__":
    disk_residual()

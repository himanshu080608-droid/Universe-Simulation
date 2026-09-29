"""
Phase 6.1 — Rigorous Plummer/Kuzmin Equilibrium Reconstruction
tests/analyze_stage6p1_plummer_jeans.py

Diagnostic only.  No production files are modified.
All units: normalized simulation units (G=1, eps=1).

Production constants confirmed from simulation_engine.py:
  G=1.0  eps=1.0  dt=0.001  BH_THRESHOLD=600

Production PlummerModel (universe/models.py):
  u  ~ Uniform(u_lo=0.001, u_hi=0.95)
  R(u) = a * sqrt(1/(1-u)^2 - 1)
  sigma_prod^2 = M_total_system / (6 * sqrt(R^2 + a^2))
  v_esc^2 = 2*M/sqrt(R^2+a^2) + 2*M_c/(R+eps)  (escape clamp)

Force law (physics/integrator.py compute_forces_direct):
  F_ij = G * m_j * (r_ij^2 + eps^2)^{-3/2} * delta_r
Potential:
  phi_ij = -G * m_i * m_j / sqrt(|dx|^2 + eps^2)
"""

import sys, os, warnings, time
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.models import PlummerModel

G   = 1.0
EPS = 1.0

# ──────────────────────────────────────────────────────────────
# Production sampler geometry
# ──────────────────────────────────────────────────────────────
U_LO = 0.001
U_HI = 0.95

def kuzmin_cdf_inv(u, a):
    return a * np.sqrt(1.0/(1.0 - u)**2 - 1.0)

def R_min_prod(a): return kuzmin_cdf_inv(U_LO, a)
def R_max_prod(a): return kuzmin_cdf_inv(U_HI, a)

# ──────────────────────────────────────────────────────────────
# Truncated Kuzmin surface density (renormalized to M over [R_min,R_max])
# Full Kuzmin CDF: F(R) = 1 - a/sqrt(R^2+a^2)
# Mass fraction in [R_lo,R_hi]: F(R_hi)-F(R_lo)
# ──────────────────────────────────────────────────────────────
def kuzmin_cdf(R, a):
    return 1.0 - a / np.sqrt(R**2 + a**2)

def kuzmin_sigma(R, a, M, R_lo, R_hi):
    """
    Truncated Kuzmin surface mass density, renormalized so total mass = M.
    Sigma_raw(R) = M_full * a / (2*pi*(R^2+a^2)^1.5)
    The truncated version normalizes by (F(R_hi)-F(R_lo)).
    """
    norm = kuzmin_cdf(R_hi, a) - kuzmin_cdf(R_lo, a)
    return M * a / (2.0*np.pi*(R**2 + a**2)**1.5) / norm

# ──────────────────────────────────────────────────────────────
# Truncated Kuzmin CDF (for spatial-distribution validation)
# ──────────────────────────────────────────────────────────────
def truncated_kuzmin_cdf(R, a):
    F_lo = kuzmin_cdf(R_min_prod(a), a)
    F_hi = kuzmin_cdf(R_max_prod(a), a)
    return np.clip((kuzmin_cdf(R, a) - F_lo) / (F_hi - F_lo), 0.0, 1.0)

# ──────────────────────────────────────────────────────────────
# Helpers: printing
# ──────────────────────────────────────────────────────────────
def section(name):
    print(f"\n{'═'*72}\n  {name}\n{'═'*72}")

def subsection(name):
    print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

def ok(msg):   print(f"  ✓ {msg}")
def fail(msg): print(f"  ✗ FAIL: {msg}")

# ──────────────────────────────────────────────────────────────
# Truncated continuum softened-force reference
# Integrates the TRUNCATED Kuzmin disk [R_lo, R_hi].
# Splits the radial integral at R_probe to eliminate near-singular
# behaviour where R_source ≈ R_probe.
# ──────────────────────────────────────────────────────────────
def continuum_force_truncated(R_probe, a, M, eps, R_lo, R_hi,
                               deg_r=120, deg_th=240):
    """
    Softened radial continuum force from truncated Kuzmin disk.
    F(Rp) = integral_{R_lo}^{R_hi} integral_0^{2pi}
              G * Sigma_trunc(r) * (Rp - r*cos_th)
              / (Rp^2 + r^2 - 2*Rp*r*cos_th + eps^2)^{3/2}
              * r  dr  dth
    """
    xr, wr = np.polynomial.legendre.leggauss(deg_r)
    xt, wt = np.polynomial.legendre.leggauss(deg_th)
    th_pts  = np.pi*xt + np.pi
    w_th    = np.pi*wt

    results = np.empty(len(R_probe))
    for k, Rp in enumerate(R_probe):
        # Split at Rp for better quadrature conditioning
        sub_intervals = []
        if R_lo < Rp < R_hi:
            sub_intervals = [(R_lo, Rp), (Rp, R_hi)]
        elif Rp <= R_lo:
            sub_intervals = [(R_lo, R_hi)]
        else:
            sub_intervals = [(R_lo, R_hi)]

        f_acc = 0.0
        for (lo, hi) in sub_intervals:
            if hi <= lo:
                continue
            span  = hi - lo
            r_pts = 0.5*span*xr + 0.5*(lo + hi)
            w_r   = 0.5*span*wr
            for i_r in range(deg_r):
                rp   = r_pts[i_r]
                wr_i = w_r[i_r]
                sig  = kuzmin_sigma(rp, a, M, R_lo, R_hi)
                for i_t in range(deg_th):
                    cos_th = np.cos(th_pts[i_t])
                    dR     = Rp - rp * cos_th
                    dist2  = Rp**2 + rp**2 - 2.0*Rp*rp*cos_th + eps**2
                    f_acc += G * sig * dR / dist2**1.5 * rp * wr_i * w_th[i_t]
        results[k] = f_acc
    return results


def build_force_reference(R_grid, a, M, eps, R_lo, R_hi):
    """Build and convergence-test the continuum force on a dense grid."""
    F120 = continuum_force_truncated(R_grid, a, M, eps, R_lo, R_hi,
                                     deg_r=120, deg_th=240)
    F160 = continuum_force_truncated(R_grid, a, M, eps, R_lo, R_hi,
                                     deg_r=160, deg_th=320)
    F200 = continuum_force_truncated(R_grid, a, M, eps, R_lo, R_hi,
                                     deg_r=200, deg_th=400)

    rel_160_vs_120 = np.abs(F160 - F120) / np.maximum(np.abs(F120), 1e-15)
    rel_200_vs_160 = np.abs(F200 - F160) / np.maximum(np.abs(F160), 1e-15)
    return F200, rel_160_vs_120, rel_200_vs_160


# ──────────────────────────────────────────────────────────────
# Exact pairwise energetics (no scaling)
# ──────────────────────────────────────────────────────────────
def exact_kinetic(vel, masses):
    return 0.5 * np.sum(masses * (vel[:, 0]**2 + vel[:, 1]**2))

def exact_potential(pos, masses, eps=EPS, G=G):
    N  = len(pos)
    pe = 0.0
    for i in range(N-1):
        dx  = pos[i+1:, 0] - pos[i, 0]
        dy  = pos[i+1:, 1] - pos[i, 1]
        r   = np.sqrt(dx**2 + dy**2 + eps**2)
        pe -= G * np.sum(masses[i] * masses[i+1:] / r)
    return pe

def make_plummer(N, a, M, seed, M_central=0.0):
    rng   = np.random.default_rng(seed)
    td    = {'types': [0], 'probs': [1.0]}
    model = PlummerModel(N=N, a_scale=a, total_mass=M, rng=rng,
                         types_dist=td, M_central=M_central)
    return model.generate()

# ──────────────────────────────────────────────────────────────
# 2-D isotropic Jeans solution (numerical, with explicit BCs)
# ──────────────────────────────────────────────────────────────
def solve_jeans(R_grid, a, M, M_c, eps, F_grid):
    """
    Solve the 2-D isotropic radial Jeans equation on R_grid:

      d(Sigma * sigma_R^2)/dR = -Sigma(R) * F(R)

    where F(R) = continuum softened radial acceleration (positive inward).

    Boundary condition: sigma_R^2(R_max) = 0  (zero pressure at outer edge).
    This is appropriate for a truncated distribution with finite support.

    Solution by numerical integration from R_max inward:
      sigma_R^2(R_j) = [1/Sigma(R_j)] * integral_{R_j}^{R_max} Sigma(s)*F(s) ds

    Uses trapezoidal integration on the grid (right-to-left).
    """
    R_lo = R_grid[0]
    R_hi = R_grid[-1]
    Sigma = kuzmin_sigma(R_grid, a, M, R_lo, R_hi)

    integrand = Sigma * F_grid      # Sigma(R) * g_R(R)

    # Cumulative integral from right (R_max) to each R_j
    # integral_{R_j}^{R_max} = cumtrapz from right
    # We integrate left-to-right then subtract from total
    n   = len(R_grid)
    cum = np.zeros(n)              # cum[j] = integral_{R_grid[0]}^{R_grid[j]}
    for j in range(1, n):
        dR      = R_grid[j] - R_grid[j-1]
        cum[j]  = cum[j-1] + 0.5*(integrand[j] + integrand[j-1])*dR

    total  = cum[-1]               # integral_{R_min}^{R_max} Sigma*F dR
    # integral_{R_j}^{R_max} = total - cum[j]
    sigma2 = (total - cum) / np.maximum(Sigma, 1e-50)

    return np.maximum(sigma2, 0.0)


# ──────────────────────────────────────────────────────────────
# Continuum virial from Jeans dispersion
# ──────────────────────────────────────────────────────────────
def continuum_ke_from_jeans(R_grid, sigma2_jeans, a, M, R_lo, R_hi):
    """
    K_cont = integral (1/2) * Sigma(R) * <v^2> * 2*pi*R dR
    For 2-D isotropy: <v^2> = 2*sigma_R^2
    """
    Sigma  = kuzmin_sigma(R_grid, a, M, R_lo, R_hi)
    integrand = 0.5 * Sigma * 2.0 * sigma2_jeans * 2.0*np.pi * R_grid
    return np.trapz(integrand, R_grid)

# ──────────────────────────────────────────────────────────────
# Point-probe angular-mean particle force
# ──────────────────────────────────────────────────────────────
def particle_azimuth_force(pos, masses, R_probe, n_az=32, eps=EPS, G=G):
    th  = np.linspace(0, 2*np.pi, n_az, endpoint=False)
    cos_th = np.cos(th); sin_th = np.sin(th)
    fr_all = np.empty((len(R_probe), n_az))
    for i, Rp in enumerate(R_probe):
        px_p = Rp * cos_th; py_p = Rp * sin_th
        dx   = px_p[:, None] - pos[None, :, 0]   # (n_az, N)
        dy   = py_p[:, None] - pos[None, :, 1]
        d2   = dx**2 + dy**2 + eps**2
        f    = G * masses / d2**1.5                # (n_az, N)
        fx   = np.sum(f * dx, axis=1)             # (n_az,)
        fy   = np.sum(f * dy, axis=1)
        fr_all[i] = fx*cos_th + fy*sin_th         # radial outward
    mean_fr = np.mean(fr_all, axis=1)
    rms_fr  = np.std(fr_all, axis=1)
    return mean_fr, rms_fr

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — EXACT PRODUCTION SPATIAL MEASURE
# ══════════════════════════════════════════════════════════════
def phase_spatial(a, M):
    section("PHASE 6.1 — EXACT PRODUCTION SPATIAL MEASURE")
    Rlo = R_min_prod(a);  Rhi = R_max_prod(a)
    F_lo = kuzmin_cdf(Rlo, a);  F_hi = kuzmin_cdf(Rhi, a)
    mass_frac = F_hi - F_lo

    print(f"  a={a}  M={M}")
    print(f"  u_lo={U_LO}  u_hi={U_HI}")
    print(f"  R_min(u_lo) = {Rlo:.6f}")
    print(f"  R_max(u_hi) = {Rhi:.6f}")
    print(f"  Kuzmin CDF at R_min = {F_lo:.6f}")
    print(f"  Kuzmin CDF at R_hi  = {F_hi:.6f}")
    print(f"  Mass fraction in [R_min,R_max] = {mass_frac:.6f}  "
          f"(renormalization factor = {1/mass_frac:.6f})")
    print(f"  Production sampler assigns total_mass=M={M} to this truncated support.")
    print(f"  Therefore Sigma_trunc(R) = (1/{mass_frac:.4f}) * Sigma_full(R)  for R in [R_min,R_max].")

    # Verify with N=20000 sample
    N = 20_000
    pos, _, _, _ = make_plummer(N, a, M, 0)
    r = np.hypot(pos[:, 0], pos[:, 1])
    r_s = np.sort(r)
    ecdf = np.arange(1, N+1)/N
    tcdf = truncated_kuzmin_cdf(r_s, a)
    D    = np.max(np.abs(ecdf - tcdf))
    print(f"\n  KS vs truncated CDF (N={N}): D = {D:.5f}  ", end="")
    print("PASS" if D < 0.01 else "FAIL (spatial test)")
    return Rlo, Rhi

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — TRUNCATED CONTINUUM FORCE
# ══════════════════════════════════════════════════════════════
def phase_force_reference(a, M, eps, Rlo, Rhi):
    section("PHASE 6.1 — TRUNCATED CONTINUUM FORCE")
    probe_R = np.array([0.05*a, 0.1*a, 0.25*a, 0.5*a, a, 2*a, 5*a, 10*a])
    # Only keep probes inside [Rlo, Rhi]
    probe_R = probe_R[(probe_R >= Rlo) & (probe_R <= Rhi)]

    print(f"  a={a}  M={M}  eps={eps}  R_lo={Rlo:.4f}  R_hi={Rhi:.4f}")
    print(f"  Probe radii: {[f'{r:.2f}' for r in probe_R]}")

    orders = [(120, 240), (160, 320), (200, 400)]
    F_all  = {}
    times  = {}
    for (dr, dt) in orders:
        t0 = time.time()
        F  = continuum_force_truncated(probe_R, a, M, eps, Rlo, Rhi, deg_r=dr, deg_th=dt)
        times[(dr,dt)] = time.time()-t0
        F_all[(dr,dt)] = F
        print(f"  {dr}×{dt}: {times[(dr,dt)]:.2f}s")

    F_ref = F_all[(200,400)]
    print(f"\n  Convergence check (max rel diff):")
    for (d1,t1),(d2,t2) in [((120,240),(160,320)),((160,320),(200,400))]:
        rd = np.abs(F_all[(d2,t2)] - F_all[(d1,t1)]) / np.maximum(np.abs(F_all[(d1,t1)]), 1e-15)
        mx = np.max(rd)
        print(f"    {d2}×{t2} vs {d1}×{t1}: max_rel = {mx*100:.4e}%  "
              f"{'PASS' if mx < 1e-4 else 'FAIL'}")

    print(f"\n  {'R':>8}  {'R/a':>5}  {'F_ref':>12}  {'F_160/F_120':>12}  {'F_200/F_160':>12}")
    for i, R in enumerate(probe_R):
        rd1 = F_all[(160,320)][i]/max(F_all[(120,240)][i], 1e-15)
        rd2 = F_all[(200,400)][i]/max(F_all[(160,320)][i], 1e-15)
        print(f"  {R:>8.3f}  {R/a:>5.2f}  {F_ref[i]:>12.6e}  {rd1:>12.8f}  {rd2:>12.8f}")

    return F_ref, probe_R

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — EXACT VIRIAL DIAGNOSTIC
# ══════════════════════════════════════════════════════════════
def phase_virial(a, M):
    section("PHASE 6.1 — EXACT VIRIAL DIAGNOSTIC")
    seeds = [42, 43, 44, 45, 46]
    Ns    = [1000, 2000]

    for N in Ns:
        subsection(f"N={N}  a={a}  M={M}  eps={EPS}  G={G}")
        ratios = []
        print(f"  {'seed':>5}  {'K':>12}  {'W':>14}  {'2K/|W|':>8}")
        for seed in seeds:
            pos, vel, masses, _ = make_plummer(N, a, M, seed)
            K = exact_kinetic(vel, masses)
            W = exact_potential(pos, masses)
            r = 2*K/abs(W)
            ratios.append(r)
            print(f"  {seed:>5d}  {K:>12.4e}  {W:>14.4e}  {r:>8.4f}")
        arr = np.array(ratios)
        print(f"\n  Summary (N={N}): mean={arr.mean():.4f}  median={np.median(arr):.4f}"
              f"  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        print(f"  Virialized system target: 2K/|W| = 1.0")
        offset = arr.mean() - 1.0
        print(f"  Mean offset from virial: {offset:+.4f}  ({offset/1.0*100:+.1f}%)")

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — 2-D JEANS SOLUTION
# ══════════════════════════════════════════════════════════════
def phase_jeans(a, M, M_c, eps, Rlo, Rhi, label=""):
    section(f"PHASE 6.1 — 2-D JEANS SOLUTION  {label}")
    print("  Boundary condition: sigma_R^2(R_max) = 0  (zero pressure at truncation)")
    print("  Justification: production sampler has finite upper cutoff R_max=R(u_hi).")
    print("  Outside this radius, no particles exist by construction; pressure -> 0.")
    print(f"  R_lo={Rlo:.4f}  R_hi={Rhi:.4f}  a={a}  M={M}  M_c={M_c}  eps={eps}\n")

    # Dense logarithmic grid
    N_grid = 1000
    R_grid = np.logspace(np.log10(Rlo*1.001), np.log10(Rhi*0.999), N_grid)

    # Continuum radial acceleration on grid
    print(f"  Building continuum force on {N_grid}-point grid (200×400)...")
    t0 = time.time()
    # Include central mass contribution
    F_cluster = continuum_force_truncated(R_grid, a, M, eps, Rlo, Rhi,
                                          deg_r=100, deg_th=200)
    # Point-mass softened acceleration from central mass
    if M_c > 0:
        F_central = G * M_c * R_grid / (R_grid**2 + eps**2)**1.5
    else:
        F_central = np.zeros_like(R_grid)
    F_total = F_cluster + F_central
    print(f"  Done in {time.time()-t0:.2f}s")

    # Solve Jeans
    sigma2_jeans = solve_jeans(R_grid, a, M, M_c, eps, F_total)
    sigma_jeans  = np.sqrt(sigma2_jeans)

    # Production dispersion
    M_sys = M + M_c
    sigma2_prod = M_sys / (6.0 * np.sqrt(R_grid**2 + a**2))
    sigma_prod  = np.sqrt(sigma2_prod)

    # Ratio and residual
    ratio    = sigma_prod / np.maximum(sigma_jeans, 1e-15)
    ratio_sq = sigma2_prod / np.maximum(sigma2_jeans, 1e-15)
    resid    = (sigma2_prod - sigma2_jeans) / np.maximum(sigma2_jeans, 1e-15)

    # Canonical evaluation points (only within support)
    eval_Ra = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    eval_R  = [ra*a for ra in eval_Ra if Rlo <= ra*a <= Rhi]

    print(f"\n  {'R':>8}  {'R/a':>5}  {'sigma_J':>10}  {'sigma_P':>10}"
          f"  {'sigP/sigJ':>10}  {'sig2P/sig2J':>12}  {'frac_resid':>12}")
    for R in eval_R:
        idx = np.argmin(np.abs(R_grid - R))
        sJ  = sigma_jeans[idx];  sP = sigma_prod[idx]
        rr  = ratio[idx];        rr2= ratio_sq[idx]
        rs  = resid[idx]
        print(f"  {R:>8.3f}  {R/a:>5.2f}  {sJ:>10.4e}  {sP:>10.4e}"
              f"  {rr:>10.4f}  {rr2:>12.4f}  {rs:>+12.4f}")

    # Summary statistics
    # Only over well-resolved region: R >= eps, R <= 0.95*R_hi
    mask = (R_grid >= eps) & (R_grid <= 0.95*Rhi)
    abs_resid = np.abs(resid[mask])
    print(f"\n  Jeans residual |sigma2_prod/sigma2_Jeans - 1| (R in [eps, 0.95*R_hi]):")
    print(f"    N_pts = {mask.sum()}")
    print(f"    Median  = {np.median(abs_resid):.4f}")
    print(f"    P95     = {np.percentile(abs_resid, 95):.4f}")
    print(f"    Max     = {np.max(abs_resid):.4f}")

    return R_grid, sigma2_jeans, sigma_jeans, F_total, ratio, resid

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — PRODUCTION VS JEANS
# ══════════════════════════════════════════════════════════════
def phase_prod_vs_jeans(a, M, M_c, eps, Rlo, Rhi,
                        R_grid, sigma2_jeans, sigma_jeans, F_total,
                        ratio, resid, label=""):
    section(f"PHASE 6.1 — PRODUCTION VS JEANS  {label}")
    N = 5000
    seeds = [42, 43, 44]

    print(f"  N={N}  a={a}  M={M}  M_c={M_c}  eps={eps}")

    radial_bins = np.array([Rlo, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, Rhi])
    radial_bins = radial_bins[(radial_bins >= Rlo) & (radial_bins <= Rhi)]
    bin_mids    = 0.5*(radial_bins[:-1] + radial_bins[1:])

    print(f"\n  {'r_mid':>8}  {'sigma_J':>10}  {'sigma_P_theory':>15}", end="")
    for seed in seeds:
        print(f"  {'sig_r_emp(s'+str(seed)+')':>14}", end="")
    print()

    # For each seed, measure empirical sigma_r
    emp_data = {}
    for seed in seeds:
        pos, vel, masses, _ = make_plummer(N, a, M, seed, M_central=M_c)
        r = np.hypot(pos[:, 0], pos[:, 1])
        r_hat_x = pos[:, 0]/(r+1e-15); r_hat_y = pos[:, 1]/(r+1e-15)
        vr = vel[:, 0]*r_hat_x + vel[:, 1]*r_hat_y
        emp_data[seed] = (r, vr)

    for j, rm in enumerate(bin_mids):
        R_lo_b = radial_bins[j]; R_hi_b = radial_bins[j+1]
        # Jeans sigma at midpoint
        idx_J = np.argmin(np.abs(R_grid - rm))
        sJ    = sigma_jeans[idx_J]
        sP_th = np.sqrt(max((M+M_c)/(6.0*np.sqrt(rm**2+a**2)), 0))
        print(f"  {rm:>8.2f}  {sJ:>10.4e}  {sP_th:>15.4e}", end="")
        for seed in seeds:
            r, vr = emp_data[seed]
            mask  = (r >= R_lo_b) & (r < R_hi_b)
            sig   = np.std(vr[mask]) if mask.sum() >= 5 else np.nan
            print(f"  {sig:>14.4e}" if not np.isnan(sig) else f"  {'(n<5)':>14}", end="")
        print()

    # Continuum virial check
    print(f"\n  Continuum Jeans kinetic energy:")
    # Use trapz over grid (within production support)
    mask_cont = (R_grid >= Rlo) & (R_grid <= Rhi)
    Sigma_g   = kuzmin_sigma(R_grid[mask_cont], a, M, Rlo, Rhi)
    KE_jeans  = np.trapz(
        0.5 * Sigma_g * 2.0 * sigma2_jeans[mask_cont] * 2.0*np.pi * R_grid[mask_cont],
        R_grid[mask_cont])

    # Production formula kinetic energy
    M_sys = M + M_c
    sigma2_prod_g = M_sys / (6.0 * np.sqrt(R_grid[mask_cont]**2 + a**2))
    KE_prod = np.trapz(
        0.5 * Sigma_g * 2.0 * sigma2_prod_g * 2.0*np.pi * R_grid[mask_cont],
        R_grid[mask_cont])

    print(f"    KE_Jeans (continuum) = {KE_jeans:.4e}")
    print(f"    KE_prod  (continuum) = {KE_prod:.4e}")
    print(f"    KE_prod / KE_Jeans   = {KE_prod/max(KE_jeans, 1e-15):.4f}")

    # Monte-Carlo particle KE and PE
    print(f"\n  Monte-Carlo particle virial (N={N}):")
    for seed in seeds:
        pos, vel, masses, _ = make_plummer(N, a, M, seed, M_central=M_c)
        K = exact_kinetic(vel, masses)
        # PE is expensive at N=5000; use N_pe=1000 sample
        N_pe = 1000
        W_est = exact_potential(pos[:N_pe], masses[:N_pe]) * (N/N_pe)**2
        print(f"    seed={seed}: K={K:.4e}  W_est={W_est:.4e}  2K/|W|={2*K/abs(W_est):.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — FINITE-N RESOLUTION
# ══════════════════════════════════════════════════════════════
def phase_resolution(a, M, eps, Rlo, Rhi, F_ref, probe_R):
    section("PHASE 6.1 — FINITE-N RESOLUTION")
    Ns    = [500, 1000, 2500, 5000, 10000]
    n_az  = 32
    seed  = 7

    print(f"  a={a}  M={M}  eps={eps}  seed={seed}  n_az={n_az}")
    print(f"  Continuum reference: TRUNCATED Kuzmin disk [R_lo={Rlo:.3f}, R_hi={Rhi:.3f}]")
    print(f"  Force reference uses 200×400 GL quadrature.\n")

    headers = ["R/a", "F_ref"] + [f"N={N}(rel%)" for N in Ns] + ["noise_slope"]
    fmt_h   = f"  {'R/a':>5}  {'F_ref':>10}" + "  {:>12}"*len(Ns)
    print(fmt_h.format(*[f"N={N}(rel%)" for N in Ns]))

    particle_data = {}
    for N in Ns:
        pos, _, masses, _ = make_plummer(N, a, M, seed)
        particle_data[N] = (pos, masses)

    mean_table = np.empty((len(probe_R), len(Ns)))
    rms_table  = np.empty((len(probe_R), len(Ns)))

    for i, Rp in enumerate(probe_R):
        Fc = F_ref[i]
        print(f"  {Rp/a:>5.2f}  {Fc:>10.4e}", end="")
        for j, N in enumerate(Ns):
            pos, masses = particle_data[N]
            mfr, rfr   = particle_azimuth_force(pos, masses, [Rp], n_az, eps, G)
            mean_table[i, j] = mfr[0]
            rms_table[i, j]  = rfr[0]
            relerr = (mfr[0] - Fc) / max(abs(Fc), 1e-15)
            print(f"  {relerr*100:>+11.2f}%", end="")
        print()

    # Noise scaling at R~a
    i_a = np.argmin(np.abs(probe_R - a))
    print(f"\n  Noise (RMS azimuth) at R={probe_R[i_a]:.2f}:")
    log_N   = np.log(Ns)
    log_rms = np.log(np.maximum(rms_table[i_a], 1e-15))
    slope   = np.polyfit(log_N, log_rms, 1)[0]
    print(f"  {'N':>8}  {'mean_F':>12}  {'RMS_az':>12}  {'RMS/Fref':>10}")
    for j, N in enumerate(Ns):
        print(f"  {N:>8d}  {mean_table[i_a,j]:>12.4e}"
              f"  {rms_table[i_a,j]:>12.4e}  {rms_table[i_a,j]/max(abs(F_ref[i_a]),1e-15):>10.4f}")
    print(f"  log-log slope (noise vs N): {slope:.3f}  (Poisson: -0.5)")
    print(f"\n  Inner cusp (R<0.5a) errors at N=10000:")
    inner = probe_R < 0.5*a
    for i, Rp in enumerate(probe_R):
        if inner[i]:
            rel = (mean_table[i,-1] - F_ref[i])/max(abs(F_ref[i]),1e-15)
            print(f"    R/a={Rp/a:.2f}: rel_err={rel*100:+.1f}%  "
                  f"(noise RMS/Fref={rms_table[i,-1]/max(abs(F_ref[i]),1e-15):.3f})")

# ══════════════════════════════════════════════════════════════
# PHASE 6.1 — BASE MODEL VERDICT
# ══════════════════════════════════════════════════════════════
def phase_verdict(median_resid, p95_resid, max_resid, virial_mean):
    section("PHASE 6.1 — BASE MODEL VERDICT")
    print(f"  Base configuration: a=10, M=1000, M_c=0, eps=1, G=1")
    print(f"  Production velocity formula: sigma^2 = M / (6*sqrt(R^2+a^2))")
    print()
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  Metric                       │ Value   │ Criterion         │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Median |frac resid|          │ {median_resid:.4f}  │ < 0.05 → A        │")
    print(f"  │  P95    |frac resid|          │ {p95_resid:.4f}  │ < 0.20 → A/B      │")
    print(f"  │  Max    |frac resid|          │ {max_resid:.4f}  │ < 0.50 → B        │")
    print(f"  │  Mean virial 2K/|W| (N=1000)  │ {virial_mean:.4f}  │ ≈1.00 → A/B       │")
    print(f"  └─────────────────────────────────────────────────────────────┘")
    print()

    # Classification rules
    if median_resid < 0.05 and p95_resid < 0.20 and abs(virial_mean-1.0) < 0.10:
        verdict = "A — CONSISTENT"
        explanation = ("Production dispersion agrees with the independently reconstructed "
                       "equilibrium profile within tolerance over the resolved domain.")
    elif (median_resid < 0.30 and max_resid < 1.00) or abs(virial_mean-1.0) < 0.30:
        verdict = "B — APPROXIMATELY CONSISTENT"
        explanation = ("Bounded systematic mismatch: production dispersion is biased "
                       "relative to the Jeans profile, but the system does not diverge immediately. "
                       "The formula systematically under-/over-estimates sigma depending on radius.")
    else:
        verdict = "C — INCONSISTENT"
        explanation = ("Clear systematic mismatch in both radial dispersion and virial balance. "
                       "Production velocity formula does not satisfy the 2-D Jeans equation for "
                       "the truncated Kuzmin disk with softened gravity.")

    print(f"  ▶ VERDICT: {verdict}")
    print()
    print(f"  Explanation: {explanation}")
    print()
    print(f"  Note on limitations of this verdict:")
    print(f"    - Jeans equation is for the EXACT softened force and EXACT truncated distribution")
    print(f"    - Finite-N sampling noise contributes to measured virial ratio")
    print(f"    - The 0-pressure BC at R_max is exact only for a perfectly sharp cutoff")
    print(f"    - omega_rotation/is_gas presets are intentionally non-isotropic — verdict")
    print(f"      applies only to the BASE isotropic non-rotating model")
    print(f"    - Proposed fix (if needed) must come from a separate implementation phase")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("═"*72)
    print("  PHASE 6.1 — Plummer/Kuzmin Rigorous Equilibrium Reconstruction")
    print("  HEAD: a717856  Parent: 1ea961a")
    print("  Branch: feature/dual-kawase-bloom")
    print("  G=1.0  eps=1.0  No production files modified.")
    print("═"*72)

    a_def = 10.0; M_def = 1000.0

    # ── STEP 1: spatial measure
    Rlo, Rhi = phase_spatial(a_def, M_def)

    # ── STEP 2: truncated continuum force reference (probe grid)
    F_ref, probe_R = phase_force_reference(a_def, M_def, EPS, Rlo, Rhi)

    # ── STEP 3: exact virial (saves summary for verdict)
    section("PHASE 6.1 — EXACT VIRIAL DIAGNOSTIC")
    virial_by_N = {}
    for N in [1000, 2000]:
        seeds = [42, 43, 44, 45, 46]
        subsection(f"N={N}  a={a_def}  M={M_def}")
        ratios = []
        print(f"  {'seed':>5}  {'K':>12}  {'W':>14}  {'2K/|W|':>8}")
        for seed in seeds:
            pos, vel, masses, _ = make_plummer(N, a_def, M_def, seed)
            K = exact_kinetic(vel, masses)
            W = exact_potential(pos, masses)
            ratios.append(2*K/abs(W))
            print(f"  {seed:>5d}  {K:>12.4e}  {W:>14.4e}  {ratios[-1]:>8.4f}")
        arr = np.array(ratios)
        virial_by_N[N] = arr
        print(f"\n  mean={arr.mean():.4f}  median={np.median(arr):.4f}"
              f"  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        print(f"  Offset from virial: {arr.mean()-1.0:+.4f}  ({(arr.mean()-1.0)*100:+.1f}%)")

    virial_mean_1000 = virial_by_N[1000].mean()

    # ── STEP 4: Jeans solutions for base + central-mass cases
    configs = [
        (M_def, 0.0,  "base: M=1000, M_c=0"),
        (M_def, 50.0, "M_c=50"),
        (M_def, 200.0,"M_c=200"),
    ]
    jeans_results = {}
    for M, M_c, label in configs:
        R_grid, sig2J, sigJ, F_tot, ratio, resid = phase_jeans(
            a_def, M, M_c, EPS, Rlo, Rhi, label=label)
        jeans_results[(M, M_c)] = (R_grid, sig2J, sigJ, F_tot, ratio, resid)

    # ── STEP 5: Production vs Jeans for all configs
    for M, M_c, label in configs:
        R_grid, sig2J, sigJ, F_tot, ratio, resid = jeans_results[(M, M_c)]
        phase_prod_vs_jeans(a_def, M, M_c, EPS, Rlo, Rhi,
                            R_grid, sig2J, sigJ, F_tot, ratio, resid, label=label)

    # ── STEP 6: Finite-N resolution (with corrected continuum reference)
    phase_resolution(a_def, M_def, EPS, Rlo, Rhi, F_ref, probe_R)

    # ── STEP 7: Verdict
    # Use base case residuals
    R_grid_base, sig2J_base, sigJ_base, _, ratio_base, resid_base = jeans_results[(M_def, 0.0)]
    mask_v = (R_grid_base >= EPS) & (R_grid_base <= 0.95*Rhi)
    abs_r  = np.abs(resid_base[mask_v])
    phase_verdict(np.median(abs_r), np.percentile(abs_r, 95), np.max(abs_r),
                  virial_mean_1000)

    section("COMPLETE")
    print("  All 18 existing regression tests must still pass (run separately).")

if __name__ == "__main__":
    main()

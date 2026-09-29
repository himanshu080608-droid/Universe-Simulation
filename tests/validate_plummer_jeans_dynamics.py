"""
tests/validate_plummer_jeans_dynamics.py

Phase 6.2 — Validation of the Phase 6.1 analytic Jeans initialization.

Diagnostic only.  Does NOT modify production code.

HEAD: a323f18  branch: feature/dual-kawase-bloom
G=1.0  eps=1.0  dt=0.001  direct-force N<=300  BH for N=1000+
"""

import sys, os, warnings, time
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from universe.models import PlummerModel, EPS
from physics.integrator import compute_forces_direct, compute_energy

G  = 1.0
DT = 0.001

# ──────────────────────────────────────────────────────────────
# Old-formula sigma (for fair comparison without re-running git)
# ──────────────────────────────────────────────────────────────
def old_sigma_1d(r, M, M_c, a):
    M_sys = M + M_c
    return np.sqrt(M_sys / (6.0 * np.sqrt(r**2 + a**2) + 1e-9))

# ──────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────
def make_new(N, a, M, seed, M_c=0.0):
    rng = np.random.default_rng(seed)
    td  = {'types': [0], 'probs': [1.0]}
    m   = PlummerModel(N=N, a_scale=a, total_mass=M, rng=rng,
                       types_dist=td, M_central=M_c)
    return m.generate()

def make_old(N, a, M, seed, M_c=0.0):
    """Replicate old initialization exactly, using same RNG order as production."""
    rng = np.random.default_rng(seed)
    td  = {'types': [0], 'probs': [1.0]}
    # Reproduce IDENTICAL spatial positions (same rng calls)
    u   = rng.uniform(0.001, 0.95, N)
    r   = a * np.sqrt(1.0 / (1.0 - u)**2 - 1.0)
    th  = rng.uniform(0, 2*np.pi, N)
    pos = np.column_stack((r * np.cos(th), r * np.sin(th)))
    # Old sigma
    sigma_1d = old_sigma_1d(r, M, M_c, a)
    vx = rng.normal(0, sigma_1d)
    vy = rng.normal(0, sigma_1d)
    # escape clamp (unchanged)
    v_esc_cluster_sq = 2.0 * M / np.sqrt(r**2 + a**2 + 1e-9)
    v_esc_central_sq = 2.0 * M_c / (r + 1e-9)
    v_esc   = np.sqrt(v_esc_cluster_sq + v_esc_central_sq)
    speeds  = np.sqrt(vx**2 + vy**2)
    clamp   = np.minimum(1.0, 0.95 * v_esc / (speeds + 1e-9))
    vx *= clamp; vy *= clamp
    vel = np.column_stack((vx, vy))
    masses = np.full(N, M / N)
    # mass segregation / types (plain)
    types = rng.choice(td['types'], p=td['probs'], size=N)
    return pos, vel, masses, types

# ──────────────────────────────────────────────────────────────
# Virial: W = sum m_i dot(r_i, a_i)   (virial theorem quantity)
# ──────────────────────────────────────────────────────────────
def compute_virial_W(pos, masses):
    N   = len(masses)
    acc = compute_forces_direct(pos, masses, N, G, EPS)
    W   = np.sum(masses * (pos[:, 0]*acc[:, 0] + pos[:, 1]*acc[:, 1]))
    return W

def kinetic(vel, masses):
    return 0.5 * np.sum(masses * (vel[:,0]**2 + vel[:,1]**2))

def total_momentum(vel, masses):
    return np.sqrt(np.sum((masses[:,None]*vel).sum(axis=0)**2))

# ──────────────────────────────────────────────────────────────
# Radial velocity dispersion in bins
# ──────────────────────────────────────────────────────────────
def radial_sigma_bins(pos, vel, a, bins=None):
    if bins is None:
        bins = [0, 0.5*a, a, 2*a, 5*a, 10*a, 20*a]
    r     = np.hypot(pos[:,0], pos[:,1])
    r_hat = pos / (r[:,None] + 1e-15)
    vr    = (vel * r_hat).sum(axis=1)
    result= []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (r >= lo) & (r < hi)
        if mask.sum() >= 3:
            result.append((0.5*(lo+hi)/a, float(np.std(vr[mask]))))
        else:
            result.append((0.5*(lo+hi)/a, float('nan')))
    return result

# ──────────────────────────────────────────────────────────────
# Section helpers
# ──────────────────────────────────────────────────────────────
def section(name):
    print(f"\n{'═'*72}\n  {name}\n{'═'*72}")

def subsection(name):
    print(f"\n  ── {name}")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — Initialization comparison
# ══════════════════════════════════════════════════════════════
def section1(a=10.0, M=1000.0, M_c=0.0, seed=42):
    section("1 — INITIALIZATION COMPARISON  (old vs new σ formula)")
    print(f"  a={a}  M={M}  M_c={M_c}  seed={seed}  eps={EPS}  G={G}")
    print(f"\n  {'N':>6}  {'formula':>5}  {'K':>10}  {'W':>12}  {'2K/|W|':>8}"
          f"  {'|p|':>10}")

    rows = {}
    for N in [1000, 2000, 5000]:
        for label, gen in [("old", make_old), ("new", make_new)]:
            pos, vel, masses, _ = gen(N, a, M, seed, M_c)
            K = kinetic(vel, masses)
            W = compute_virial_W(pos, masses)
            ratio = 2*K / abs(W)
            p     = total_momentum(vel, masses)
            rows[(N, label)] = (K, W, ratio, p)
            print(f"  {N:>6}  {label:>5}  {K:>10.4e}  {W:>12.4e}  "
                  f"{ratio:>8.4f}  {p:>10.4e}")
        N_key = N
        Ko, Wo, ro, _ = rows[(N, 'old')]
        Kn, Wn, rn, _ = rows[(N, 'new')]
        delta_r = rn - ro
        print(f"  {'':>6}  {'Δ':>5}  {'':>10}  {'':>12}  {delta_r:>+8.4f}")
    return rows

# ══════════════════════════════════════════════════════════════
# SECTION 2 — Short dynamical test  (N=300, 1000 steps, direct)
# ══════════════════════════════════════════════════════════════
def section2(a=10.0, M=1000.0, N=300, n_steps=1000, dt=DT):
    section("2 — SHORT DYNAMICAL TEST  (N=300, 1000 KDK steps, direct force)")
    print(f"  N={N}  a={a}  M={M}  dt={dt}  eps={EPS}")

    for seed in [42, 43]:
        subsection(f"seed={seed}")
        pos, vel, masses, _ = make_new(N, a, M, seed)
        masses = masses.astype(np.float64)
        pos    = pos.astype(np.float64)
        vel    = vel.astype(np.float64)

        def snap():
            acc = compute_forces_direct(pos, masses, N, G, EPS)
            E   = compute_energy(pos, vel, masses, N, G, EPS)
            K   = kinetic(vel, masses)
            W   = np.sum(masses * (pos[:,0]*acc[:,0] + pos[:,1]*acc[:,1]))
            r   = np.hypot(pos[:,0], pos[:,1])
            return E, K, W, r

        E0, K0, W0, r0 = snap()
        med0 = np.median(r0)
        q100 = np.percentile(r0, 10); q900 = np.percentile(r0, 90)

        snapshots = {}

        # Initial acc for KDK
        acc = compute_forces_direct(pos, masses, N, G, EPS)

        for step in range(1, n_steps+1):
            # KDK leapfrog
            vel += 0.5*dt * acc
            pos += dt * vel
            acc  = compute_forces_direct(pos, masses, N, G, EPS)
            vel += 0.5*dt * acc

            if step in (0, 250, 500, 1000):
                snapshots[step] = snap()

        # Collect step=0 too
        E_final, Kf, Wf, r_final = snap()
        r_med_f = np.median(r_final)
        bound_f = np.sum(
            0.5*masses*(vel[:,0]**2+vel[:,1]**2) +
            (-G*masses*np.sum(masses)/(np.hypot(r_final,0)+EPS))
            < 0
        )

        print(f"    Step snapshots (2K/|W|):")
        for s, (E,K,W,r) in sorted(snapshots.items()):
            rmed = np.median(r); rq10=np.percentile(r,10); rq90=np.percentile(r,90)
            ratio = 2*K/abs(W) if abs(W)>1e-15 else float('nan')
            print(f"      step={s:>4}: 2K/|W|={ratio:.4f}  "
                  f"r_med={rmed:.3f}  [q10={rq10:.2f}, q90={rq90:.2f}]")

        dE = abs(E_final - E0) / max(abs(E0), 1e-15)
        dr = abs(r_med_f - med0) / max(med0, 1e-15)
        bound_frac = np.sum(compute_energy(
            pos, vel, masses, N, G, EPS) < 0) / N  # per-particle approximate

        print(f"    Summary:")
        print(f"      E0={E0:.4e}  E_final={E_final:.4e}  |ΔE/E0|={dE:.3e}")
        print(f"      r_med: {med0:.3f} → {r_med_f:.3f}  (Δ={dr*100:.1f}%)")
        print(f"      q10: {q100:.2f}→{np.percentile(r_final,10):.2f}  "
              f"q90: {q900:.2f}→{np.percentile(r_final,90):.2f}")
        print(f"      |p_final|={total_momentum(vel,masses):.3e}")

# ══════════════════════════════════════════════════════════════
# SECTION 3 — Production SimulationEngine path  (N=1000, 250 steps)
# ══════════════════════════════════════════════════════════════
def section3(a=10.0, M=1000.0, N=1000, n_steps=250):
    section("3 — PRODUCTION SimulationEngine PATH  (N=1000, 250 steps, BH)")
    from simulation_engine import SimulationEngine, BH_THRESHOLD
    print(f"  N={N}  steps={n_steps}  BH_THRESHOLD={BH_THRESHOLD}")
    print(f"  N {'<' if N < BH_THRESHOLD else '>='} BH_THRESHOLD → "
          f"{'direct' if N < BH_THRESHOLD else 'Barnes-Hut'} path")

    for seed in [42, 43]:
        pos, vel, masses, types = make_new(N, a, M, seed)
        eng = SimulationEngine(pos.copy(), vel.copy(), masses.copy(), types,
                               use_bh=(N >= BH_THRESHOLD))
        eng.warmup()

        e0   = eng.energy
        r0   = np.hypot(eng.pos[:,0], eng.pos[:,1])
        med0 = np.median(r0)

        for _ in range(n_steps):
            eng.step()

        e1   = eng.energy
        r1   = np.hypot(eng.pos[:,0], eng.pos[:,1])
        med1 = np.median(r1)
        dE   = abs(e1 - e0) / max(abs(e0), 1e-15)

        nan_pos = np.any(np.isnan(eng.pos)) or np.any(np.isinf(eng.pos))
        nan_vel = np.any(np.isnan(eng.vel)) or np.any(np.isinf(eng.vel))

        # Determinism check
        pos2, vel2, masses2, types2 = make_new(N, a, M, seed)
        eng2 = SimulationEngine(pos2.copy(), vel2.copy(), masses2.copy(), types2,
                                use_bh=(N >= BH_THRESHOLD))
        eng2.warmup()
        for _ in range(n_steps):
            eng2.step()
        deterministic = np.allclose(eng.pos, eng2.pos)

        # Expansion/contraction gate: median radius stays within 3x of initial
        stable = (med1 > med0 * 0.2) and (med1 < med0 * 5.0)

        print(f"\n  seed={seed}:")
        print(f"    E0={e0:.4e}  E_final={e1:.4e}  |ΔE/E0|={dE:.3e}")
        print(f"    r_med: {med0:.3f} → {med1:.3f}")
        print(f"    NaN/Inf pos={nan_pos}  vel={nan_vel}")
        print(f"    Deterministic: {deterministic}")
        print(f"    Stable (0.2x–5x expansion): {stable}")
        print(f"    {'PASS' if not nan_pos and not nan_vel and deterministic and stable else 'FAIL'}")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — Special presets
# ══════════════════════════════════════════════════════════════
def section4():
    section("4 — SPECIAL PRESETS")
    from universe.generator import build_universe
    presets = ["omega_centauri", "pleiades_m45", "gravothermal_catastrophe"]

    for preset in presets:
        subsection(preset)
        for seed in [42, 43]:
            pos, vel, masses, types = build_universe(preset, seed=seed)

            nan_p   = np.any(np.isnan(pos))  or np.any(np.isinf(pos))
            nan_v   = np.any(np.isnan(vel))  or np.any(np.isinf(vel))
            mass_ok = np.all(masses > 0) and np.all(np.isfinite(masses))

            # Determinism
            pos2, vel2, _, _ = build_universe(preset, seed=seed)
            determ  = (np.array_equal(pos2, pos) and np.array_equal(vel2, vel))

            K = kinetic(vel, masses)
            W = compute_virial_W(pos, masses)
            ratio = 2*K/abs(W) if abs(W) > 1e-15 else float('nan')

            status = "PASS" if not nan_p and not nan_v and mass_ok and determ else "FAIL"
            print(f"    seed={seed}: N={len(masses):,}  2K/|W|={ratio:.3f}  "
                  f"finite_pos={not nan_p}  finite_vel={not nan_v}  "
                  f"determ={determ}  → {status}")

# ══════════════════════════════════════════════════════════════
# SECTION 5 — Acceptance summary
# ══════════════════════════════════════════════════════════════
def acceptance_summary(rows):
    section("ACCEPTANCE SUMMARY")
    print("  Criterion A: new 2K/|W| closer to 1.0 than old?")
    all_pass = True
    for N in [1000, 2000, 5000]:
        ro = rows[(N,'old')][2]
        rn = rows[(N,'new')][2]
        d_old = abs(ro - 1.0)
        d_new = abs(rn - 1.0)
        passes = d_new < d_old
        all_pass = all_pass and passes
        print(f"    N={N}: old={ro:.4f}(Δ={d_old:.4f})  new={rn:.4f}(Δ={d_new:.4f})  "
              f"{'PASS ✓' if passes else 'FAIL ✗'}")
    print()
    print(f"  Overall acceptance: {'PASS ✓' if all_pass else 'FAIL ✗'}")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("═"*72)
    print("  Phase 6.2 — Plummer Jeans Dynamics Validation")
    print("  HEAD: a323f18  branch: feature/dual-kawase-bloom")
    print("  G=1.0  eps=1.0  dt=0.001  No production files modified.")
    print("═"*72)

    rows = section1()
    section2()
    section3()
    section4()
    acceptance_summary(rows)

    section("COMPLETE")

if __name__ == "__main__":
    main()

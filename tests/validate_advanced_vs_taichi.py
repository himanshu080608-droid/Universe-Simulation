"""
tests/validate_advanced_vs_taichi.py

Phase: diagnostic-only validation.
HEAD: 643e8e4  branch: feature/dual-kawase-bloom

Validates AdvancedTaichiEngine against TaichiEngine using:
  - 2-body softened circular orbit (deterministic, from test_advanced_engine_state.py)
  - N=300 Plummer/Jeans IC  seed=42  (from validate_plummer_jeans_dynamics.py)

All measurements use physics.integrator.compute_energy() and compute_momentum()
on float64-cast exported arrays.  engine.energy is never used for comparison.

PASS/FAIL/INVALID semantics are documented inline.
"""

import sys, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── physics helpers (float64, no GPU dependency) ─────────────────────────────
from physics.integrator import compute_energy, compute_momentum

# ── engine imports — single ti.init() already fired by either import ─────────
try:
    from advanced_engine import AdvancedTaichiEngine
    from taichi_engine    import TaichiEngine
    import taichi as ti
    _ENGINES_OK = True
    _ENGINES_ERR = ""
except Exception as exc:
    _ENGINES_OK = False
    _ENGINES_ERR = str(exc)

# ── ICs ───────────────────────────────────────────────────────────────────────
def _two_body_ic(eps=1.0, G=1.0):
    """Softened circular 2-body (from test_advanced_engine_state.py)."""
    m, d = 1.0, 4.0
    omega = np.sqrt(G * 2.0 * m / (d**2 + eps**2)**1.5)
    r, v  = d / 2.0, omega * d / 2.0
    pos   = np.array([[-r, 0.0], [ r, 0.0]], dtype=np.float32)
    vel   = np.array([[ 0.0, -v], [0.0,  v]], dtype=np.float32)
    mass  = np.array([m, m],                  dtype=np.float32)
    types = np.array([0, 0],                  dtype=np.int32)
    return pos, vel, mass, types

def _plummer_ic(N=300, seed=42):
    """Plummer/Jeans N=300 seed=42 (from validate_plummer_jeans_dynamics.py)."""
    from universe.models import PlummerModel
    rng = np.random.default_rng(seed)
    td  = {'types': [0], 'probs': [1.0]}
    m   = PlummerModel(N=N, a_scale=10.0, total_mass=1000.0, rng=rng, types_dist=td)
    pos, vel, masses, types = m.generate()
    return (pos.astype(np.float32), vel.astype(np.float32),
            masses.astype(np.float32), types.astype(np.int32))

# ── energy/momentum using integrator (float64) ───────────────────────────────
def _energy(pos, vel, mass, G=1.0, eps=1.0):
    p64 = pos.astype(np.float64)
    v64 = vel.astype(np.float64)
    m64 = mass.astype(np.float64)
    return compute_energy(p64, v64, m64, len(mass), G, eps)

def _momentum(vel, mass):
    v64 = vel.astype(np.float64)
    m64 = mass.astype(np.float64)
    px, py = compute_momentum(v64, m64, len(mass))
    return np.array([px, py], dtype=np.float64)

# ── engine factory — always float32 ──────────────────────────────────────────
def _adv(pos, vel, mass, types, dt=0.001):
    return AdvancedTaichiEngine(
        pos.copy(), vel.copy(), mass.copy(), types.copy(),
        G=1.0, eps=1.0, dt=dt)

def _tai(pos, vel, mass, types, dt=0.001):
    return TaichiEngine(
        pos.copy(), vel.copy(), mass.copy(), types.copy(),
        G=1.0, eps=1.0, dt=dt)

# ── RMS helper ────────────────────────────────────────────────────────────────
def _rms(arr): return float(np.sqrt(np.mean(arr**2)))

# ── formatting ────────────────────────────────────────────────────────────────
def section(s):  print(f"\n{'═'*72}\n  {s}\n{'═'*72}")
def subsec(s):   print(f"\n  ── {s}")

# ── core diagnostic loop ──────────────────────────────────────────────────────
def run_comparison(case_name, pos0, vel0, mass, types,
                   checkpoints, gates_fn,
                   is_2body=False, G=1.0, eps=1.0, dt=0.001):
    """
    Runs both engines step-by-step from identical ICs, collecting metrics
    at each checkpoint.  Returns (overall_pass, results_list, invalid_reason).
    """
    if not _ENGINES_OK:
        return None, [], f"Engine import failed: {_ENGINES_ERR}"

    N   = len(mass)
    R0  = float(np.sqrt(np.mean(np.sum(pos0**2, axis=1))))
    V0  = float(np.sqrt(np.mean(np.sum(vel0**2, axis=1))))
    M   = float(mass.sum())
    P0  = _momentum(vel0, mass)
    E0  = _energy(pos0, vel0, mass, G, eps)

    if not np.isfinite(E0):
        return None, [], f"E0 not finite: {E0}"

    init_sep = None
    if is_2body:
        init_sep = float(np.linalg.norm(pos0[1] - pos0[0]))

    # Build engines (warmup already done by init)
    engA = _adv(pos0, vel0, mass, types, dt)
    engT = _tai(pos0, vel0, mass, types, dt)

    # ── determinism: build two extra engines for bitwise check ───────────────
    engA2 = _adv(pos0, vel0, mass, types, dt)
    engT2 = _tai(pos0, vel0, mass, types, dt)

    results    = []
    all_pass   = True
    invalid    = ""
    step_cur   = 0
    cp_sorted  = sorted(checkpoints)

    for cp in cp_sorted:
        steps_needed = cp - step_cur
        if steps_needed < 0:
            invalid = f"checkpoint {cp} < current step {step_cur}"
            break
        if steps_needed > 0:
            for _ in range(steps_needed):
                engA.step(n_substeps=1, speed_mult=1.0)
                engT.step(n_substeps=1, speed_mult=1.0)
                engA2.step(n_substeps=1, speed_mult=1.0)
                engT2.step(n_substeps=1, speed_mult=1.0)
            step_cur = cp

        posA = engA.pos; velA = engA.vel
        posT = engT.pos; velT = engT.vel

        # ── validity check ────────────────────────────────────────────────────
        for lbl, p, v in [("Adv", posA, velA), ("Tai", posT, velT)]:
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
                invalid = f"NaN/Inf in {lbl} at step {cp}"
                break
        if invalid: break

        # ── energy (float64) ─────────────────────────────────────────────────
        try:
            EA = _energy(posA, velA, mass, G, eps)
            ET = _energy(posT, velT, mass, G, eps)
        except Exception as exc:
            invalid = f"compute_energy failed at step {cp}: {exc}"; break

        if not (np.isfinite(EA) and np.isfinite(ET)):
            invalid = f"Non-finite energy at step {cp}: EA={EA} ET={ET}"; break

        drift_A  = abs(EA - E0) / abs(E0)
        drift_T  = abs(ET - E0) / abs(E0)
        cross_E  = abs(EA - ET) / abs(E0)

        # ── momentum (float64) ───────────────────────────────────────────────
        try:
            PA = _momentum(velA, mass)
            PT = _momentum(velT, mass)
        except Exception as exc:
            invalid = f"compute_momentum failed at step {cp}: {exc}"; break

        mom_driftA = float(np.linalg.norm(PA - P0)) / (M * V0 + 1e-30)
        mom_driftT = float(np.linalg.norm(PT - P0)) / (M * V0 + 1e-30)

        # ── state disagreement ────────────────────────────────────────────────
        dpos = np.linalg.norm(posA - posT, axis=1)
        dvel = np.linalg.norm(velA - velT, axis=1)

        rms_pos = _rms(dpos) / (R0 + 1e-30)
        p95_pos = float(np.percentile(dpos, 95)) / (R0 + 1e-30)
        max_pos = float(np.max(dpos)) / (R0 + 1e-30)
        rms_vel = _rms(dvel) / (V0 + 1e-30)
        p95_vel = float(np.percentile(dvel, 95)) / (V0 + 1e-30)
        max_vel = float(np.max(dvel)) / (V0 + 1e-30)

        sep_err = None
        if is_2body and init_sep is not None:
            sep_A = float(np.linalg.norm(posA[1] - posA[0]))
            sep_T = float(np.linalg.norm(posT[1] - posT[0]))
            sep_err = abs(sep_A - sep_T) / init_sep

        # ── determinism ───────────────────────────────────────────────────────
        posA2 = engA2.pos; velA2 = engA2.vel
        posT2 = engT2.pos; velT2 = engT2.vel
        determ_A = (np.array_equal(posA, posA2) and np.array_equal(velA, velA2))
        determ_T = (np.array_equal(posT, posT2) and np.array_equal(velT, velT2))

        # ── interval stats for Advanced engine ───────────────────────────────
        intervals   = engA.ti_step_interval.to_numpy()
        mean_int    = float(np.mean(intervals))
        max_int     = int(np.max(intervals))
        active_frac = float(np.mean(intervals == 1))  # fraction at min block

        # ── gate evaluation ───────────────────────────────────────────────────
        row = dict(
            cp=cp, t=cp*dt,
            drift_A=drift_A, drift_T=drift_T, cross_E=cross_E,
            mom_driftA=mom_driftA, mom_driftT=mom_driftT,
            rms_pos=rms_pos, p95_pos=p95_pos, max_pos=max_pos,
            rms_vel=rms_vel, p95_vel=p95_vel, max_vel=max_vel,
            sep_err=sep_err,
            determ_A=determ_A, determ_T=determ_T,
            mean_int=mean_int, max_int=max_int, active_frac=active_frac,
        )
        failures = gates_fn(row)
        row['failures'] = failures
        if failures:
            all_pass = False
        results.append(row)

    if invalid:
        return None, results, invalid
    return all_pass, results, ""

# ── gate functions ────────────────────────────────────────────────────────────
def _gates_2body(r):
    f = []
    if r['rms_pos']  > 1e-4:  f.append(f"rms_pos={r['rms_pos']:.3e} > 1e-4")
    if r['max_pos']  > 1e-3:  f.append(f"max_pos={r['max_pos']:.3e} > 1e-3")
    if r['rms_vel']  > 1e-4:  f.append(f"rms_vel={r['rms_vel']:.3e} > 1e-4")
    if r['max_vel']  > 1e-3:  f.append(f"max_vel={r['max_vel']:.3e} > 1e-3")
    if r['sep_err']  is not None and r['sep_err'] > 1e-3:
        f.append(f"sep_err={r['sep_err']:.3e} > 1e-3")
    if r['drift_A']  > 1e-4:  f.append(f"drift_A={r['drift_A']:.3e} > 1e-4")
    if r['drift_T']  > 1e-4:  f.append(f"drift_T={r['drift_T']:.3e} > 1e-4")
    if r['cross_E']  > 1e-4:  f.append(f"cross_E={r['cross_E']:.3e} > 1e-4")
    if r['mom_driftA'] > 1e-5: f.append(f"mom_driftA={r['mom_driftA']:.3e} > 1e-5")
    if r['mom_driftT'] > 1e-5: f.append(f"mom_driftT={r['mom_driftT']:.3e} > 1e-5")
    if not r['determ_A']: f.append("AdvancedEngine not deterministic")
    if not r['determ_T']: f.append("TaichiEngine not deterministic")
    return f

def _gates_n300(r):
    f = []
    if r['rms_pos']  > 1e-2:  f.append(f"rms_pos={r['rms_pos']:.3e} > 1e-2")
    if r['p95_pos']  > 2e-2:  f.append(f"p95_pos={r['p95_pos']:.3e} > 2e-2")
    if r['rms_vel']  > 1e-2:  f.append(f"rms_vel={r['rms_vel']:.3e} > 1e-2")
    if r['p95_vel']  > 2e-2:  f.append(f"p95_vel={r['p95_vel']:.3e} > 2e-2")
    if r['drift_A']  > 1e-3:  f.append(f"drift_A={r['drift_A']:.3e} > 1e-3")
    if r['drift_T']  > 1e-3:  f.append(f"drift_T={r['drift_T']:.3e} > 1e-3")
    if r['cross_E']  > 1e-3:  f.append(f"cross_E={r['cross_E']:.3e} > 1e-3")
    if r['mom_driftA'] > 1e-5: f.append(f"mom_driftA={r['mom_driftA']:.3e} > 1e-5")
    if r['mom_driftT'] > 1e-5: f.append(f"mom_driftT={r['mom_driftT']:.3e} > 1e-5")
    if not r['determ_A']: f.append("AdvancedEngine not deterministic")
    if not r['determ_T']: f.append("TaichiEngine not deterministic")
    return f

# ── print table ───────────────────────────────────────────────────────────────
def print_table(results, is_2body=False):
    hdr = (f"  {'cp':>5} {'t':>7}  {'driftA':>9} {'driftT':>9} {'crossE':>9}"
           f"  {'rms_pos':>9} {'max_pos':>9}"
           f"  {'rms_vel':>9} {'max_vel':>9}"
           f"  {'momA':>9} {'momT':>9}")
    if is_2body: hdr += f"  {'sep_err':>9}"
    hdr += f"  {'intM':>5} {'intX':>4} {'actF':>6}  status"
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for r in results:
        status = "PASS" if not r['failures'] else f"FAIL({len(r['failures'])})"
        line = (f"  {r['cp']:>5} {r['t']:>7.4f}"
                f"  {r['drift_A']:>9.3e} {r['drift_T']:>9.3e} {r['cross_E']:>9.3e}"
                f"  {r['rms_pos']:>9.3e} {r['max_pos']:>9.3e}"
                f"  {r['rms_vel']:>9.3e} {r['max_vel']:>9.3e}"
                f"  {r['mom_driftA']:>9.3e} {r['mom_driftT']:>9.3e}")
        if is_2body:
            se = r['sep_err'] if r['sep_err'] is not None else float('nan')
            line += f"  {se:>9.3e}"
        line += f"  {r['mean_int']:>5.1f} {r['max_int']:>4d} {r['active_frac']:>6.3f}  {status}"
        print(line)
        if r['failures']:
            for fmsg in r['failures']:
                print(f"      ✗ {fmsg}")

# ── performance benchmark ─────────────────────────────────────────────────────
def benchmark(label, N, pos0, vel0, mass, types,
              n_warmup=100, n_blocks=5, block_steps=200):
    if not _ENGINES_OK:
        print(f"  SKIP (engines unavailable)"); return

    subsec(f"Performance — {label}  N={N}")
    engA = _adv(pos0, vel0, mass, types)
    engT = _tai(pos0, vel0, mass, types)

    # warmup (excludes JIT compilation — engines already compiled at import)
    for _ in range(n_warmup):
        engA.step(n_substeps=1, speed_mult=1.0)
        engT.step(n_substeps=1, speed_mult=1.0)

    def _time_engine(eng):
        times = []
        for _ in range(n_blocks):
            t0 = time.perf_counter()
            for _ in range(block_steps):
                eng.step(n_substeps=1, speed_mult=1.0)
            times.append((time.perf_counter() - t0) * 1000.0 / block_steps)
        return np.median(times)

    ms_A = _time_engine(engA)
    ms_T = _time_engine(engT)

    # time compute_energy separately
    pos_A64 = engA.pos.astype(np.float64)
    vel_A64 = engA.vel.astype(np.float64)
    m64     = mass.astype(np.float64)
    t0 = time.perf_counter()
    for _ in range(20):
        compute_energy(pos_A64, vel_A64, m64, N, 1.0, 1.0)
    ms_E = (time.perf_counter()-t0)*1000.0/20

    ratio = ms_A / ms_T if ms_T > 0 else float('inf')
    print(f"    Advanced:  {ms_A:.3f} ms/step")
    print(f"    Taichi:    {ms_T:.3f} ms/step")
    print(f"    Adv/Tai ratio: {ratio:.3f}x")
    print(f"    compute_energy: {ms_E:.3f} ms/call  (N={N}, CPU Numba)")

# ═══════════════════════════════════════════════════════════════════════════════
def main():
    section("Advanced Engine vs TaichiEngine — Diagnostic Validation")
    print(f"  HEAD: 643e8e4   branch: feature/dual-kawase-bloom")
    print(f"  G=1.0  eps=1.0  dt=0.001  n_substeps=1  speed_mult=1.0")
    if not _ENGINES_OK:
        print(f"\n  INVALID: {_ENGINES_ERR}")
        return

    overall_status = "PASS"

    # ─── CASE 1: 2-body ───────────────────────────────────────────────────────
    section("CASE 1 — 2-body softened circular orbit")
    pos0, vel0, mass, types = _two_body_ic()
    checkpoints_2b = [1, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"  N=2  initial_sep={np.linalg.norm(pos0[1]-pos0[0]):.4f}  "
          f"E0={_energy(pos0,vel0,mass):.6e}  "
          f"|P0|={np.linalg.norm(_momentum(vel0,mass)):.3e}")

    ok2, res2, inv2 = run_comparison(
        "2-body", pos0, vel0, mass, types,
        checkpoints_2b, _gates_2body, is_2body=True)

    if ok2 is None:
        print(f"\n  STATUS: INVALID — {inv2}")
        overall_status = "INVALID"
    else:
        print_table(res2, is_2body=True)
        case_pass = ok2
        first_fail = next((r for r in res2 if r['failures']), None)
        if not case_pass:
            overall_status = "FAIL"
            if first_fail:
                print(f"\n  First failure: step={first_fail['cp']}  t={first_fail['t']:.4f}")
                for f in first_fail['failures']:
                    print(f"    ✗ {f}")

    # ─── CASE 2: N=300 Plummer ────────────────────────────────────────────────
    section("CASE 2 — N=300 Plummer/Jeans  seed=42")
    pos0p, vel0p, massp, typesp = _plummer_ic(N=300, seed=42)
    checkpoints_p = [1, 32, 64, 128, 256]
    E0p = _energy(pos0p, vel0p, massp)
    P0p = _momentum(vel0p, massp)
    print(f"  N=300  E0={E0p:.6e}  |P0|={np.linalg.norm(P0p):.3e}")

    okP, resP, invP = run_comparison(
        "N300", pos0p, vel0p, massp, typesp,
        checkpoints_p, _gates_n300, is_2body=False)

    if okP is None:
        print(f"\n  STATUS: INVALID — {invP}")
        overall_status = "INVALID"
    else:
        print_table(resP, is_2body=False)
        first_fail_p = next((r for r in resP if r['failures']), None)
        if not okP:
            overall_status = "FAIL"
            if first_fail_p:
                print(f"\n  First failure: step={first_fail_p['cp']}  t={first_fail_p['t']:.4f}")
                for f in first_fail_p['failures']:
                    print(f"    ✗ {f}")

    # ─── PERFORMANCE BENCHMARKS ───────────────────────────────────────────────
    section("PERFORMANCE BENCHMARKS")
    for N_bm in [256, 1000]:
        pos_b, vel_b, mass_b, types_b = _plummer_ic(N=N_bm, seed=0)
        benchmark(f"Plummer", N_bm, pos_b, vel_b, mass_b, types_b)

    # ─── FINAL VERDICT ────────────────────────────────────────────────────────
    section(f"FINAL STATUS: {overall_status}")
    if overall_status == "PASS":
        print("  All gates passed at all checkpoints.")
        print("  Both engines are deterministic.")
    elif overall_status == "FAIL":
        print("  One or more numerical gates failed.")
    else:
        print("  Diagnostic could not complete.")

if __name__ == "__main__":
    main()

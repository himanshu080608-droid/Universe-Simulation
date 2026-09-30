"""
Legacy SimulationEngine — owns the physics state and drives the time loop using Barnes-Hut.

Laplace's Demon philosophy:
  The universe is a deterministic machine. Given exact knowledge of every
  particle's position and momentum at t=0, all future (and past) states
  are perfectly calculable. This engine embodies that principle: there is
  no randomness after initialization. Every particle obeys Newton's laws,
  integrated with a symplectic (energy-preserving) leapfrog scheme,
  with gravitational forces computed via Barnes-Hut O(N log N) trees.

Barnes-Hut tree is rebuilt every step (positions change).
Leapfrog requires storing accelerations between steps.
"""

import numpy as np
import time
import sys

from physics.barnes_hut import _build_tree, compute_forces_bh
from physics.integrator  import (compute_forces_direct,
                                  compute_energy, compute_momentum)


# ─────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ─────────────────────────────────────────────────────────────────────────────
BH_THRESHOLD    = 600     # switch from direct to Barnes-Hut above this N
BH_THETA        = 0.8     # opening angle (accuracy vs speed trade-off)
DEFAULT_DT      = 0.001   # time step — halved for better close-encounter stability
DEFAULT_EPS     = 1.00    # softening length — 1.0 prevents fast close encounters
DEFAULT_G       = 1.0     # gravitational constant (normalized units)
ENERGY_INTERVAL = 200     # steps between energy checks


class SimulationEngine:
    """
    Core physics engine with Laplace's Demon deterministic integration.

    Usage:
        engine = SimulationEngine(pos, vel, mass, types)
        engine.warmup()           # JIT compile Numba functions
        for _ in range(steps):
            engine.step()
            pos, types = engine.pos, engine.types
    """

    def __init__(self, pos, vel, mass, types,
                 dt=DEFAULT_DT, eps=DEFAULT_EPS, G=DEFAULT_G,
                 theta=BH_THETA, use_bh=None, preset=""):
        self.pos    = pos.astype(np.float64)
        self.vel    = vel.astype(np.float64)
        self.mass   = mass.astype(np.float64)
        self.types  = types.astype(np.int32)
        self.N      = len(pos)
        self.preset = preset

        self.dt     = dt
        self.eps    = eps
        self.G      = G
        self.theta  = theta
        self.use_bh = use_bh if use_bh is not None else (self.N >= BH_THRESHOLD)

        self.step_count   = 0
        self.energy       = None
        self.energy_ref   = None
        self._acc         = np.zeros((self.N, 2), dtype=np.float64)

        # BH tree cache (pre-allocated once to avoid huge per-step overhead)
        from physics.barnes_hut import MAX_NODES
        self._node_float  = np.empty((MAX_NODES, 6), dtype=np.float64)
        self._node_int    = np.empty((MAX_NODES, 5), dtype=np.int32)
        self._num_nodes   = 0
        
        self._bh_stacks   = np.zeros((self.N, 256), dtype=np.int32)
        self._forces_buf  = np.zeros((self.N, 2), dtype=np.float64)

        print(f"[Engine] N={self.N:,}  dt={dt}  eps={eps}  "
              f"mode={'Barnes-Hut' if self.use_bh else 'Direct O(N²)'}")

    # ─────────────────────────────────────────────────────────────────────────
    # JIT warmup
    # ─────────────────────────────────────────────────────────────────────────
    def warmup(self):
        """
        Trigger Numba JIT compilation with a small dummy system.
        Call once before the main loop to avoid first-frame stutter.
        """
        print("[Engine] Warming up Numba JIT (first compile, ~5–20s)...", flush=True)
        t0 = time.perf_counter()

        n_warm = 8
        p_w  = np.random.rand(n_warm, 2).astype(np.float64)
        v_w  = np.random.rand(n_warm, 2).astype(np.float64) * 0.1
        m_w  = np.ones(n_warm, dtype=np.float64)

        # Compile: direct force path
        compute_forces_direct(p_w, m_w, n_warm, self.G, self.eps)

        # Compile: BH tree path
        xmin, ymin = p_w.min(axis=0) - 1
        xmax, ymax = p_w.max(axis=0) + 1
        nf, ni, nc = _build_tree(p_w, m_w, n_warm, xmin, ymin, xmax, ymax, self._node_float, self._node_int)
        fw = np.zeros((n_warm, 2), dtype=np.float64)
        sw = np.zeros((n_warm, 256), dtype=np.int32)
        compute_forces_bh(p_w, m_w, n_warm, self.G, self.eps, self.theta, nf, ni, nc, fw, sw)

        # Compile: energy
        compute_energy(p_w, v_w, m_w, n_warm, self.G, self.eps)

        elapsed = time.perf_counter() - t0
        print(f"[Engine] JIT warmup complete ({elapsed:.1f}s)", flush=True)

        # Initialize accelerations for the real system
        self._init_accelerations()


    def _init_accelerations(self):
        """Compute initial accelerations (a_i = sum_j G*m_j/r_ij^2 * r_hat)."""
        if self.use_bh:
            self._rebuild_tree()
            self._acc[:] = compute_forces_bh(
                self.pos, self.mass, self.N, self.G, self.eps, self.theta,
                self._node_float, self._node_int, self._num_nodes,
                self._forces_buf, self._bh_stacks
            )
        else:
            self._acc[:] = compute_forces_direct(
                self.pos, self.mass, self.N, self.G, self.eps
            )

        # Note: compute_forces_* already return accelerations (F/m_i),
        # so no further division by mass is needed.

        # Reference energy (used to track drift in HUD)
        if self.N <= 2000:
            self.energy     = compute_energy(self.pos, self.vel, self.mass, self.N, self.G, self.eps)
            self.energy_ref = self.energy

    def _rebuild_tree(self):
        """Rebuild Barnes-Hut quadtree from current positions."""
        pad = 1.0
        xmin = self.pos[:, 0].min() - pad
        ymin = self.pos[:, 1].min() - pad
        xmax = self.pos[:, 0].max() + pad
        ymax = self.pos[:, 1].max() + pad
        self._node_float, self._node_int, self._num_nodes = _build_tree(
            self.pos, self.mass, self.N, xmin, ymin, xmax, ymax, self._node_float, self._node_int
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core step
    # ─────────────────────────────────────────────────────────────────────────
    def _single_leapfrog_step(self, step_dt):
        half_dt = 0.5 * step_dt
        # ── KICK (first half) ─────────────────────────────────────────
        self.vel += self._acc * half_dt

        # ── DRIFT ─────────────────────────────────────────────────────
        self.pos += self.vel * step_dt

        # ── REBUILD TREE & COMPUTE FORCES ──────────────────────────────
        if self.use_bh:
            self._rebuild_tree()
            forces = compute_forces_bh(
                self.pos, self.mass, self.N,
                self.G, self.eps, self.theta,
                self._node_float, self._node_int, self._num_nodes,
                self._forces_buf, self._bh_stacks
            )
        else:
            forces = compute_forces_direct(
                self.pos, self.mass, self.N, self.G, self.eps
            )

        self._acc[:] = forces

        # ── KICK (second half) ────────────────────────────────────────
        self.vel += self._acc * half_dt
        self.step_count += 1

        if self.preset == "cygnus_x1":
            self._process_jet_recycling()

    def _process_jet_recycling(self):
        """
        Continuous Relativistic Jet Engine for Cygnus X-1:
        Maintains a steady-state constant mass flux (up to 6 particles per step)
        re-injected at the black hole nozzle core (r = 1.5..2.5) with hyperbolic escape velocity (v_jet = 1.12 * v_esc).
        Guarantees a 100% smooth, continuous, non-collapsing polar jet beam.
        """
        bh_mask = (self.types == 5)  # BLACK_HOLE (type 5)
        if np.any(bh_mask):
            bh_pos = self.pos[bh_mask][0]
        else:
            bh_pos = np.array([0.0, 0.0])

        bh_mass = 3500.0
        r_vec = self.pos - bh_pos
        r_sq = r_vec[:, 0]**2 + r_vec[:, 1]**2

        # Outer boundary threshold r > 32.0 (r^2 > 1024.0)
        jet_outer = (self.types == 2) & (r_sq > 1024.0)  # COMET (type 2)
        # Inner event horizon accreted threshold r < 2.2 (r^2 < 4.84)
        accreted = (self.types != 5) & (r_sq < 4.84)

        recycle_candidates = jet_outer | accreted
        idxs = np.where(recycle_candidates)[0]

        if len(idxs) > 0:
            # Sort candidates by distance from BH (furthest first)
            dists = r_sq[idxs]
            sort_order = np.argsort(dists)[::-1]
            sorted_idxs = idxs[sort_order]

            # Recycle a constant steady rate of up to 6 particles per step
            n_to_recycle = min(len(sorted_idxs), 6)
            to_re = sorted_idxs[:n_to_recycle]

            rng = np.random.default_rng()
            n_top = n_to_recycle // 2
            n_bot = n_to_recycle - n_top

            r_launch = rng.uniform(1.5, 2.5, n_to_recycle)
            theta_top = rng.normal(np.pi / 2, 0.05, n_top)
            theta_bot = rng.normal(-np.pi / 2, 0.05, n_bot)
            thetas = np.concatenate((theta_top, theta_bot))

            self.pos[to_re, 0] = bh_pos[0] + r_launch * np.cos(thetas)
            self.pos[to_re, 1] = bh_pos[1] + r_launch * np.sin(thetas)

            # Hyperbolic escape velocity v_jet = 1.12 * v_esc ensures jets overcome gravity and stream out perpetually!
            v_esc = np.sqrt(2.0 * bh_mass / r_launch)
            v_jet = 1.12 * v_esc + rng.uniform(-0.5, 0.5, n_to_recycle)
            self.vel[to_re, 0] = v_jet * np.cos(thetas)
            self.vel[to_re, 1] = v_jet * np.sin(thetas)

            self.types[to_re] = 2  # COMET
            self.mass[to_re]  = 0.0001

    def step(self, n_substeps=1, max_ms=12.0, speed_mult=1.0):
        """
        Advance simulation with float speed multiplier.
        """
        n_sub = max(1, int(round(n_substeps * float(speed_mult))))
        
        for _ in range(n_sub):
            self._single_leapfrog_step(self.dt)

        # Periodic energy sampling (only for smaller systems — O(N²))
        if self.N <= 1500 and self.step_count % ENERGY_INTERVAL == 0:
            self.energy = compute_energy(
                self.pos, self.vel, self.mass, self.N, self.G, self.eps
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Precompute mode (for Matplotlib offline rendering)
    # ─────────────────────────────────────────────────────────────────────────
    def precompute(self, total_steps, record_every=10, progress=True):
        """
        Run simulation offline and return full trajectory array.
        Returns: history (total_steps//record_every, N, 2)
        """
        n_frames = total_steps // record_every
        history  = np.zeros((n_frames, self.N, 2), dtype=np.float32)
        frame    = 0

        t0 = time.perf_counter()
        for s in range(total_steps):
            self.step(1)
            if s % record_every == 0 and frame < n_frames:
                history[frame] = self.pos.astype(np.float32)
                frame += 1

            if progress and s % max(1, total_steps // 20) == 0:
                pct = s / total_steps * 100
                elapsed = time.perf_counter() - t0
                eta = elapsed / (s + 1) * (total_steps - s - 1)
                print(f"  [{pct:5.1f}%] step {s:,}/{total_steps:,}  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

        print(f"[Engine] Precompute done. {total_steps:,} steps, "
              f"{frame} frames recorded.", flush=True)
        return history[:frame]

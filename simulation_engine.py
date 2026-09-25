"""
SimulationEngine — owns the physics state and drives the time loop.

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
BH_THETA        = 0.6     # opening angle (accuracy vs speed trade-off)
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
                 theta=BH_THETA, use_bh=None):
        self.pos    = pos.astype(np.float64)
        self.vel    = vel.astype(np.float64)
        self.mass   = mass.astype(np.float64)
        self.types  = types.astype(np.int32)
        self.N      = len(pos)

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
        compute_forces_bh(p_w, m_w, n_warm, self.G, self.eps, self.theta, nf, ni, nc)

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
                self._node_float, self._node_int, self._num_nodes
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
    def step(self, n_substeps=1):
        """
        Advance simulation by n_substeps Kick-Drift-Kick leapfrog steps.

        BUG FIX: the BH tree MUST be rebuilt from the drifted positions before
        the second force evaluation. Using the pre-drift tree for post-drift
        forces injects energy every step and causes particles to escape.
        """
        half_dt = 0.5 * self.dt
        for _ in range(n_substeps):
            # ── KICK (first half) ─────────────────────────────────────────
            self.vel += self._acc * half_dt

            # ── DRIFT ─────────────────────────────────────────────────────
            self.pos += self.vel * self.dt

            # ── REBUILD TREE from NEW positions, then compute forces ───────
            # Critical: tree must reflect particle positions AFTER the drift.
            if self.use_bh:
                self._rebuild_tree()          # ← uses self.pos (now updated)
                forces = compute_forces_bh(
                    self.pos, self.mass, self.N,
                    self.G, self.eps, self.theta,
                    self._node_float, self._node_int, self._num_nodes
                )
            else:
                forces = compute_forces_direct(
                    self.pos, self.mass, self.N, self.G, self.eps
                )

            # ── UPDATE ACCELERATIONS ──────────────────────────────────────
            # compute_forces_* return accelerations directly (G*m_j/r² per m_i)
            self._acc[:] = forces

            # ── KICK (second half) ────────────────────────────────────────
            self.vel += self._acc * half_dt

            self.step_count += 1

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

import numpy as np
import time

from physics.hermite_fmm import build_hermite_tree, compute_acc_jerk_bh, MAX_NODES
from physics.integrator import compute_energy

ENERGY_INTERVAL = 100

class HermiteEngine:
    def __init__(self, pos, vel, mass, types, dt=0.001, eps=1.0, G=1.0, theta=0.8, preset=""):
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

        self.step_count   = 0
        self.energy       = None
        self.energy_ref   = None
        
        self._acc         = np.zeros((self.N, 2), dtype=np.float64)
        self._jerk        = np.zeros((self.N, 2), dtype=np.float64)
        
        # Predicted positions and velocities for Hermite step
        self._pos_p       = np.zeros_like(self.pos)
        self._vel_p       = np.zeros_like(self.vel)

        # BH tree cache
        self._node_float  = np.empty((MAX_NODES, 8), dtype=np.float64)
        self._node_int    = np.empty((MAX_NODES, 5), dtype=np.int32)
        self._num_nodes   = 0

        self._bh_stacks   = np.zeros((self.N, 256), dtype=np.int32)
        self._acc_buf     = np.zeros((self.N, 2), dtype=np.float64)
        self._jerk_buf    = np.zeros((self.N, 2), dtype=np.float64)

        print(f"[Hermite Engine] N={self.N:,}  dt={dt}  eps={eps}  mode=Barnes-Hut (O(N) Multipole)")
        
        # Initial force calculation
        self._rebuild_tree()
        self._compute_acc_jerk(self.pos, self.vel, self._acc, self._jerk)

    def warmup(self):
        print("[Hermite Engine] Warming up Numba JIT (first compile, ~5-20s)...", flush=True)
        t0 = time.perf_counter()
        
        n_warm = 8
        p_w = np.random.rand(n_warm, 2).astype(np.float64)
        v_w = np.random.rand(n_warm, 2).astype(np.float64) * 0.1
        m_w = np.ones(n_warm, dtype=np.float64)
        
        xmin, ymin = p_w.min(axis=0) - 1
        xmax, ymax = p_w.max(axis=0) + 1
        
        nf, ni, num = build_hermite_tree(
            p_w, v_w, m_w, n_warm, xmin, ymin, xmax, ymax,
            self._node_float, self._node_int
        )
        
        fw = np.zeros((n_warm, 2), dtype=np.float64)
        jw = np.zeros((n_warm, 2), dtype=np.float64)
        sw = np.zeros((n_warm, 256), dtype=np.int32)
        
        compute_acc_jerk_bh(
            p_w, v_w, m_w, n_warm, self.G, self.eps, self.theta,
            nf, ni, fw, jw, sw
        )
        
        compute_energy(p_w, v_w, m_w, n_warm, self.G, self.eps)
        
        elapsed = time.perf_counter() - t0
        print(f"[Hermite Engine] JIT warmup complete ({elapsed:.1f}s)", flush=True)

    def _rebuild_tree(self):
        xmin, ymin = self.pos.min(axis=0) - 1
        xmax, ymax = self.pos.max(axis=0) + 1
        nf, ni, num = build_hermite_tree(
            self.pos, self.vel, self.mass, self.N, 
            xmin, ymin, xmax, ymax, 
            self._node_float, self._node_int
        )
        self._node_float = nf
        self._node_int = ni
        self._num_nodes = num

    def _compute_acc_jerk(self, p, v, a_out, j_out):
        compute_acc_jerk_bh(
            p, v, self.mass, self.N, self.G, self.eps, self.theta,
            self._node_float, self._node_int, a_out, j_out, self._bh_stacks
        )

    def _single_hermite_step(self, dt):
        # 1. Predict
        dt2 = dt * dt
        dt3 = dt2 * dt
        
        # x_p = x + v*dt + 0.5*a*dt^2 + (1/6)*j*dt^3
        self._pos_p[:] = self.pos + self.vel * dt + 0.5 * self._acc * dt2 + (1.0 / 6.0) * self._jerk * dt3
        # v_p = v + a*dt + 0.5*j*dt^2
        self._vel_p[:] = self.vel + self._acc * dt + 0.5 * self._jerk * dt2
        
        # 2. Evaluate forces at predicted positions
        # Temporarily swap state so _rebuild_tree uses predicted positions/velocities
        orig_pos = self.pos.copy()
        orig_vel = self.vel.copy()
        self.pos, self.vel = self._pos_p, self._vel_p
        
        self._rebuild_tree()
        self._compute_acc_jerk(self.pos, self.vel, self._acc_buf, self._jerk_buf)
        
        # Restore state by value, NOT by reference!
        self.pos[:] = orig_pos
        self.vel[:] = orig_vel
        
        # 3. Correct
        a1 = self._acc_buf
        j1 = self._jerk_buf
        
        # v_c = v + 0.5*(a + a1)*dt + (1/12)*(j - j1)*dt^2
        self.vel += 0.5 * (self._acc + a1) * dt + (1.0 / 12.0) * (self._jerk - j1) * dt2
        
        # x_c = x + 0.5*(v + v_c)*dt + (1/12)*(a - a1)*dt^2
        # (orig_vel is completely untouched because we used [:] to restore self.vel)
        self.pos += 0.5 * (self.vel + orig_vel) * dt + (1.0 / 12.0) * (self._acc - a1) * dt2
        
        # Update current acc/jerk to the newly calculated ones (for next step's prediction)
        self._acc[:] = a1
        self._jerk[:] = j1

    def step(self, n_substeps=1, max_ms=12.0, speed_mult=1.0):
        # Scale substeps rather than dt to guarantee orbital stability at high speeds
        n_sub = max(1, int(round(n_substeps * float(speed_mult))))
        
        for _ in range(n_sub):
            self._single_hermite_step(self.dt)
            self.step_count += 1

        if self.N <= 1500 and self.step_count % ENERGY_INTERVAL == 0:
            e_tot = compute_energy(self.pos, self.vel, self.mass, self.N, self.G, self.eps)
            if self.energy_ref is None and e_tot != 0:
                self.energy_ref = e_tot
            
            # Simple smooth low-pass
            if self.energy is None:
                self.energy = e_tot
            else:
                self.energy = self.energy * 0.9 + e_tot * 0.1

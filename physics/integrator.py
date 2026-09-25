"""
Leapfrog / Kick-Drift-Kick integrator — symplectic, energy-conserving.
This is the mathematical backbone of Laplace's Demon:
  given exact state at t=0, deterministically predict all future states.

For N-body gravity at scale we use Barnes-Hut approximated forces.
Direct O(N²) fallback is available for small N.
"""

import numpy as np
from numba import njit, prange
from .barnes_hut import _build_tree, compute_forces_bh


# ──────────────────────────────────────────────────────────────────────────────
# Direct O(N²) — used for N < ~512 or as ground-truth check
# ──────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True, parallel=True)
def compute_forces_direct(pos, mass, N, G, eps):
    """Direct pairwise gravity. O(N²) — exact but slow for large N."""
    forces = np.zeros((N, 2), dtype=np.float64)
    eps2   = eps * eps
    for i in prange(N):
        fx = 0.0
        fy = 0.0
        for j in range(N):
            if i != j:
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                r2 = dx*dx + dy*dy + eps2
                f  = G * mass[j] * (r2 ** -1.5)
                fx += f * dx
                fy += f * dy
        forces[i, 0] = fx
        forces[i, 1] = fy
    return forces


# ──────────────────────────────────────────────────────────────────────────────
# Leapfrog KDK integrator — single step
# ──────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def leapfrog_step(pos, vel, acc, mass, N, dt, G, eps, theta,
                  use_bh, node_float, node_int, num_nodes):
    """
    Kick-Drift-Kick leapfrog step.
      v(t + dt/2) = v(t) + a(t) * dt/2          [half kick]
      x(t + dt)   = x(t) + v(t+dt/2) * dt        [drift]
      a(t + dt)   = F(x(t+dt)) / m               [recompute forces]
      v(t + dt)   = v(t+dt/2) + a(t+dt) * dt/2   [half kick]
    """
    half_dt = 0.5 * dt

    # Half kick
    for i in range(N):
        vel[i, 0] += acc[i, 0] * half_dt
        vel[i, 1] += acc[i, 1] * half_dt

    # Drift
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt

    # Recompute forces
    if use_bh:
        forces = compute_forces_bh(pos, mass, N, G, eps, theta,
                                   node_float, node_int, num_nodes)
    else:
        forces = compute_forces_direct(pos, mass, N, G, eps)

    # Update accelerations & half kick
    for i in range(N):
        acc[i, 0] = forces[i, 0] / mass[i]
        acc[i, 1] = forces[i, 1] / mass[i]
        vel[i, 0] += acc[i, 0] * half_dt
        vel[i, 1] += acc[i, 1] * half_dt

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Energy & momentum diagnostics (Laplace demon observer)
# ──────────────────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def compute_energy(pos, vel, mass, N, G, eps):
    """Total mechanical energy: KE + PE."""
    eps2 = eps * eps
    ke = 0.0
    pe = 0.0
    for i in range(N):
        v2 = vel[i, 0]**2 + vel[i, 1]**2
        ke += 0.5 * mass[i] * v2
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            r  = np.sqrt(dx*dx + dy*dy + eps2)
            pe -= G * mass[i] * mass[j] / r
    return ke + pe


@njit(cache=True, fastmath=True)
def compute_momentum(vel, mass, N):
    """Total linear momentum vector."""
    px = 0.0
    py = 0.0
    for i in range(N):
        px += mass[i] * vel[i, 0]
        py += mass[i] * vel[i, 1]
    return px, py

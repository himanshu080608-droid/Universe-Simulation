"""
Barnes-Hut Quadtree for O(N log N) gravitational force computation.
Uses Numba-accelerated flat array tree representation for maximum speed.

Philosophy: Laplace's Demon sees every particle. We approximate distant clusters
as single point masses — the universe's own optimization trick.
"""

import numpy as np
from numba import njit, float64, int32, prange
from numba.typed import List


# ──────────────────────────────────────────────────────────────────────────────
# Flat-array Barnes-Hut Tree (cache-friendly, Numba-compatible)
# ──────────────────────────────────────────────────────────────────────────────
# Node layout per slot (10 floats + 4 ints = stored as float array with views):
#   [cx, cy, half_size, total_mass, com_x, com_y, child_sw, child_se, child_nw, child_ne, count, body_idx]
#
# We use two parallel arrays: float_data and int_data

MAX_NODES = 8_000_000   # pre-allocated node pool


@njit(cache=True, fastmath=True)
def _build_tree(pos, mass, N, x_min, y_min, x_max, y_max, node_float, node_int):
    """
    Build a Barnes-Hut quadtree from scratch using pre-allocated arrays.
    Returns flat node arrays.

    node_float: shape (MAX_NODES, 6)  → [cx, cy, half, total_mass, com_x, com_y]
    node_int:   shape (MAX_NODES, 5)  → [sw, se, nw, ne, body_idx]  (-1 = empty)
    """
    node_count = np.zeros(1, dtype=np.int32)   # next free slot

    def alloc_node(cx, cy, half):
        idx = node_count[0]
        if idx >= MAX_NODES - 1:
            return MAX_NODES - 1 # return the last safe slot if we run out
        node_count[0] += 1
        node_float[idx, 0] = cx
        node_float[idx, 1] = cy
        node_float[idx, 2] = half
        node_float[idx, 3] = 0.0   # total_mass
        node_float[idx, 4] = 0.0   # com_x
        node_float[idx, 5] = 0.0   # com_y
        node_int[idx, 0] = -1
        node_int[idx, 1] = -1
        node_int[idx, 2] = -1
        node_int[idx, 3] = -1
        node_int[idx, 4] = -1
        return idx


    cx0 = (x_min + x_max) * 0.5
    cy0 = (y_min + y_max) * 0.5
    half0 = max(x_max - cx0, y_max - cy0) * 1.001  # slight pad

    root = alloc_node(cx0, cy0, half0)

    def child_quadrant(node_idx, px, py):
        """Returns child slot index (0=sw,1=se,2=nw,3=ne) and new cx,cy,half."""
        cx = node_float[node_idx, 0]
        cy = node_float[node_idx, 1]
        h  = node_float[node_idx, 2] * 0.5
        if px < cx:
            if py < cy:
                q = 0; ncx = cx - h; ncy = cy - h
            else:
                q = 2; ncx = cx - h; ncy = cy + h
        else:
            if py < cy:
                q = 1; ncx = cx + h; ncy = cy - h
            else:
                q = 3; ncx = cx + h; ncy = cy + h
        return q, ncx, ncy, h

    def insert(node_idx, body_i):
        px = pos[body_i, 0]
        py = pos[body_i, 1]
        m  = mass[body_i]

        # Iterative insertion to avoid Numba recursion depth limits
        # Increased to 128 to comfortably handle extreme float64 ratios
        stack = np.zeros(128, dtype=np.int32)
        sp = 0
        stack[sp] = node_idx
        sp += 1

        while sp > 0:
            sp -= 1
            nidx = stack[sp]

            existing = node_int[nidx, 4]  # body_idx

            if existing == -1 and node_int[nidx, 0] == -1:
                # Empty leaf → store body
                node_int[nidx, 4] = body_i
                node_float[nidx, 3] = m
                node_float[nidx, 4] = px * m
                node_float[nidx, 5] = py * m
                break
            else:
                # Internal node or occupied leaf
                if existing >= 0:
                    # Occupied leaf → subdivide: re-insert existing body
                    node_int[nidx, 4] = -2   # mark as internal
                    eq, ecx, ecy, eh = child_quadrant(nidx, pos[existing, 0], pos[existing, 1])
                    if node_int[nidx, eq] == -1:
                        child_node = alloc_node(ecx, ecy, eh)
                        node_int[nidx, eq] = child_node
                    child = node_int[nidx, eq]
                    node_int[child, 4] = existing
                    node_float[child, 3] = mass[existing]
                    node_float[child, 4] = pos[existing, 0] * mass[existing]
                    node_float[child, 5] = pos[existing, 1] * mass[existing]

                # Update COM for this node
                node_float[nidx, 3] += m
                node_float[nidx, 4] += px * m
                node_float[nidx, 5] += py * m

                # Push body into appropriate child
                q, ncx, ncy, nh = child_quadrant(nidx, px, py)
                if node_int[nidx, q] == -1:
                    child_node = alloc_node(ncx, ncy, nh)
                    node_int[nidx, q] = child_node
                
                # Prevent stack overflow from duplicate/precision-lost coordinates
                if sp >= 126:
                    break
                    
                sp += 1
                stack[sp - 1] = node_int[nidx, q]

    for i in range(N):
        insert(root, i)

    return node_float, node_int, node_count[0]


@njit(cache=True, fastmath=True, parallel=True)
def compute_forces_bh(pos, mass, N, G, eps, theta,
                      node_float, node_int, num_nodes):
    """
    Compute gravitational forces using Barnes-Hut approximation.
    theta = opening angle (0.5–0.9). Smaller = more accurate, slower.
    """
    forces = np.zeros((N, 2), dtype=np.float64)
    eps2   = eps * eps

    # Preallocate stacks for each thread to avoid allocation inside parallel loop
    stacks = np.zeros((N, 256), dtype=np.int32)

    for i in prange(N):
        px = pos[i, 0]
        py = pos[i, 1]
        fx = 0.0
        fy = 0.0

        stack = stacks[i]
        sp = 0
        stack[sp] = 0   # start from root
        sp += 1

        while sp > 0:
            sp -= 1
            nidx = stack[sp]

            tm  = node_float[nidx, 3]
            if tm == 0.0:
                continue

            com_x = node_float[nidx, 4] / tm
            com_y = node_float[nidx, 5] / tm
            half  = node_float[nidx, 2]

            dx = com_x - px
            dy = com_y - py
            r2 = dx*dx + dy*dy + eps2

            body_idx = node_int[nidx, 4]

            # Leaf with single body
            if body_idx >= 0 and node_int[nidx, 0] == -1:
                if body_idx != i:
                    inv_r3 = G * tm * (r2 ** -1.5)
                    fx += inv_r3 * dx
                    fy += inv_r3 * dy
                continue

            # Approximate if s/r < theta (far enough away)
            # Check using squared distances to avoid expensive sqrt on failed checks
            if 4.0 * half * half < theta * theta * r2:
                inv_r3 = G * tm * (r2 ** -1.5)
                fx += inv_r3 * dx
                fy += inv_r3 * dy
            else:
                # Recurse into children
                for q in range(4):
                    child = node_int[nidx, q]
                    if child != -1:
                        stack[sp] = child
                        sp += 1

        forces[i, 0] = fx
        forces[i, 1] = fy

    return forces

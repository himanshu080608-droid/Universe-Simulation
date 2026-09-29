import numpy as np
from numba import njit, prange

MAX_NODES = 8_000_000

@njit(cache=True, fastmath=True)
def build_hermite_tree(pos, vel, mass, N, x_min, y_min, x_max, y_max, node_float, node_int):
    """
    node_float: (MAX_NODES, 8) -> [cx, cy, half, total_mass, com_x, com_y, com_vx, com_vy]
    node_int:   (MAX_NODES, 5) -> [sw, se, nw, ne, body_idx]
    """
    node_count = np.zeros(1, dtype=np.int32)
    
    def alloc_node(cx, cy, half):
        idx = node_count[0]
        if idx >= MAX_NODES - 1: return -1
        node_count[0] += 1
        node_float[idx, 0] = cx
        node_float[idx, 1] = cy
        node_float[idx, 2] = half
        node_float[idx, 3] = 0.0 # m
        node_float[idx, 4] = 0.0 # com_x
        node_float[idx, 5] = 0.0 # com_y
        node_float[idx, 6] = 0.0 # com_vx
        node_float[idx, 7] = 0.0 # com_vy
        node_int[idx, 0] = -1
        node_int[idx, 1] = -1
        node_int[idx, 2] = -1
        node_int[idx, 3] = -1
        node_int[idx, 4] = -1
        return idx

    cx0 = (x_min + x_max) * 0.5
    cy0 = (y_min + y_max) * 0.5
    half0 = max(x_max - cx0, y_max - cy0) * 1.001
    root = alloc_node(cx0, cy0, half0)
    if root == -1:
        return node_float, node_int, 0

    def child_quadrant(node_idx, px, py):
        cx = node_float[node_idx, 0]
        cy = node_float[node_idx, 1]
        h  = node_float[node_idx, 2] * 0.5
        if px < cx:
            if py < cy: return 0, cx - h, cy - h, h
            else:       return 2, cx - h, cy + h, h
        else:
            if py < cy: return 1, cx + h, cy - h, h
            else:       return 3, cx + h, cy + h, h

    def insert(node_idx, body_i, stack):
        px = pos[body_i, 0]
        py = pos[body_i, 1]
        vx = vel[body_i, 0]
        vy = vel[body_i, 1]
        m  = mass[body_i]
        
        sp = 0
        stack[sp] = node_idx
        sp += 1
        
        while sp > 0:
            sp -= 1
            nidx = stack[sp]
            if nidx == -1: break
            existing = node_int[nidx, 4]
            
            if existing == -1 and node_int[nidx, 0] == -1:
                node_int[nidx, 4] = body_i
                node_float[nidx, 3] = m
                node_float[nidx, 4] = px * m
                node_float[nidx, 5] = py * m
                node_float[nidx, 6] = vx * m
                node_float[nidx, 7] = vy * m
                break
            else:
                if existing >= 0:
                    node_int[nidx, 4] = -2
                    eq, ecx, ecy, eh = child_quadrant(nidx, pos[existing, 0], pos[existing, 1])
                    if node_int[nidx, eq] == -1:
                        child_node = alloc_node(ecx, ecy, eh)
                        if child_node == -1: break
                        node_int[nidx, eq] = child_node
                    child = node_int[nidx, eq]
                    node_int[child, 4] = existing
                    em = mass[existing]
                    node_float[child, 3] = em
                    node_float[child, 4] = pos[existing, 0] * em
                    node_float[child, 5] = pos[existing, 1] * em
                    node_float[child, 6] = vel[existing, 0] * em
                    node_float[child, 7] = vel[existing, 1] * em

                node_float[nidx, 3] += m
                node_float[nidx, 4] += px * m
                node_float[nidx, 5] += py * m
                node_float[nidx, 6] += vx * m
                node_float[nidx, 7] += vy * m

                q, ncx, ncy, nh = child_quadrant(nidx, px, py)
                if node_int[nidx, q] == -1:
                    child_node = alloc_node(ncx, ncy, nh)
                    if child_node == -1: break
                    node_int[nidx, q] = child_node
                
                if sp >= 126: break
                sp += 1
                stack[sp - 1] = node_int[nidx, q]

    insert_stack = np.empty(128, dtype=np.int32)
    for i in range(N):
        insert(root, i, insert_stack)

    return node_float, node_int, node_count[0]


@njit(cache=True, fastmath=True, parallel=True)
def compute_acc_jerk_bh(pos, vel, mass, N, G, eps, theta,
                        node_float, node_int, acc_buf, jerk_buf, stacks):
    eps2 = eps * eps
    for i in prange(N):
        px = pos[i, 0]
        py = pos[i, 1]
        vx = vel[i, 0]
        vy = vel[i, 1]
        ax = 0.0
        ay = 0.0
        jx = 0.0
        jy = 0.0
        
        stack = stacks[i]
        sp = 0
        stack[sp] = 0
        sp += 1
        
        while sp > 0:
            sp -= 1
            nidx = stack[sp]
            tm = node_float[nidx, 3]
            if tm == 0.0: continue
            
            com_x = node_float[nidx, 4] / tm
            com_y = node_float[nidx, 5] / tm
            com_vx = node_float[nidx, 6] / tm
            com_vy = node_float[nidx, 7] / tm
            half = node_float[nidx, 2]
            
            dx = com_x - px
            dy = com_y - py
            dvx = com_vx - vx
            dvy = com_vy - vy
            
            r2 = dx*dx + dy*dy + eps2
            body_idx = node_int[nidx, 4]
            
            # Leaf
            if body_idx >= 0 and node_int[nidx, 0] == -1:
                if body_idx != i:
                    inv_r3 = (G * tm) / (r2 * np.sqrt(r2))
                    ax += inv_r3 * dx
                    ay += inv_r3 * dy
                    # Jerk: G m (dv/r^3 - 3 r (r.dv) / r^5)
                    r_dot_v = dx*dvx + dy*dvy
                    term2 = 3.0 * r_dot_v / r2
                    jx += inv_r3 * (dvx - dx * term2)
                    jy += inv_r3 * (dvy - dy * term2)
                continue
                
            # Multipole Approximation
            if 4.0 * half * half < theta * theta * r2:
                inv_r3 = (G * tm) / (r2 * np.sqrt(r2))
                ax += inv_r3 * dx
                ay += inv_r3 * dy
                r_dot_v = dx*dvx + dy*dvy
                term2 = 3.0 * r_dot_v / r2
                jx += inv_r3 * (dvx - dx * term2)
                jy += inv_r3 * (dvy - dy * term2)
            else:
                for q in range(4):
                    child = node_int[nidx, q]
                    if child != -1:
                        stack[sp] = child
                        sp += 1
                        
        acc_buf[i, 0] = ax
        acc_buf[i, 1] = ay
        jerk_buf[i, 0] = jx
        jerk_buf[i, 1] = jy


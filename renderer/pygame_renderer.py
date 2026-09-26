"""
Pygame-based real-time renderer for the universe simulation.

Features:
  - GPU-accelerated pixel blitting via pygame.surfarray
  - Color-coded particle types:
      STAR=solar-gold, PLANET=electric-cyan, COMET=pale-mint,
      ROCKY=terracotta-copper, GAS_GIANT=neon-magenta
  - Size hierarchy: comet (1px) <= planet (1px) < rocky planet (1-2px) < gas giant (2-4px) < star (3-6px) < BH (6-12px)
  - Particle trails using additive blending (motion blur)
  - Mouse drag pan, scroll wheel zoom
  - Keyboard controls: SPACE=pause, +/-=speed, R=reset view,
                        [/]=zoom, T=clear trails, ESC=quit
  - HUD overlay: frame time, particle count, zoom, energy drift
"""

import sys
import time
import math
import numpy as np
import pygame
import pygame.surfarray as surfarray
import pygame.gfxdraw

from universe.generator import (
    STAR, PLANET, COMET, ROCKY, GAS_GIANT, BLACK_HOLE,
    RED_GIANT, BLUE_STRAGGLER, WHITE_DWARF, NEUTRON_STAR, EMISSION_STAR,
    RED_DWARF, PIXEL_COLORS
)


# ─────────────────────────────────────────────────────────────────────────────
# Color table: type → (R, G, B)  pre-loaded as a lookup array for speed
# Index order MUST match type integer values (0..10)
# ─────────────────────────────────────────────────────────────────────────────
_COLOR_LUT = np.array([
    PIXEL_COLORS[STAR],           # 0 — warm yellow-gold
    PIXEL_COLORS[PLANET],         # 1 — sky blue
    PIXEL_COLORS[COMET],          # 2 — icy mint-green
    PIXEL_COLORS[ROCKY],          # 3 — terracotta orange
    PIXEL_COLORS[GAS_GIANT],      # 4 — deep royal purple / indigo
    PIXEL_COLORS[BLACK_HOLE],     # 5 — radiant crimson red
    PIXEL_COLORS[RED_GIANT],      # 6 — fiery coral red
    PIXEL_COLORS[BLUE_STRAGGLER], # 7 — electric sapphire blue
    PIXEL_COLORS[WHITE_DWARF],    # 8 — brilliant pearl white
    PIXEL_COLORS[NEUTRON_STAR],   # 9 — electric neon magenta
    PIXEL_COLORS[EMISSION_STAR],  # 10 — vivid emerald lime green
    PIXEL_COLORS[RED_DWARF],      # 11 — deep crimson red (M-dwarf)
], dtype=np.uint8)   # shape (12, 3)

_COLOR_TUPLES = {t: PIXEL_COLORS[t] for t in range(12)}


def _type_to_rgb(types):
    """Vectorized type → RGB mapping. Returns (N, 3) uint8 array."""
    return _COLOR_LUT[types]


class Camera:
    """Handles world→screen coordinate transform with pan/zoom (Standard Right-Handed Cartesian +Y UP)."""
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.zoom   = 20.0       # pixels per world unit
        self.offset = np.array([width / 2, height / 2], dtype=np.float64)

    def world_to_screen(self, pos):
        """
        pos: (N,2) world coords (Standard Cartesian +X right, +Y UP)
        returns: (N,2) screen pixel coords (Pygame screen +X right, +Y DOWN)
        """
        pos = np.asarray(pos)
        if pos.ndim == 1:
            return np.array([pos[0] * self.zoom + self.offset[0],
                             self.offset[1] - pos[1] * self.zoom], dtype=np.float64)
        sp = np.empty_like(pos)
        sp[:, 0] = pos[:, 0] * self.zoom + self.offset[0]
        sp[:, 1] = self.offset[1] - pos[:, 1] * self.zoom
        return sp

    def screen_to_world(self, px, py):
        """
        px, py: screen pixel coords
        returns: (2,) world coords in standard Cartesian (+X right, +Y UP)
        """
        return np.array([(px - self.offset[0]) / self.zoom,
                         (self.offset[1] - py) / self.zoom], dtype=np.float64)

    def zoom_at(self, factor, screen_x, screen_y):
        """Zoom centered on a screen point (enlarged range 0.0001x to 100,000x)."""
        world_pt = self.screen_to_world(screen_x, screen_y)
        self.zoom  *= factor
        self.zoom   = np.clip(self.zoom, 0.0001, 100000.0)
        # Keep world_pt under cursor
        self.offset = np.array([screen_x - world_pt[0] * self.zoom,
                                screen_y + world_pt[1] * self.zoom], dtype=np.float64)

    def pan(self, dx, dy):
        self.offset += np.array([dx, dy])

    def reset(self, pos):
        """Auto-fit all particles into view."""
        if len(pos) == 0:
            return
        cx = (pos[:, 0].max() + pos[:, 0].min()) / 2
        cy = (pos[:, 1].max() + pos[:, 1].min()) / 2
        span = max(pos[:, 0].max() - pos[:, 0].min(), pos[:, 1].max() - pos[:, 1].min()) + 1e-6
        self.zoom   = min(self.w, self.h) * 0.85 / span
        self.offset = np.array([self.w / 2 - cx * self.zoom,
                                 self.h / 2 + cy * self.zoom], dtype=np.float64)


class ParticleTrailBuffer:
    """
    GPU-style additive blending trail buffer.
    Each frame: fade the buffer by a decay factor, then paint new dots.
    """
    def __init__(self, width, height, decay=0.88):
        self.w = width
        self.h = height
        self.decay  = decay
        # Float buffer for smooth additive blending
        self.buffer = np.zeros((width, height, 3), dtype=np.float32)

    def update(self, sx, sy, colors, radii=None):
        """
        sx, sy: screen x/y arrays (int), already clipped to screen
        colors: (N,3) uint8 RGB
        """
        self.buffer *= self.decay

        # Splat particles onto buffer
        N = len(sx)
        for i in range(N):
            x, y = sx[i], sy[i]
            if 0 <= x < self.w and 0 <= y < self.h:
                r, g, b = colors[i]
                self.buffer[x, y, 0] = min(255.0, self.buffer[x, y, 0] + r)
                self.buffer[x, y, 1] = min(255.0, self.buffer[x, y, 1] + g)
                self.buffer[x, y, 2] = min(255.0, self.buffer[x, y, 2] + b)

    def get_surface_array(self):
        return self.buffer.astype(np.uint8)


from numba import njit

@njit(parallel=True, fastmath=True)
def _fast_splat(buffer, fx, fy, prev_fx, prev_fy, colors, w, h, intensity, draw_lines):
    """
    1-pixel thick, gap-free, native DDA line splatting for trails.
    Draws perfectly thin and sharp trails.
    """
    N = len(fx)
    for i in prange(N):
        x1_f = fx[i]
        y1_f = fy[i]
        x0_f = prev_fx[i]
        y0_f = prev_fy[i]

        dx = x1_f - x0_f
        dy = y1_f - y0_f

        if draw_lines:
            dist_cheb = max(abs(dx), abs(dy))
            steps = int(dist_cheb) + 1
            dist = math.sqrt(dx*dx + dy*dy)
        else:
            dist = 0.0
            steps = 1
            dx, dy = 0.0, 0.0
            x0_f, y0_f = x1_f, y1_f

        density = steps / max(1.0, dist) if dist > 0.0 else 1.0
        adj_intensity = intensity / density

        r = float(colors[i, 0]) * adj_intensity
        g = float(colors[i, 1]) * adj_intensity
        b = float(colors[i, 2]) * adj_intensity

        for step in range(steps):
            if steps == 1:
                x_f = x1_f
                y_f = y1_f
            else:
                t = step / (steps - 1)
                x_f = x0_f + dx * t
                y_f = y0_f + dy * t

            ix = int(math.floor(x_f))
            iy = int(math.floor(y_f))

            if 0 <= ix < w and 0 <= iy < h:
                buffer[ix, iy, 0] = min(255.0, buffer[ix, iy, 0] + r)
                buffer[ix, iy, 1] = min(255.0, buffer[ix, iy, 1] + g)
                buffer[ix, iy, 2] = min(255.0, buffer[ix, iy, 2] + b)


# Pixel radii per particle type (Max and Min bounds for zoom scaling).
# Guaranteed size hierarchy at ALL zoom levels (zoomed out or zoomed in):
#   comet (1px) <= planet/asteroid (1px) < rocky planet (1-2px) < gas giant (2-4px) < star (3-6px) < BH (6-12px)
_TYPE_MAX_RADIUS = {
    BLACK_HOLE:     12,  # Crimson Red core with photon ring & event horizon
    RED_GIANT:      10,  # Deep Ruby Red expanded giant envelope
    BLUE_STRAGGLER:  8,  # Luminous Ice Blue star
    EMISSION_STAR:   8,  # Aquatic Emerald Teal emission star
    STAR:            6,  # Warm Cream Gold main sequence star
    RED_DWARF:       4,  # Deep Crimson Red M-type dwarf
    GAS_GIANT:       3,  # Electric Royal Violet gas giant (Jupiter/Saturn scale)
    WHITE_DWARF:     3,  # Diamond Pearl White compact remnant
    NEUTRON_STAR:    3,  # Soft Violet-Indigo pulsar
    ROCKY:           2,  # Terracotta Rust terrestrial rocky planet (Earth/Mars scale)
    PLANET:          1,  # Soft Cyan / Sky Blue orbital disk dot
    COMET:           1,  # Pale Ice Mint comet trail dot
}

_TYPE_MIN_RADIUS = {
    BLACK_HOLE:      3,
    RED_GIANT:       1,
    BLUE_STRAGGLER:  1,
    EMISSION_STAR:   1,
    STAR:            0,  # 0px radius (renders as 1-pixel dot) when zoomed out to prevent blob clumping
    RED_DWARF:       0,
    GAS_GIANT:       0,
    WHITE_DWARF:     0,
    NEUTRON_STAR:    0,
    ROCKY:           0,
    PLANET:          0,
    COMET:           0,
}


# Trail decay presets: (decay_value, label)
TRAIL_PRESETS = [
    (0.00, "Off"),
    (0.50, "Short"),
    (0.75, "Medium"),
    (0.85, "Standard"),
    (0.92, "Long"),
    (0.96, "Extra Long"),
    (0.995, "Super"),
]


from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _shift_trail_buffer_subpixel(buffer, shift_x, shift_y, w, h):
    """
    Sub-pixel bilinear shifting of float32 trail buffer.
    Shift_x, shift_y: continuous float screen pixel displacement.
    Eliminates integer quantization jitter & zigzag trail artifacts during camera pan/spectating!
    """
    if shift_x == 0.0 and shift_y == 0.0:
        return
    new_buf = np.zeros_like(buffer)
    for x in prange(w):
        src_x = float(x) - shift_x
        if 0.0 <= src_x < w - 1.0:
            x0 = int(src_x)
            x1 = x0 + 1
            wx1 = src_x - x0
            wx0 = 1.0 - wx1
            for y in range(h):
                src_y = float(y) - shift_y
                if 0.0 <= src_y < h - 1.0:
                    y0 = int(src_y)
                    y1 = y0 + 1
                    wy1 = src_y - y0
                    wy0 = 1.0 - wy1
                    w00 = wx0 * wy0
                    w10 = wx1 * wy0
                    w01 = wx0 * wy1
                    w11 = wx1 * wy1
                    for c in range(3):
                        new_buf[x, y, c] = (buffer[x0, y0, c] * w00 +
                                            buffer[x1, y0, c] * w10 +
                                            buffer[x0, y1, c] * w01 +
                                            buffer[x1, y1, c] * w11)
    buffer[:] = new_buf


def _shift_trail_buffer(buffer, dx, dy):
    """
    Shift trail buffer by discrete (dx, dy) screen pixels.
    """
    if dx == 0 and dy == 0:
        return
    w, h = buffer.shape[0], buffer.shape[1]
    new_buf = np.zeros_like(buffer)

    src_x0 = max(0, -int(dx))
    src_x1 = min(w, w - int(dx))
    dst_x0 = max(0, int(dx))
    dst_x1 = min(w, w + int(dx))

    src_y0 = max(0, -int(dy))
    src_y1 = min(h, h - int(dy))
    dst_y0 = max(0, int(dy))
    dst_y1 = min(h, h + int(dy))

    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        new_buf[dst_x0:dst_x1, dst_y0:dst_y1] = buffer[src_x0:src_x1, src_y0:src_y1]

    buffer[:] = new_buf


_TYPE_NAMES = {
    BLACK_HOLE:     "Black Hole Core",
    STAR:           "G-Type Main-Sequence Star",
    RED_GIANT:      "Red Giant Star",
    BLUE_STRAGGLER: "Blue Straggler Star",
    WHITE_DWARF:    "White Dwarf Remnant",
    NEUTRON_STAR:   "Pulsar / Neutron Star",
    EMISSION_STAR:  "Wolf-Rayet Emission Star",
    RED_DWARF:      "M-Type Red Dwarf Star",
    ROCKY:          "Terrestrial Rocky Planet",
    GAS_GIANT:      "Jovian Gas Giant",
    PLANET:         "Orbital Disk / Asteroid",
    COMET:          "Comet / Jet Plasma",
}


class PygameRenderer:
    """
    Main renderer. Owns the Pygame window and event loop.
    Calls a physics step callback each frame.
    """
    def __init__(self, width=1600, height=900,
                 trail_decay=0.90,
                 title="Laplace's Demon — Universe Sandbox"):
        pygame.init()
        pygame.display.set_caption(title)
        self.w = width
        self.h = height
        self.screen  = pygame.display.set_mode((width, height),
                                                pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)

        self.camera  = Camera(width, height)
        self.trail_buffer = np.zeros((width, height, 3), dtype=np.float32)

        # Trail size preset index (default: Standard 0.90)
        self.trail_index = 4
        self.decay = TRAIL_PRESETS[self.trail_index][0]

        # Interaction & Spectator Tracking state
        self.paused           = False
        self.speed_mult       = 1.0
        self.tracked_particle = None   # int particle index or None
        self._drag            = False
        self._drag_last       = (0, 0)

        # Stats
        self.frame_times = []
        self.total_frames = 0
        self.energy_ref  = None   # set after first energy sample

    def _handle_events(self, pos):
        """Returns True to keep running, False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_u:                  # U → unselect spectator tracking
                    self.tracked_particle = None
                    self.paused = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_mult = min(self.speed_mult * 2.0, 512.0)
                elif event.key == pygame.K_MINUS:
                    self.speed_mult = max(self.speed_mult / 2.0, 0.0625)
                elif event.key in (pygame.K_COMMA, pygame.K_k):   # , or K → decrease trail size
                    self.trail_index = max(0, self.trail_index - 1)
                    self.decay = TRAIL_PRESETS[self.trail_index][0]
                elif event.key in (pygame.K_PERIOD, pygame.K_l):  # . or L → increase trail size
                    self.trail_index = min(len(TRAIL_PRESETS) - 1, self.trail_index + 1)
                    self.decay = TRAIL_PRESETS[self.trail_index][0]
                elif event.key == pygame.K_r:
                    self.camera.reset(pos)
                    self.tracked_particle = None
                    self.trail_buffer[:] = 0
                elif event.key == pygame.K_t:
                    self.trail_buffer[:] = 0
                elif event.key == pygame.K_RIGHTBRACKET:   # ] → zoom in
                    self.camera.zoom_at(1.30, self.w // 2, self.h // 2)
                    self.trail_buffer[:] = 0
                elif event.key == pygame.K_LEFTBRACKET:    # [ → zoom out
                    self.camera.zoom_at(1 / 1.30, self.w // 2, self.h // 2)
                    self.trail_buffer[:] = 0
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                # If spectating, lock zoom center at screen center
                zx, zy = (self.w // 2, self.h // 2) if self.tracked_particle is not None else (mx, my)
                factor = 1.15 if event.y > 0 else 1 / 1.15
                self.camera.zoom_at(factor, zx, zy)
                self.trail_buffer[:] = 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Feature: When simulation is PAUSED, click selects a celestial entity to spectate!
                    if self.paused and len(pos) > 0:
                        mx, my = event.pos
                        sp = self.camera.world_to_screen(pos)
                        d2 = (sp[:, 0] - mx)**2 + (sp[:, 1] - my)**2
                        min_idx = int(np.argmin(d2))
                        if d2[min_idx] <= 25.0**2:
                            self.tracked_particle = min_idx
                            # Immediately center camera on selected particle
                            target_world = pos[min_idx]
                            new_offset = np.array([
                                self.w / 2 - target_world[0] * self.camera.zoom,
                                self.h / 2 + target_world[1] * self.camera.zoom
                            ], dtype=np.float64)
                            _shift_trail_buffer_subpixel(self.trail_buffer,
                                                         float(new_offset[0] - self.camera.offset[0]),
                                                         float(new_offset[1] - self.camera.offset[1]),
                                                         self.w, self.h)
                            self.camera.offset = new_offset
                            continue

                    self._drag = True
                    self._drag_last = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._drag = False
            elif event.type == pygame.MOUSEMOTION:
                if self._drag:
                    # Dragging camera cancels spectator tracking to restore free camera mode
                    self.tracked_particle = None
                    dx = event.pos[0] - self._drag_last[0]
                    dy = event.pos[1] - self._drag_last[1]
                    self.camera.pan(dx, dy)
                    self._drag_last = event.pos
                    # Shift trail buffer in lockstep with camera pan
                    _shift_trail_buffer(self.trail_buffer, dx, dy)
            elif event.type == pygame.VIDEORESIZE:
                self.w, self.h = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.w, self.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                self.trail_buffer = np.zeros((self.w, self.h, 3), dtype=np.float32)
                self.camera.w = self.w
                self.camera.h = self.h
        return True

    def render_frame(self, pos, types, mass=None, energy=None, step=0, steps_per_frame=1):
        """
        Render one display frame.
        pos:   (N,2) world positions
        types: (N,)  int32 particle type
        mass:  (N,)  float64 particle mass (optional, used for smart star sizing)
        """
        t0 = time.perf_counter()

        # Track previous positions for smooth trail interpolation
        if not hasattr(self, 'prev_pos') or len(self.prev_pos) != len(pos):
            self.prev_pos = pos.copy()

        # Spectator tracking lock: continuously keep tracked particle centered with sub-pixel precision
        if self.tracked_particle is not None:
            if 0 <= self.tracked_particle < len(pos):
                target_world = pos[self.tracked_particle]
                new_offset = np.array([
                    self.w / 2 - target_world[0] * self.camera.zoom,
                    self.h / 2 + target_world[1] * self.camera.zoom
                ], dtype=np.float64)
                shift_x = float(new_offset[0] - self.camera.offset[0])
                shift_y = float(new_offset[1] - self.camera.offset[1])
                if abs(shift_x) > 1e-4 or abs(shift_y) > 1e-4:
                    _shift_trail_buffer_subpixel(self.trail_buffer, shift_x, shift_y, self.w, self.h)
                self.camera.offset = new_offset
            else:
                self.tracked_particle = None

        # Measure frame delta time for framerate-independent trail decay
        dt = t0 - getattr(self, '_last_frame_time', t0 - 1.0 / 60.0)
        self._last_frame_time = t0
        dt_clipped = float(np.clip(dt, 0.001, 0.10))

        # World → screen (keep continuous float64 positions for sub-pixel anti-aliasing)
        sp = self.camera.world_to_screen(pos)
        fx = sp[:, 0]
        fy = sp[:, 1]
        sx = fx.astype(np.int32)
        sy = fy.astype(np.int32)
        
        # Previous screen positions using CURRENT camera (ensures trails pan with the camera)
        sp_prev = self.camera.world_to_screen(self.prev_pos)
        fx_prev = sp_prev[:, 0]
        fy_prev = sp_prev[:, 1]
        
        self.prev_pos = pos.copy()

        # Colors
        colors = _type_to_rgb(types)

        # Framerate-independent decay calibrated to 60 FPS standard (t_decay constant in seconds)
        effective_decay = float(self.decay ** (dt_clipped * 60.0)) if self.decay > 0 else 0.0
        self.trail_buffer *= effective_decay

        # Compute zoom-scaled splat intensity to prevent fuzzy blobs when zoomed out
        zoom = self.camera.zoom
        splat_intensity = float(np.clip((zoom / 30.0) ** 0.5, 0.35, 1.0))
        
        # Disable line streaks entirely if trails are set to Off
        draw_lines = bool(self.decay > 0.0)

        # ── Pass 1: 1-Pixel Thick DDA Line Splatting ─────
        _fast_splat(self.trail_buffer, fx, fy, fx_prev, fy_prev, colors, self.w, self.h, splat_intensity, draw_lines)

        # Push trail buffer to Pygame surface FIRST
        surf_array = np.clip(self.trail_buffer, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(surf_array)
        self.screen.blit(surf, (0, 0))

        # ── Pass 2: stars, planets, comets & black hole → anti-aliased filled circles ──
        # Draw directly onto screen buffer using gfxdraw for anti-aliased round edges
        # Sort indices by type integer so non-BH particles are drawn first, BLACK_HOLE last on top!
        if len(types) > 0:
            idxs = np.arange(len(types))
            type_order = types[idxs]
            sorted_order = np.argsort(type_order)
            idxs = idxs[sorted_order]

            zoom = self.camera.zoom
            w, h = self.w, self.h
            screen = self.screen

            for i in idxs:
                x, y = int(round(sx[i])), int(round(sy[i]))
                if -30 <= x < w + 30 and -30 <= y < h + 30:
                    t = int(types[i])
                    max_r = _TYPE_MAX_RADIUS.get(t, 2)
                    min_r = _TYPE_MIN_RADIUS.get(t, 1)
                    col   = _COLOR_TUPLES.get(t, (200, 200, 200))

                    # Dynamic smooth zoom scaling between min_r and max_r
                    r_val = min_r + (max_r - min_r) * np.clip((zoom - 6.0) / 35.0, 0.0, 1.0)
                    r = int(round(r_val))

                    # Mass-based stellar scaling for stars & supermassive black holes
                    if mass is not None and t in (STAR, RED_GIANT, BLUE_STRAGGLER, EMISSION_STAR, BLACK_HOLE, RED_DWARF):
                        m_val = float(mass[i])
                        if t == BLACK_HOLE:
                            m_scale = np.clip(m_val ** 0.3, 1.0, 5.0)
                        else:
                            m_scale = np.clip(m_val ** 0.25, 0.6, 2.0)
                        r = int(round(r * m_scale))

                    if t == BLACK_HOLE:
                        # 1. Outer Photon Sphere Glow Ring (Crisp White Boundary)
                        pygame.gfxdraw.aacircle(screen, x, y, r + 2, (255, 255, 255))
                        # 2. Accretion Core (Radiant Crimson)
                        pygame.gfxdraw.filled_circle(screen, x, y, r, col)
                        pygame.gfxdraw.aacircle(screen, x, y, r, col)
                        # 3. Pitch-Black Event Horizon Shadow Core
                        if r >= 4:
                            r_bh = max(2, int(r * 0.75))
                            pygame.gfxdraw.filled_circle(screen, x, y, r_bh, (0, 0, 0))
                            pygame.gfxdraw.aacircle(screen, x, y, r_bh, (0, 0, 0))
                    else:
                        if r > 0:
                            pygame.gfxdraw.filled_circle(screen, x, y, r, col)
                            pygame.gfxdraw.aacircle(screen, x, y, r, col)
                        elif r == 0:
                            pygame.gfxdraw.pixel(screen, x, y, col)

            # Draw Spectator Target Reticle over tracked particle
            if self.tracked_particle is not None and 0 <= self.tracked_particle < len(types):
                tx = int(round(sx[self.tracked_particle]))
                ty = int(round(sy[self.tracked_particle]))
                ttype = int(types[self.tracked_particle])
                t_max_r = _TYPE_MAX_RADIUS.get(ttype, 3)
                tr = max(4, min(t_max_r, int(t_max_r * zoom / 20.0)))

                # Pulse reticle animation
                pulse = int(math.sin(time.perf_counter() * 8.0) * 3.0)
                r_reticle = max(8, tr + 8 + pulse)

                # Cyan targeting double ring & crosshair ticks
                pygame.gfxdraw.aacircle(screen, tx, ty, r_reticle, (0, 240, 255))
                pygame.gfxdraw.aacircle(screen, tx, ty, r_reticle + 1, (0, 200, 255))

                l = 6
                pygame.draw.line(screen, (0, 255, 255), (tx - r_reticle - l, ty), (tx - r_reticle + 2, ty), 2)
                pygame.draw.line(screen, (0, 255, 255), (tx + r_reticle - 2, ty), (tx + r_reticle + l, ty), 2)
                pygame.draw.line(screen, (0, 255, 255), (tx, ty - r_reticle - l), (tx, ty - r_reticle + 2), 2)
                pygame.draw.line(screen, (0, 255, 255), (tx, ty + r_reticle - 2), (tx, ty + r_reticle + l), 2)

        # ── HUD overlay ──────────────────────────────────────────────
        self._draw_hud(pos, types, mass, energy, step, steps_per_frame, t0)

        pygame.display.flip()
        self.total_frames += 1

    def _draw_hud(self, pos, types, mass, energy, step, spf, t0):
        dt_ms = (time.perf_counter() - t0) * 1000
        self.frame_times.append(dt_ms)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        avg_ms = sum(self.frame_times) / len(self.frame_times)

        N = len(pos)
        n_bh        = int(np.sum(types == BLACK_HOLE))
        n_stars     = int(np.sum(types == STAR))
        n_red_giant = int(np.sum(types == RED_GIANT))
        n_blue_strag = int(np.sum(types == BLUE_STRAGGLER))
        n_white_dwarf = int(np.sum(types == WHITE_DWARF))
        n_neutron   = int(np.sum(types == NEUTRON_STAR))
        n_emission  = int(np.sum(types == EMISSION_STAR))
        n_red_dwarf = int(np.sum(types == RED_DWARF))
        n_rocky     = int(np.sum(types == ROCKY))
        n_gas       = int(np.sum(types == GAS_GIANT))
        n_planets   = int(np.sum(types == PLANET))
        n_comets    = int(np.sum(types == COMET))

        total_all_stars = n_stars + n_red_giant + n_blue_strag + n_white_dwarf + n_neutron + n_emission + n_red_dwarf

        # Energy drift
        energy_str = "N/A"
        if energy is not None:
            if self.energy_ref is None:
                self.energy_ref = energy
            drift = abs((energy - self.energy_ref) / (abs(self.energy_ref) + 1e-30)) * 100
            energy_str = f"{drift:.4f}%"

        zoom_str = f"{self.camera.zoom:.2f}x"
        trail_label = TRAIL_PRESETS[self.trail_index][1]

        bh_str = f" BH:{n_bh:,}" if n_bh > 0 else ""

        # Format stellar counts breakdown
        stellar_parts = []
        if n_stars > 0: stellar_parts.append(f"G-Star:{n_stars:,}")
        if n_red_giant > 0: stellar_parts.append(f"RedGiant:{n_red_giant:,}")
        if n_blue_strag > 0: stellar_parts.append(f"BlueStar:{n_blue_strag:,}")
        if n_red_dwarf > 0: stellar_parts.append(f"RedDwarf:{n_red_dwarf:,}")
        if n_white_dwarf > 0: stellar_parts.append(f"WhiteDwarf:{n_white_dwarf:,}")
        if n_neutron > 0: stellar_parts.append(f"Pulsar:{n_neutron:,}")
        if n_emission > 0: stellar_parts.append(f"WolfRayet:{n_emission:,}")

        if len(stellar_parts) > 0:
            stars_detail_str = "  ".join(stellar_parts)
        else:
            stars_detail_str = f"Stars:{total_all_stars:,}"

        non_stars_parts = []
        if n_rocky > 0: non_stars_parts.append(f"Rocky:{n_rocky:,}")
        if n_gas > 0: non_stars_parts.append(f"Gas:{n_gas:,}")
        if n_planets > 0: non_stars_parts.append(f"Disk:{n_planets:,}")
        if n_comets > 0: non_stars_parts.append(f"Comet:{n_comets:,}")
        non_stars_str = ("  " + "  ".join(non_stars_parts)) if len(non_stars_parts) > 0 else ""

        # Spectator tracking info
        tracking_info = None
        if self.tracked_particle is not None and 0 <= self.tracked_particle < N:
            t_type = int(types[self.tracked_particle])
            t_name = _TYPE_NAMES.get(t_type, "Celestial Entity")
            t_m = float(mass[self.tracked_particle]) if mass is not None else 1.0
            tracking_info = f"SPECTATING: {t_name} #{self.tracked_particle} (Mass: {t_m:.3f}) | Press 'U' to Unselect"

        paused_hint = None
        if self.paused:
            if self.tracked_particle is None:
                paused_hint = "SIMULATION PAUSED — Click any celestial body to select & spectate"
            else:
                paused_hint = "SIMULATION PAUSED — Press SPACE to resume & spectate tracked body"

        lines = [
            ("LAPLACE'S DEMON — UNIVERSE SANDBOX", (180, 120, 255), self.font_lg),
            (f"Step: {step:,}   SPF: {spf}   FPS: {self.clock.get_fps():.0f}   Speed: {self.speed_mult:g}x", (200, 200, 200), self.font_sm),
            (f"Particles: {N:,}{bh_str}  {stars_detail_str}{non_stars_str}", (200, 200, 200), self.font_sm),
            (f"Zoom: {zoom_str}   Trail: {trail_label} ({self.decay:.3f})   Frame: {avg_ms:.1f}ms", (200, 200, 200), self.font_sm),
        ]
        if tracking_info:
            lines.append((tracking_info, (0, 240, 255), self.font_sm))
        if paused_hint:
            lines.append((paused_hint, (255, 220, 80), self.font_sm))
        lines.append((f"Energy drift: {energy_str}", (150, 255, 150), self.font_sm))

        y = 8
        for text, color, font in lines:
            surf = font.render(text, True, color)
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            self.screen.blit(bg, (6, y - 2))
            self.screen.blit(surf, (11, y))
            y += surf.get_height() + 4

        # Controls reminder (bottom left)
        controls = [
            "SPACE: pause/resume   Click (when paused): spectate body   U: unselect tracking",
            "+/-: speed (0.06x-512x)   ,/.: trail size   Scroll/[]: zoom   Drag: pan   R: reset",
        ]
        y = self.h - 34
        for c in controls:
            s = self.font_sm.render(c, True, (140, 140, 160))
            bg = pygame.Surface((s.get_width() + 10, s.get_height() + 4), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 140))
            self.screen.blit(bg, (6, y - 2))
            self.screen.blit(s, (11, y))
            y += 18

    def auto_fit(self, pos):
        self.camera.reset(pos)

    def tick(self, fps=60):
        self.clock.tick(fps)

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
    RED_DWARF, ASTEROID, DUST, PIXEL_COLORS
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
    PIXEL_COLORS[ASTEROID],       # 12 — dusty grey
    PIXEL_COLORS[DUST],           # 13 — ethereal gas/dust tracer
    (128,  30, 255),              # 14 — Dark Matter debug violet
], dtype=np.uint8)   # shape (15, 3)

_COLOR_TUPLES = {t: PIXEL_COLORS.get(t, (128, 30, 255)) for t in range(15)}

# Float radii allow for sub-pixel blending (Gaussian soft-dot for tiny particles)
# Index: 0:STAR, 1:PLANET, 2:COMET, 3:ROCKY, 4:GAS_GIANT, 5:BLACK_HOLE, 6:RED_GIANT, 7:BLUE_STRAGGLER, 8:WHITE_DWARF, 9:NEUTRON_STAR, 10:EMISSION_STAR, 11:RED_DWARF, 12:ASTEROID, 13:DUST, 14:DARK_MATTER
_RADIUS_MAX_LUT = np.array([12.0, 3.0, 1.0, 4.0, 6.0, 40.0, 15.0, 20.0, 4.0, 3.0, 15.0, 8.0, 1.0, 0.6, 1.0], dtype=np.float32)
_RADIUS_MIN_LUT = np.array([1.5, 1.0, 0.5, 1.5, 2.0, 3.0, 2.0, 2.5, 1.0, 1.5, 2.0, 1.2, 0.5, 0.3, 0.5], dtype=np.float32)
_BLOOM_INTENSITY = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0], dtype=np.float32)
_BLOOM_SPREAD    = np.array([1.5,  2.0,  2.0,  2.0,  2.0,  0.0, 1.5,  0.8,  1.5,  1.5,  1.5,  1.5,  2.0,  2.0, 0.0], dtype=np.float32)
_MASS_SCALE_LUT  = np.array([100.0, 1.0, 1.0, 1.0, 100.0, 1.0, 30.0, 20.0, 80.0, 80.0, 80.0, 300.0, 1.0, 1.0, 1.0], dtype=np.float32)

def _type_to_rgb(types):
    """Vectorized type → RGB mapping. Returns (N, 3) uint8 array."""
    return _COLOR_LUT[types]


class Camera:
    """Handles world→screen coordinate transform with pan/zoom (Standard Right-Handed Cartesian +Y UP)."""
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.zoom   = 20.0       # pixels per world unit
        self.target_zoom = 20.0
        self.zoom_center = (width / 2, height / 2)
        self.is_zooming = False
        self.offset = np.array([width / 2, height / 2], dtype=np.float64)

    def update_smooth_zoom(self):
        """Called every frame to glide zoom to target_zoom."""
        self.is_zooming = False
        if abs(self.target_zoom - self.zoom) / self.zoom > 0.001:
            diff = self.target_zoom - self.zoom
            step = diff * 0.15  # lerp speed
            factor = (self.zoom + step) / self.zoom
            self.zoom_at(factor, self.zoom_center[0], self.zoom_center[1])
            self.is_zooming = True
        else:
            if self.zoom != self.target_zoom:
                factor = self.target_zoom / self.zoom
                self.zoom_at(factor, self.zoom_center[0], self.zoom_center[1])

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
        """Auto-fit all particles into view, ignoring extreme outliers."""
        if len(pos) == 0:
            return
            
        # Use percentiles to ignore ejected stars that ruin the zoom
        x_min, x_max = np.percentile(pos[:, 0], [2, 98])
        y_min, y_max = np.percentile(pos[:, 1], [2, 98])
        
        # Fallback if the simulation collapsed into a single point
        if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
            x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
            y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
            
        cx = (x_max + x_min) / 2
        cy = (y_max + y_min) / 2
        span = max(x_max - x_min, y_max - y_min) + 1e-6
        
        self.zoom   = min(self.w, self.h) * 0.85 / span
        self.target_zoom = self.zoom
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
        # Reinhard Extended Tonemapping for beautiful HDR
        # Preserves colors much better than simple exponential exposure
        exposure = 0.005
        L_white = 3.0 # Brightness level where everything clips to pure white
        
        # Scale by exposure
        mapped = self.buffer * exposure
        
        # Extended Reinhard
        mapped = mapped * (1.0 + (mapped / (L_white * L_white))) / (1.0 + mapped)
        
        # Convert to 0-255 range and clip
        mapped = np.clip(mapped * 255.0, 0, 255)
        return mapped.astype(np.uint8)


from numba import njit

@njit(parallel=True, fastmath=True)
def _fast_draw_heads(buffer, fx, fy, types, mass, zoom, w, h, colors, max_r_lut, min_r_lut, bloom_int_lut, bloom_spr_lut, mass_scale_lut, show_dm):
    N = len(fx)
    # --- PASS 1: Draw everything EXCEPT Black Holes ---
    for i in prange(N):
        t = types[i]
        
        if t == 14: # DARK_MATTER (always perfectly invisible, only felt via gravity)
            continue
            
        if t == 5:
            continue
            
        x_f = fx[i]
        y_f = fy[i]
        ix = int(math.floor(x_f))
        iy = int(math.floor(y_f))
        
        max_r = max_r_lut[t]
        min_r = min_r_lut[t]
        
        m_val = 1.0
        if mass is not None:
            m_val = float(mass[i])
            
        r_val = min_r
        
        scale_div = mass_scale_lut[t]
        if scale_div > 0.0:
            r_val += (m_val / scale_div)
            
        r_val = min(r_val, max_r)
        
        is_stellar = (t == 0) or (t >= 6 and t <= 11)
        r_draw = r_val * (2.0 if is_stellar else 1.2)
        r_int = int(round(r_draw))
        r_sq_base = float(r_val * r_val)
        
        c_r = float(colors[i, 0])
        c_g = float(colors[i, 1])
        c_b = float(colors[i, 2])
        
        if r_int <= 0:
            if 0 <= ix < w and 0 <= iy < h:
                buffer[ix, iy, 0] += c_r
                buffer[ix, iy, 1] += c_g
                buffer[ix, iy, 2] += c_b
        else:
            r_sq_draw = r_int * r_int
            for dx in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    dist2_px = dx*dx + dy*dy
                    if dist2_px <= r_sq_draw:
                        px = ix + dx
                        py = iy + dy
                        if 0 <= px < w and 0 <= py < h:
                            dist2_norm = dist2_px / r_sq_base if r_sq_base > 0 else 0.0
                            
                            # Steep exponential dropoff creates a sharp, bright core but very little overlapping bloom
                            factor = math.exp(-dist2_norm * 8.0)
                            if bloom_int_lut[t] > 0.0:
                                factor += bloom_int_lut[t] * math.exp(-dist2_norm * bloom_spr_lut[t])
                            
                            factor = min(factor, 1.0)
                            buffer[px, py, 0] += c_r * factor
                            buffer[px, py, 1] += c_g * factor
                            buffer[px, py, 2] += c_b * factor

    # --- PASS 2: Draw ONLY Black Holes (On Top) ---
    telescope_power = max(0.0, min(1.0, (zoom - 5.0) / 15.0))
    void_fade = 1.0 - telescope_power
    bh_zoom_factor = max(0.4, min(1.0, zoom / 15.0))
    
    for i in prange(N):
        t = types[i]
        if t != 5:
            continue
            
        x_f = fx[i]
        y_f = fy[i]
        ix = int(math.floor(x_f))
        iy = int(math.floor(y_f))
        
        max_r = max_r_lut[t]
        min_r = min_r_lut[t]
        
        m_val = 1.0
        if mass is not None:
            m_val = float(mass[i])
            
        # Shrink the apparent size of the black hole slightly when zoomed out
        r_val = min_r + (math.log1p(m_val) * 0.8) * bh_zoom_factor
        r_val = min(r_val, max_r)
        
        r_draw = r_val * 1.5
        r_int = int(round(r_draw))
        r_sq_base = float(r_val * r_val)
        
        if r_int > 0:
            r_sq_draw = r_int * r_int
            for dx in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    dist2_px = dx*dx + dy*dy
                    if dist2_px <= r_sq_draw:
                        px = ix + dx
                        py = iy + dy
                        if 0 <= px < w and 0 <= py < h:
                            dist2_norm = dist2_px / r_sq_base if r_sq_base > 0 else 0.0
                            dist = math.sqrt(dist2_norm)
                            
                            if dist < 0.35:
                                # Event Horizon (Fades based on telescope power)
                                buffer[px, py, 0] = buffer[px, py, 0] * void_fade
                                buffer[px, py, 1] = buffer[px, py, 1] * void_fade
                                buffer[px, py, 2] = buffer[px, py, 2] * void_fade
                            elif dist < 0.5:
                                # Photon Ring (Bright White-Gold)
                                factor = (dist - 0.35) / 0.15
                                intensity = math.sin(factor * math.pi)
                                buffer[px, py, 0] += 255.0 * intensity
                                buffer[px, py, 1] += 200.0 * intensity
                                buffer[px, py, 2] += 150.0 * intensity
                            else:
                                # Outer Accretion Disk (Soft Fading Orange-Gold)
                                factor = 1.0 - ((dist - 0.5) / 1.0) # stretch fade over remaining distance
                                if factor > 0:
                                    factor = factor * factor
                                    buffer[px, py, 0] += 200.0 * factor
                                    buffer[px, py, 1] += 100.0 * factor
                                    buffer[px, py, 2] += 40.0 * factor


@njit(parallel=True, fastmath=True)
def _fast_splat(buffer, fx, fy, prev_fx, prev_fy, types, colors, w, h, intensity, draw_lines):
    """
    1-pixel thick, gap-free, native DDA line splatting for trails.
    Draws perfectly thin and sharp trails.
    """
    N = len(fx)
    for i in prange(N):
        if types[i] == 14: # DARK_MATTER
            continue
            
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
                buffer[ix, iy, 0] += r
                buffer[ix, iy, 1] += g
                buffer[ix, iy, 2] += b


# Pixel radii per particle type (Max and Min bounds for zoom scaling).
# Guaranteed size hierarchy at ALL zoom levels (zoomed out or zoomed in):
#   comet (1px) <= planet/asteroid (1px) < rocky planet (1-2px) < gas giant (2-4px) < star (3-6px) < BH (6-12px)
_TYPE_MAX_RADIUS = {
    BLACK_HOLE:      6,  # Crimson Red core with photon ring & event horizon
    RED_GIANT:       4,  # Deep Ruby Red expanded giant envelope
    BLUE_STRAGGLER:  4,  # Luminous Ice Blue star
    EMISSION_STAR:   4,  # Aquatic Emerald Teal emission star
    STAR:            3,  # Warm Cream Gold main sequence star
    RED_DWARF:       2,  # Deep Crimson Red M-type dwarf
    GAS_GIANT:       2,  # Electric Royal Violet gas giant (Jupiter/Saturn scale)
    WHITE_DWARF:     2,  # Diamond Pearl White compact remnant
    NEUTRON_STAR:    2,  # Soft Violet-Indigo pulsar
    ROCKY:           1,  # Terracotta Rust terrestrial rocky planet (Earth/Mars scale)
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
                 title="Laplace's Demon — Universe Sandbox", preset_name=None):
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

        # Determine initial trail preset index based on requested decay
        self.trail_index = 4 # Default to standard
        for i, preset in enumerate(TRAIL_PRESETS):
            if abs(preset[0] - trail_decay) < 1e-4:
                self.trail_index = i
                break
                
        self.decay = trail_decay

        # Interaction & Spectator Tracking state
        self.paused           = False
        self.speed_mult       = 1.0
        self.tracked_particle = None   # int particle index or None
        self._drag            = False
        self.show_dark_matter = False

        self.bloom_int = _BLOOM_INTENSITY.copy()
        self.bloom_spr = _BLOOM_SPREAD.copy()
        self.mass_scale = _MASS_SCALE_LUT.copy()
        self.max_r = _RADIUS_MAX_LUT.copy()
        self.min_r = _RADIUS_MIN_LUT.copy()
        self._apply_preset_visuals(preset_name)

    def _apply_preset_visuals(self, preset):
        if not preset:
            return
        if preset == "pleiades_m45":
            self.bloom_int[7] = 0.35 # Extreme blue stragglers
            self.bloom_spr[7] = 0.5
            self.bloom_int[0] = 0.15 # Brighter G-stars
            self.bloom_int[11] = 0.05
            
            # Since total mass is small, individual m_val is ~0.4
            self.mass_scale[7] = 0.05 # 0.4 / 0.05 = 8 pixels bonus
            self.mass_scale[0] = 0.30  # 0.4 / 0.30 = 1.3 pixels bonus
            self.mass_scale[11] = 0.80 # 0.4 / 0.80 = 0.5 pixels bonus
            
            self.min_r[0] = 2.0  # Force minimum base size for G-stars
            self.min_r[11] = 1.0 # Force minimum base size for Red Dwarfs
            self.max_r[0] = 6.0 # Cap G-stars so they don't blow up
            
        elif preset == "trappist_1":
            self.min_r[11] = 6.0 # Force Red Dwarf to be large
            self.min_r[3]  = 2.5 # Force Rocky planets to be clearly visible
            self.bloom_int[11] = 0.25 # Give Red Dwarf some glow

        elif preset == "omega_centauri":
            self.bloom_int[8] = 0.20 
            self.bloom_int[7] = 0.25
            
            # Total mass is huge, individual m_val is ~40
            self.mass_scale[7] = 5.0
            self.mass_scale[0] = 15.0
            self.mass_scale[11] = 40.0
            self.mass_scale[8] = 10.0 # White dwarfs
            
        elif preset == "milkomeda":
            self.bloom_int[0] = 0.10
        elif preset == "castor_sextuple":
            self.bloom_int[0] = 0.30
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
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_mult = min(self.speed_mult * 2.0, 16.0)
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
                elif event.key == pygame.K_d:
                    self.show_dark_matter = not self.show_dark_matter
                elif event.key == pygame.K_t:
                    self.trail_buffer[:] = 0
                elif event.key == pygame.K_RIGHTBRACKET:   # ] → zoom in
                    self.camera.target_zoom = np.clip(self.camera.target_zoom * 1.30, 0.0001, 100000.0)
                    self.camera.zoom_center = (self.w // 2, self.h // 2)
                elif event.key == pygame.K_LEFTBRACKET:    # [ → zoom out
                    self.camera.target_zoom = np.clip(self.camera.target_zoom / 1.30, 0.0001, 100000.0)
                    self.camera.zoom_center = (self.w // 2, self.h // 2)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                # If spectating, lock zoom center at screen center
                zx, zy = (self.w // 2, self.h // 2) if self.tracked_particle is not None else (mx, my)
                factor = 1.15 if event.y > 0 else 1 / 1.15
                self.camera.target_zoom = np.clip(self.camera.target_zoom * factor, 0.0001, 100000.0)
                self.camera.zoom_center = (zx, zy)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._drag = True
                    self._drag_last = event.pos
                    self._drag_start_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._drag = False
                    # Feature: When simulation is PAUSED, clicking (without dragging) selects a celestial entity to spectate!
                    if hasattr(self, '_drag_start_pos'):
                        dx = event.pos[0] - self._drag_start_pos[0]
                        dy = event.pos[1] - self._drag_start_pos[1]
                        if self.paused and len(pos) > 0 and (dx*dx + dy*dy) < 25.0:
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

        # Smooth zoom transition
        self.camera.update_smooth_zoom()
        if self.camera.is_zooming:
            self.trail_buffer[:] = 0

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
        _fast_splat(self.trail_buffer, fx, fy, fx_prev, fy_prev, types, colors, self.w, self.h, splat_intensity, draw_lines)

        # ── Pass 2: Fast Numba Additive Heads ────────────────────────
        # Copy trail buffer to avoid leaving permanent blobs
        display_buffer = self.trail_buffer.copy()
        
        # Optionally pass a dummy array for mass if None to satisfy numba typing
        mass_array = mass if mass is not None else np.zeros(0, dtype=np.float64)
        _fast_draw_heads(display_buffer, fx, fy, types, mass_array if mass is not None else None, zoom, self.w, self.h, colors, self.max_r, self.min_r, self.bloom_int, self.bloom_spr, self.mass_scale, self.show_dark_matter)

        # Draw Spectator Target Reticle
        if self.tracked_particle is not None and 0 <= self.tracked_particle < len(types):
            tx, ty = int(round(sx[self.tracked_particle])), int(round(sy[self.tracked_particle]))
            ttype = int(types[self.tracked_particle])
            t_max_r = _RADIUS_MAX_LUT[ttype]
            tr = max(4, min(t_max_r, int(t_max_r * zoom / 20.0)))
            pulse = int(math.sin(time.perf_counter() * 8.0) * 3.0)
            r_reticle = max(8, tr + 8 + pulse)
            
            # Simple bounds check for reticle drawing on numpy buffer
            for angle in np.linspace(0, 2*np.pi, 30):
                px = int(tx + r_reticle * math.cos(angle))
                py = int(ty + r_reticle * math.sin(angle))
                if 0 <= px < self.w and 0 <= py < self.h:
                    display_buffer[px, py] = [0, 240, 255]

        # Push display buffer to Pygame surface
        surf_array = np.clip(display_buffer, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(surf_array)
        self.screen.blit(surf, (0, 0))

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
        n_asteroids = int(np.sum(types == ASTEROID))

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
        if n_asteroids > 0: non_stars_parts.append(f"Asteroid:{n_asteroids:,}")
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

        universe_time = step * 0.001
        time_per_sec = spf * self.speed_mult * 0.001 * self.clock.get_fps()
        
        dm_str = "  (DM Vis: ON)" if self.show_dark_matter else ""
        lines = [
            ("LAPLACE'S DEMON — UNIVERSE SANDBOX", (180, 120, 255), self.font_lg),
            (f"Step: {step:,}   SPF: {spf}   FPS: {self.clock.get_fps():.0f}   Speed: {self.speed_mult:g}x{dm_str}", (200, 200, 200), self.font_sm),
            (f"Univ Time: {universe_time:.2f}   Time/sec: {time_per_sec:.3f}/s", (180, 220, 255), self.font_sm),
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
            "+/-: speed (0.06x-16x)   ,/.: trail size   Scroll/[]: zoom   Drag: pan   R: reset",
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

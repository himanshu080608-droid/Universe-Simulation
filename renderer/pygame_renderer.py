"""
Pygame-based real-time renderer for the universe simulation.

Features:
  - GPU-accelerated pixel blitting via pygame.surfarray
  - Color-coded particle types (star=gold, planet=blue, comet=cyan)
  - Particle trails using additive blending (motion blur)
  - Mouse drag pan, scroll wheel zoom
  - Keyboard controls: SPACE=pause, +/-=speed, R=reset view,
                        1/2/3/4=switch presets (requires restart)
  - HUD overlay: frame time, particle count, zoom, energy drift
"""

import sys
import time
import math
import numpy as np
import pygame
import pygame.surfarray as surfarray

from universe.generator import STAR, PLANET, COMET, PIXEL_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Color table: type → (R, G, B)  pre-loaded as a lookup array for speed
# ─────────────────────────────────────────────────────────────────────────────
_COLOR_LUT = np.array([
    PIXEL_COLORS[STAR],
    PIXEL_COLORS[PLANET],
    PIXEL_COLORS[COMET],
], dtype=np.uint8)   # shape (3, 3)


def _type_to_rgb(types):
    """Vectorized type → RGB mapping. Returns (N, 3) uint8 array."""
    return _COLOR_LUT[types]


class Camera:
    """Handles world→screen coordinate transform with pan/zoom."""
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.zoom   = 20.0       # pixels per world unit
        self.offset = np.array([width / 2, height / 2], dtype=np.float64)

    def world_to_screen(self, pos):
        """pos: (N,2) world coords → (N,2) screen pixel coords (float)."""
        return pos * self.zoom + self.offset

    def screen_to_world(self, px, py):
        return ((np.array([px, py]) - self.offset) / self.zoom)

    def zoom_at(self, factor, screen_x, screen_y):
        """Zoom centered on a screen point."""
        world_pt = self.screen_to_world(screen_x, screen_y)
        self.zoom  *= factor
        self.zoom   = np.clip(self.zoom, 1.0, 5000.0)
        # Keep world_pt under cursor
        self.offset = np.array([screen_x, screen_y]) - world_pt * self.zoom

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
                                 self.h / 2 - cy * self.zoom], dtype=np.float64)


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

@njit(cache=True, fastmath=True)
def _fast_splat(buffer, sx, sy, colors, w, h):
    """
    JIT-compiled particle splat — native C speed.
    """
    N = len(sx)
    for i in range(N):
        x = sx[i]
        y = sy[i]
        if 0 <= x < w and 0 <= y < h:
            buffer[x, y, 0] += colors[i, 0]
            buffer[x, y, 1] += colors[i, 1]
            buffer[x, y, 2] += colors[i, 2]


# Pixel radii per particle type — scale purely with zoom, no fixed minimum.
# At low zoom (galaxy view) stars render as 1-2px; at high zoom they grow.
_TYPE_RADIUS = {
    STAR:   3,   # max 3px — scales down with zoom
    PLANET: 2,   # max 2px
    COMET:  0,   # always single pixel via fast splat
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
        self.decay    = trail_decay

        # Interaction state
        self.paused     = False
        self.speed_mult = 1
        self._drag      = False
        self._drag_last = (0, 0)

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
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
                    self.speed_mult = min(self.speed_mult * 2, 64)
                elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                    self.speed_mult = max(self.speed_mult // 2, 1)
                elif event.key == pygame.K_r:
                    self.camera.reset(pos)
                elif event.key == pygame.K_t:
                    self.trail_buffer[:] = 0
                elif event.key == pygame.K_RIGHTBRACKET:   # ] → zoom in
                    self.camera.zoom_at(1.20, self.w // 2, self.h // 2)
                elif event.key == pygame.K_LEFTBRACKET:    # [ → zoom out
                    self.camera.zoom_at(1 / 1.20, self.w // 2, self.h // 2)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                factor = 1.12 if event.y > 0 else 1 / 1.12
                self.camera.zoom_at(factor, mx, my)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._drag = True
                    self._drag_last = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._drag = False
            elif event.type == pygame.MOUSEMOTION:
                if self._drag:
                    dx = event.pos[0] - self._drag_last[0]
                    dy = event.pos[1] - self._drag_last[1]
                    self.camera.pan(dx, dy)
                    self._drag_last = event.pos
            elif event.type == pygame.VIDEORESIZE:
                self.w, self.h = event.w, event.h
                self.screen = pygame.display.set_mode(
                    (self.w, self.h), pygame.RESIZABLE | pygame.DOUBLEBUF)
                self.trail_buffer = np.zeros((self.w, self.h, 3), dtype=np.float32)
                self.camera.w = self.w
                self.camera.h = self.h
        return True

    def render_frame(self, pos, types, energy=None, step=0, steps_per_frame=1):
        """
        Render one display frame.
        pos:   (N,2) world positions
        types: (N,)  int32 particle type
        """
        t0 = time.perf_counter()

        # World → screen
        sp = self.camera.world_to_screen(pos)
        sx = sp[:, 0].astype(np.int32)
        sy = sp[:, 1].astype(np.int32)

        # Colors
        colors = _type_to_rgb(types)

        # Trail buffer: fade + splat
        self.trail_buffer *= self.decay

        # ── Pass 1: Fast splat ALL particles onto trail buffer (vectorized) ─────
        _fast_splat(self.trail_buffer, sx, sy, colors, self.w, self.h)

        # ── Pass 2: stars & planets → small filled circles ──────────────────
        # Draw onto a temp surface, then blend into the trail buffer.
        # Only runs for non-comet particles — negligible cost at typical N.
        comet_mask = (types == COMET)
        big_mask = ~comet_mask
        if np.any(big_mask):
            circle_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            circle_surf.fill((0, 0, 0, 0))
            idxs = np.where(big_mask)[0]
            zoom = self.camera.zoom

            for i in idxs:
                t = int(types[i])
                base_r = _TYPE_RADIUS.get(t, 1)
                # Scale with zoom: r grows as user zooms in, stays <=base_r at zoom<=20.
                r = max(1, min(int(base_r * zoom / 20.0), base_r))
                x, y = int(sx[i]), int(sy[i])
                if 0 <= x < self.w and 0 <= y < self.h:
                    col = tuple(int(c) for c in colors[i])
                    pygame.draw.circle(circle_surf, col + (220,), (x, y), r)
            self.screen.blit(circle_surf, (0, 0))


        # Push buffer to Pygame surface
        surf_array = np.clip(self.trail_buffer, 0, 255).astype(np.uint8)
        # surfarray expects (w, h, 3)
        surf = pygame.surfarray.make_surface(surf_array)
        self.screen.blit(surf, (0, 0))

        # ── HUD overlay ──────────────────────────────────────────────
        self._draw_hud(pos, types, energy, step, steps_per_frame, t0)

        pygame.display.flip()
        self.total_frames += 1

    def _draw_hud(self, pos, types, energy, step, spf, t0):
        dt_ms = (time.perf_counter() - t0) * 1000
        self.frame_times.append(dt_ms)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        avg_ms = sum(self.frame_times) / len(self.frame_times)

        N = len(pos)
        n_stars   = int(np.sum(types == STAR))
        n_planets = int(np.sum(types == PLANET))
        n_comets  = int(np.sum(types == COMET))

        # Energy drift
        energy_str = "N/A"
        if energy is not None:
            if self.energy_ref is None:
                self.energy_ref = energy
            drift = abs((energy - self.energy_ref) / (abs(self.energy_ref) + 1e-30)) * 100
            energy_str = f"{drift:.4f}%"

        zoom_str = f"{self.camera.zoom:.1f}x"
        speed_str = f"×{self.speed_per_frame}" if hasattr(self, 'speed_per_frame') else ""

        lines = [
            ("LAPLACE'S DEMON — UNIVERSE SANDBOX", (180, 120, 255), self.font_lg),
            (f"Step: {step:,}   SPF: {spf}   FPS: {self.clock.get_fps():.0f}", (200, 200, 200), self.font_sm),
            (f"Particles: {N:,}  ⭐ {n_stars:,}  🪐 {n_planets:,}  ☄ {n_comets:,}", (200, 200, 200), self.font_sm),
            (f"Zoom: {zoom_str}  Frame: {avg_ms:.1f}ms", (200, 200, 200), self.font_sm),
            (f"Energy drift: {energy_str}", (150, 255, 150), self.font_sm),
        ]

        y = 8
        for text, color, font in lines:
            surf = font.render(text, True, color)
            # Dark background pill
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 160))
            self.screen.blit(bg, (6, y - 2))
            self.screen.blit(surf, (11, y))
            y += surf.get_height() + 4

        # Controls reminder (bottom left)
        controls = [
            "SPACE: pause   +/-: speed   R: reset view   T: clear trails",
            "[/]: zoom out/in   Scroll: zoom   Drag: pan   ESC: quit",
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

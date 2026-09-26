"""
Matplotlib fallback renderer.
Precomputes a trajectory history then plays it back as an animation.
Good for: exporting videos, systems that can't run Pygame.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from universe.generator import (
    STAR, PLANET, COMET, ROCKY, GAS_GIANT, BLACK_HOLE,
    RED_GIANT, BLUE_STRAGGLER, WHITE_DWARF, NEUTRON_STAR, EMISSION_STAR,
    PIXEL_COLORS
)


def _rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


_MPL_COLORS = {t: _rgb_to_hex(*PIXEL_COLORS[t]) for t in range(11)}

_TYPE_SIZES = {
    BLACK_HOLE:     50,
    RED_GIANT:      25,
    BLUE_STRAGGLER: 20,
    EMISSION_STAR:  20,
    STAR:           18,
    GAS_GIANT:      15,
    ROCKY:           8,
    WHITE_DWARF:    10,
    NEUTRON_STAR:   10,
    PLANET:          6,
    COMET:           2,
}


class MatplotlibRenderer:
    """
    Offline renderer: runs physics first, then animates.
    Usage:
        renderer = MatplotlibRenderer(history, types, masses)
        renderer.show()          # interactive
        renderer.save("out.mp4") # export video
    """
    def __init__(self, history, types, masses,
                 stride=10, tail_length=50, fps=30):
        """
        history: (steps, N, 2) float64 array of positions
        types:   (N,)   int32 particle types
        masses:  (N,)   float64 masses
        """
        self.hist   = history[::stride]
        self.types  = types
        self.masses = masses
        self.tail   = tail_length
        self.fps    = fps
        self.n_frames, self.N, _ = self.hist.shape

        self.fig, self.ax = plt.subplots(figsize=(14, 9), facecolor='#050510')
        self._setup_canvas()

    def _setup_canvas(self):
        ax = self.ax
        ax.set_facecolor('#050510')
        ax.tick_params(colors='#444466')
        for spine in ax.spines.values():
            spine.set_edgecolor('#222244')
        ax.set_title("Laplace's Demon — Universe Sandbox", color='#9988ff',
                     fontsize=16, pad=12, fontfamily='monospace')
        ax.set_xlabel("x [AU]", color='#666688', fontsize=10)
        ax.set_ylabel("y [AU]", color='#666688', fontsize=10)

        # Compute bounds from full history
        all_pos = self.hist.reshape(-1, 2)
        xmid = (all_pos[:, 0].max() + all_pos[:, 0].min()) / 2
        ymid = (all_pos[:, 1].max() + all_pos[:, 1].min()) / 2
        span = max(all_pos[:, 0].max() - all_pos[:, 0].min(), all_pos[:, 1].max() - all_pos[:, 1].min()) * 0.6
        ax.set_xlim(xmid - span, xmid + span)
        ax.set_ylim(ymid - span, ymid + span)

        # Create scatter plots per type
        self.scatters = {}
        type_vals = [STAR, PLANET, COMET]
        type_labels = {STAR: "Stars", PLANET: "Planets", COMET: "Comets"}
        for t in type_vals:
            mask = self.types == t
            if not np.any(mask):
                continue
            sc = ax.scatter(
                self.hist[0, mask, 0], self.hist[0, mask, 1],
                s=_TYPE_SIZES[t], c=_MPL_COLORS[t],
                alpha=0.85, lw=0, label=type_labels[t], zorder=5
            )
            self.scatters[t] = (sc, mask)

        # Trail lines (sampled subset to avoid millions of lines)
        max_trail_particles = min(self.N, 400)
        trail_idx = np.random.choice(self.N, max_trail_particles, replace=False)
        self.trail_lines = []
        for i in trail_idx:
            t = self.types[i]
            ln, = ax.plot([], [], '-', lw=0.5, color=_MPL_COLORS[t], alpha=0.25, zorder=2)
            self.trail_lines.append((ln, i))

        ax.legend(loc='upper right', facecolor='#0a0a20', labelcolor='white',
                  fontsize=10, framealpha=0.7)

        # Info text
        self.info_text = ax.text(
            0.01, 0.98, '', transform=ax.transAxes,
            color='#aaaacc', fontsize=9, va='top', fontfamily='monospace'
        )

    def _update(self, frame):
        pos = self.hist[frame]
        tail_start = max(0, frame - self.tail)
        tail_slice = self.hist[tail_start:frame]

        for t, (sc, mask) in self.scatters.items():
            sc.set_offsets(pos[mask])

        for ln, i in self.trail_lines:
            if len(tail_slice) > 0:
                ln.set_data(tail_slice[:, i, 0], tail_slice[:, i, 1])

        self.info_text.set_text(
            f"Frame {frame}/{self.n_frames}   N={self.N:,}\n"
            f"Stars={np.sum(self.types==STAR):,}  "
            f"Planets={np.sum(self.types==PLANET):,}  "
            f"Comets={np.sum(self.types==COMET):,}"
        )
        return [sc for sc, _ in self.scatters.values()] + \
               [ln for ln, _ in self.trail_lines] + [self.info_text]

    def _init(self):
        return []

    def show(self):
        self.ani = FuncAnimation(
            self.fig, self._update, init_func=self._init,
            frames=self.n_frames, interval=1000//self.fps, blit=True
        )
        plt.tight_layout()
        plt.show()

    def save(self, filepath="universe_simulation.mp4", dpi=150):
        print(f"[Renderer] Saving to {filepath} ...")
        writer = FFMpegWriter(fps=self.fps, bitrate=3000)
        self.ani = FuncAnimation(
            self.fig, self._update, init_func=self._init,
            frames=self.n_frames, blit=True
        )
        self.ani.save(filepath, writer=writer, dpi=dpi)
        print(f"[Renderer] Saved → {filepath}")

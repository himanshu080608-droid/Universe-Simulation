#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LAPLACE'S DEMON — UNIVERSE SANDBOX SIMULATION                     ║
║          N-Body Gravity · Barnes-Hut O(N log N) · Leapfrog KDK             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Laplace's Demon is a thought experiment by Pierre-Simon Laplace (1814):
  "An intellect which at a certain moment would know all forces that set
   nature in motion, and all positions of all items of which nature is
   composed, if this intellect were also vast enough to submit these data
   to analysis, it would embrace in a single formula the movements of the
   greatest bodies of the universe and those of the tiniest atom."

This simulation IS that intellect — a deterministic, closed-form numerical
integration of N gravitating particles from t=0 to t=∞.

Usage:
    python main.py                        # interactive Pygame (twin galaxies)
    python main.py --preset solar_system  # solar system preset
    python main.py --preset chaos         # 5 colliding clusters
    python main.py --preset laplace       # ordered Laplacian rings
    python main.py --N 2000               # custom particle count
    python main.py --mode mpl             # matplotlib offline renderer
    python main.py --mode mpl --save out.mp4  # export video
    python main.py --no-bh                # disable Barnes-Hut (small N only)
"""

import argparse
import sys
import os
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Laplace's Demon — N-Body Universe Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--preset", choices=["twin_galaxies", "solar_system",
                                         "chaos", "laplace"],
                   default="twin_galaxies",
                   help="Initial condition preset (default: twin_galaxies)")
    p.add_argument("--N", type=int, default=5000,
                   help="Target particle count (default: 5000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--mode", choices=["pygame", "mpl"], default="pygame",
                   help="Renderer: 'pygame' = live, 'mpl' = offline (default: pygame)")
    p.add_argument("--dt", type=float, default=0.001,
                   help="Leapfrog time step (default: 0.001)")
    p.add_argument("--eps", type=float, default=1.0,
                   help="Gravitational softening (default: 1.0)")
    p.add_argument("--theta", type=float, default=0.6,
                   help="Barnes-Hut opening angle (default: 0.6)")
    p.add_argument("--no-bh", action="store_true",
                   help="Disable Barnes-Hut (O(N²) direct sum — small N only)")
    p.add_argument("--spf", type=int, default=2,
                   help="Physics steps per display frame (default: 2)")
    p.add_argument("--fps", type=int, default=60,
                   help="Target display FPS (default: 60)")
    p.add_argument("--trail-decay", type=float, default=0.90,
                   help="Trail fade factor per frame, 0=off 1=infinite (default: 0.90)")
    # Matplotlib/offline options
    p.add_argument("--mpl-steps", type=int, default=20000,
                   help="Total physics steps for mpl precompute (default: 20000)")
    p.add_argument("--mpl-stride", type=int, default=5,
                   help="Record every N steps for mpl (default: 5)")
    p.add_argument("--save", type=str, default=None,
                   help="If set, save mpl animation to this file (e.g. out.mp4)")
    p.add_argument("--width",  type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Pygame real-time loop
# ──────────────────────────────────────────────────────────────────────────────
def run_pygame(engine, args):
    from renderer.pygame_renderer import PygameRenderer

    renderer = PygameRenderer(
        width=args.width, height=args.height,
        trail_decay=args.trail_decay
    )
    if args.preset == "twin_galaxies":
        # Ignore comets (type 2) for auto-fit so the initial zoom is tight 
        # around the spectacular stellar disks, making them look large.
        mask = engine.types != 2
        renderer.auto_fit(engine.pos[mask])
    else:
        renderer.auto_fit(engine.pos)
    spf = args.spf
    print(f"[Main] Starting Pygame loop. SPF={spf}  FPS target={args.fps}")
    print( "[Main] Controls: SPACE=pause  +/-=speed  [/]=zoom  R=reset view  T=clear trails  ESC=quit")


    running = True
    while running:
        if not renderer._handle_events(engine.pos):
            break

        if not renderer.paused:
            engine.step(spf * renderer.speed_mult)


        renderer.render_frame(
            engine.pos, engine.types,
            energy=engine.energy,
            step=engine.step_count,
            steps_per_frame=spf
        )
        renderer.tick(args.fps)

    import pygame
    pygame.quit()
    print("[Main] Simulation ended.")


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib offline loop
# ──────────────────────────────────────────────────────────────────────────────
def run_mpl(engine, args):
    from renderer.mpl_renderer import MatplotlibRenderer

    print(f"[Main] Precomputing {args.mpl_steps:,} steps "
          f"(recording every {args.mpl_stride})...")
    history = engine.precompute(args.mpl_steps, record_every=args.mpl_stride)

    renderer = MatplotlibRenderer(
        history, engine.types, engine.mass,
        stride=1, tail_length=60, fps=30
    )
    if args.save:
        renderer.save(args.save)
    else:
        renderer.show()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   LAPLACE'S DEMON — Universe Sandbox Simulation          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Preset  : {args.preset}")
    print(f"  Particles: {args.N:,}")
    print(f"  Renderer: {args.mode}")
    print(f"  Seed    : {args.seed}")
    print()

    # ── 1. Generate universe ──────────────────────────────────────────────────
    print("[Main] Generating particle initial conditions...", flush=True)
    from universe.generator import build_universe
    pos, vel, mass, types = build_universe(
        preset=args.preset, N_total=args.N, seed=args.seed
    )
    N_actual = len(pos)
    print(f"[Main] Generated {N_actual:,} particles.", flush=True)

    # ── 2. Build engine ───────────────────────────────────────────────────────
    from simulation_engine import SimulationEngine
    use_bh = not args.no_bh
    engine = SimulationEngine(
        pos, vel, mass, types,
        dt=args.dt, eps=args.eps, G=1.0,
        theta=args.theta, use_bh=use_bh
    )

    # ── 3. JIT warmup ─────────────────────────────────────────────────────────
    engine.warmup()

    # ── 4. Launch renderer ────────────────────────────────────────────────────
    if args.mode == "pygame":
        run_pygame(engine, args)
    else:
        run_mpl(engine, args)


if __name__ == "__main__":
    main()

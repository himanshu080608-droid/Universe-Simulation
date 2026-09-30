#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          LAPLACE'S DEMON — UNIVERSE SANDBOX SIMULATION                     ║
║          N-Body Gravity · Taichi GPU Hermite · Ahmad-Cohen Block Steps     ║
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
    python3 main.py                             # default live Pygame mode (AdvancedTaichiEngine, twin_galaxies)
    python3 main.py --preset twin_galaxies      # two colliding spiral galaxies with dark matter halos
    python3 main.py --preset solar_system       # Sun + 8 major planets + asteroid & Oort belt
    python3 main.py --preset alpha_centauri     # Alpha Centauri A/B binary + Proxima & exoplanets
    python3 main.py --preset milkomeda          # Andromeda + Milky Way + M33 triple galactic merger
    python3 main.py --preset stephans_quintet   # Stephan's Quintet (Compact Galaxy Group)
    python3 main.py --preset messier13          # M13 100% stellar cluster (Red Giants, Pulsars, Blue Stragglers)
    python3 main.py --preset messier31          # M31 Andromeda standalone spiral galaxy
    python3 main.py --preset chaos              # 5 galaxy clusters on collision course
    python3 main.py --preset laplace            # ordered Laplacian concentric rings
    python3 main.py --preset ngc1052_df2        # Dark-Matter-Free Ultra-Diffuse Galaxy
    python3 main.py --preset castor_sextuple    # 6-star Hierarchical Resonance System
    python3 main.py --preset hd98800_polar      # Quadruple Star with Polar Protoplanetary Disk
    python3 main.py --preset trappist_1         # 7-Planet Resonant Laplace Chain around Red Dwarf
    python3 main.py --preset gravothermal_catastrophe # Core Collapse of a Uniform Cluster
    python3 main.py --preset wr104_pinwheel     # The Pinwheel Nebula (Colliding Wind Binary)
    python3 main.py --preset omega_centauri     # Core-Collapsed Globular with IMBH
    python3 main.py --preset pleiades_m45       # Pleiades Open Cluster (The Seven Sisters)
    python3 main.py --preset hirayama_family    # Asteroid Disruption & Keplerian Shear
    python3 main.py --preset dark_matter_halo_merger # Dark Matter Halo Merger (Violent Relaxation)
    python3 main.py --preset great_attractor    # The Great Attractor (Cosmic Filaments)
    
    python3 main.py --seed random               # use a random layout seed (default is 42)
    python3 main.py --N 8000                    # custom particle count
    python3 main.py --mode mpl                  # matplotlib offline renderer (legacy engine only)
    python3 main.py --mode mpl --save out.mp4   # export video
    python3 main.py --advanced                  # Use AdvancedTaichiEngine (default)
    python3 main.py --taichi                    # Use fixed-step TaichiEngine
    python3 main.py --legacy-leapfrog           # Use older Barnes-Hut SimulationEngine
    
    # Media Generation (saves to output/ directory)
    python3 record_screenshots.py               # generate high-res zoom PNGs
    python3 record_pygame.py                    # generate animated GIFs

Controls (Pygame Mode):
    Left Drag          : Pan camera viewport freely
    Left Click (Paused): Spectate entity (locks camera onto star, planet, or black hole)
    Scroll Wheel       : Smooth exponential zoom centered at cursor
    U / ESC            : Unselect target spectating / return to free camera
    SPACE              : Pause / resume simulation
    + / -              : Acceleration speed multiplier (0.0625x to 512x)
    , / . (or K / L)   : Adjust particle motion trail persistence
    T                  : Clear trail buffer
    R                  : Reset camera view & center galaxy
"""

import argparse
import sys
import os
import numpy as np
from universe.registry import PresetRegistry
from universe.production_presets import register_all

# Populate registry before argparse uses it
register_all()

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Laplace's Demon — N-Body Universe Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--preset", choices=PresetRegistry.list_presets(),
                   default="twin_galaxies",
                   help="Initial condition preset layout (default: twin_galaxies)")
    p.add_argument("--N", type=int, default=5000,
                   help="Target particle count (default: 5000)")
    p.add_argument("--seed", type=str, default="42",
                   help="Random seed for reproducibility, or 'random' for a new seed every run (default: 42)")
    p.add_argument("--mode", choices=["pygame", "mpl"], default="pygame",
                   help="Renderer: 'pygame' = live, 'mpl' = offline (default: pygame)")
    p.add_argument("--dt", type=float, default=0.001,
                   help="Integration time step (default: 0.001 for Hermite)")
    p.add_argument("--eps", type=float, default=1.0,
                   help="Gravitational softening (default: 1.0)")
    p.add_argument("--theta", type=float, default=0.8,
                   help="Barnes-Hut opening angle (default: 0.8)")
    p.add_argument("--legacy-leapfrog", action="store_true",
                   help="Use the older 2nd-Order Leapfrog Integrator instead of AdvancedTaichiEngine")
    p.add_argument("--taichi", action="store_true",
                   help="Use Taichi GPU Hermite engine (Metal/CUDA)")
    p.add_argument("--advanced", action="store_true",
                   help="Use the Ultimate Engine (Taichi GPU + Ahmad-Cohen Block Time-Steps) [also the default]")
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
    p.add_argument("--width",  type=int, default=1600,
                   help="Display width (default: 1600)")
    p.add_argument("--height", type=int, default=900,
                   help="Display height (default: 900)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Pygame real-time loop
# ──────────────────────────────────────────────────────────────────────────────
def run_pygame(engine, args):
    from renderer.pygame_renderer import PygameRenderer

    preset_def = PresetRegistry.get(args.preset)
    semantics = preset_def.get_render_semantics()

    renderer = PygameRenderer(
        width=args.width, height=args.height,
        trail_decay=args.trail_decay,
        preset_name=args.preset,
        render_semantics=semantics
    )
    spf = args.spf

    # Smart initial zoom: fit only the inner stellar disk (non-comet, within
    # the 80th-percentile radius from the centroid) so the view starts
    # comfortably zoomed in — not a tiny dot, not an overwhelming wall of stars.
    # Comets (type 2) are ignored because their outer halo would zoom the camera
    # Exclude designated types from auto-fit (e.g. comets, dark matter)
    exclude_mask = np.isin(engine.types, semantics.get('auto_fit_exclude_types', []))
    _mask = ~exclude_mask
    _fit_pos = engine.pos[_mask] if np.any(_mask) else engine.pos
    if len(_fit_pos) > 0:
        _cx = np.median(_fit_pos[:, 0])
        _cy = np.median(_fit_pos[:, 1])
        _r  = np.linalg.norm(_fit_pos - np.array([_cx, _cy]), axis=1)
        # Use 80th-percentile radius so outlier stars don't force a tiny zoom
        _r80  = np.percentile(_r, 80)
        _span = max(_r80 * 2.0, 1.0)   # diameter of the inner disk
        _zoom = min(args.width, args.height) * 0.50 / _span
        renderer.camera.zoom   = _zoom
        renderer.camera.target_zoom = _zoom
        renderer.camera.offset = np.array([
            args.width  / 2 - _cx * _zoom,
            args.height / 2 + _cy * _zoom,
        ], dtype=np.float64)

    print("[Main] Controls: Left-Drag=pan  Click(paused)=spectate entity  Scroll=smooth zoom  U/ESC=unselect  SPACE=pause  +/-=speed  T=clear trails  R=reset")


    running = True
    while running:
        if not renderer._handle_events(engine.pos):
            break

        if not renderer.paused:
            engine.step(spf, speed_mult=renderer.speed_mult)


        renderer.render_frame(
            engine.pos, engine.types, mass=engine.mass,
            energy=engine.energy,
            step=engine.step_count,
            steps_per_frame=spf
        )
        renderer.tick(args.fps)

    import pygame
    pygame.quit()
    print("[Main] Simulation ended.")
    sys.exit(0)


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
    import random
    if args.seed.lower() == 'random':
        args.seed = random.randint(0, 999999)
    else:
        args.seed = int(args.seed)

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
    preset_def = PresetRegistry.get(args.preset)
    G_sim = preset_def.unit_system.G
    eps_sim = args.eps if args.eps != 1.0 else preset_def.unit_system.eps

    if args.legacy_leapfrog:
        from simulation_engine import SimulationEngine
        engine = SimulationEngine(
            pos, vel, mass, types,
            dt=args.dt, eps=eps_sim, G=G_sim,
            theta=args.theta, use_bh=True,
            preset=args.preset
        )

        # ── 3. JIT warmup ─────────────────────────────────────────────────────────
        engine.warmup()
    elif args.taichi:
        from taichi_engine import TaichiEngine
        engine = TaichiEngine(
            pos, vel, mass, types,
            dt=args.dt, eps=eps_sim, G=G_sim
        )
        # Taichi compiles kernels on first call (already done in __init__)
        print("[Taichi Engine] GPU initialization complete.", flush=True)
    else:
        # Default: AdvancedTaichiEngine (selected explicitly via --advanced or by default)
        from advanced_engine import AdvancedTaichiEngine
        engine = AdvancedTaichiEngine(
            pos, vel, mass, types,
            dt=args.dt, eps=eps_sim, G=G_sim
        )
        print("[Advanced Engine] GPU Block Time-Step initialization complete.", flush=True)

    # ── 4. Launch renderer ────────────────────────────────────────────────────
    if args.mode == "pygame":
        run_pygame(engine, args)
    else:
        run_mpl(engine, args)


if __name__ == "__main__":
    main()

# 🌌 Laplace's Demon — Universe Sandbox Simulation

> *"An intellect which at a certain moment would know all forces that set nature in motion,
> and all positions of all items of which nature is composed... would embrace in a single formula
> the movements of the greatest bodies of the universe and those of the tiniest atom."*
> — Pierre-Simon Laplace, 1814

A high-performance N-body gravitational simulation of a sandbox universe with thousands of particles, powered by Barnes-Hut O(N log N) tree gravity and Leapfrog symplectic integration.

---

## Features

| Feature | Detail |
|---|---|
| **Gravity algorithm** | Barnes-Hut quadtree — O(N log N) |
| **Integrator** | Leapfrog KDK (Kick-Drift-Kick) — symplectic, energy-preserving |
| **JIT acceleration** | Numba `@njit` for all physics hot paths |
| **Particle types** | ⭐ Stars (gold), 🪐 Planets (blue), ☄ Comets (cyan) |
| **Renderers** | Pygame real-time (additive trails) + Matplotlib offline |
| **Controls** | Zoom, pan, pause, speed up/down, trail clear |
| **Presets** | Twin galaxies, Solar system, Chaos, Laplace rings |
| **Particle count** | 500–10,000+ (Barnes-Hut makes large N tractable) |

---

## Installation

```bash
pip install numpy numba pygame matplotlib scipy
```

---

## Usage

### Real-time Pygame (recommended)

```bash
# Default: 5,000 particles, twin galaxy collision
python main.py

# Solar system: central star + planet belts + Oort cloud
python main.py --preset solar_system --N 3000

# 5 chaotic colliding clusters
python main.py --preset chaos --N 8000

# Ordered Laplacian determinism rings
python main.py --preset laplace --N 4000

# Larger simulation (needs patience for JIT warmup)
python main.py --N 10000 --spf 1
```

### Offline Matplotlib (export-friendly)

```bash
# Interactive matplotlib animation
python main.py --mode mpl --N 1000 --mpl-steps 5000

# Export to video (requires ffmpeg)
python main.py --mode mpl --N 1000 --mpl-steps 5000 --save universe.mp4
```

---

## Controls (Pygame mode)

| Key / Action | Effect |
|---|---|
| `SPACE` | Pause / resume |
| `+` / `↑` | Double simulation speed |
| `-` / `↓` | Halve simulation speed |
| `R` | Reset camera to fit all particles |
| `T` | Clear particle trails |
| `Scroll wheel` | Zoom in/out |
| `Left drag` | Pan camera |
| `ESC` | Quit |

---

## Physics Parameters

| Flag | Default | Description |
|---|---|---|
| `--dt` | `0.002` | Leapfrog time step (smaller = more accurate, slower) |
| `--eps` | `0.35` | Gravitational softening (prevents singularities at close range) |
| `--theta` | `0.6` | Barnes-Hut opening angle (0.3=accurate, 0.9=fast) |
| `--spf` | `2` | Physics steps per display frame |
| `--trail-decay` | `0.90` | Trail fade factor (0=no trails, 1=eternal) |

---

## Architecture

```
Universe Simulation/
├── main.py                    # Entry point + CLI
├── simulation_engine.py       # Physics loop, BH tree driver
├── physics/
│   ├── barnes_hut.py          # Barnes-Hut quadtree (Numba JIT)
│   └── integrator.py          # Leapfrog KDK + energy diagnostics
├── universe/
│   └── generator.py           # Particle generator: stars, planets, comets
└── renderer/
    ├── pygame_renderer.py      # Real-time Pygame with trails + HUD
    └── mpl_renderer.py         # Offline Matplotlib + video export
```

---

## Performance Notes

- **Numba JIT warmup**: First run compiles the LLVM kernels (~5–20s). Subsequent runs are instant.
- **Barnes-Hut threshold**: Automatically enabled for N ≥ 600. Below that, direct O(N²) is used.
- **Energy sampling**: Only done for N ≤ 1,500 (O(N²) operation). Disabled for larger systems.
- **Trail rendering**: Uses vectorized `np.add.at` for O(N) GPU-style additive blending.

---

## Laplace's Demon Philosophy

Laplace's determinism posits that the universe is a **closed, deterministic machine**.
The leapfrog integrator is *symplectic* — it preserves the phase-space volume of Hamiltonian
mechanics, meaning **energy drift is bounded** (not accumulated) over long runs.
The HUD shows live energy drift percentage as a measure of how faithfully the demon's
knowledge is being preserved.

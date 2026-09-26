# Laplace's Demon — N-Body Universe Simulation

> *"An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed... would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for it, nothing would be uncertain and the future just like the past would be present before its eyes."*  
> — **Pierre-Simon Laplace**, *Essai philosophique sur les probabilités* (1814)

A deterministic, ultra-high-performance 2D N-body gravitational sandbox simulation capable of rendering thousands of gravitating particles in real-time. Built with **Barnes-Hut O(N log N) quadtree gravity**, **parallel Numba JIT kernels**, **symplectic Leapfrog KDK integration**, and **zero-allocation memory architecture**.

---

## Features

- **Numba JIT Acceleration**: Multi-threaded C-speed physics hot paths using Numba `@njit(parallel=True)` and SIMD fastmath operations.
- **Barnes-Hut O(N log N) Quadtree**: Scalable gravitational force calculations for 5,000+ particles in real-time.
- **Symplectic Integrator**: Leapfrog KDK (Kick-Drift-Kick) scheme guaranteeing long-term energy conservation without energy drift.
- **Vectorized Real-Time Renderer**: Blazing-fast Pygame particle trails, additive blending, glow effects, and interactive HUD.
- **Zero-Allocation Architecture**: Global pre-allocated node pools eliminating garbage collection pauses across tens of thousands of frames.
- **Precision & Stability Safeguards**: Stack-depth limits (126 levels) and floating-point safeguards against deep-space slingshot stack overflows.
- **Dual Rendering Modes**: Real-time interactive Pygame sandbox + headless offline Matplotlib renderer with video export (`.mp4`).

---

## Simulation Presets

| Preset | Command | Description |
| :--- | :--- | :--- |
| **Twin Galaxies** | `python3 main.py --preset twin_galaxies` | Two colliding spiral galaxies with central supermassive black holes, disk stars, and comet halos. |
| **Solar System** | `python3 main.py --preset solar_system` | Sun orbited by 8 Major Planets (Mercury to Neptune), packed Asteroid Belt, and Oort Cloud. |
| **Alpha Centauri** | `python3 main.py --preset alpha_centauri` | Alpha Centauri A/B binary star pair + Proxima Centauri (M-dwarf Red Star) + Proxima b/c exoplanets + debris belt. |
| **Milkomeda** | `python3 main.py --preset milkomeda` | Future 4.5B year merger of Andromeda (M31) + Milky Way + Triangulum (M33) dwarf satellite galaxy. |
| **Cygnus X-1** | `python3 main.py --preset cygnus_x1` | Stellar-mass Black Hole + Blue Supergiant companion star (HDE 226868) + accretion disk & bipolar jets. |
| **Messier 13** | `python3 main.py --preset messier13` | M13 Hercules Globular Cluster modeled with **100% Stellar Archetypes** (Red Giants, Pulsars, White Dwarfs, Blue Stragglers) with radial mass segregation. |
| **Messier 31** | `python3 main.py --preset messier31` | M31 Andromeda Galaxy with central SMBH, logarithmic spiral arm structure, and stellar bulge. |
| **Chaos** | `python3 main.py --preset chaos` | 5 chaotic galaxy clusters on a collision course producing intricate tidal tails and gravitational slingshots. |
| **Laplace** | `python3 main.py --preset laplace` | Structured concentric deterministic rings demonstrating phase space symmetry and order under gravity. |

---

## Stellar Archetypes & Visual Palette

The simulation uses an astrophysically accurate, visually distinct, non-overlapping color palette across all celestial bodies:

| Archetype | Description & Astrophysical Basis | Visual Color | RGB Code |
| :--- | :--- | :--- | :--- |
| **`STAR`** | Sun-like G/K Main Sequence | Warm Cream Gold | `(255, 220, 120)` |
| **`RED_DWARF`** | M-Type Red Dwarf Star | Deep Crimson Red | `(180, 20, 40)` |
| **`RED_GIANT`** | Evolved Red Giants & Supergiants | Deep Ruby Coral | `(255, 90, 50)` |
| **`BLUE_STRAGGLER`** | Hot O/B-type Stars & Blue Stragglers | Luminous Ice Blue | `(100, 190, 255)` |
| **`WHITE_DWARF`** | Compact Stellar Remnants | Diamond Pearl White | `(245, 250, 255)` |
| **`NEUTRON_STAR`** | Relativistic Pulsars / Magnetars | Soft Violet-Indigo | `(190, 140, 255)` |
| **`EMISSION_STAR`** | Wolf-Rayet Stars ($[\text{O III}]\ \lambda 5007\,\text{Å}$) | Aquatic Emerald-Teal | `(80, 240, 170)` |
| **`BLACK_HOLE`** | Accreting Core + Photon Sphere Ring | Crimson Red + White Ring + Void Core | `(255, 50, 70)` |
| **`PLANET`** | Accretion / Orbital Disk Particles | Soft Sky Blue | `(90, 180, 255)` |
| **`COMET`** | Relativistic Polar Jets & Oort Cloud | Icy Aquamarine | `(140, 255, 220)` |
| **`ROCKY`** | Terrestrial Rocky Planets | Terracotta Rust | `(220, 130, 70)` |
| **`GAS_GIANT`** | Jovian Gas / Ice Giants | Soft Royal Lavender | `(180, 110, 240)` |

---

## Controls (Pygame Mode)

| Input / Key | Action |
| :--- | :--- |
| **Left Click (when Paused)** | **Select & Spectate Entity**: Lock camera onto any clicked celestial body (Black Hole, Star, Planet, Comet) |
| **U** | **Unselect Spectator Tracking**: Return to normal free-panning camera mode |
| **Left Click + Drag** | Pan camera viewport (cancels target tracking to restore free camera) |
| **Mouse Scroll Wheel** | Zoom in / Zoom out centered at cursor (or centered on spectated body) |
| **]** / **[** | Zoom in / Zoom out towards window center |
| **SPACE** | Pause / Resume physics simulation |
| **+** / **=** | Increase animation speed multiplier (up to **512x**) |
| **-** | Decrease animation speed multiplier (slow motion down to **0.0625x** / **1/16th speed**) |
| **,** / **.** (or **K** / **L**) | Adjust Motion Trail Length (**Off** to **Super**) |
| **T** | Clear particle motion trail buffer |
| **R** | Reset camera view & unselect target tracking |
| **Window Resize** | Resize viewport dynamically |
| **ESC** | Clear spectator target / Exit simulation |

---

## Installation

Ensure you have Python 3.9+ installed, then install the required dependencies:

```bash
pip3 install numpy numba pygame matplotlib scipy
```

---

## Usage Guide

### Real-Time Interactive Simulation (Pygame)

```bash
# Default launch (Twin Galaxies, 5,000 particles)
python3 main.py

# Launch all 9 presets:
python3 main.py --preset twin_galaxies      # Two colliding spiral galaxies with central SMBHs
python3 main.py --preset solar_system       # Sun + 8 major planets + Asteroid belt & Oort Cloud
python3 main.py --preset alpha_centauri     # Alpha Centauri A/B binary + Proxima Centauri & exoplanets
python3 main.py --preset milkomeda          # Andromeda + Milky Way + M33 dwarf satellite merger
python3 main.py --preset cygnus_x1          # Black hole accretion disk + companion star + polar jets
python3 main.py --preset messier13          # M13 100% Stellar Cluster (Red Giants, Pulsars, Blue Stragglers)
python3 main.py --preset messier31          # M31 Andromeda standalone spiral galaxy
python3 main.py --preset chaos              # 5 chaotic galaxy clusters on collision course
python3 main.py --preset laplace            # Structured concentric deterministic rings

# Custom particle count & physics tuning:
python3 main.py --preset messier13 --N 8000
python3 main.py --preset cygnus_x1 --N 5000 --dt 0.0005 --spf 4
python3 main.py --N 10000 --trail-decay 0.95
python3 main.py --preset twin_galaxies --seed random
```

### Offline Matplotlib & Video Export

```bash
# Interactive Matplotlib view
python3 main.py --mode mpl --N 1000 --mpl-steps 5000

# Export simulation run to high-quality MP4 (requires ffmpeg)
python3 main.py --mode mpl --N 1000 --mpl-steps 5000 --save universe.mp4
```

---

## Configuration & CLI Options

```text
Options:
  --preset        {twin_galaxies,solar_system,chaos,laplace,alpha_centauri,milkomeda,cygnus_x1,messier13,messier31}
                  Initial condition layout (default: twin_galaxies)
  --N INT         Target particle count (default: 5000)
  --seed STR      Random seed for deterministic initialization, or 'random' (default: "42")
  --mode          {pygame,mpl} Rendering interface (default: pygame)
  --dt FLOAT      Leapfrog integration timestep (default: 0.001)
  --eps FLOAT     Gravitational softening factor (default: 1.0)
  --theta FLOAT   Barnes-Hut opening angle criteria (default: 0.6)
  --no-bh         Disable Barnes-Hut quadtree and force direct O(N²) calculation
  --spf INT       Physics sub-steps per display frame (default: 2)
  --fps INT       Target render frame rate (default: 60)
  --trail-decay   Trail fade persistence [0.0 to 1.0] (default: 0.90)
  --width INT     Window width in pixels (default: 1600)
  --height INT    Window height in pixels (default: 900)
```

---

## Project Architecture

```
Universe Simulation/
├── main.py                 # Entry point, CLI parser, and execution orchestrator
├── simulation_engine.py    # Physics step engine, zero-alloc BH tree builder & dispatcher
├── physics/
│   ├── barnes_hut.py       # Parallel Barnes-Hut quadtree force evaluator (Numba JIT)
│   └── integrator.py       # Symplectic Leapfrog KDK integration & Hamiltonian energy check
├── universe/
│   └── generator.py        # All 8 universe preset generators (Galaxies, Solar System, Cygnus X-1, etc.)
└── renderer/
    ├── pygame_renderer.py # High-performance vectorized Pygame visualizer with HUD & trails
    └── mpl_renderer.py    # Headless Matplotlib animator & video exporter
```

---

## Physics & Optimization Highlights

### 1. Symplectic Phase-Space Preservation
Unlike standard Euler or Runge-Kutta integrators which damp or explode orbital systems over time, Laplace's Demon utilizes the **Kick-Drift-Kick (KDK) Leapfrog scheme**:

```math
v^{n+1/2} = v^n + a^n \frac{\Delta t}{2}
```
```math
x^{n+1} = x^n + v^{n+1/2} \Delta t
```
```math
v^{n+1} = v^{n+1/2} + a^{n+1} \frac{\Delta t}{2}
```

This preserves the Hamiltonian phase-space volume, keeping total energy drift `dE/E_0` virtually zero over millions of steps.

### 2. Parallel Barnes-Hut Tree Calculation
Gravitational acceleration is computed in `O(N log N)` time using quadtrees. When a tree node satisfies the opening criteria `r / d < theta`, the entire subtree is approximated by its center of mass, evaluated across multi-core CPU threads using Numba `prange`.

### 3. Zero-Allocation Memory Pipeline
Instead of dynamically instantiating Python quadtree objects every frame, node data structures are laid out flat in contiguous array buffers (`_node_float`, `_node_int`). Rebuilding the tree costs `O(N)` flat array overwrites without heap memory allocations or garbage collection hits.

### 4. Numerical Stability & Softening
Close encounters between high-velocity particles can cause numerical divergence or precision loss. Softening factor `epsilon` prevents gravitational singularities:

```math
a_i = G \sum_{j \neq i} m_j \frac{r_{ji}}{(r_{ji}^2 + \epsilon^2)^{3/2}}
```

Tree depth is hard-clamped at 126 levels to handle edge cases where particles are ejected into deep space.

---

## License

Distributed under the MIT License.

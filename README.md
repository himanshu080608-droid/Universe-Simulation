# Laplace's Demon — N-Body Universe Simulation

> *"An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed... would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for it, nothing would be uncertain and the future just like the past would be present before its eyes."*  
> — **Pierre-Simon Laplace**, *Essai philosophique sur les probabilités* (1814)

A deterministic, ultra-high-performance 2D N-body gravitational sandbox simulation capable of rendering thousands of gravitating particles in real-time. Built with **Barnes-Hut O(N log N) quadtree gravity**, **parallel Numba JIT kernels**, **symplectic Leapfrog KDK integration**, **dynamic Telescope Resolution Shader**, and **zero-allocation memory architecture**.

---

## Key Features

- **Telescope Resolution Mechanic (Optical Zoom Shader)**: Real-time dynamic optical resolution simulation. When zoomed out, black hole event horizons blend smoothly into surrounding stellar glare (simulating low-magnification optical limits). Zooming in increases effective telescope resolving power, revealing the pitch-black void event horizon, white/gold photon sphere ring, and surrounding accretion disk.
- **Multi-Pass Shading Engine**:
  - **Pass 1**: Renders all stars, planets, and nebular particles into an additive motion trail buffer with anti-aliased sub-pixel point rendering.
  - **Pass 2**: Renders Black Hole event horizons, photon spheres, and accretion glows over stellar backdrops with proper Z-ordering to eliminate background punched holes or visual clipping.
- **Decoupled Camera Controls & Target Spectating**:
  - **Free Viewport Panning**: Smooth left-click drag to pan anywhere in the simulation.
  - **Click-to-Spectate**: Mouse release with movement < 5 pixels locks the camera onto any targeted star, planet, or black hole for smooth camera tracking.
  - **Unselect**: Press `U` or `ESC` to release spectator tracking and return to free camera mode.
- **Smooth Zoom Interpolation & Flash Prevention**: Exponential zoom lerping (`target_zoom`) smooths camera transitions across all scales (0.0001x to 100,000x). Trail buffer resets are synchronized during camera movement to prevent visual flickering.
- **Numba JIT Acceleration**: Multi-threaded C-speed physics hot paths using Numba `@njit(parallel=True)` and SIMD fastmath operations.
- **Barnes-Hut O(N log N) Quadtree**: Scalable gravitational force calculations for 5,000+ particles in real-time.
- **Symplectic Integrator**: Leapfrog KDK (Kick-Drift-Kick) scheme guaranteeing long-term energy conservation without energy drift.
- **Zero-Allocation Architecture**: Global pre-allocated node pools eliminating garbage collection pauses across tens of thousands of frames.
- **Automated Visual Regression Test Suite**: Includes a headless test suite (`tests/test_visuals.py`) to verify macro and micro visual output across all presets.
- **Dual Rendering Modes**: Real-time interactive Pygame sandbox + headless offline Matplotlib renderer with video export (`.mp4`).

---

## Simulation Presets

| Preset | Command | Description |
| :--- | :--- | :--- |
| **Twin Galaxies** | `python3 main.py --preset twin_galaxies` | Two colliding spiral galaxies with central supermassive black holes, disk stars, and comet halos. |
| **Solar System** | `python3 main.py --preset solar_system` | Sun orbited by 8 Major Planets (Mercury to Neptune), packed Asteroid Belt, and Oort Cloud. |
| **Alpha Centauri** | `python3 main.py --preset alpha_centauri` | Alpha Centauri A/B binary star pair + Proxima Centauri (M-dwarf Red Star) + Proxima b/c exoplanets + debris belt. |
| **Milkomeda** | `python3 main.py --preset milkomeda` | Future 4.5B year merger of Andromeda (M31) + Milky Way + Triangulum (M33) with optimized initial galaxy separation. |
| **Cygnus X-1** | `python3 main.py --preset cygnus_x1` | Stellar-mass Black Hole + Blue Supergiant companion star (HDE 226868) + accretion disk & bipolar relativistic jets. |
| **Messier 13** | `python3 main.py --preset messier13` | M13 Hercules Globular Cluster modeled with **100% Stellar Archetypes** (Red Giants, Pulsars, White Dwarfs, Blue Stragglers) with radial mass segregation. |
| **Messier 31** | `python3 main.py --preset messier31` | M31 Andromeda Galaxy with central SMBH, logarithmic spiral arm structure, and stellar bulge. |
| **Chaos** | `python3 main.py --preset chaos` | 5 chaotic galaxy clusters on a collision course producing intricate tidal tails and gravitational slingshots. |
| **Laplace** | `python3 main.py --preset laplace` | Structured concentric deterministic rings demonstrating phase space symmetry and order under gravity. |
| **NGC 1052-DF2** | `python3 main.py --preset ngc1052_df2` | **Ultra-Diffuse Galaxy:** A ghostly, dark-matter-free dwarf galaxy. Particle counts are strictly capped (N/5) and distributed across a Virial-balanced Gaussian cloud to accurately reflect its sparse, nearly transparent real-world visual density. |
| **Castor Sextuple** | `python3 main.py --preset castor_sextuple` | 6-star Hierarchical Resonance System surrounded by a distant, mathematically stable circumbinary asteroid belt. |
| **HD 98800** | `python3 main.py --preset hd98800_polar` | Quadruple Star system featuring a perpendicular Polar Protoplanetary Disk. |
| **TRAPPIST-1** | `python3 main.py --preset trappist_1` | 7-Planet Resonant Laplace Chain orbiting a central ultra-cool Red Dwarf. |
| **Shepherd Moons** | `python3 main.py --preset shepherd_moons` | Saturnian-style ring system where shepherd moons carve distinct dark orbital gaps within the planetary rings. |
| **WR 104** | `python3 main.py --preset wr104_pinwheel` | The Pinwheel Nebula: A colliding-wind binary producing a continuous, rotating Archimedean spiral of hot dust. |
| **Omega Centauri** | `python3 main.py --preset omega_centauri` | Massive Core-Collapsed Globular Cluster harboring a central Intermediate-Mass Black Hole (IMBH). |
| **Pleiades (M45)** | `python3 main.py --preset pleiades_m45` | Open Cluster (The Seven Sisters) featuring luminous young blue stars surrounded by a wispy reflection nebula. |
| **HL Tauri** | `python3 main.py --preset hl_tauri` | Young protoplanetary dust disk featuring distinct annular gaps carved by unseen forming protoplanets. |

---

## Stellar Archetypes & Visual Palette

The simulation uses an astrophysically accurate, visually distinct, non-overlapping color palette across all celestial bodies:

| Archetype | Description & Astrophysical Basis | Visual Color | RGB Code |
| :--- | :--- | :--- | :--- |
| **`STAR`** | Sun-like G/K Main Sequence Stars | Warm Cream Gold | `(255, 220, 120)` |
| **`RED_DWARF`** | M-Type Red Dwarf Star | Deep Crimson Red | `(180, 20, 40)` |
| **`RED_GIANT`** | Evolved Red Giants & Supergiants | Deep Ruby Coral | `(255, 90, 50)` |
| **`BLUE_STRAGGLER`** | Hot O/B-type Stars & Blue Stragglers | Luminous Ice Blue | `(100, 190, 255)` |
| **`WHITE_DWARF`** | Compact Stellar Remnants | Diamond Pearl White | `(245, 250, 255)` |
| **`NEUTRON_STAR`** | Relativistic Pulsars / Magnetars | Soft Violet-Indigo | `(190, 140, 255)` |
| **`EMISSION_STAR`** | Wolf-Rayet Stars ($[\text{O III}]\ \lambda 5007\,\text{Å}$) | Aquatic Emerald-Teal | `(80, 240, 170)` |
| **`BLACK_HOLE`** | Accreting Core + Photon Sphere Ring | Pitch-Black Void + Gold Ring | Dynamic Shader |
| **`PLANET`** | Accretion / Orbital Disk Particles | Soft Sky Blue | `(90, 180, 255)` |
| **`COMET`** | Relativistic Polar Jets & Oort Cloud | Icy Aquamarine | `(140, 255, 220)` |
| **`ROCKY`** | Terrestrial Rocky Planets | Terracotta Rust | `(220, 130, 70)` |
| **`GAS_GIANT`** | Jovian Gas / Ice Giants | Soft Royal Lavender | `(180, 110, 240)` |
| **`ASTEROID`** | Small Solar/Orbital Asteroids | Dusty Grey | `(160, 160, 160)` |

---

## Controls (Pygame Mode)

| Input / Key | Action |
| :--- | :--- |
| **Left Click Drag** | **Pan Viewport**: Drag mouse to pan camera freely across space |
| **Left Click Release (< 5px drag)** | **Select & Spectate Entity**: Lock camera center onto clicked celestial body (paused) |
| **U** / **ESC** | **Unselect Tracking**: Release spectator target and return to free camera mode |
| **Mouse Scroll Wheel** | **Smooth Zoom**: Exponential zoom centered at mouse cursor (or spectated body) |
| **]** / **[** | Zoom in / Zoom out towards window center |
| **SPACE** | Pause / Resume physics simulation |
| **+** / **=** | Increase animation speed multiplier (up to **16x**) |
| **-** | Decrease animation speed multiplier (slow motion down to **0.0625x** / **1/16th speed**) |
| **,** / **.** (or **K** / **L**) | Adjust Motion Trail Length (**Off** to **Super**) |
| **T** | Clear particle motion trail buffer |
| **R** | Reset camera view to default disk fitting |
| **Window Resize** | Dynamically update viewport bounds and aspect ratio |

---

## Installation

Ensure you have Python 3.9+ installed, then install the required dependencies:

```bash
pip3 install numpy numba pygame matplotlib scipy pillow
```

---

## Usage Guide

### Real-Time Interactive Simulation (Pygame)

```bash
# Default launch (Twin Galaxies, 5,000 particles)
python3 main.py

# Launch presets:
python3 main.py --preset twin_galaxies      # Two colliding spiral galaxies with central SMBHs
python3 main.py --preset solar_system       # Sun + 8 major planets + Asteroid belt & Oort Cloud
python3 main.py --preset alpha_centauri     # Alpha Centauri A/B binary + Proxima Centauri & exoplanets
python3 main.py --preset milkomeda          # Andromeda + Milky Way + M33 dwarf satellite merger
python3 main.py --preset cygnus_x1          # Black hole accretion disk + companion star + polar jets
python3 main.py --preset messier13          # M13 100% Stellar Cluster (Red Giants, Pulsars, Blue Stragglers)
python3 main.py --preset messier31          # M31 Andromeda standalone spiral galaxy
python3 main.py --preset chaos              # 5 chaotic galaxy clusters on collision course
python3 main.py --preset laplace            # Structured concentric deterministic rings
python3 main.py --preset ngc1052_df2        # Dark-Matter-Free Ultra-Diffuse Galaxy
python3 main.py --preset castor_sextuple    # 6-star Hierarchical Resonance System
python3 main.py --preset hd98800_polar      # Quadruple Star with Polar Protoplanetary Disk
python3 main.py --preset trappist_1         # 7-Planet Resonant Laplace Chain around Red Dwarf
python3 main.py --preset shepherd_moons     # Saturnian Ring Gaps & Shepherd Moons
python3 main.py --preset wr104_pinwheel     # The Pinwheel Nebula (Colliding Wind Binary)
python3 main.py --preset omega_centauri     # Core-Collapsed Globular with IMBH
python3 main.py --preset pleiades_m45       # Pleiades Open Cluster (The Seven Sisters)
python3 main.py --preset hl_tauri           # Protoplanetary Disk with Annular Gaps

# Custom particle count & physics tuning:
python3 main.py --preset messier13 --N 8000
python3 main.py --preset cygnus_x1 --N 5000 --dt 0.0005 --spf 4
python3 main.py --N 10000 --trail-decay 0.95
python3 main.py --preset twin_galaxies --seed random
```

### Automated Regression Test Suite

For contributors and developers, run the full test suite (which includes physics, integration, and Pygame visual regressions running headlessly in CI):

```bash
python3 -m unittest discover tests -v
```

### Headless Visual Regression Test Suite

Run the automated visual test suite to generate frame snapshots and verified visual GIFs for macro/micro views of all presets:

```bash
python3 tests/test_visuals.py
```

Outputs will be saved in `tests/output/`.
    
### High-Resolution Screenshots and GIFs
    
You can generate high-resolution screenshots at multiple zoom levels, or record animated GIFs of specific presets. All media is saved to the `output/` directory to keep your workspace clean.

```bash
# Generate high-resolution Pygame screenshots at various zoom levels (saves to output/)
python3 record_screenshots.py

# Record animated GIFs of presets (saves to output/)
python3 record_pygame.py
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
  --preset        {twin_galaxies,solar_system,chaos,laplace,alpha_centauri,milkomeda,cygnus_x1,messier13,messier31,ngc1052_df2,castor_sextuple,hd98800_polar,trappist_1,shepherd_moons,wr104_pinwheel,omega_centauri,pleiades_m45,hl_tauri}
                  Initial condition layout (default: twin_galaxies)
  --N INT         Target particle count (default: 5000)
  --seed STR      Random seed for deterministic initialization, or 'random' (default: "42")
  --mode          {pygame,mpl} Rendering interface (default: pygame)
  --dt FLOAT      Leapfrog integration timestep (default: 0.001)
  --eps FLOAT     Gravitational softening factor (default: 1.0)
  --theta FLOAT   Barnes-Hut opening angle criteria (default: 0.6)
  --legacy-leapfrog Use the older 2nd-Order Leapfrog Integrator instead of Hermite
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
│   └── generator.py        # All 9 universe preset generators (Galaxies, Solar System, Cygnus X-1, etc.)
├── renderer/
│   ├── pygame_renderer.py # Multi-pass Pygame renderer, telescope shader, trails & camera
│   └── mpl_renderer.py    # Headless Matplotlib animator & video exporter
└── tests/
    └── test_visuals.py    # Automated headless visual testing & visual regression suite
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

### 2. Telescope Resolution Shader & Multi-Pass Rendering
Black holes present a unique optical challenge: when viewed from interstellar distances (zoomed out), stellar glare blurs out the central void. When viewed under high magnification (zoomed in), the central event horizon and surrounding photon ring are resolved.

The renderer calculates `telescope_power` as a function of screen scale `zoom`:
```math
\text{telescope\_power} = \text{clamp}\left(\frac{\text{zoom}}{\text{ZOOM\_THRESHOLD}}, 0.0, 1.0\right)
```
Pass 1 blits all stellar bodies into the additive trail buffer. Pass 2 composite-blends the black hole void, photon ring, and accretion disc over the backdrop scaled by `telescope_power`.

### 3. Parallel Barnes-Hut Tree Calculation
Gravitational acceleration is computed in `O(N log N)` time using quadtrees. When a tree node satisfies the opening criteria `r / d < theta`, the entire subtree is approximated by its center of mass, evaluated across multi-core CPU threads using Numba `prange`.

### 4. Zero-Allocation Memory Pipeline
Instead of dynamically instantiating Python quadtree objects every frame, node data structures are laid out flat in contiguous array buffers (`_node_float`, `_node_int`). Rebuilding the tree costs `O(N)` flat array overwrites without heap memory allocations or garbage collection hits.

---

## License

Distributed under the MIT License.

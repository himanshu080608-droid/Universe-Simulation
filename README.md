# Laplace's Demon — N-Body Universe Simulation

> *"An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed... would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for it, nothing would be uncertain and the future just like the past would be present before its eyes."*  
> — **Pierre-Simon Laplace**, *Essai philosophique sur les probabilités* (1814)

A deterministic, ultra-high-performance 2D N-body gravitational sandbox simulation capable of rendering thousands of gravitating particles in real-time. Built with **Taichi GPU acceleration**, **Ahmad-Cohen block timesteps**, **4th-order Hermite integration**, **dynamic Telescope Resolution Shader**, and fallback **Barnes-Hut O(N log N)** capabilities.

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
- **GPU Architecture**: Default `AdvancedTaichiEngine` leverages Taichi Metal/CUDA for exact $O(N^2)$ direct gravity across thousands of cores.
- **Block Time-Steps (Ahmad-Cohen)**: Assigns custom dt to each particle. Fast binaries update 32x per frame, while slow particles update 1x per frame, dropping workload by 90%.
- **4th-Order Hermite Integrator**: High-precision solver ensuring stability even during tight encounters.
- **Legacy Symplectic Integrator**: O(N log N) Barnes-Hut Leapfrog KDK engine with zero-allocation memory available via `--legacy-leapfrog`.
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
| **Stephan's Quintet** | `python3 main.py --preset stephans_quintet` | Stephan's Quintet (Compact Galaxy Group). |
| **Messier 13** | `python3 main.py --preset messier13` | M13 Hercules Globular Cluster modeled with **100% Stellar Archetypes** (Red Giants, Pulsars, White Dwarfs, Blue Stragglers) with radial mass segregation. |
| **Messier 31** | `python3 main.py --preset messier31` | M31 Andromeda Galaxy with central SMBH, logarithmic spiral arm structure, and stellar bulge. |
| **Chaos** | `python3 main.py --preset chaos` | 5 chaotic galaxy clusters on a collision course producing intricate tidal tails and gravitational slingshots. |
| **Laplace** | `python3 main.py --preset laplace` | Structured concentric deterministic rings demonstrating phase space symmetry and order under gravity. |
| **NGC 1052-DF2** | `python3 main.py --preset ngc1052_df2` | **Ultra-Diffuse Galaxy:** A ghostly, dark-matter-free dwarf galaxy. Particle counts are strictly capped (N/5) and distributed across a Virial-balanced Gaussian cloud to accurately reflect its sparse, nearly transparent real-world visual density. |
| **Castor Sextuple** | `python3 main.py --preset castor_sextuple` | 6-star Hierarchical Resonance System surrounded by a distant, mathematically stable circumbinary asteroid belt. |
| **HD 98800** | `python3 main.py --preset hd98800_polar` | Quadruple Star system featuring a perpendicular Polar Protoplanetary Disk. |
| **TRAPPIST-1** | `python3 main.py --preset trappist_1` | 7-Planet Resonant Laplace Chain orbiting a central ultra-cool Red Dwarf. |
| **Gravothermal Catastrophe** | `python3 main.py --preset gravothermal_catastrophe` | Core Collapse of a Uniform Cluster. |
| **WR 104** | `python3 main.py --preset wr104_pinwheel` | The Pinwheel Nebula: A colliding-wind binary producing a continuous, rotating Archimedean spiral of hot dust. |
| **Omega Centauri** | `python3 main.py --preset omega_centauri` | Massive Core-Collapsed Globular Cluster harboring a central Intermediate-Mass Black Hole (IMBH). |
| **Pleiades (M45)** | `python3 main.py --preset pleiades_m45` | Open Cluster (The Seven Sisters) featuring luminous young blue stars surrounded by a wispy reflection nebula. |
| **Hirayama Family** | `python3 main.py --preset hirayama_family` | Asteroid Disruption & Keplerian Shear. |
| **Dark Matter Halo Merger** | `python3 main.py --preset dark_matter_halo_merger` | Dark Matter Halo Merger (Violent Relaxation). |
| **Great Attractor** | `python3 main.py --preset great_attractor` | The Great Attractor (Cosmic Filaments). |

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
python3 main.py --preset stephans_quintet   # Stephan's Quintet (Compact Galaxy Group)
python3 main.py --preset messier13          # M13 100% Stellar Cluster (Red Giants, Pulsars, Blue Stragglers)
python3 main.py --preset messier31          # M31 Andromeda standalone spiral galaxy
python3 main.py --preset chaos              # 5 chaotic galaxy clusters on collision course
python3 main.py --preset laplace            # Structured concentric deterministic rings
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

# Custom particle count & physics tuning:
python3 main.py --preset messier13 --N 8000
python3 main.py --preset chaos --N 5000 --dt 0.0005 --spf 4
python3 main.py --N 10000 --trail-decay 0.95
python3 main.py --preset twin_galaxies --seed random
```

### Automated Regression Test Suite

For contributors and developers, run the full test suite (which covers physics/integration checks plus Advanced-engine/Pygame runtime, preset compatibility, and visual regression tests running headlessly in CI):

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
  --preset        {twin_galaxies,solar_system,chaos,laplace,alpha_centauri,milkomeda,stephans_quintet,messier13,messier31,ngc1052_df2,castor_sextuple,hd98800_polar,trappist_1,gravothermal_catastrophe,wr104_pinwheel,omega_centauri,pleiades_m45,hirayama_family,dark_matter_halo_merger,great_attractor}
                  Initial condition preset layout (default: twin_galaxies)
  --N INT         Target particle count (default: 5000)
  --seed STR      Random seed for reproducibility, or 'random' for a new seed every run (default: 42)
  --mode          {pygame,mpl} Renderer: 'pygame' = live, 'mpl' = offline (default: pygame)
  --dt FLOAT      Integration time step (default: 0.001 for Hermite)
  --eps FLOAT     Gravitational softening (default: 1.0)
  --theta FLOAT   Barnes-Hut opening angle (default: 0.8)
  --legacy-leapfrog Use the older 2nd-Order Leapfrog Integrator instead of AdvancedTaichiEngine
  --taichi        Use Taichi GPU Hermite engine (Metal/CUDA)
  --advanced      Use the Ultimate Engine (Taichi GPU + Ahmad-Cohen Block Time-Steps) [also the default]
  --spf INT       Physics steps per display frame (default: 2)
  --fps INT       Target display FPS (default: 60)
  --trail-decay   Trail fade factor per frame, 0=off 1=infinite (default: 0.90)
  --mpl-steps     Total physics steps for mpl precompute (default: 20000)
  --mpl-stride    Record every N steps for mpl (default: 5)
  --save          If set, save mpl animation to this file (e.g. out.mp4)
  --width INT     Display width (default: 1600)
  --height INT    Display height (default: 900)
```

---

## Project Architecture

```
Universe Simulation/
├── main.py                 # Entry point, CLI parser, and execution orchestrator
├── advanced_engine.py      # Default engine: Taichi GPU, Hermite integrator, Ahmad-Cohen block timesteps
├── taichi_engine.py        # Alternative engine: Taichi GPU, fixed-step Hermite integrator
├── simulation_engine.py    # Legacy engine: Barnes-Hut Leapfrog (selected via --legacy-leapfrog)
├── physics/
│   ├── barnes_hut.py       # Legacy parallel Barnes-Hut quadtree force evaluator
│   └── integrator.py       # Legacy Leapfrog KDK integration & Hamiltonian energy check
├── universe/
│   └── generator.py        # All 20 universe preset generators (Galaxies, Solar System, etc.)
├── renderer/
│   ├── pygame_renderer.py # Multi-pass Pygame renderer, telescope shader, trails & camera
│   └── mpl_renderer.py    # Headless Matplotlib animator & video exporter
└── tests/
    └── test_visuals.py    # Automated headless visual testing & visual regression suite
```

---

## Physics & Optimization Highlights

### 1. High-Performance GPU Integration and Block Timesteps
The default `AdvancedTaichiEngine` leverages Taichi to evaluate exact $O(N^2)$ direct gravity across thousands of cores, bypassing slow tree branching. It pairs this with a 4th-order Hermite integrator and an **Ahmad-Cohen Block Time-Step Scheme**. Fast-moving particles receive tiny timesteps, while slow particles are updated infrequently, reducing total computational workload by 90% without sacrificing precision.

*(The legacy Barnes-Hut Symplectic Leapfrog KDK scheme is preserved in `simulation_engine.py` for comparative study.)*

### 2. Telescope Resolution Shader & Multi-Pass Rendering
Black holes present a unique optical challenge: when viewed from interstellar distances (zoomed out), stellar glare blurs out the central void. When viewed under high magnification (zoomed in), the central event horizon and surrounding photon ring are resolved.

The renderer calculates `telescope_power` as a function of screen scale `zoom`:
```math
\text{telescope\_power} = \text{clamp}\left(\frac{\text{zoom}}{\text{ZOOM\_THRESHOLD}}, 0.0, 1.0\right)
```
Pass 1 blits all stellar bodies into the additive trail buffer. Pass 2 composite-blends the black hole void, photon ring, and accretion disc over the backdrop scaled by `telescope_power`.

### 3. Exact Direct Gravity on GPU
Instead of CPU quadtrees, the primary engines rely on massive parallelization via Taichi. By fully utilizing Metal/CUDA acceleration for $O(N^2)$ calculations with softened direct gravity $F = G \frac{m_1 m_2}{r^2 + \epsilon^2}$, the simulation achieves near-perfect momentum conservation and handles extremely dense configurations, like core-collapsed globular clusters, seamlessly in real-time.

### 4. Legacy CPU Optimization
For non-GPU environments (via `--legacy-leapfrog`), gravitational acceleration is computed in $O(N \log N)$ time using Barnes-Hut quadtrees. Node data structures are laid out flat in contiguous array buffers (`_node_float`, `_node_int`), eliminating heap memory allocations or garbage collection hits during the tree rebuild process.

---

## License

Distributed under the MIT License.

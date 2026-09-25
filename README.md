# 🌌 Laplace's Demon — N-Body Universe Simulation

> *"An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed... would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for it, nothing would be uncertain and the future just like the past would be present before its eyes."*  
> — **Pierre-Simon Laplace**, *Essai philosophique sur les probabilités* (1814)

A deterministic, ultra-high-performance 2D N-body gravitational sandbox simulation capable of rendering thousands of gravitating particles in real-time. Built with **Barnes-Hut $O(N \log N)$ quadtree gravity**, **parallel Numba JIT kernels**, **symplectic Leapfrog KDK integration**, and **zero-allocation memory architecture**.

---

## ✨ Features

- ⚡ **Numba JIT Acceleration**: Multi-threaded C-speed physics hot paths using Numba `@njit(parallel=True)` and SIMD fastmath operations.
- 🌳 **Barnes-Hut $O(N \log N)$ Quadtree**: Scalable gravitational force calculations for $5,000+$ particles in real-time.
- 🔄 **Symplectic Integrator**: Leapfrog KDK (Kick-Drift-Kick) scheme guaranteeing long-term energy conservation without energy drift.
- 🎨 **Vectorized Real-Time Renderer**: Blazing-fast Pygame particle trails, additive blending, glow effects, and interactive HUD.
- 🚀 **Zero-Allocation Architecture**: Global pre-allocated node pools eliminating garbage collection pauses across tens of thousands of frames.
- 🛡️ **Precision & Stability Safeguards**: Stack-depth limits (126 levels) and floating-point safeguards against deep-space slingshot stack overflows.
- 🎥 **Dual Rendering Modes**: Real-time interactive Pygame sandbox + headless offline Matplotlib renderer with video export (`.mp4`).

---

## 🌌 Simulation Presets

| Preset | Command | Description |
| :--- | :--- | :--- |
| **Twin Galaxies** | `python main.py --preset twin_galaxies` | Two massive spiral galaxies with central supermassive black holes, accretion disks, and orbiting planetary systems locked in a collision course. |
| **Solar System** | `python main.py --preset solar_system` | A central star orbited by multiple major planets, moons, packed main asteroid belt, and an outer Oort cloud. |
| **Chaos** | `python main.py --preset chaos` | 5 chaotic, high-density particle clusters colliding at speed, producing intricate tidal tails and gravitational slingshots. |
| **Laplace** | `python main.py --preset laplace` | Highly structured concentric deterministic rings demonstrating phase space symmetry and order under gravity. |

---

## 🎮 Controls (Pygame Mode)

| Action / Key | Function |
| :--- | :--- |
| **Mouse Drag (Left Click)** | Pan the camera viewport |
| **Mouse Scroll** | Zoom in / Zoom out smoothly |
| `SPACE` | Pause / Resume physics step |
| `+` / `↑` | Increase simulation speed (double time step multiplier) |
| `-` / `↓` | Decrease simulation speed (halve time step multiplier) |
| `R` | Reset camera view to frame all active particles |
| `T` | Clear all particle trail visual buffers |
| `ESC` | Exit simulation |

---

## 🛠️ Installation

Ensure you have Python 3.9+ installed, then install the required dependencies:

```bash
pip install numpy numba pygame matplotlib scipy
```

---

## 🚀 Usage Guide

### Real-Time Interactive Simulation (Pygame)

```bash
# Default launch (Twin Galaxies, 5,000 particles)
python main.py

# Launch Solar System preset with 3,000 bodies
python main.py --preset solar_system --N 3000

# Launch Chaos preset with 8,000 bodies
python main.py --preset chaos --N 8000

# High-precision mode (smaller time step dt, more sub-steps per frame)
python main.py --preset chaos --N 5000 --dt 0.0005 --spf 4

# Custom particle count & trail persistence
python main.py --N 10000 --trail-decay 0.95
```

### Offline Matplotlib & Video Export

```bash
# Interactive Matplotlib view
python main.py --mode mpl --N 1000 --mpl-steps 5000

# Export simulation run to high-quality MP4 (requires ffmpeg)
python main.py --mode mpl --N 1000 --mpl-steps 5000 --save universe.mp4
```

---

## ⚙️ Configuration & CLI Options

```text
Options:
  --preset        {twin_galaxies,solar_system,chaos,laplace} Initial condition layout (default: twin_galaxies)
  --N INT         Target particle count (default: 5000)
  --seed INT      Random seed for deterministic initialization (default: 42)
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

## 🏛️ Project Architecture

```
Universe Simulation/
├── main.py                 # Entry point, CLI parser, and execution orchestrator
├── simulation_engine.py    # Physics step engine, zero-alloc BH tree builder & dispatcher
├── physics/
│   ├── barnes_hut.py       # Parallel Barnes-Hut quadtree force evaluator (Numba JIT)
│   └── integrator.py       # Symplectic Leapfrog KDK integration & Hamiltonian energy check
├── universe/
│   └── generator.py        # Galaxy, Solar System, Chaos, and Laplace preset generators
└── renderer/
    ├── pygame_renderer.py # High-performance vectorized Pygame visualizer with HUD & trails
    └── mpl_renderer.py    # Headless Matplotlib animator & video exporter
```

---

## 🔬 Physics & Optimization Highlights

1. **Symplectic Phase-Space Preservation**:
   Unlike standard Euler or Runge-Kutta integrators which damp or explode orbital systems over time, Laplace's Demon utilizes the **Kick-Drift-Kick (KDK) Leapfrog scheme**:
   $$\mathbf{v}^{n+1/2} = \mathbf{v}^n + \mathbf{a}^n \frac{\Delta t}{2}$$
   $$\mathbf{x}^{n+1} = \mathbf{x}^n + \mathbf{v}^{n+1/2} \Delta t$$
   $$\mathbf{v}^{n+1} = \mathbf{v}^{n+1/2} + \mathbf{a}^{n+1} \frac{\Delta t}{2}$$
   This preserves the Hamiltonian phase-space volume, keeping total energy $\Delta E / E_0$ virtually zero over millions of steps.

2. **Parallel Barnes-Hut Tree Calculation**:
   Gravitational acceleration is computed in $O(N \log N)$ time using quadtrees. When a tree node satisfies the opening criteria $r / d < \theta$, the entire subtree is approximated by its center of mass, evaluated across multi-core CPU threads using Numba `prange`.

3. **Zero-Allocation Memory Pipeline**:
   Instead of dynamically instantiating Python quadtree objects every frame, node data structures are laid out flat in contiguous array buffers (`_node_float`, `_node_int`). Rebuilding the tree costs $O(N)$ flat array overwrites without heap memory allocations or garbage collection hits.

4. **Numerical Stability Safeguards**:
   Close encounters between high-velocity particles can cause numerical divergence or precision loss. Softening factor $\epsilon$ prevents gravitational singularities:
   $$\mathbf{a}_i = G \sum_{j \neq i} m_j \frac{\mathbf{r}_{ji}}{(r_{ji}^2 + \epsilon^2)^{3/2}}$$
   Tree depth is hard-clamped at 126 levels to handle edge cases where particles are ejected into deep space.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

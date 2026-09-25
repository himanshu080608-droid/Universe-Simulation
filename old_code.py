import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from abc import ABC, abstractmethod

# =====================================================================
# 1. THE CELESTIAL BLUEPRINT (System Abstraction Layer)
# =====================================================================
class SpaceSystem(ABC):
    """Abstract base class establishing structural blueprints for any universe setup."""
    def __init__(self):
        self.positions = np.empty((0, 2), dtype=np.float64)
        self.velocities = np.empty((0, 2), dtype=np.float64)
        self.masses = np.empty((0,), dtype=np.float64)

    @abstractmethod
    def generate(self):
        """Must be implemented by subclasses to define custom geometric generation math."""
        pass

    def append_system(self, pos, vel, mass):
        """Helper to combine or shift coordinate states across multiple object layers."""
        self.positions = np.vstack((self.positions, pos))
        self.velocities = np.vstack((self.velocities, vel))
        self.masses = np.concatenate((self.masses, mass))


# =====================================================================
# 2. CONCRETE SYSTEM EXTENSIONS (Stars & Planets)
# =====================================================================
class DenseParticleStarSystem(SpaceSystem):
    """Generates a soft, composite 'Star' made out of many tiny point masses."""
    def __init__(self, center=[0.0, 0.0], drift_velocity=[0.0, 0.0], num_particles=50, radius=0.45, total_mass=40.0):
        super().__init__()
        self.center = np.array(center, dtype=np.float64)
        self.drift_velocity = np.array(drift_velocity, dtype=np.float64)
        self.num_particles = num_particles
        self.radius = radius
        self.particle_mass = total_mass / num_particles

    def generate(self):
        indices = np.arange(self.num_particles) + 0.5
        r = self.radius * np.sqrt(indices / self.num_particles)
        theta = indices * np.pi * (1 + 5**0.5)  # Golden ratio distribution
        
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center
        
        # Internal spinning velocity matrix
        spin_speed = 0.4
        vel = np.column_stack((-spin_speed * np.sin(theta), spin_speed * np.cos(theta))) + self.drift_velocity
        mass = np.full(self.num_particles, self.particle_mass)
        
        self.append_system(pos, vel, mass)
        return self


class PlanetSystem(SpaceSystem):
    """Generates a distinct set of discrete planets in stable circular orbits."""
    def __init__(self, center=[0.0, 0.0], orbit_radius=2.2, num_planets=8, planet_mass=4.0, G=1.0, central_mass_estimate=80.0):
        super().__init__()
        self.center = np.array(center, dtype=np.float64)
        self.orbit_radius = orbit_radius
        self.num_planets = num_planets
        self.planet_mass = planet_mass
        self.G = G
        self.central_mass_estimate = central_mass_estimate

    def generate(self):
        angles = np.linspace(0, 2 * np.pi, self.num_planets, endpoint=False)
        orbital_speed = np.sqrt(self.G * self.central_mass_estimate / self.orbit_radius)
        
        pos = np.column_stack((self.orbit_radius * np.cos(angles), self.orbit_radius * np.sin(angles))) + self.center
        vel = np.column_stack((-orbital_speed * np.sin(angles), orbital_speed * np.cos(angles)))
        mass = np.full(self.num_planets, self.planet_mass)
        
        self.append_system(pos, vel, mass)
        return self


class Universe(SpaceSystem):
    """A concrete master system acting purely as a sandbox framework to fuse components."""
    def generate(self):
        return self


# =====================================================================
# 3. HIGH-PERFORMANCE ISOLATED COMPUTATION LAYER (Pure NumPy Interface)
# =====================================================================
@njit
def run_physics_core_engine(pos, vel, mass, steps, N, dt, G, eps):
    """
    Completely isolated Numba compiler function. 
    Accepts ONLY primitive float arrays to guarantee compile-phase execution speeds.
    """
    history = np.zeros((steps, N, 2))
    for t in range(steps):
        history[t] = pos.copy()
        forces = np.zeros((N, 2))
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    rx = pos[j, 0] - pos[i, 0]
                    ry = pos[j, 1] - pos[i, 1]
                    r_mag = np.sqrt(rx**2 + ry**2) + eps
                    f_mag = (G * mass[i] * mass[j]) / (r_mag**3)
                    forces[i, 0] += f_mag * rx
                    forces[i, 1] += f_mag * ry
        
        for i in range(N):
            ax = forces[i, 0] / mass[i]
            ay = forces[i, 1] / mass[i]
            pos[i, 0] += vel[i, 0] * dt
            pos[i, 1] += vel[i, 1] * dt
            vel[i, 0] += ax * dt
            vel[i, 1] += ay * dt
    return history


# =====================================================================
# 4. VISUALIZATION FRAMEWORK (Matplotlib Canvas Architecture)
# =====================================================================
class SimulationVisualizer:
    """Renders the dark space multi-system cosmos layer cleanly with lifecycle hooks."""
    def __init__(self, history, masses, stride=50, tail_length=35, fps=60):
        self.animated_history = history[::stride]
        self.masses = masses
        self.tail_length = tail_length
        self.fps = fps
        self.total_frames, self.N, _ = self.animated_history.shape
        
        self.fig, self.ax = plt.subplots(figsize=(8, 8), facecolor='black')
        self.ani = None # Persistent class reference handle protecting against garbage collection
        
        self._setup_canvas()
        self._setup_render_elements()

    def _setup_canvas(self):
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.axis('off')
        self.ax.set_title("Laplace's Demon: Completely Abstracted Cosmos Engine", color='white', fontsize=14)
        
    def _setup_render_elements(self):
        self.colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.N))
        self.lines = [self.ax.plot([], [], '-', lw=0.6, color=self.colors[i], alpha=0.3)[0] for i in range(self.N)]
        self.scatters = self.ax.scatter(self.animated_history[0, :, 0], self.animated_history[0, :, 1], 
                                        s=np.clip(self.masses * 30, 15, 200), c=self.colors, zorder=3)

    def _init_render(self):
        for line in self.lines:
            line.set_data([], [])
        self.scatters.set_offsets(np.empty((0, 2)))
        return self.lines + [self.scatters]

    def _update_frame(self, frame_idx):
        current_positions = self.animated_history[frame_idx]
        start_tail_idx = max(0, frame_idx - self.tail_length)
        history_slice = self.animated_history[start_tail_idx:frame_idx]
        
        for i in range(self.N):
            self.lines[i].set_data(history_slice[:, i, 0], history_slice[:, i, 1])
            
        self.scatters.set_offsets(current_positions)
        return self.lines + [self.scatters]

    def show(self):
        self.ani = FuncAnimation(
            self.fig, 
            self._update_frame, 
            init_func=self._init_render,
            frames=self.total_frames, 
            interval=int(1000 / self.fps), 
            blit=True
        )
        plt.show()


# =====================================================================
# 5. MAIN SYSTEM ORCHESTRATION PIPELINE
# =====================================================================
if __name__ == "__main__":
    print("Step 1: Setting up modular celestial coordinates...", flush=True)
    universe_sandbox = Universe().generate()
    
    # Build structural sub-systems
    star_a = DenseParticleStarSystem(center=[-2.5, 0.0], drift_velocity=[0.3, -0.05], num_particles=50, radius=0.45).generate()
    star_b = DenseParticleStarSystem(center=[2.5, 0.0], drift_velocity=[-0.3, 0.05], num_particles=50, radius=0.45).generate()
    planets = PlanetSystem(center=[0.0, 0.0], orbit_radius=2.2, num_planets=8, planet_mass=4.0, central_mass_estimate=80.0).generate()
    
    # Merge arrays into master container
    universe_sandbox.append_system(star_a.positions, star_a.velocities, star_a.masses)
    universe_sandbox.append_system(star_b.positions, star_b.velocities, star_b.masses)
    universe_sandbox.append_system(planets.positions, planets.velocities, planets.masses)

    # Extract clean raw configurations safely
    raw_pos = universe_sandbox.positions.copy()
    raw_vel = universe_sandbox.velocities.copy()
    raw_mass = universe_sandbox.masses.copy()
    N_bodies = len(raw_pos)

    print(f"Step 2: JIT Compiling and simulating {N_bodies} bodies matrix...", flush=True)
    
    # Fire the compiler passing standard floating primitives only
    trajectory_data = run_physics_core_engine(
        raw_pos, raw_vel, raw_mass, steps=30000, N=N_bodies, dt=0.0003, G=1.0, eps=0.22
    )

    print("Step 3: Compiling complete! Initializing live canvas layout framework...", flush=True)
    global_visualizer = SimulationVisualizer(trajectory_data, raw_mass, stride=60, tail_length=40)
    global_visualizer.show()

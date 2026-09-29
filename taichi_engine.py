import taichi as ti
import numpy as np

# Initialize Taichi for GPU acceleration (Apple Metal / CUDA)
ti.init(arch=ti.gpu, fast_math=True)

@ti.data_oriented
class TaichiEngine:
    """
    GPU-Accelerated 4th-Order Hermite Integrator using Apple Metal / CUDA via Taichi.
    For N=5000, an O(N^2) direct summation on the GPU is vastly faster than 
    a Barnes-Hut O(N log N) tree on the CPU because the GPU has thousands of cores
    and zero branching overhead.
    """
    def __init__(self, pos, vel, mass, types, G=1.0, eps=1.0, dt=0.001):
        self.N = len(mass)
        self.G = G
        self.eps = eps
        self.dt = dt
        self.step_count = 0
        
        # NumPy interface arrays (used for Pygame rendering)
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.array(vel, dtype=np.float32)
        self.mass = np.array(mass, dtype=np.float32)
        self.types = np.array(types, dtype=np.int32)
        
        self.energy = 0.0
        self.energy_ref = None
        
        # Taichi GPU fields
        self.ti_pos = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_vel = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_mass = ti.field(dtype=ti.f32, shape=self.N)
        
        # Acceleration and Jerk fields
        self.ti_acc = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_jerk = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        
        # Old values for prediction-correction
        self.ti_old_pos = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_old_vel = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_old_acc = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        self.ti_old_jerk = ti.Vector.field(2, dtype=ti.f32, shape=self.N)
        
        # Load data to GPU
        self.ti_pos.from_numpy(self.pos)
        self.ti_vel.from_numpy(self.vel)
        self.ti_mass.from_numpy(self.mass)
        
        # Initial physics state
        self.compute_acc_jerk()
        
    @ti.kernel
    def compute_acc_jerk(self):
        """
        O(N^2) Direct Summation on GPU.
        Computes perfectly exact Gravity and Jerk.
        """
        for i in range(self.N):
            p_i = self.ti_pos[i]
            v_i = self.ti_vel[i]
            
            a_i = ti.Vector([0.0, 0.0])
            j_i = ti.Vector([0.0, 0.0])
            
            for j in range(self.N):
                if i != j:
                    p_j = self.ti_pos[j]
                    v_j = self.ti_vel[j]
                    m_j = self.ti_mass[j]
                    
                    dp = p_j - p_i
                    dv = v_j - v_i
                    
                    r2 = dp.dot(dp) + self.eps * self.eps
                    r = ti.sqrt(r2)
                    r3 = r2 * r
                    
                    inv_r3 = (self.G * m_j) / r3
                    a_i += inv_r3 * dp
                    
                    # Jerk term
                    r_dot_v = dp.dot(dv)
                    term2 = 3.0 * r_dot_v / r2
                    j_i += inv_r3 * (dv - dp * term2)
                    
            self.ti_acc[i] = a_i
            self.ti_jerk[i] = j_i

    @ti.kernel
    def _predict(self, dt: ti.f32):
        dt2 = dt * dt
        dt3 = dt2 * dt
        for i in range(self.N):
            # Save old state
            self.ti_old_pos[i] = self.ti_pos[i]
            self.ti_old_vel[i] = self.ti_vel[i]
            self.ti_old_acc[i] = self.ti_acc[i]
            self.ti_old_jerk[i] = self.ti_jerk[i]
            
            # Predict
            self.ti_pos[i] += self.ti_vel[i] * dt + 0.5 * self.ti_acc[i] * dt2 + (1.0/6.0) * self.ti_jerk[i] * dt3
            self.ti_vel[i] += self.ti_acc[i] * dt + 0.5 * self.ti_jerk[i] * dt2

    @ti.kernel
    def _correct(self, dt: ti.f32):
        dt2 = dt * dt
        for i in range(self.N):
            # Correct velocity
            a0 = self.ti_old_acc[i]
            j0 = self.ti_old_jerk[i]
            a1 = self.ti_acc[i]
            j1 = self.ti_jerk[i]
            
            v0 = self.ti_old_vel[i]
            self.ti_vel[i] = v0 + 0.5 * (a0 + a1) * dt + (1.0 / 12.0) * (j0 - j1) * dt2
            
            # Correct position
            self.ti_pos[i] = self.ti_old_pos[i] + 0.5 * (v0 + self.ti_vel[i]) * dt + (1.0 / 12.0) * (a0 - a1) * dt2
            
    def step(self, n_substeps=1, max_ms=12.0, speed_mult=1.0):
        n_sub = max(1, int(round(n_substeps * float(speed_mult))))
        
        for _ in range(n_sub):
            self._predict(self.dt)
            self.compute_acc_jerk()
            self._correct(self.dt)
            self.step_count += 1
            
        # Download GPU data back to CPU RAM for Pygame to render
        self.pos = self.ti_pos.to_numpy()
        self.vel = self.ti_vel.to_numpy()

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.generator import GalacticDisk, build_universe
from universe.models import ExponentialDisk
from simulation_engine import SimulationEngine

class TestGalaxyDynamics(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.G = 1.0

    def test_1_deterministic_velocities(self):
        """Test 1: Same seed and parameters must reproduce the same initial velocities exactly."""
        g1 = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=np.random.default_rng(42))
        _, vel1, _, _ = g1.generate()
        
        g2 = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=np.random.default_rng(42))
        _, vel2, _, _ = g2.generate()
        
        np.testing.assert_array_equal(vel1, vel2)

    def test_2_circular_speed_consistency(self):
        """Test 2: Compare generated tangential speeds against the target circular-speed model."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=10000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        # Isolate stars
        star_pos = pos[1:]
        star_vel = vel[1:]
        
        r = np.linalg.norm(star_pos, axis=1)
        v_tan = np.linalg.norm(star_vel, axis=1) # Since drift=0 and radial component is small (just dispersion)
        
        # Calculate theoretical circular speed
        # R_d = disk_radius * 0.2
        R_d = 14.0 * 0.2
        x_min = (0.05 * 14.0) / R_d
        x_max = 14.0 / R_d
        
        def exp_disk_cdf(x): return 1.0 - (1.0 + x) * np.exp(-x)
        F0 = exp_disk_cdf(x_min)
        F1 = exp_disk_cdf(x_max)
        
        star_mass_total = (3000.0 * 0.55) * 0.90 # M = 3000. smbh = 0.45. disk = 0.55. stars = 90% of disk
        frac = (exp_disk_cdf(r / R_d) - F0) / (F1 - F0)
        star_M_enc = star_mass_total * np.clip(frac, 0.0, 1.0)
        M_enc = (3000.0 * 0.45) + star_M_enc
        
        # Softened theoretical v_circ
        eps = 1.0
        r2 = r*r + eps*eps
        expected_v_circ = np.sqrt(self.G * M_enc * (r**2) / (r2**1.5))
        
        # Dispersion is added: rng.normal(0, 0.03 * v_circ).
        # We can test the RMS error.
        error = np.abs(v_tan - expected_v_circ) / expected_v_circ
        mean_error = np.mean(error)
        
        # Mean relative error should be close to the dispersion magnitude (0.03 * sqrt(2/pi) = 0.023)
        self.assertLess(mean_error, 0.05, f"Tangential speed deviation too high: {mean_error:.4f}")

    def test_3_and_4_force_balance_and_radial_coverage(self):
        """Test 3 & 4: Force balance test against the actual repository force law (v^2/R vs a_rad) at multiple radii."""
        # Use ExponentialDisk which has a known NFW halo + SMBH + Disk, which exercises more force terms
        g = ExponentialDisk(N=5000, r_scale=2.0, M_central=100.0, M_disk=1000.0, rng=self.rng, types_dist={'types': [1], 'probs': [1.0]})
        pos, vel, mass, types = g.generate()
        
        # ExponentialDisk only generates the disk particles, it assumes the central mass exists.
        # We must add the central mass to the arrays before running the N-body engine!
        pos = np.vstack(([[0.0, 0.0]], pos))
        vel = np.vstack(([[0.0, 0.0]], vel))
        mass = np.concatenate(([100.0], mass))
        types = np.concatenate(([0], types))
        
        # Generate accelerations using the exact SimulationEngine!
        engine = SimulationEngine(pos, vel, mass, types, G=self.G, eps=1.0, dt=0.01)
        engine._init_accelerations()
        
        # Skip the central mass for force evaluation
        r_vec = pos[1:]
        r_mag = np.linalg.norm(r_vec, axis=1)
        
        # Radial inward acceleration magnitude
        r_hat = r_vec / r_mag[:, None]
        a_rad = -np.sum(engine._acc[1:] * r_hat, axis=1)
        
        # Tangential speed v_tan
        v_tan = np.linalg.norm(vel[1:], axis=1)
        
        # Centripetal acceleration from velocity: a_cent = v^2 / r
        a_cent = (v_tan**2) / r_mag
        
        # Compare a_rad and a_cent across different radial bins
        for r_min, r_max, name in [(0, 1.0, "inner"), (1.0, 4.0, "intermediate"), (4.0, 10.0, "outer")]:
            mask = (r_mag >= r_min) & (r_mag < r_max)
            if np.sum(mask) == 0: continue
            
            mean_a_rad = np.mean(a_rad[mask])
            mean_a_cent = np.mean(a_cent[mask])
            
            # Since ExponentialDisk spherical approximation is used, force balance might not be mathematically
            # exact to 1e-6 (because a flat disk exerts a slightly different force field than a sphere),
            # but it should be quantitatively accurate enough (e.g. < 15% error).
            rel_error = abs(mean_a_rad - mean_a_cent) / mean_a_rad
            self.assertLess(rel_error, 0.15, f"{name} disk force balance error too high: {rel_error:.4f}")

    def test_5_boundness(self):
        """Test 5: Measure fraction of initialized particles exceeding local escape speed."""
        g = GalacticDisk(center=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        # Using a simple spherical approximation for escape velocity
        star_pos = pos[1:]
        star_vel = vel[1:]
        r = np.linalg.norm(star_pos, axis=1)
        speed = np.linalg.norm(star_vel, axis=1)
        
        # M_total = 3000
        # v_esc ~ sqrt(2 G M / r)
        v_esc = np.sqrt(2.0 * self.G * 3000.0 / r)
        
        unbound = np.sum(speed > v_esc)
        unbound_fraction = unbound / len(speed)
        
        self.assertLess(unbound_fraction, 0.01, f"Too many unbound particles initialized: {unbound_fraction:.4f}")

    def test_6_velocity_dispersion_sanity(self):
        """Test 6: Measure radial and tangential velocity dispersion relative to circular speed."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=5000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        star_pos = pos[1:]
        star_vel = vel[1:]
        r_mag = np.linalg.norm(star_pos, axis=1)
        r_hat = star_pos / r_mag[:, None]
        
        # Tangential vector (2D cross product equivalent)
        t_hat = np.column_stack((-r_hat[:, 1], r_hat[:, 0]))
        
        v_rad = np.sum(star_vel * r_hat, axis=1)
        v_tan = np.sum(star_vel * t_hat, axis=1)
        
        # Median circular speed
        median_v_circ = np.median(v_tan)
        
        # Velocity dispersion
        sigma_rad = np.std(v_rad)
        
        # GalacticDisk injects sigma = 0.03 * v_circ
        measured_ratio = sigma_rad / median_v_circ
        
        # Should be roughly 0.03
        self.assertTrue(0.01 < measured_ratio < 0.05, f"Velocity dispersion ratio {measured_ratio:.4f} outside bounds")

    def test_7_early_time_structural_stability(self):
        """Test 7: Run a short deterministic simulation to ensure it doesn't immediately collapse or explode."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        r_initial = np.linalg.norm(pos[1:], axis=1)
        median_r_initial = np.median(r_initial)
        
        engine = SimulationEngine(pos, vel, mass, types, dt=0.01)
        engine.precompute(total_steps=50, record_every=50, progress=False)
        
        pos_final = engine.pos
        r_final = np.linalg.norm(pos_final[1:], axis=1)
        median_r_final = np.median(r_final)
        
        # The median radius shouldn't change drastically in 50 steps. 
        # A 15% threshold allows for standard N-body relaxation from the spherical approximation
        rel_change = abs(median_r_final - median_r_initial) / median_r_initial
        self.assertLess(rel_change, 0.15, f"Structure rapidly collapsed/exploded. Rel change: {rel_change:.4f}")

    def test_8_preset_coverage(self):
        """Test 8: Exercise real galaxy presets."""
        for preset in ["twin_galaxies", "messier31"]:
            pos, vel, mass, types = build_universe(preset, N_total=500, seed=self.seed)
            self.assertTrue(len(pos) > 0, f"Preset {preset} generated no particles")
            self.assertFalse(np.any(np.isnan(vel)), f"Preset {preset} generated NaN velocities")

if __name__ == '__main__':
    unittest.main()

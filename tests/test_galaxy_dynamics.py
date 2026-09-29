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

    def _force_balance_validation(self, pos, vel, mass, types, R_d, name_prefix):
        """Helper to compute per-particle force balance error."""
        engine = SimulationEngine(pos, vel, mass, types, G=self.G, eps=1.0, dt=0.01)
        engine._init_accelerations()
        
        r_vec = pos[1:] # Exclude SMBH
        v_vec = vel[1:]
        acc = engine._acc[1:]
        
        r_mag = np.linalg.norm(r_vec, axis=1)
        r_hat = r_vec / r_mag[:, None]
        
        t_hat = np.column_stack((-r_hat[:, 1], r_hat[:, 0]))
        v_rad = np.sum(v_vec * r_hat, axis=1)
        v_tan = np.sum(v_vec * t_hat, axis=1)
        
        a_engine = -np.sum(acc * r_hat, axis=1)
        a_model = (v_tan**2) / r_mag
        
        # relative error
        rel_error = (a_model - a_engine) / a_engine
        abs_rel_error = np.abs(rel_error)
        
        bins = [
            ("inner", r_mag < 1.0),
            ("intermediate", (r_mag >= 1.0) & (r_mag < 4.0)),
            ("outer", r_mag >= 4.0)
        ]
        
        max_errors = {}
        for bin_name, mask in bins:
            if not np.any(mask): continue
            errs = abs_rel_error[mask]
            med = np.median(errs)
            rms = np.sqrt(np.mean(errs**2))
            p95 = np.percentile(errs, 95)
            mx = np.max(errs)
            
            print(f"[{name_prefix}] {bin_name} | Med: {med:.4f} | RMS: {rms:.4f} | P95: {p95:.4f} | Max: {mx:.4f}")
            max_errors[bin_name] = med
            
            # Acceptance criteria for median error: under 15% 
            self.assertLess(med, 0.15, f"{name_prefix} {bin_name} median force balance error too high: {med:.4f}")
            
        return max_errors, v_tan, v_rad, r_mag, a_engine

    def test_2_force_balance_direct(self):
        """Test 2: Direct-force validation (ground truth, N < 600) for GalacticDisk."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=300, n_planets=50, n_comets=50, rng=self.rng)
        pos, vel, mass, types = g.generate()
        self._force_balance_validation(pos, vel, mass, types, R_d=14.0*0.2, name_prefix="Direct_GalacticDisk")

    def test_3_force_balance_bh(self):
        """Test 3: Barnes-Hut validation (N > 600) for GalacticDisk."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=200, n_comets=200, rng=self.rng)
        pos, vel, mass, types = g.generate()
        self._force_balance_validation(pos, vel, mass, types, R_d=14.0*0.2, name_prefix="BH_GalacticDisk")

    def test_4_force_balance_exponential_disk(self):
        """Test 4: Direct-force validation for ExponentialDisk."""
        g = ExponentialDisk(N=400, r_scale=2.0, M_central=100.0, M_disk=1000.0, rng=self.rng, types_dist={'types': [1], 'probs': [1.0]})
        pos, vel, mass, types = g.generate()
        
        # ExponentialDisk generates only disk, we must add the central mass.
        pos = np.vstack(([[0.0, 0.0]], pos))
        vel = np.vstack(([[0.0, 0.0]], vel))
        mass = np.concatenate(([100.0], mass))
        types = np.concatenate(([0], types))
        
        self._force_balance_validation(pos, vel, mass, types, R_d=2.0, name_prefix="Direct_ExponentialDisk")

    def test_5_dispersion_and_boundness(self):
        """Test 5: Velocity dispersion metrics and boundness diagnostic."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=2000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        engine = SimulationEngine(pos, vel, mass, types, G=self.G, eps=1.0, dt=0.01)
        engine._init_accelerations()
        
        # Using exact engine potential energy would be ideal, but engine._acc provides forces.
        # We can approximate boundness using the a_rad * r as a potential surrogate (virial).
        # v_esc = sqrt(2 * a_rad * r)
        r_vec = pos[1:]
        v_vec = vel[1:]
        acc = engine._acc[1:]
        
        r_mag = np.linalg.norm(r_vec, axis=1)
        r_hat = r_vec / r_mag[:, None]
        t_hat = np.column_stack((-r_hat[:, 1], r_hat[:, 0]))
        
        a_rad = -np.sum(acc * r_hat, axis=1)
        v_rad = np.sum(v_vec * r_hat, axis=1)
        v_tan = np.sum(v_vec * t_hat, axis=1)
        
        sigma_rad = np.std(v_rad)
        sigma_phi = np.std(v_tan) # this is dispersion around the mean rotation
        v_circ = np.median(np.abs(v_tan))
        
        # Dispersion is injected as 3% of v_circ.
        ratio_rad = sigma_rad / v_circ
        print(f"[Dispersion] sigma_R / v_circ: {ratio_rad:.4f}, sigma_phi / v_circ: {sigma_phi/v_circ:.4f}")
        self.assertTrue(0.01 < ratio_rad < 0.05)
        
        # Numerical escape diagnostic based on actual softened forces
        # Potential roughly ~ a_rad * r (for a 1/r potential, this is exact. For softened it's an approx).
        # So v_esc ~ sqrt(2 * a_rad * r)
        v_esc_approx = np.sqrt(2.0 * a_rad * r_mag)
        speed = np.linalg.norm(v_vec, axis=1)
        
        unbound = np.sum(speed > v_esc_approx)
        unbound_fraction = unbound / len(speed)
        print(f"[Boundness] Diagnostic unbound fraction: {unbound_fraction:.4f}")
        self.assertLess(unbound_fraction, 0.05)

    def test_6_short_run_structural_stability(self):
        """Test 6: Short deterministic N-body simulation measuring quantiles."""
        g = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        engine = SimulationEngine(pos, vel, mass, types, dt=0.01)
        
        def record_quantiles():
            r = np.linalg.norm(engine.pos[1:], axis=1)
            return np.percentile(r, [10, 50, 90])
            
        initial_q = record_quantiles()
        
        engine.precompute(total_steps=25, record_every=25, progress=False)
        mid_q = record_quantiles()
        
        engine.precompute(total_steps=25, record_every=25, progress=False)
        final_q = record_quantiles()
        
        print(f"[Stability] Initial Quantiles: {initial_q}")
        print(f"[Stability] Mid Quantiles: {mid_q}")
        print(f"[Stability] Final Quantiles: {final_q}")
        
        rel_change_median = abs(final_q[1] - initial_q[1]) / initial_q[1]
        self.assertLess(rel_change_median, 0.15)

if __name__ == '__main__':
    unittest.main()

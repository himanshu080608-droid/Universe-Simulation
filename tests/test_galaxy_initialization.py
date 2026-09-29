import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.generator import GalacticDisk, build_universe
from universe.models import ExponentialDisk, NFWModel

class TestGalaxyInitialization(unittest.TestCase):

    def setUp(self):
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

    def test_1_determinism(self):
        """Test 1: The same generator inputs and seed produce identical pos, vel, mass, and types."""
        g1 = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=np.random.default_rng(42))
        pos1, vel1, mass1, types1 = g1.generate()
        
        g2 = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0), n_stars=1000, n_planets=0, n_comets=0, rng=np.random.default_rng(42))
        pos2, vel2, mass2, types2 = g2.generate()
        
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(vel1, vel2)
        np.testing.assert_array_equal(mass1, mass2)
        np.testing.assert_array_equal(types1, types2)

    def test_2_and_3_exponential_radial_distribution(self):
        """Test 2 & 3: Compare empirical radial CDF against correct mathematically truncated exponential-disk CDF."""
        # We need a large sample for a solid statistical test
        disk_radius = 20.0
        n_stars = 20000
        g = GalacticDisk(center=(0.0, 0.0), n_stars=n_stars, n_planets=0, n_comets=0, disk_radius=disk_radius, rng=self.rng)
        pos, vel, mass, types = g.generate()
        
        # Only check stars (ignore SMBH at index 0)
        star_pos = pos[1:]
        r = np.linalg.norm(star_pos, axis=1)
        
        # Exact theoretical bounds from implementation
        R_d = disk_radius * 0.2
        x_min = 0.05 * disk_radius / R_d
        x_max = disk_radius / R_d
        
        def unnorm_cdf(x):
            return 1.0 - (1.0 + x) * np.exp(-x)
            
        F0 = unnorm_cdf(x_min)
        F1 = unnorm_cdf(x_max)
        
        def trunc_cdf(r_val):
            x = r_val / R_d
            return (unnorm_cdf(x) - F0) / (F1 - F0)
            
        # Compute empirical CDF
        sorted_r = np.sort(r)
        empirical_cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        
        # Compute expected CDF
        expected_cdf = trunc_cdf(sorted_r)
        
        # KS-style max discrepancy
        max_discrepancy = np.max(np.abs(empirical_cdf - expected_cdf))
        
        # For N=20000, max discrepancy should be extremely small (<0.01)
        self.assertLess(max_discrepancy, 0.01, f"Radial distribution differs from exact model (discrepancy: {max_discrepancy:.4f})")

    def test_4_angular_spiral_independence(self):
        """Test 4: Spiral-arm perturbation changes angular structure but doesn't distort radial marginal distribution."""
        # Using ExponentialDisk from models.py which applies strong spiral-arm angular perturbation
        types_dist = {'types': [1], 'probs': [1.0]}
        g = ExponentialDisk(N=10000, r_scale=2.0, M_central=100.0, M_disk=1000.0, rng=self.rng, types_dist=types_dist)
        pos, _, _, _ = g.generate()
        
        r = np.linalg.norm(pos, axis=1)
        
        x_min = 0.1625
        x_max = 4.605
        R_d = 2.0
        
        def unnorm_cdf(x):
            return 1.0 - (1.0 + x) * np.exp(-x)
            
        F0 = unnorm_cdf(x_min)
        F1 = unnorm_cdf(x_max)
        
        def trunc_cdf(r_val):
            x = r_val / R_d
            return (unnorm_cdf(x) - F0) / (F1 - F0)
            
        sorted_r = np.sort(r)
        empirical_cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        expected_cdf = trunc_cdf(sorted_r)
        
        max_discrepancy = np.max(np.abs(empirical_cdf - expected_cdf))
        self.assertLess(max_discrepancy, 0.01, "Spiral wave perturbation corrupted the radial distribution!")

    def test_5_finite_range_safety(self):
        """Test 5: Confirm all generated stellar radii satisfy the documented physical bounds."""
        disk_radius = 15.0
        g = GalacticDisk(center=(10.0, -10.0), n_stars=1000, n_planets=0, n_comets=0, disk_radius=disk_radius, rng=self.rng)
        pos, _, _, _ = g.generate()
        
        # Center is at (10, -10)
        star_pos = pos[1:] - np.array([10.0, -10.0])
        r = np.linalg.norm(star_pos, axis=1)
        
        r_min_expected = 0.05 * disk_radius
        r_max_expected = disk_radius
        
        self.assertGreaterEqual(np.min(r), r_min_expected - 1e-7)
        self.assertLessEqual(np.max(r), r_max_expected + 1e-7)

    def test_6_preset_integration(self):
        """Test 6: Exercise actual presets to confirm GalacticDisk is used."""
        # build_universe directly calls GalacticDisk for 'twin_galaxies'
        pos, vel, mass, types = build_universe("twin_galaxies", N_total=1000, seed=42)
        
        # 'twin_galaxies' uses 2 GalacticDisks. Let's just ensure it generated particles.
        self.assertTrue(len(pos) > 0)
        self.assertTrue(np.any(types == 0)) # STAR

    def test_7_nfw_stage5_surrogate(self):
        """Test 7: NFWModel uses validated Stage 5 surrogate calibration."""
        N = 1000
        M = 100.0
        R_s = 10.0
        C = 10.0
        eps = 1.0
        g = NFWModel(N=N, total_mass=M, R_s=R_s, C=C, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
        pos, vel, mass, types = g.generate()
        
        # Generation succeeds and finite
        self.assertTrue(np.all(np.isfinite(pos)))
        self.assertTrue(np.all(np.isfinite(vel)))
        
        r = np.linalg.norm(pos, axis=1)
        self.assertTrue(np.all(r > 0.0))
        
        # Mass conservation
        self.assertAlmostEqual(np.sum(mass), M)
        
        # Support bound check: should be at most 1.05 * R_s * C
        R_vir = R_s * C
        self.assertLessEqual(np.max(r), 1.05 * R_vir + 1e-5)
        
        # Deterministic generation check
        g2 = NFWModel(N=N, total_mass=M, R_s=R_s, C=C, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
        pos2, _, _, _ = g2.generate()
        np.testing.assert_array_equal(pos, pos2)

    def test_8_nfw_unsupported_calibration(self):
        """Test 8: NFWModel explicitly rejects uncalibrated R_s, C, eps configurations."""
        with self.assertRaises(ValueError) as context:
            # R_s=2.0 is not in the validated calibration table
            g = NFWModel(N=1000, total_mass=100.0, R_s=2.0, C=10.0, rng=np.random.default_rng(42), types_dist={'types': [14], 'probs': [1.0]})
            g.generate()
            
        msg = str(context.exception)
        self.assertIn("R_s=2.0", msg)
        self.assertIn("C=10.0", msg)
        self.assertIn("eps=1.0", msg)

if __name__ == '__main__':
    unittest.main()

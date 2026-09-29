"""
tests/test_plummer_jeans.py

Regression tests for the Phase 6.1 analytic Jeans velocity calibration
in PlummerModel._jeans_sigma1d().

Tested invariants (all mathematically established, none hard-coding equilibrium
dispersion into the production test yet):
  1. Deterministic output for fixed seed.
  2. All positions and velocities finite.
  3. Exact mass conservation.
  4. sigma_1d > 0 everywhere in the production radial support.
  5. Analytic Jeans profile at a=10, M=1000, eps=1 matches the Phase 6.1
     numerical quadrature reference within 4 % at R/a = 0.5, 1, 2, 5, 10.
  6. Monotonically reasonable: sigma decreases with radius at large R.
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.models import PlummerModel

# ── helpers ──────────────────────────────────────────────────────────────────
def _make_model(N=500, a=10.0, M=1000.0, seed=42, M_central=0.0):
    rng = np.random.default_rng(seed)
    td  = {'types': [0], 'probs': [1.0]}
    return PlummerModel(N=N, a_scale=a, total_mass=M, rng=rng,
                        types_dist=td, M_central=M_central)


class TestPlummerJeansSigma(unittest.TestCase):
    """Tests for _jeans_sigma1d() in isolation."""

    def test_sigma_positive(self):
        """sigma_1d >= 0 everywhere; > 0 strictly inside the production support."""
        m = _make_model()
        a   = m.a
        # Use u strictly inside [0.001, 0.94] — excludes the R=R_max boundary where
        # the zero-pressure BC gives sigma=0 by construction.
        u   = np.linspace(0.001, 0.94, 200)
        r   = a * np.sqrt(1.0 / (1.0 - u)**2 - 1.0)
        sig = m._jeans_sigma1d(r)
        self.assertTrue(np.all(sig >= 0),
                        f"sigma_1d has negative values: min={sig.min():.4e}")
        self.assertTrue(np.all(sig > 0),
                        f"sigma_1d has non-positive values inside support: min={sig.min():.4e}")

    def test_sigma_finite(self):
        """sigma_1d is finite everywhere in production radial support."""
        m = _make_model()
        a   = m.a
        u   = np.linspace(0.001, 0.95, 200)
        r   = a * np.sqrt(1.0 / (1.0 - u)**2 - 1.0)
        sig = m._jeans_sigma1d(r)
        self.assertFalse(np.any(np.isnan(sig)), "sigma_1d contains NaN")
        self.assertFalse(np.any(np.isinf(sig)), "sigma_1d contains Inf")

    def test_sigma_with_central_mass(self):
        """sigma_1d >= 0, and > 0 strictly inside support, with M_c=50 and M_c=200."""
        for M_c in (50.0, 200.0):
            with self.subTest(M_central=M_c):
                m = _make_model(M_central=M_c)
                a = m.a
                # Exclude R=R_max boundary (sigma=0 by zero-pressure BC)
                u = np.linspace(0.001, 0.94, 100)
                r = a * np.sqrt(1.0 / (1.0 - u)**2 - 1.0)
                sig = m._jeans_sigma1d(r)
                self.assertTrue(np.all(sig >= 0))
                self.assertTrue(np.all(sig > 0))
                self.assertFalse(np.any(np.isnan(sig)))

    def test_phase61_reference(self):
        """
        Analytic Jeans profile vs Phase 6.1 numerical quadrature reference.

        Reference: tests/analyze_stage6p1_plummer_jeans.py, HEAD a717856.
        Parameters: a=10, M=1000, M_c=0, eps=1, G=1.
        Acceptance criterion: |sigma_analytic / sigma_ref - 1| < 4 %.
        """
        # Phase 6.1 quadrature Jeans reference (160×320 GL, split-at-probe)
        # Columns: R,  sigma_Jeans
        phase61_ref = {
             5.0: 4.4775,   # R/a = 0.5
            10.0: 4.0986,   # R/a = 1.0
            20.0: 3.3616,   # R/a = 2.0
            50.0: 2.2604,   # R/a = 5.0
           100.0: 1.5655,   # R/a = 10.0
        }
        tol = 0.04          # 4 %

        m   = _make_model(a=10.0, M=1000.0, M_central=0.0)
        r_probe = np.array(sorted(phase61_ref.keys()))
        sig_analytic = m._jeans_sigma1d(r_probe)

        for i, R in enumerate(r_probe):
            sig_ref  = phase61_ref[R]
            sig_got  = sig_analytic[i]
            frac_err = abs(sig_got / sig_ref - 1.0)
            self.assertLess(
                frac_err, tol,
                f"R/a={R/10:.1f}: sigma_analytic={sig_got:.4f} vs ref={sig_ref:.4f}"
                f" → {frac_err*100:.2f}% > {tol*100:.0f}%"
            )

    def test_sigma_decreases_at_large_r(self):
        """sigma_1d is monotonically decreasing for R > a (outer pressure-supported region)."""
        m = _make_model()
        a = m.a
        r_outer = np.logspace(np.log10(a), np.log10(a * 15), 50)
        sig = m._jeans_sigma1d(r_outer)
        # Allow a small tolerance for near-flat regions
        self.assertTrue(
            np.all(np.diff(sig) < 0.01 * sig[:-1]),
            "sigma_1d is not monotonically decreasing for R > a"
        )


class TestPlummerJeansGenerate(unittest.TestCase):
    """End-to-end tests for PlummerModel.generate() with the new sigma formula."""

    def test_generate_deterministic(self):
        """Identical seed produces identical (pos, vel, masses, types)."""
        for seed in (42, 99):
            with self.subTest(seed=seed):
                pos1, vel1, m1, t1 = _make_model(seed=seed).generate()
                pos2, vel2, m2, t2 = _make_model(seed=seed).generate()
                np.testing.assert_array_equal(pos1, pos2)
                np.testing.assert_array_equal(vel1, vel2)
                np.testing.assert_array_equal(m1,   m2)
                np.testing.assert_array_equal(t1,   t2)

    def test_generate_finite(self):
        """All positions and velocities are finite."""
        pos, vel, masses, _ = _make_model().generate()
        self.assertFalse(np.any(np.isnan(pos)),  "pos contains NaN")
        self.assertFalse(np.any(np.isinf(pos)),  "pos contains Inf")
        self.assertFalse(np.any(np.isnan(vel)),  "vel contains NaN")
        self.assertFalse(np.any(np.isinf(vel)),  "vel contains Inf")

    def test_generate_mass_conservation(self):
        """Total mass equals specified M within floating-point tolerance."""
        M = 1000.0
        _, _, masses, _ = _make_model(M=M).generate()
        np.testing.assert_allclose(
            masses.sum(), M, rtol=1e-10,
            err_msg="Total mass not conserved"
        )

    def test_generate_velocities_positive_rms(self):
        """RMS velocity > 0 (sigma actually applied)."""
        _, vel, _, _ = _make_model(N=1000).generate()
        rms = np.sqrt(np.mean(vel**2))
        self.assertGreater(rms, 0.0, "RMS velocity is zero — sigma not applied")

    def test_generate_with_central_mass(self):
        """generate() with M_central > 0 produces finite outputs."""
        for M_c in (50.0, 200.0):
            with self.subTest(M_central=M_c):
                pos, vel, masses, _ = _make_model(M_central=M_c).generate()
                self.assertFalse(np.any(np.isnan(vel)))
                self.assertFalse(np.any(np.isinf(vel)))

    def test_generate_support_bounds(self):
        """All particles lie within the production radial support [R_min, R_max]."""
        m = _make_model(N=2000)
        pos, _, _, _ = m.generate()
        r = np.hypot(pos[:, 0], pos[:, 1])
        a = m.a
        # Gas flag is off; R_max from u_hi=0.95 (add 1% margin for float precision)
        R_max = a * np.sqrt(1.0 / (1.0 - 0.95)**2 - 1.0) * 1.01
        self.assertTrue(np.all(r > 0),     "Some particles have r=0")
        self.assertTrue(np.all(r < R_max), f"Particle outside R_max: max_r={r.max():.2f} R_max={R_max:.2f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

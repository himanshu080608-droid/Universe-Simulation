"""
tests/test_advanced_engine_state.py

Regression test for the AdvancedTaichiEngine state-contract fix:
    sync_active_to_pred() must be called after active_force_and_correct()
    so that engine.pos / engine.vel reflect the corrected physical state,
    not the pre-correction Taylor predictor.

Tests:
  1. Deterministic: two identical runs from same ICs produce identical pos/vel.
  2. State contract: engine.pos matches ti_pos (corrected GPU field) for every
     particle whose block-interval divides sys_step (active particles).
     Before the fix, active particles in engine.pos would differ from ti_pos.
  3. Finite: no NaN / Inf in pos, vel, or energy after one step.
  4. Energy: engine.energy is finite and non-zero after one step.

The test uses a 2-body softened circular orbit as a deterministic, minimal IC.
Taichi initializes GPU on first import; the test is skipped if taichi is
unavailable or fails to initialize (e.g., CI without GPU).
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── graceful skip if taichi / GPU not available ───────────────────────────────
_TAICHI_OK = False
_SKIP_MSG  = ""
try:
    import taichi as ti
    # Try CPU fallback so tests run in headless/CI environments too
    try:
        ti.init(arch=ti.gpu, fast_math=True, log_level=ti.WARN)
    except Exception:
        ti.init(arch=ti.cpu, fast_math=True, log_level=ti.WARN)
    from advanced_engine import AdvancedTaichiEngine
    _TAICHI_OK = True
except Exception as exc:
    _SKIP_MSG = f"Taichi/GPU unavailable: {exc}"


# ── 2-body softened circular orbit IC ────────────────────────────────────────
def _two_body_ic(eps=1.0, G=1.0):
    """
    Two equal masses m=1 in a softened circular binary.
    Separation d=4.0, omega=sqrt(G*(2m)/(d^2+eps^2)^1.5).
    Returns (pos, vel, mass, types) as float32-compatible arrays.
    """
    m     = 1.0
    d     = 4.0
    omega = np.sqrt(G * 2.0 * m / (d**2 + eps**2)**1.5)
    r     = d / 2.0        # each body's CoM radius
    v     = omega * r      # each body's speed

    pos   = np.array([[-r, 0.0], [ r, 0.0]], dtype=np.float32)
    vel   = np.array([[ 0.0, -v], [0.0,  v]], dtype=np.float32)
    mass  = np.array([m, m],                  dtype=np.float32)
    types = np.array([0, 0],                  dtype=np.int32)
    return pos, vel, mass, types


@unittest.skipUnless(_TAICHI_OK, _SKIP_MSG)
class TestAdvancedEngineStateContract(unittest.TestCase):
    """Verify pos/vel export reflects corrected GPU state after each step."""

    def _make_engine(self):
        pos, vel, mass, types = _two_body_ic()
        return AdvancedTaichiEngine(pos, vel, mass, types, G=1.0, eps=1.0, dt=0.001)

    def test_finite_after_one_step(self):
        """pos, vel, energy must all be finite after one global step."""
        eng = self._make_engine()
        eng.step()
        self.assertFalse(np.any(np.isnan(eng.pos)), "pos contains NaN")
        self.assertFalse(np.any(np.isinf(eng.pos)), "pos contains Inf")
        self.assertFalse(np.any(np.isnan(eng.vel)), "vel contains NaN")
        self.assertFalse(np.any(np.isinf(eng.vel)), "vel contains Inf")
        self.assertTrue(np.isfinite(eng.energy),    f"energy={eng.energy} not finite")

    def test_energy_nonzero(self):
        """Energy must be non-zero (particles have KE + PE)."""
        eng = self._make_engine()
        eng.step()
        self.assertNotEqual(eng.energy, 0.0, "energy is zero after one step")

    def test_state_contract_pos(self):
        """
        Core fix test: engine.pos must match ti_pos for every active particle.

        For sys_step=1 and max_mult=32, a particle is active when
        sys_step % interval == 0.  With dt=0.001 and Aarseth criterion,
        the 2-body system should assign interval=1 (smallest dt, close orbit),
        so ALL particles are active every step.

        If sync_active_to_pred() were missing, engine.pos would contain the
        Taylor-predictor values, which differ from ti_pos by the Hermite correction.
        """
        eng  = self._make_engine()
        eng.step()

        # Read corrected GPU state directly
        corrected_pos = eng.ti_pos.to_numpy()   # shape (N, 2)
        exported_pos  = eng.pos                  # what the renderer sees

        intervals = eng.ti_step_interval.to_numpy()
        sys_step  = eng.sys_step

        for i in range(eng.N):
            if sys_step % intervals[i] == 0:
                # This particle was active; exported must equal corrected
                np.testing.assert_allclose(
                    exported_pos[i], corrected_pos[i], atol=1e-6,
                    err_msg=(f"Particle {i} (active): "
                             f"engine.pos={exported_pos[i]} "
                             f"ti_pos={corrected_pos[i]} differ — "
                             f"state-contract violation")
                )

    def test_state_contract_vel(self):
        """Same state-contract check for velocities."""
        eng  = self._make_engine()
        eng.step()

        corrected_vel = eng.ti_vel.to_numpy()
        exported_vel  = eng.vel

        intervals = eng.ti_step_interval.to_numpy()
        sys_step  = eng.sys_step

        for i in range(eng.N):
            if sys_step % intervals[i] == 0:
                np.testing.assert_allclose(
                    exported_vel[i], corrected_vel[i], atol=1e-6,
                    err_msg=(f"Particle {i} (active): "
                             f"engine.vel={exported_vel[i]} "
                             f"ti_vel={corrected_vel[i]} differ — "
                             f"state-contract violation")
                )

    def test_deterministic(self):
        """
        Two identical runs from the same IC produce byte-identical pos/vel
        (Taichi GPU kernels must be deterministic for same inputs).
        """
        def run():
            eng = self._make_engine()
            eng.step()
            eng.step()
            return eng.pos.copy(), eng.vel.copy()

        p1, v1 = run()
        p2, v2 = run()

        np.testing.assert_array_equal(p1, p2, err_msg="pos not deterministic")
        np.testing.assert_array_equal(v1, v2, err_msg="vel not deterministic")

    def test_step_count_increments(self):
        """step_count increments by n_sub per step() call."""
        eng = self._make_engine()
        self.assertEqual(eng.step_count, 0)
        eng.step()
        self.assertEqual(eng.step_count, 1)
        eng.step(n_substeps=3)
        self.assertEqual(eng.step_count, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

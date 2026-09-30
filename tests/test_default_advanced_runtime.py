"""
tests/test_default_advanced_runtime.py

Final default-engine runtime validation phase.
Validates the actual execution path of AdvancedTaichiEngine via main.main().
"""

import sys
import os
import unittest
import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import renderer.pygame_renderer
RealPygameRenderer = renderer.pygame_renderer.PygameRenderer
import main
from universe.generator import build_universe

render_frame_calls = 0
captured_engine = None

class HeadlessTestRenderer(RealPygameRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = 0

    def _handle_events(self, pos):
        if self.frames >= 3:
            return False
        return super()._handle_events(pos)

    def render_frame(self, *args, **kwargs):
        global render_frame_calls
        render_frame_calls += 1
        self.frames += 1
        return super().render_frame(*args, **kwargs)


class TestDefaultAdvancedRuntime(unittest.TestCase):
    def setUp(self):
        global render_frame_calls, captured_engine
        render_frame_calls = 0
        captured_engine = None
        self.original_renderer = renderer.pygame_renderer.PygameRenderer
        renderer.pygame_renderer.PygameRenderer = HeadlessTestRenderer
        
    def tearDown(self):
        renderer.pygame_renderer.PygameRenderer = self.original_renderer
        
    def _run_main_headless(self, seed="42"):
        old_argv = sys.argv
        sys.argv = [
            "main.py",
            "--preset", "laplace",
            "--N", "64",
            "--seed", str(seed),
            "--mode", "pygame",
            "--spf", "1",
            "--fps", "60",
            "--width", "320",
            "--height", "240"
        ]
        original_run_pygame = main.run_pygame
        
        def run_pygame_wrapper(engine, args):
            global captured_engine
            captured_engine = engine
            try:
                original_run_pygame(engine, args)
            finally:
                pass
                
        main.run_pygame = run_pygame_wrapper
        
        try:
            main.main()
        except SystemExit as e:
            self.assertEqual(e.code, 0, f"SystemExit with nonzero code: {e.code}")
        finally:
            main.run_pygame = original_run_pygame
            sys.argv = old_argv

    def test_default_advanced_runtime(self):
        global render_frame_calls, captured_engine
        
        # Get baseline initial conditions
        pos_initial, vel_initial, _, _ = build_universe("laplace", N_total=64, seed=42)
        pos_initial = pos_initial.astype(np.float32)
        vel_initial = vel_initial.astype(np.float32)
        
        # First Run
        self._run_main_headless(seed="42")
        
        engine1 = captured_engine
        calls1 = render_frame_calls
        
        # Validations
        self.assertIsNotNone(engine1, "Engine was not captured")
        self.assertEqual(type(engine1).__name__, "AdvancedTaichiEngine")
        
        self.assertEqual(calls1, 3, f"Expected 3 render_frame calls, got {calls1}")
        self.assertEqual(engine1.step_count, 3)
        self.assertEqual(engine1.sys_step, 3)
        
        N = engine1.N
        self.assertEqual(engine1.pos.shape, (N, 2))
        self.assertEqual(engine1.vel.shape, (N, 2))
        self.assertEqual(engine1.mass.shape, (N,))
        self.assertEqual(engine1.types.shape, (N,))
        
        self.assertEqual(N, len(engine1.mass))
        self.assertEqual(N, len(engine1.types))
        
        self.assertTrue(np.all(np.isfinite(engine1.pos)), "pos contains NaN/Inf")
        self.assertTrue(np.all(np.isfinite(engine1.vel)), "vel contains NaN/Inf")
        self.assertTrue(np.all(np.isfinite(engine1.mass)), "mass contains NaN/Inf")
        self.assertTrue(np.isfinite(engine1.energy), "energy is not finite")
        
        self.assertTrue(np.all(engine1.mass > 0), "non-positive mass found")
        self.assertGreater(N, 0, "engine state is empty")
        
        self.assertFalse(np.array_equal(pos_initial, engine1.pos), "pos did not change from IC")
        self.assertFalse(np.array_equal(vel_initial, engine1.vel), "vel did not change from IC")
        
        pos_run1 = engine1.pos.copy()
        vel_run1 = engine1.vel.copy()
        
        # Second Run for determinism
        render_frame_calls = 0
        captured_engine = None
        
        self._run_main_headless(seed="42")
        
        engine2 = captured_engine
        calls2 = render_frame_calls
        
        self.assertIsNotNone(engine2, "Engine was not captured in run 2")
        self.assertEqual(calls2, 3)
        self.assertEqual(engine2.step_count, 3)
        
        self.assertTrue(np.array_equal(pos_run1, engine2.pos), "deterministic repeat differs in pos")
        self.assertTrue(np.array_equal(vel_run1, engine2.vel), "deterministic repeat differs in vel")

if __name__ == "__main__":
    unittest.main(verbosity=2)

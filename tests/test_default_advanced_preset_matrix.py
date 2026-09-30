"""
tests/test_default_advanced_preset_matrix.py

Headless validation matrix for the Pygame runtime on AdvancedTaichiEngine,
testing multiple representative generator presets to ensure compatibility,
finiteness, determinism, and state advancement.
"""
import sys
import os
import time
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
captured_renderer = None
captured_trail_buffers = []
captured_surface_sizes = []
render_times = []
engine_step_times = []
frame_times = []

class HeadlessTestRenderer(RealPygameRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = 0
        global captured_renderer
        captured_renderer = self

    def _handle_events(self, pos):
        if self.frames >= 3:
            return False
        return super()._handle_events(pos)

    def render_frame(self, *args, **kwargs):
        global render_frame_calls, captured_engine
        
        t0 = time.perf_counter()
        ret = super().render_frame(*args, **kwargs)
        render_times.append(time.perf_counter() - t0)
        
        self.frames += 1
        render_frame_calls += 1
        
        if hasattr(self, 'trail_buffer'):
            captured_trail_buffers.append(self.trail_buffer.copy())
            
        if hasattr(self, 'screen'):
            captured_surface_sizes.append(self.screen.get_size())
            
        return ret


class TestPresetMatrix(unittest.TestCase):
    def setUp(self):
        self.original_renderer = renderer.pygame_renderer.PygameRenderer
        renderer.pygame_renderer.PygameRenderer = HeadlessTestRenderer
        
    def tearDown(self):
        renderer.pygame_renderer.PygameRenderer = self.original_renderer
        
    def _reset_globals(self):
        global render_frame_calls, captured_engine, captured_renderer
        global captured_trail_buffers, captured_surface_sizes
        global render_times, engine_step_times, frame_times
        
        render_frame_calls = 0
        captured_engine = None
        captured_renderer = None
        captured_trail_buffers.clear()
        captured_surface_sizes.clear()
        render_times.clear()
        engine_step_times.clear()
        frame_times.clear()

    def _run_headless(self, preset):
        old_argv = sys.argv
        sys.argv = [
            "main.py",
            "--preset", preset,
            "--N", "2000",
            "--seed", "42",
            "--mode", "pygame",
            "--spf", "1",
            "--fps", "60",
            "--width", "640",
            "--height", "480"
        ]
        
        original_run_pygame = main.run_pygame
        
        def run_pygame_wrapper(engine, args):
            global captured_engine
            captured_engine = engine
            
            original_step = engine.step
            def timed_step(*s_args, **s_kwargs):
                t0 = time.perf_counter()
                res = original_step(*s_args, **s_kwargs)
                engine_step_times.append(time.perf_counter() - t0)
                return res
            engine.step = timed_step
            
            t_run_start = time.perf_counter()
            try:
                original_run_pygame(engine, args)
            finally:
                frame_times.append(time.perf_counter() - t_run_start)
                
        main.run_pygame = run_pygame_wrapper
        
        try:
            main.main()
        except SystemExit as e:
            self.assertEqual(e.code, 0, f"SystemExit with nonzero code: {e.code}")
        finally:
            main.run_pygame = original_run_pygame
            sys.argv = old_argv

    def test_preset_matrix(self):
        presets = [
            "solar_system",
            "alpha_centauri",
            "twin_galaxies",
            "messier13",
            "messier31",
            "castor_sextuple",
            "omega_centauri",
            "great_attractor"
        ]
        
        print("\n" + "="*85)
        print(f"{'PRESET':<18} | {'ACTUAL N':>8} | {'FINAL ENERGY':>14} | {'DETR?':>5} | {'ADV?':>4} | {'RNDR?':>5}")
        print("-" * 85)
        
        for preset in presets:
            with self.subTest(preset=preset):
                # ── RUN 1 ──
                self._reset_globals()
                
                pos_initial, vel_initial, _, _ = build_universe(preset, N_total=2000, seed=42)
                pos_initial = pos_initial.astype(np.float32)
                vel_initial = vel_initial.astype(np.float32)
                
                self._run_headless(preset)
                
                engine1 = captured_engine
                renderer1 = captured_renderer
                
                self.assertIsNotNone(engine1, f"{preset}: Engine not captured")
                self.assertIsNotNone(renderer1, f"{preset}: Renderer not captured")
                self.assertEqual(type(engine1).__name__, "AdvancedTaichiEngine")
                
                N = engine1.N
                self.assertGreater(N, 0, f"{preset}: N must be > 0")
                self.assertEqual(engine1.step_count, 3)
                self.assertEqual(engine1.sys_step, 3)
                
                self.assertEqual(engine1.pos.shape, (N, 2))
                self.assertEqual(engine1.vel.shape, (N, 2))
                self.assertEqual(engine1.mass.shape, (N,))
                self.assertEqual(engine1.types.shape, (N,))
                
                self.assertTrue(np.all(np.isfinite(engine1.pos)), f"{preset}: pos non-finite")
                self.assertTrue(np.all(np.isfinite(engine1.vel)), f"{preset}: vel non-finite")
                self.assertTrue(np.all(np.isfinite(engine1.mass)), f"{preset}: mass non-finite")
                self.assertTrue(np.isfinite(engine1.energy), f"{preset}: energy non-finite")
                self.assertTrue(np.all(engine1.mass > 0), f"{preset}: mass must be strictly positive")
                
                self.assertEqual(render_frame_calls, 3)
                
                self.assertEqual(len(captured_trail_buffers), 3)
                for tb in captured_trail_buffers:
                    self.assertTrue(np.all(np.isfinite(tb)), f"{preset}: trail buffer non-finite")
                    self.assertTrue(np.all(tb >= -1.0), f"{preset}: trail buffer negative")
                    self.assertTrue(np.all(tb <= 1e6), f"{preset}: trail buffer too large")
                
                self.assertEqual(len(captured_surface_sizes), 3)
                for sz in captured_surface_sizes:
                    self.assertEqual(sz, (640, 480))
                    
                advanced = not (np.array_equal(pos_initial, engine1.pos) and np.array_equal(vel_initial, engine1.vel))
                self.assertTrue(advanced, f"{preset}: state did not change from initial")
                
                pos_run1 = engine1.pos.copy()
                vel_run1 = engine1.vel.copy()
                energy_run1 = engine1.energy
                
                # ── RUN 2 (Determinism) ──
                self._reset_globals()
                self._run_headless(preset)
                
                engine2 = captured_engine
                
                self.assertEqual(render_frame_calls, 3)
                self.assertTrue(np.array_equal(pos_run1, engine2.pos), f"{preset}: pos determinism failed")
                self.assertTrue(np.array_equal(vel_run1, engine2.vel), f"{preset}: vel determinism failed")
                
                print(f"{preset:<18} | {N:>8} | {energy_run1:>14.4e} | {'YES':>5} | {'YES':>4} | {'YES':>5}")

        print("="*85 + "\n")

if __name__ == "__main__":
    unittest.main(verbosity=2)

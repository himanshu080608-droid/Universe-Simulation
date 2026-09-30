"""
tests/test_default_advanced_pygame_production.py

Pygame-only phase validating the actual Pygame renderer loop
working with the AdvancedTaichiEngine production runtime.
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

# Global test state
render_times = []
engine_step_times = []
frame_times = []

render_frame_calls = 0
captured_engine = None
captured_renderer = None
captured_trail_buffers = []
captured_surface_sizes = []

class HeadlessTestRenderer(RealPygameRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = 0
        global captured_renderer
        captured_renderer = self

    def _handle_events(self, pos):
        if self.frames >= 10:
            return False
        return super()._handle_events(pos)

    def render_frame(self, *args, **kwargs):
        global render_frame_calls, captured_engine
        
        t0 = time.perf_counter()
        ret = super().render_frame(*args, **kwargs)
        render_times.append(time.perf_counter() - t0)
        
        self.frames += 1
        render_frame_calls += 1
        
        # Capture renderer buffer stats for this frame
        if hasattr(self, 'trail_buffer'):
            captured_trail_buffers.append(self.trail_buffer.copy())
            
        if hasattr(self, 'screen'):
            captured_surface_sizes.append(self.screen.get_size())
            
        return ret


class TestDefaultAdvancedPygameProduction(unittest.TestCase):
    def setUp(self):
        global render_frame_calls, captured_engine, captured_renderer
        global render_times, engine_step_times, frame_times
        global captured_trail_buffers, captured_surface_sizes
        
        render_frame_calls = 0
        captured_engine = None
        captured_renderer = None
        render_times.clear()
        engine_step_times.clear()
        frame_times.clear()
        captured_trail_buffers.clear()
        captured_surface_sizes.clear()
        
        self.original_renderer = renderer.pygame_renderer.PygameRenderer
        renderer.pygame_renderer.PygameRenderer = HeadlessTestRenderer
        
    def tearDown(self):
        renderer.pygame_renderer.PygameRenderer = self.original_renderer

    def _run_main_headless(self, seed="42"):
        old_argv = sys.argv
        sys.argv = [
            "main.py",
            "--preset", "twin_galaxies",
            "--N", "5000",
            "--seed", str(seed),
            "--mode", "pygame",
            "--spf", "1",
            "--fps", "60",
            "--width", "800",
            "--height", "600"
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

    def test_pygame_production_runtime(self):
        global render_frame_calls, captured_engine, captured_renderer
        
        pos_initial, vel_initial, _, _ = build_universe("twin_galaxies", N_total=5000, seed=42)
        pos_initial = pos_initial.astype(np.float32)
        vel_initial = vel_initial.astype(np.float32)
        
        t_total_start = time.perf_counter()
        self._run_main_headless(seed="42")
        total_runtime = time.perf_counter() - t_total_start
        
        engine1 = captured_engine
        renderer1 = captured_renderer
        
        self.assertIsNotNone(engine1)
        self.assertIsNotNone(renderer1)
        self.assertEqual(type(engine1).__name__, "AdvancedTaichiEngine")
        
        self.assertEqual(engine1.N, 3788)
        self.assertEqual(engine1.step_count, 10)
        self.assertEqual(engine1.sys_step, 10)
        
        self.assertEqual(engine1.pos.shape, (3788, 2))
        self.assertEqual(engine1.vel.shape, (3788, 2))
        self.assertEqual(engine1.mass.shape, (3788,))
        self.assertEqual(engine1.types.shape, (3788,))
        
        self.assertTrue(np.all(np.isfinite(engine1.pos)))
        self.assertTrue(np.all(np.isfinite(engine1.vel)))
        self.assertTrue(np.all(np.isfinite(engine1.mass)))
        self.assertTrue(np.isfinite(engine1.energy))
        self.assertTrue(np.all(engine1.mass > 0))
        
        self.assertFalse(np.array_equal(pos_initial, engine1.pos))
        self.assertFalse(np.array_equal(vel_initial, engine1.vel))
        
        self.assertEqual(render_frame_calls, 10)
        
        self.assertEqual(len(captured_trail_buffers), 10)
        for tb in captured_trail_buffers:
            self.assertTrue(np.all(np.isfinite(tb)), "trail_buffer contains non-finite values")
            self.assertTrue(np.all(tb >= -1.0), "trail buffer channel out of range (too low)")
            self.assertTrue(np.all(tb <= 1e6), "trail buffer channel out of range (too high)")
            
        self.assertEqual(len(captured_surface_sizes), 10)
        for sz in captured_surface_sizes:
            self.assertEqual(sz, (800, 600))
            
        intervals = engine1.ti_step_interval.to_numpy()
        mean_interval = intervals.mean()
        max_interval = intervals.max()
        active_frac = np.mean(intervals == 1)
        
        print(f"\n--- Performance Observation (N=5000, twin_galaxies) ---")
        print(f"Total 10-frame runtime   : {total_runtime:.3f} s")
        if render_times:
            print(f"Median render_frame time : {np.median(render_times)*1000:.2f} ms")
        if engine_step_times:
            print(f"Median engine.step time  : {np.median(engine_step_times)*1000:.2f} ms")
        if frame_times:
            print(f"Total pygame loop run    : {frame_times[0]:.3f} s")
            
        print(f"Final Energy             : {engine1.energy:.6e}")
        print(f"Mean Adv Block Interval  : {mean_interval:.2f}")
        print(f"Max Adv Block Interval   : {max_interval}")
        print(f"Min-Interval Fraction @ Frame10: {active_frac:.4f}")
        print("-------------------------------------------------------\n")

        pos_run1 = engine1.pos.copy()
        vel_run1 = engine1.vel.copy()
        
        render_frame_calls = 0
        captured_engine = None
        captured_renderer = None
        captured_trail_buffers.clear()
        captured_surface_sizes.clear()
        
        self._run_main_headless(seed="42")
        
        self.assertEqual(render_frame_calls, 10)
        self.assertTrue(np.array_equal(pos_run1, captured_engine.pos))
        self.assertTrue(np.array_equal(vel_run1, captured_engine.vel))
        
if __name__ == "__main__":
    unittest.main(verbosity=2)

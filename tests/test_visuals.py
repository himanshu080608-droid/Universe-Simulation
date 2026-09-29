import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import unittest
import numpy as np
import pygame
import zipfile
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.generator import build_universe, STAR, DARK_MATTER
from renderer.pygame_renderer import PygameRenderer
from simulation_engine import SimulationEngine

# --- Metrics Helpers ---
def get_roi(arr, cy, cx, radius):
    h, w = arr.shape[:2]
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius))
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius))
    return arr[y0:y1, x0:x1]

def get_non_bg_pixels(arr):
    return np.any(arr > 0, axis=-1)

def get_pixel_bounds(arr):
    mask = get_non_bg_pixels(arr)
    if not np.any(mask):
        return 0, 0, 0, 0
    y_idx, x_idx = np.where(mask)
    return np.min(y_idx), np.max(y_idx), np.min(x_idx), np.max(x_idx)

def count_saturated_pixels(arr):
    # Count pixels where any channel is 255
    return np.sum(np.any(arr >= 250, axis=-1))

class TestPygameRendererVisuals(unittest.TestCase):
    
    def setUp(self):
        pygame.init()
        # Helper to create a fresh renderer and patch HUD
        def create_renderer(*args, **kwargs):
            if 'preset_name' not in kwargs:
                kwargs['preset_name'] = "twin_galaxies"
            r = PygameRenderer(*args, **kwargs)
            # Patch _draw_hud to no-op
            self._original_draw_hud = r._draw_hud
            r._draw_hud = lambda *a, **k: None
            return r
        self.create_renderer = create_renderer

    def tearDown(self):
        pygame.quit()

    def test_A_determinism(self):
        pos = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float64)
        types = np.array([STAR, STAR], dtype=np.int32)
        mass = np.array([100.0, 100.0], dtype=np.float64)
        
        # Instance 1
        r1 = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r1.camera.zoom = 5.0
        r1.camera.offset = np.array([50.0, 50.0], dtype=np.float64)
        r1.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr1 = pygame.surfarray.array3d(r1.screen).copy()
        
        # Instance 2 (fresh state)
        r2 = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r2.camera.zoom = 5.0
        r2.camera.offset = np.array([50.0, 50.0], dtype=np.float64)
        r2.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr2 = pygame.surfarray.array3d(r2.screen).copy()
        
        np.testing.assert_array_equal(arr1, arr2)

    def test_B_dark_matter_invisibility(self):
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([DARK_MATTER], dtype=np.int32)
        mass = np.array([1000.0], dtype=np.float64)
        
        r = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r.camera.zoom = 5.0
        r.camera.offset = np.array([50.0, 50.0], dtype=np.float64)
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        self.assertEqual(np.max(arr), 0)

    def test_C_bright_source_bloom(self):
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([STAR], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        r = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r.camera.zoom = 10.0
        r.camera.offset = np.array([50.0, 50.0], dtype=np.float64) 
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2)) 
        
        roi = get_roi(arr, 50, 50, 40)
        
        self.assertTrue(np.max(roi) > 0, "Source is not rendered")
        
        # Check boundary of ROI is empty
        self.assertTrue(np.all(roi[0, :] == 0), "Glow reached ROI top edge")
        self.assertTrue(np.all(roi[-1, :] == 0), "Glow reached ROI bottom edge")
        self.assertTrue(np.all(roi[:, 0] == 0), "Glow reached ROI left edge")
        self.assertTrue(np.all(roi[:, -1] == 0), "Glow reached ROI right edge")

    def test_D_bloom_separation(self):
        # Two stars at x = -2.0, and x = 2.0. With zoom=10, they are 40 pixels apart on screen.
        pos = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        types = np.array([STAR, STAR], dtype=np.int32)
        mass = np.array([200.0, 200.0], dtype=np.float64)
        
        r = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r.camera.zoom = 10.0
        r.camera.offset = np.array([50.0, 50.0], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2)) 
        
        slice_y = 50
        intensities = np.mean(arr[slice_y, :, :], axis=1) 
        
        peaks = []
        for x in range(1, 99):
            if intensities[x] > 50 and intensities[x] > intensities[x-1] and intensities[x] > intensities[x+1]:
                peaks.append(x)
                
        self.assertEqual(len(peaks), 2, "Expected exactly two peaks")
        separation = abs(peaks[0] - peaks[1])
        self.assertGreater(separation, 30, f"Peaks are not separated enough (measured {separation}px)")

    def test_E_bloom_disabled(self):
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([STAR], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        r = self.create_renderer(width=100, height=100, trail_decay=0.0)
        r.camera.zoom = 10.0
        r.camera.offset = np.array([50.0, 50.0], dtype=np.float64)
        
        # Disable bloom using the renderer's actual API
        r.bloom_int[:] = 0.0
        r.bloom_spr[:] = 0.0
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2))
        
        roi = get_roi(arr, 50, 50, 40)
        min_y, max_y, min_x, max_x = get_pixel_bounds(roi)
        height = max_y - min_y
        width = max_x - min_x
        
        self.assertLessEqual(height, 26, "Source is too large without bloom")
        self.assertLessEqual(width, 26, "Source is too large without bloom")

    def test_F_small_edge_cases(self):
        # Empty input
        r1 = self.create_renderer(width=16, height=16, trail_decay=0.0)
        r1.render_frame(np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int32), mass=np.zeros(0, dtype=np.float64))
        self.assertEqual(np.max(pygame.surfarray.array3d(r1.screen)), 0)
        
        # Out of bounds
        r2 = self.create_renderer(width=16, height=16, trail_decay=0.0)
        r2.camera.zoom = 10.0
        r2.camera.offset = np.array([8.0, 8.0], dtype=np.float64)
        pos = np.array([[-100.0, -100.0], [100.0, 100.0]], dtype=np.float64)
        types = np.array([STAR, STAR], dtype=np.int32)
        mass = np.array([500.0, 500.0], dtype=np.float64)
        r2.render_frame(pos, types, mass=mass)
        self.assertEqual(np.max(pygame.surfarray.array3d(r2.screen)), 0)

    def test_H_multiscale_bloom_halo(self):
        """Test 1 - bounded halo: Measure glow radius around a single source."""
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([STAR], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        r = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r.camera.zoom = 5.0
        r.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2))
        
        # Center is at 100, 100
        intensities = np.mean(arr[100, 100:], axis=1) # slice from center to right
        
        # Find where intensity drops to 0 (or very close, say < 2)
        radius = 0
        for i, val in enumerate(intensities):
            if val < 2.0:
                radius = i
                break
                
        # The bloom should be finite and within the configured design range 
        # (dual kawase spreads it wide but finite based on level count and image bounds).
        # We expect a soft glow radius up to 40-80 pixels for a 200px window with 5 levels.
        self.assertGreater(radius, 5, "Bloom radius is too small, Kawase should spread it")
        self.assertLess(radius, 95, "Bloom radius hit the boundary of the 200px window without fading")

    def test_I_multiscale_bloom_monotonic(self):
        """Test 2 - monotonic contribution: Verify smooth spatial falloff without rings."""
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([STAR], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        r = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r.camera.zoom = 5.0
        r.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2))
        
        intensities = np.mean(arr[100, 100:180], axis=1) # scan outward
        
        # Verify it falls off monotonically from the core outwards
        # (Dual-kawase is generally smooth)
        for i in range(1, len(intensities)):
            # allow a tiny float/rounding wiggle room (e.g. 1 pixel intensity)
            self.assertLessEqual(intensities[i], intensities[i-1] + 1.5, f"Non-monotonic glow found at distance {i}")

    def test_J_multiscale_bloom_saturation(self):
        """Test 4 - saturation protection: Measure saturated pixels in a dense source."""
        # Create a dense cluster of 50 stars at the same location to simulate a galactic core
        pos = np.zeros((50, 2), dtype=np.float64)
        types = np.zeros(50, dtype=np.int32) # STAR
        mass = np.full(50, 100.0, dtype=np.float64)
        
        r = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r.camera.zoom = 5.0
        r.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        
        # Find pixels at maximum intensity (>=250 for at least one channel)
        saturated_pixels = count_saturated_pixels(arr)
        total_pixels = 200 * 200
        saturated_percentage = (saturated_pixels / total_pixels) * 100
        
        # The core itself will be white/saturated because it's a dense cluster,
        # but the bloom must not wash out the whole screen.
        # So we expect saturated_percentage to be > 0 but << 100%
        self.assertGreater(saturated_percentage, 0.0)
        # 10% of a 200x200 window is a circle of radius ~35, which is quite large for the core.
        # The bloom protection should prevent it from exceeding this.
        self.assertLess(saturated_percentage, 10.0, "Bloom washed out the screen (saturation protection failed)")


    def test_K_overlapping_source_determinism(self):
        """Test overlapping-source determinism with a dense cluster."""
        pos = np.zeros((100, 2), dtype=np.float64) # All in same spot
        types = np.full(100, STAR, dtype=np.int32)
        mass = np.full(100, 100.0, dtype=np.float64)
        
        r1 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r1.camera.zoom = 5.0
        r1.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        r1.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr1 = pygame.surfarray.array3d(r1.screen).copy()
        
        r2 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r2.camera.zoom = 5.0
        r2.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        r2.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr2 = pygame.surfarray.array3d(r2.screen).copy()
        
        np.testing.assert_array_equal(arr1, arr2)

    def test_L_bloom_strength_control(self):
        """Verify that doubling the configured bloom_int increases the total screen brightness."""
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([STAR], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        # Base renderer
        r1 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r1.camera.zoom = 5.0
        r1.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        # Ensure base bloom is active
        r1.bloom_int[STAR] = 0.5
        r1.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr1 = pygame.surfarray.array3d(r1.screen)
        sum1 = np.sum(arr1)
        
        # High bloom renderer
        r2 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r2.camera.zoom = 5.0
        r2.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        r2.bloom_int[STAR] = 1.0 # Double strength
        r2.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr2 = pygame.surfarray.array3d(r2.screen)
        sum2 = np.sum(arr2)
        
        self.assertGreater(sum2, sum1 * 1.1, "Bloom strength control failed to increase brightness")

    def test_M_non_stellar_bloom_exclusion(self):
        """Verify that non-stellar objects (like planets) do not bloom even if bloom_int is high."""
        from universe.generator import ROCKY
        pos = np.array([[0.0, 0.0]], dtype=np.float64)
        types = np.array([ROCKY], dtype=np.int32)
        mass = np.array([500.0], dtype=np.float64)
        
        r1 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r1.camera.zoom = 5.0
        r1.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        r1.bloom_int[ROCKY] = 10.0 # Extreme bloom intent
        r1.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr1 = pygame.surfarray.array3d(r1.screen)
        
        r2 = self.create_renderer(width=200, height=200, trail_decay=0.0)
        r2.camera.zoom = 5.0
        r2.camera.offset = np.array([100.0, 100.0], dtype=np.float64)
        r2.bloom_int[ROCKY] = 0.0 # No bloom intent
        r2.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr2 = pygame.surfarray.array3d(r2.screen)
        
        # Arrays must be identical because ROCKY is excluded from the bloom pipeline
        np.testing.assert_array_equal(arr1, arr2)

    def test_N_realistic_saturation_protection(self):
        """Verify saturation protection using an actual dense galaxy preset."""
        preset = "omega_centauri"
        pos, vel, mass, types = build_universe(preset, N_total=2000, seed=42)
        
        r = self.create_renderer(width=600, height=600, trail_decay=0.0, preset_name=preset)
        r.camera.zoom = 10.0
        r.camera.offset = np.array([300.0, 300.0], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        
        # Count saturated pixels over the whole frame
        saturated_pixels = count_saturated_pixels(arr)
        total_pixels = 600 * 600
        saturated_percentage = (saturated_pixels / total_pixels) * 100
        
        # The center of the galaxy should have high mean brightness
        center_roi = arr[250:350, 250:350]
        mean_brightness = np.mean(center_roi)
        
        self.assertGreater(mean_brightness, 50.0, "Central region too dim")
        self.assertLess(saturated_percentage, 15.0, "Bloom blown out across galaxy")


    def test_G_baseline_castor_sextuple(self):
        preset = "castor_sextuple"
        baseline_path = os.path.join(os.path.dirname(__file__), f"baselines/{preset}_t0_scene_post_bloom.png")
        if not os.path.exists(baseline_path):
            self.skipTest(f"Baseline {baseline_path} not found")
            
        baseline_img = np.array(Image.open(baseline_path).convert("RGB"))
        
        pos, vel, mass, types = build_universe(preset, N_total=2000, seed=42)
        r = self.create_renderer(width=800, height=800, trail_decay=0.88, preset_name=preset)
        
        cx, cy = 0.0, 0.0
        mask = types != 2
        fp = pos[mask] if np.any(mask) else pos
        if len(fp) > 0:
            cx, cy = np.median(fp[:,0]), np.median(fp[:,1])
            r_dist = np.linalg.norm(fp - np.array([cx, cy]), axis=1)
            span = max(np.percentile(r_dist, 90) * 2.5, 1.0)
        else:
            span = 50.0
            
        zoom = 800 * 0.45 / span
        r.camera.zoom = zoom
        r.camera.offset = np.array([400 - cx * zoom, 400 + cy * zoom], dtype=np.float64)
        
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        arr = pygame.surfarray.array3d(r.screen)
        arr = np.transpose(arr, (1, 0, 2))
        
        self.assertEqual(arr.shape, baseline_img.shape, "Dimension mismatch")
        diff = np.abs(arr.astype(int) - baseline_img.astype(int))
        mean_diff = np.mean(diff)
        self.assertLess(mean_diff, 1.0, f"Mean pixel difference {mean_diff} exceeds tolerance")

if __name__ == '__main__':
    unittest.main()

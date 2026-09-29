import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import time
import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.generator import build_universe
from renderer.pygame_renderer import PygameRenderer
import renderer.pygame_renderer as pr

# We will monkeypatch _fast_draw_heads to measure its specific time
original_fast_draw = pr._fast_draw_heads
draw_heads_times = []

def fast_draw_wrapper(*args, **kwargs):
    t0 = time.perf_counter()
    res = original_fast_draw(*args, **kwargs)
    t1 = time.perf_counter()
    draw_heads_times.append(t1 - t0)
    return res

pr._fast_draw_heads = fast_draw_wrapper

# We will monkeypatch pygame.transform.smoothscale to measure multiscale time
original_smoothscale = pygame.transform.smoothscale
smoothscale_times = []

def smoothscale_wrapper(*args, **kwargs):
    t0 = time.perf_counter()
    res = original_smoothscale(*args, **kwargs)
    t1 = time.perf_counter()
    smoothscale_times.append(t1 - t0)
    return res

pygame.transform.smoothscale = smoothscale_wrapper

def run_benchmark():
    pygame.init()
    
    preset = "castor_sextuple"
    pos, vel, mass, types = build_universe(preset, N_total=2000, seed=42)
    r = PygameRenderer(width=800, height=800, trail_decay=0.88, preset_name=preset)
    r._draw_hud = lambda *a, **k: None

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
    
    # Warm-up frames
    num_warmup = 20
    for _ in range(num_warmup):
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        
    draw_heads_times.clear()
    smoothscale_times.clear()
    
    # Measured frames
    num_frames = 100
    total_times = []
    
    for _ in range(num_frames):
        t0 = time.perf_counter()
        r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
        t1 = time.perf_counter()
        total_times.append(t1 - t0)
        
    pygame.quit()
    
    mean_total = np.mean(total_times)
    median_total = np.median(total_times)
    fps = 1.0 / mean_total
    
    mean_draw = np.sum(draw_heads_times) / num_frames
    mean_scale = np.sum(smoothscale_times) / num_frames
    
    print(f"Total Frames: {num_frames} (Warm-up: {num_warmup})")
    print(f"Mean Render Time: {mean_total*1000:.2f} ms")
    print(f"Median Render Time: {median_total*1000:.2f} ms")
    print(f"FPS: {fps:.1f}")
    print(f"Mean _fast_draw_heads: {mean_draw*1000:.2f} ms")
    print(f"Mean smoothscale total (Bloom): {mean_scale*1000:.2f} ms")

if __name__ == "__main__":
    run_benchmark()

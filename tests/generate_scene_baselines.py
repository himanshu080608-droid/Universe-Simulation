import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import sys
import numpy as np
import pygame
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from universe.generator import build_universe
from renderer.pygame_renderer import PygameRenderer

def generate_castor_baseline():
    out_dir = os.path.join(os.path.dirname(__file__), "baselines")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "castor_sextuple_t0_scene.png")
    
    pygame.init()
    preset = "castor_sextuple"
    pos, vel, mass, types = build_universe(preset, N_total=2000, seed=42)
    r = PygameRenderer(width=800, height=800, trail_decay=0.88, preset_name=preset)
    
    # Suppress HUD
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
    
    r.render_frame(pos, types, mass=mass, step=0, steps_per_frame=1)
    arr = pygame.surfarray.array3d(r.screen)
    arr = np.transpose(arr, (1, 0, 2))
    Image.fromarray(arr).save(out_path)
    print(f"Saved {out_path}")
    pygame.quit()

if __name__ == "__main__":
    generate_castor_baseline()

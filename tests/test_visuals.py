import sys
import os
import pygame
import numpy as np
from PIL import Image

# Force Pygame to run headlessly without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Add parent directory to path so we can import the simulation modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from universe.generator import build_universe
from renderer.pygame_renderer import PygameRenderer
from simulation_engine import SimulationEngine

def generate_test_gif(preset, output_path, N=3000, frames=20, zoom_target=None, zoom_factor=1.0):
    print(f"Testing {preset} -> {os.path.basename(output_path)}...")
    try:
        # 1. Build Universe
        pos, vel, mass, types = build_universe(preset, N_total=N, seed=42)
        
        # 2. Initialize Engine
        engine = SimulationEngine(pos, vel, mass, types, dt=0.003, eps=1.0)
        
        # 3. Initialize Pygame Renderer
        W, H = 600, 450
        renderer = PygameRenderer(width=W, height=H, trail_decay=0.88)
        
        # 4. Determine Camera Target & Zoom
        mask = types != 2 # Ignore comets/oort cloud to focus on core structures
        fp = pos[mask] if np.any(mask) else pos
        
        if zoom_target == 'black_hole':
            bh_mask = types == 5
            if np.any(bh_mask):
                cx, cy = pos[np.where(bh_mask)[0][0]]
            else:
                cx, cy = 0.0, 0.0
            span = 40.0 * zoom_factor
        elif zoom_target == 'center':
            cx, cy = 0.0, 0.0
            span = 12.0 * zoom_factor
        else:
            cx, cy = np.median(fp[:,0]), np.median(fp[:,1])
            r = np.linalg.norm(fp - np.array([cx, cy]), axis=1)
            r80 = np.percentile(r, 80) if len(r) > 0 else 10.0
            span = max(r80 * 2.5, 1.0) * zoom_factor

        zoom = min(W, H) * 0.40 / span
        renderer.camera.zoom = zoom
        renderer.camera.offset = np.array([W / 2 - cx * zoom, H / 2 + cy * zoom], dtype=np.float64)
        
        # 5. Warmup Engine (let structures settle slightly)
        for _ in range(5):
            engine.step(1, speed_mult=1.0)
            
        # 6. Render Frames
        imgs = []
        for step in range(frames):
            engine.step(1, speed_mult=1.0)
            renderer.render_frame(engine.pos, engine.types, mass=engine.mass, energy=None, step=step, steps_per_frame=1)
            
            # Extract raw pixel array from pygame surface
            arr = pygame.surfarray.array3d(renderer.screen)
            arr = np.transpose(arr, (1, 0, 2))
            img = Image.fromarray(arr)
            imgs.append(img)
            
        # 7. Save GIF
        imgs[0].save(output_path, save_all=True, append_images=imgs[1:], optimize=False, duration=50, loop=0)
        print(f"  [OK] Saved {output_path}")
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {preset}: {e}")
        return False

def main():
    # Setup output directory
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    # Define test suite: (preset_name, run_name, zoom_target, zoom_factor)
    tests = [
        # Twin Galaxies
        ('twin_galaxies', 'default', None, 1.0),
        ('twin_galaxies', 'bh_zoom', 'black_hole', 0.5),
        
        # Solar System
        ('solar_system', 'default', None, 1.0),
        ('solar_system', 'inner_zoom', 'center', 0.8),
        
        # Alpha Centauri
        ('alpha_centauri', 'default', None, 1.0),
        ('alpha_centauri', 'center_zoom', 'center', 0.5),
        
        # Milkomeda
        ('milkomeda', 'default', None, 1.0),
        ('milkomeda', 'bh_zoom', 'black_hole', 0.6),
        
        # Cygnus X-1
        ('cygnus_x1', 'default', None, 1.0),
        ('cygnus_x1', 'bh_zoom', 'black_hole', 0.3),
        
        # Messier 13 (Globular Cluster)
        ('messier13', 'default', None, 1.0),
        ('messier13', 'center_zoom', 'center', 0.4),
        
        # Messier 31 (Andromeda)
        ('messier31', 'default', None, 1.0),
        ('messier31', 'bh_zoom', 'black_hole', 0.5),
        
        # Chaos
        ('chaos', 'default', None, 1.0),
        ('chaos', 'bh_zoom', 'black_hole', 0.6),
        
        # Laplace
        ('laplace', 'default', None, 1.0),
        ('laplace', 'center_zoom', 'center', 0.5)
    ]
    
    print(f"Starting Visual Regression Test Suite ({len(tests)} scenarios)...\n")
    success = 0
    for preset, name, target, z_factor in tests:
        path = os.path.join(out_dir, f"{preset}_{name}.gif")
        if generate_test_gif(preset, path, zoom_target=target, zoom_factor=z_factor):
            success += 1
            
    print(f"\nCompleted {success}/{len(tests)} tests successfully.")
    print(f"Check the {out_dir} directory for visual results.")

if __name__ == "__main__":
    main()

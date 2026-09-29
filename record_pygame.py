import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import numpy as np
from PIL import Image

from universe.generator import build_universe
from advanced_engine import AdvancedTaichiEngine
from renderer.pygame_renderer import PygameRenderer

def record_preset(preset, output_gif, frames=120):
    print(f"Setting up Pygame for {preset}...")
    pygame.init()
    
    pos, vel, mass, types = build_universe(preset=preset, N_total=5000)
    engine = AdvancedTaichiEngine(pos, vel, mass, types, G=1.0, eps=1.0, dt=0.001)
    
    width, height = 800, 800
    renderer = PygameRenderer(width=width, height=height, trail_decay=0.0, preset_name=preset) # NO TRAILS for physics check
    
    # Setup camera
    _cx, _cy = np.median(pos[:, 0]), np.median(pos[:, 1])
    _r = np.linalg.norm(pos - np.array([_cx, _cy]), axis=1)
    _r99 = np.percentile(_r, 99)
    # Extremely wide zoom to catch any escaping particles
    _zoom = (min(width, height) * 0.40) / max(_r99 * 10.0, 1.0)
    renderer.camera.zoom = _zoom
    renderer.camera.target_zoom = _zoom
    renderer.camera.offset = np.array([width / 2 - _cx * _zoom, height / 2 + _cy * _zoom], dtype=np.float64)
    
    pil_frames = []
    print(f"Recording {frames} frames...")
    
    for i in range(frames):
        engine.step(16, speed_mult=1.0)
        renderer.render_frame(
            engine.pos, engine.types, mass=engine.mass,
            energy=engine.energy, step=engine.step_count, steps_per_frame=16
        )
        
        # Extract pixel data directly from Pygame surface
        raw_str = pygame.image.tostring(renderer.screen, "RGB")
        img = Image.frombytes("RGB", (width, height), raw_str)
        pil_frames.append(img)
        
        if i % 20 == 0:
            print(f"  Frame {i}/{frames}")
            
    print(f"Saving {output_gif}...")
    pil_frames[0].save(output_gif, format='GIF', append_images=pil_frames[1:], save_all=True, duration=33, loop=0)
    print(f"Done recording {output_gif}!")

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    record_preset("pleiades_m45", "output/pleiades_pygame.gif")
    record_preset("omega_centauri", "output/omega_pygame.gif")

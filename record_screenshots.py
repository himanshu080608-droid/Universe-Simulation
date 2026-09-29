import numpy as np
import pygame
from PIL import Image
from universe.generator import build_universe
from advanced_engine import AdvancedTaichiEngine
from renderer.pygame_renderer import PygameRenderer

def take_screenshot(preset, output_png, target_zoom=None):
    print(f"Setting up Pygame for {preset}...")
    pygame.init()
    
    pos, vel, mass, types = build_universe(preset=preset, N_total=5000)
    engine = AdvancedTaichiEngine(pos, vel, mass, types, G=1.0, eps=1.0, dt=0.001)
    
    width, height = 800, 800
    renderer = PygameRenderer(width=width, height=height, trail_decay=0.0, preset_name=preset)
    
    # Setup camera
    _cx, _cy = np.median(pos[:, 0]), np.median(pos[:, 1])
    
    if target_zoom is not None:
        _zoom = target_zoom
    else:
        _r = np.linalg.norm(pos - np.array([_cx, _cy]), axis=1)
        _r99 = np.percentile(_r, 99)
        # Extremely close zoom to see individual star sizes clearly
        _zoom = (min(width, height) * 0.40) / max(_r99 * 0.1, 0.1)
    
    if preset == "solar_system":
        _zoom = 8.0  # Zoom right into the inner planets
    elif preset == "trappist_1":
        _zoom = 50.0 # Extreme zoom for the super compact resonant chain
    elif preset == "hirayama_family":
        _zoom = 20.0
    renderer.camera.zoom = _zoom
    renderer.camera.target_zoom = _zoom
    renderer.camera.offset = np.array([width / 2 - _cx * _zoom, height / 2 + _cy * _zoom], dtype=np.float64)
    
    # Run 5 warmup frames to let renderer trail buffers (if any) initialize, and take screenshot
    for i in range(5):
        engine.step(16, speed_mult=1.0)
        renderer.render_frame(
            engine.pos, engine.types, mass=engine.mass,
            energy=engine.energy, step=engine.step_count, steps_per_frame=16
        )
        
    # Extract pixel data directly from Pygame surface
    raw_str = pygame.image.tostring(renderer.screen, "RGB")
    img = Image.frombytes("RGB", (width, height), raw_str)
    img.save(output_png, format='PNG')
    print(f"Saved {output_png}!")

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    
    presets = ["messier31"]
    
    for p in presets:
        # We will generate multiple zooms to analyze the core blowout
        zooms = [4.0, 16.0, 32.0, 64.0, 128.0]
        for z in zooms:
            # Need to wait longer for the cloud to settle and form the galaxy properly
            take_screenshot(p, f"output/shot_{p}_zoom_{z}.png", target_zoom=z)
            print(f"Captured {p} at zoom {z}x")

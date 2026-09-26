import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
from simulation_engine import SimulationEngine
from universe.generator import build_universe
from renderer.pygame_renderer import PygameRenderer
import pygame.event

def test():
    pos, vel, mass, types = build_universe(preset="twin_galaxies", N_total=500, seed=42)
    engine = SimulationEngine(pos, vel, mass, types, dt=0.001, eps=1.0, G=1.0, theta=0.6, use_bh=True, preset="twin_galaxies")
    
    renderer = PygameRenderer(width=800, height=600)
    renderer.trail_index = 7 # Permanent
    renderer.decay = 1.0
    renderer.auto_fit(engine.pos)
    
    # Run a few steps to build up trails
    for _ in range(5):
        engine.step(2, speed_mult=1.0)
        renderer.render_frame(engine.pos, engine.types, mass=engine.mass, energy=engine.energy, step=engine.step_count, steps_per_frame=2)
        
    # Simulate zoom out (Mouse wheel down)
    event = pygame.event.Event(pygame.MOUSEWHEEL, y=-1)
    pygame.event.post(event)
    
    # Handle events
    renderer._handle_events(engine.pos)
    
    # Check if trail_buffer is cleared
    import numpy as np
    assert np.max(renderer.trail_buffer) == 0.0, "Trail buffer was not cleared on zoom out!"
    
    # Render frame to confirm
    renderer.render_frame(engine.pos, engine.types, mass=engine.mass, energy=engine.energy, step=engine.step_count, steps_per_frame=2)
    
    pygame.image.save(renderer.screen, "test_zoom_screenshot.png")
    print("Test passed: Trail buffer cleared on zoom. Screenshot saved.")
    
test()

import numpy as np
import time
from universe.generator import build_universe
from advanced_engine import AdvancedTaichiEngine

def test_preset_stability(preset, target_steps=100000, report_interval=10000):
    print(f"--- Testing {preset} for {target_steps} steps ---")
    pos, vel, mass, types = build_universe(preset=preset, N_total=5000)
    
    # Initialize engine
    engine = AdvancedTaichiEngine(pos, vel, mass, types, G=1.0, eps=1.0, dt=0.001)
    
    _cx, _cy = np.median(pos[:, 0]), np.median(pos[:, 1])
    r_initial = np.percentile(np.linalg.norm(pos - np.array([_cx, _cy]), axis=1), 95)
    
    print(f"Step 0: 95th Percentile Radius = {r_initial:.2f}")
    
    start_time = time.time()
    for step in range(1, target_steps + 1):
        engine.step(1, speed_mult=1.0)
        
        if step % report_interval == 0:
            current_pos = engine.pos
            cx, cy = np.median(current_pos[:, 0]), np.median(current_pos[:, 1])
            r_current = np.percentile(np.linalg.norm(current_pos - np.array([cx, cy]), axis=1), 95)
            print(f"Step {step}: 95th Percentile Radius = {r_current:.2f}")
            
    print(f"Completed {target_steps} steps in {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    test_preset_stability("pleiades_m45", target_steps=100000)
    test_preset_stability("omega_centauri", target_steps=100000)

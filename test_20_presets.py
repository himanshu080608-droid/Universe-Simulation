import numpy as np
from universe.registry import PresetRegistry
from universe.generator import build_universe

def verify_presets():
    presets = [
        'twin_galaxies', 'solar_system', 'chaos', 'laplace', 'alpha_centauri', 'milkomeda',
        'stephans_quintet', 'messier13', 'messier31', 'ngc1052_df2', 'castor_sextuple', 'hd98800_polar',
        'trappist_1', 'gravothermal_catastrophe', 'wr104_pinwheel', 'omega_centauri', 'pleiades_m45',
        'hirayama_family', 'dark_matter_halo_merger', 'great_attractor'
    ]
    
    for p in presets:
        # Build
        pos, vel, mass, types = build_universe(p, N_total=5000, seed=42)
        pos2, vel2, mass2, types2 = build_universe(p, N_total=5000, seed=42)
        
        # Determine exact particle count
        n_actual = len(pos)
        
        # Check finites
        finite = np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and np.all(np.isfinite(mass))
        
        # Determinism
        determ = np.allclose(pos, pos2) and np.allclose(vel, vel2) and np.allclose(mass, mass2)
        
        print(f"Preset: {p:<25} | N={n_actual:<5} | Finite={finite} | Determ={determ}")

if __name__ == "__main__":
    PresetRegistry.clear()
    from universe.production_presets import register_all
    register_all()
    verify_presets()

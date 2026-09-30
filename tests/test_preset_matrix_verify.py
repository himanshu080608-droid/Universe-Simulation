import unittest
import numpy as np
import os
import hashlib
from universe.generator import build_universe
from universe.preset import PresetRegistry

class TestPresetMatrixVerification(unittest.TestCase):
    def test_all_production_presets_matrix(self):
        """Generate the compact machine-readable verification table for all 20 presets."""
        from universe.production_presets import register_all
        PresetRegistry._presets.clear()
        register_all()
        
        presets = sorted(PresetRegistry.list_presets())
        self.assertEqual(len(presets), 20, f"Must have exactly 20 presets, found {len(presets)}: {presets}")
        
        print("\n| Preset | N | N_allocated | Deterministic | Identities Preserved |")
        print("|--------|---|-------------|---------------|-----------------------|")
        
        for name in presets:
            # 1. Exact N allocation
            N = 1000
            pos1, vel1, mass1, types1 = build_universe(name, N_total=N, seed=42)
            
            # 2. Component Identity
            preset = PresetRegistry.get(name)
            ids = preset.last_identities
            
            n_allocated = len(pos1)
            identities_preserved = (len(ids) == n_allocated and n_allocated == N)
            
            # 3. Determinism
            pos2, vel2, mass2, types2 = build_universe(name, N_total=N, seed=42)
            deterministic = np.allclose(pos1, pos2) and np.allclose(vel1, vel2) and np.all(types1 == types2)
            
            print(f"| {name} | {N} | {n_allocated} | {deterministic} | {identities_preserved} |")
            
            self.assertEqual(n_allocated, N)
            self.assertTrue(identities_preserved)
            self.assertTrue(deterministic)

    def test_legacy_path_exclusion(self):
        """Legacy path is isolated."""
        import sys
        self.assertNotIn("StellarCluster", sys.modules)

if __name__ == '__main__':
    unittest.main()

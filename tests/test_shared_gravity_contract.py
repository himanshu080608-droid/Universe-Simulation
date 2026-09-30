import unittest
import numpy as np
from universe.preset import UnitSystem
from universe.components import Component
from universe.spatial import PointMassSpatial, ExponentialDiskSpatial, NFWSpatial
from universe.kinematics import DiskKinematics
from universe.generator import STAR

class TestSharedGravityContract(unittest.TestCase):
    def test_point_mass_force(self):
        spatial = PointMassSpatial(total_mass=100.0)
        F = spatial.radial_force(r=10.0, G=1.0, eps=0.0)
        self.assertAlmostEqual(F, 1.0) # 1 * 100 * 10 / 1000 = 1.0

    def test_disk_force_approximation(self):
        spatial = ExponentialDiskSpatial(r_scale=1.0, total_mass=10.0)
        # Verify it provides a bounded force using spherical approximation
        F = spatial.radial_force(r=1.0, G=1.0, eps=1.0)
        self.assertGreater(F, 0.0)

    def test_combined_kinematics_force(self):
        rng = np.random.default_rng(42)
        unit_system = UnitSystem(G_sim=1.0, eps_sim=1.0)
        
        comp_disk = Component("disk", 1.0, 10.0, {'types': [STAR], 'probs': [1.0]}, 
                              spatial_model=ExponentialDiskSpatial(1.0, 10.0), 
                              kinematic_model=DiskKinematics(velocity_dispersion=0.0))
                              
        comp_halo = Component("halo", 1.0, 100.0, {'types': [STAR], 'probs': [1.0]}, 
                              spatial_model=PointMassSpatial(100.0), # using point mass to test effect easily
                              kinematic_model=None)
                              
        # Disk only
        pos_1, r_1 = comp_disk.spatial_model.sample_positions(1, rng)
        r_scalar = r_1[0]
        vel_1 = comp_disk.kinematic_model.solve(pos_1, r_1, [comp_disk], unit_system, rng)
        speed_1 = np.linalg.norm(vel_1[0])
        
        # Disk + Halo
        vel_2 = comp_disk.kinematic_model.solve(pos_1, r_1, [comp_disk, comp_halo], unit_system, rng)
        speed_2 = np.linalg.norm(vel_2[0])
        
        # Halo should significantly increase required circular velocity
        self.assertGreater(speed_2, speed_1)

if __name__ == '__main__':
    unittest.main()

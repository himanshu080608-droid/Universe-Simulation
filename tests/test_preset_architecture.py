import unittest
import numpy as np
from universe.registry import PresetRegistry
from universe.preset import Preset, UnitSystem
from universe.components import Component
from universe.spatial import PointMassSpatial, ExponentialDiskSpatial, NFWSpatial
from universe.kinematics import PointMassKinematics, DiskKinematics, NFWKinematics
from universe.generator import build_universe, DARK_MATTER, STAR, BLACK_HOLE
from universe.validation import Diagnostics

class TestPresetArchitecture(unittest.TestCase):
    def setUp(self):
        # Only register if not already there
        if "test_stellar_only" in PresetRegistry._presets:
            return
        PresetRegistry.register(Preset(
            name="test_stellar_only",
            description="Visible stellar component only",
            components=[
                Component("stars", 1.0, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                          spatial_model=ExponentialDiskSpatial(10.0, 1000.0),
                          kinematic_model=DiskKinematics())
            ]
        ))

        # B. Dark-matter-only component
        PresetRegistry.register(Preset(
            name="test_dm_only",
            description="Dark-matter-only component",
            components=[
                Component("halo", 1.0, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                          spatial_model=NFWSpatial(10.0, 10.0, 5000.0),
                          kinematic_model=NFWKinematics())
            ]
        ))

        # C. Stellar disk + dark-matter halo
        PresetRegistry.register(Preset(
            name="test_disk_halo",
            description="Stellar disk + dark-matter halo",
            components=[
                Component("stars", 0.5, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                          spatial_model=ExponentialDiskSpatial(10.0, 1000.0),
                          kinematic_model=DiskKinematics()),
                Component("halo", 0.5, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                          spatial_model=NFWSpatial(10.0, 10.0, 5000.0),
                          kinematic_model=NFWKinematics())
            ]
        ))

        # D. Stellar disk + bulge + dark-matter halo
        PresetRegistry.register(Preset(
            name="test_disk_bulge_halo",
            description="Stellar disk + bulge + dark-matter halo",
            components=[
                Component("bulge", 0.1, 200.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                          spatial_model=PointMassSpatial(200.0),
                          kinematic_model=PointMassKinematics()),
                Component("stars", 0.4, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                          spatial_model=ExponentialDiskSpatial(10.0, 1000.0),
                          kinematic_model=DiskKinematics()),
                Component("halo", 0.5, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                          spatial_model=NFWSpatial(10.0, 10.0, 5000.0),
                          kinematic_model=NFWKinematics())
            ]
        ))

    def test_a_stellar_only(self):
        pos, vel, mass, types = build_universe("test_stellar_only", 1000, 42)
        self.assertEqual(len(pos), 1000)
        self.assertTrue(np.all(types == STAR))
        semantics = PresetRegistry.get("test_stellar_only").get_render_semantics()
        self.assertEqual(semantics['invisible_types'], [])

    def test_b_dm_only(self):
        pos, vel, mass, types = build_universe("test_dm_only", 1000, 42)
        self.assertEqual(len(pos), 1000)
        self.assertTrue(np.all(types == DARK_MATTER))
        semantics = PresetRegistry.get("test_dm_only").get_render_semantics()
        self.assertIn(DARK_MATTER, semantics['invisible_types'])

    def test_c_disk_halo(self):
        pos, vel, mass, types = build_universe("test_disk_halo", 1000, 42)
        self.assertEqual(len(pos), 1000)
        self.assertEqual(np.sum(types == STAR), 500)
        self.assertEqual(np.sum(types == DARK_MATTER), 500)
        
        # Test validation contract
        preset = PresetRegistry.get("test_disk_halo")
        diag = Diagnostics.validate_preset_initialization(preset, pos, vel, mass, types)
        self.assertTrue(diag['is_finite'])
        self.assertAlmostEqual(diag['total_mass'], 6000.0)

    def test_d_disk_bulge_halo(self):
        pos, vel, mass, types = build_universe("test_disk_bulge_halo", 1000, 42)
        self.assertEqual(len(pos), 1000)
        self.assertEqual(np.sum(types == BLACK_HOLE), 0)
        self.assertEqual(np.sum(types == STAR), 500)
        self.assertEqual(np.sum(types == DARK_MATTER), 500)
        
        # Test A3 Component Identity
        preset = PresetRegistry.get("test_disk_bulge_halo")
        slices = preset._last_slices
        self.assertEqual(slices['bulge'], (0, 100))
        self.assertEqual(slices['stars'], (100, 500))
        self.assertEqual(slices['halo'], (500, 1000))
        # Both bulge and stars could technically use the same type without losing identity
        # The prompt says "disk and bulge sharing STAR type". I will make the bulge use STAR in the test.

    def test_e_legacy_preset(self):
        pos, vel, mass, types = build_universe("twin_galaxies", 1000, 42)
        self.assertGreater(len(pos), 0)
        semantics = PresetRegistry.get("twin_galaxies").get_render_semantics()
        self.assertIn(14, semantics['invisible_types'])

    def test_shared_gravity_contract_determinism(self):
        # Verify that generating multiple times yields exactly identical output
        p1, v1, m1, t1 = build_universe("test_disk_halo", 100, 42)
        p2, v2, m2, t2 = build_universe("test_disk_halo", 100, 42)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(v1, v2)

    def test_unit_system_conversions(self):
        units = UnitSystem(L0=2.0, M0=3.0, T0=4.0)
        # V0 = L0/T0 = 0.5
        
        # Round trips
        self.assertAlmostEqual(units.length_to_sim(10.0), 5.0)
        self.assertAlmostEqual(units.length_to_physical(5.0), 10.0)
        
        self.assertAlmostEqual(units.mass_to_sim(15.0), 5.0)
        self.assertAlmostEqual(units.mass_to_physical(5.0), 15.0)
        
        self.assertAlmostEqual(units.velocity_to_sim(2.0), 4.0)
        self.assertAlmostEqual(units.velocity_to_physical(4.0), 2.0)

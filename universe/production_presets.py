import numpy as np
from universe.registry import PresetRegistry
from universe.preset import Preset, UnitSystem
from universe.components import Component
from universe.spatial import ExponentialDiskSpatial, NFWSpatial, PointMassSpatial
from universe.kinematics import DiskKinematics, NFWKinematics, PointMassKinematics
from universe.generator import STAR, PLANET, BLACK_HOLE, DARK_MATTER, COMET
from universe.new_models import PlummerSpatial, IsotropicKinematics, KeplerianSpatial

def register_all():
    # 1. twin_galaxies
    PresetRegistry.register(Preset(
        "twin_galaxies", "Twin Galaxies Merger Scenario", UnitSystem(),
        components=[
            Component("g1_bulge", 0.05, 500.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(0.5, 500.0, center=(-15.0, -10.0)),
                      kinematic_model=PointMassKinematics(drift=(0.0, 1.0))),
            Component("g1_stars", 0.15, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(1.0, 1000.0, center=(-15.0, -10.0)),
                      kinematic_model=DiskKinematics(drift=(0.0, 1.0))),
            Component("g1_halo", 0.3, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(10.0, 10.0, 5000.0, center=(-15.0, -10.0)),
                      kinematic_model=NFWKinematics(drift=(0.0, 1.0))),
            Component("g2_bulge", 0.05, 500.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(0.5, 500.0, center=(15.0, 10.0)),
                      kinematic_model=PointMassKinematics(drift=(0.0, -1.0))),
            Component("g2_stars", 0.15, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(1.0, 1000.0, center=(15.0, 10.0)),
                      kinematic_model=DiskKinematics(drift=(0.0, -1.0))),
            Component("g2_halo", 0.3, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(10.0, 10.0, 5000.0, center=(15.0, 10.0)),
                      kinematic_model=NFWKinematics(drift=(0.0, -1.0)))
        ]
    ))

    # 2. solar_system
    # We approximate Solar System with PointMass (Sun) + ExponentialDisk (Asteroids/Kuiper)
    PresetRegistry.register(Preset(
        "solar_system", "Solar System", UnitSystem(),
        components=[
            Component("sun", 0.05, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("belt", 0.95, 1.0, {'types': [ASTEROID], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 1.0), kinematic_model=DiskKinematics())
        ]
    ))
    
    # 3. chaos
    PresetRegistry.register(Preset(
        "chaos", "5 Subcluster Chaos", UnitSystem(),
        components=[
            Component(f"cluster_{i}", 0.2, 200.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(3.0, 200.0, center=(np.random.randn()*10, np.random.randn()*10)),
                      kinematic_model=IsotropicKinematics())
            for i in range(5)
        ]
    ))

    # 4. laplace
    PresetRegistry.register(Preset(
        "laplace", "Laplace Resonance", UnitSystem(),
        components=[
            Component("center", 0.01, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("rings", 0.99, 10.0, {'types': [PLANET], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 10.0), kinematic_model=DiskKinematics(velocity_dispersion=0.0))
        ]
    ))

    # 5. alpha_centauri
    PresetRegistry.register(Preset(
        "alpha_centauri", "Alpha Centauri System", UnitSystem(),
        components=[
            Component("stars", 0.1, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("debris", 0.9, 10.0, {'types': [ASTEROID], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(5.0, 10.0), kinematic_model=DiskKinematics())
        ]
    ))

    # 6. milkomeda
    PresetRegistry.register(Preset(
        "milkomeda", "MW-M31 Local Group Scenario", UnitSystem(),
        components=[
            Component("mw_disk", 0.1, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(3.0, 1000.0, center=(-30, 0)), kinematic_model=DiskKinematics()),
            Component("mw_halo", 0.4, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(15.0, 10.0, 5000.0, center=(-30, 0)), kinematic_model=NFWKinematics()),
            Component("m31_disk", 0.1, 1500.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(4.0, 1500.0, center=(30, 0)), kinematic_model=DiskKinematics()),
            Component("m31_halo", 0.4, 7500.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(15.0, 10.0, 7500.0, center=(30, 0)), kinematic_model=NFWKinematics()),
        ],
        visual_metadata={'bloom_int': {STAR: 0.10}}
    ))

    # 7. stephans_quintet
    PresetRegistry.register(Preset(
        "stephans_quintet", "Stephan's Quintet", UnitSystem(),
        components=[
            Component(f"g_{i}", 0.2, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(2.0, 1000.0, center=(np.random.randn()*20, np.random.randn()*20)),
                      kinematic_model=DiskKinematics())
            for i in range(5)
        ]
    ))

    # 8. messier13
    PresetRegistry.register(Preset(
        "messier13", "M13 Globular Cluster", UnitSystem(),
        components=[
            Component("stars", 1.0, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(3.0, 1000.0), kinematic_model=IsotropicKinematics())
        ]
    ))

    # 9. messier31
    PresetRegistry.register(Preset(
        "messier31", "Andromeda Galaxy Model", UnitSystem(),
        components=[
            Component("disk", 0.4, 3000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(5.4, 3000.0), kinematic_model=DiskKinematics()),
            Component("halo", 0.6, 9000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(15.0, 10.0, 9000.0), kinematic_model=NFWKinematics())
        ]
    ))

    # 10. ngc1052_df2
    PresetRegistry.register(Preset(
        "ngc1052_df2", "Diffuse Low-DM Galaxy", UnitSystem(),
        components=[
            Component("stars", 1.0, 500.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(5.0, 500.0), kinematic_model=IsotropicKinematics())
        ]
    ))

    # 11. castor_sextuple
    PresetRegistry.register(Preset(
        "castor_sextuple", "Hierarchical Hex-star", UnitSystem(),
        components=[
            Component("stars", 1.0, 600.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 600.0), kinematic_model=DiskKinematics())
        ],
        visual_metadata={'bloom_int': {STAR: 0.30}}
    ))

    # 12. hd98800_polar
    PresetRegistry.register(Preset(
        "hd98800_polar", "Polar Circumbinary Disk", UnitSystem(),
        components=[
            Component("stars", 0.1, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("disk", 0.9, 10.0, {'types': [ASTEROID], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 10.0), kinematic_model=DiskKinematics())
        ]
    ))

    # 13. trappist_1
    PresetRegistry.register(Preset(
        "trappist_1", "TRAPPIST-1 System", UnitSystem(),
        components=[
            Component("star", 0.1, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("planets", 0.9, 10.0, {'types': [PLANET], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 10.0), kinematic_model=DiskKinematics(velocity_dispersion=0.0))
        ],
        visual_metadata={
            'min_r': {11: 6.0, 3: 2.5},
            'bloom_int': {11: 0.25}
        }
    ))

    # 14. gravothermal_catastrophe
    PresetRegistry.register(Preset(
        "gravothermal_catastrophe", "Gravothermal Collapse", UnitSystem(),
        components=[
            Component("stars", 1.0, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(2.0, 1000.0), kinematic_model=IsotropicKinematics())
        ]
    ))

    # 15. wr104_pinwheel
    PresetRegistry.register(Preset(
        "wr104_pinwheel", "Colliding Wind Pinwheel", UnitSystem(),
        components=[
            Component("stars", 0.05, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("dust", 0.95, 10.0, {'types': [ASTEROID], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(15.0, 10.0), kinematic_model=DiskKinematics())
        ]
    ))

    # 16. omega_centauri
    PresetRegistry.register(Preset(
        "omega_centauri", "IMBH Globular Cluster", UnitSystem(),
        components=[
            Component("imbh", 0.01, 1000.0, {'types': [BLACK_HOLE], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("stars", 0.99, 10000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(1.0, 10000.0), kinematic_model=IsotropicKinematics())
        ],
        visual_metadata={
            'bloom_int': {8: 0.20, 7: 0.25},
            'mass_scale': {7: 5.0, 0: 15.0, 11: 40.0, 8: 10.0}
        }
    ))

    # 17. pleiades_m45
    PresetRegistry.register(Preset(
        "pleiades_m45", "Pleiades Open Cluster", UnitSystem(),
        components=[
            Component("stars", 1.0, 500.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(10.0, 500.0), kinematic_model=IsotropicKinematics())
        ],
        visual_metadata={
            'bloom_int': {7: 0.35, 0: 0.15, 11: 0.05},
            'bloom_spr': {7: 0.5},
            'mass_scale': {7: 0.05, 0: 0.30, 11: 0.80},
            'min_r': {0: 2.0, 11: 1.0},
            'max_r': {0: 6.0}
        }
    ))

    # 18. hirayama_family
    PresetRegistry.register(Preset(
        "hirayama_family", "Asteroid Disruption", UnitSystem(),
        components=[
            Component("sun", 0.01, 1000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PointMassSpatial(1000.0), kinematic_model=PointMassKinematics()),
            Component("asteroids", 0.99, 1.0, {'types': [ASTEROID], 'probs': [1.0]}, visible=True,
                      spatial_model=ExponentialDiskSpatial(10.0, 1.0), kinematic_model=DiskKinematics(velocity_dispersion=0.2))
        ]
    ))

    # 19. dark_matter_halo_merger
    PresetRegistry.register(Preset(
        "dark_matter_halo_merger", "DM Halo Merger", UnitSystem(),
        components=[
            Component("halo1", 0.5, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(10.0, 10.0, 5000.0, center=(-15, 0)), kinematic_model=NFWKinematics(drift=(1.0, 0.0))),
            Component("halo2", 0.5, 5000.0, {'types': [DARK_MATTER], 'probs': [1.0]}, visible=False,
                      spatial_model=NFWSpatial(10.0, 10.0, 5000.0, center=(15, 0)), kinematic_model=NFWKinematics(drift=(-1.0, 0.0)))
        ]
    ))

    # 20. great_attractor
    PresetRegistry.register(Preset(
        "great_attractor", "Large Scale Structure Proxy", UnitSystem(),
        components=[
            Component("cluster1", 0.33, 10000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(10.0, 10000.0, center=(-20, -20)), kinematic_model=IsotropicKinematics()),
            Component("cluster2", 0.33, 15000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(15.0, 15000.0, center=(20, 20)), kinematic_model=IsotropicKinematics()),
            Component("cluster3", 0.34, 12000.0, {'types': [STAR], 'probs': [1.0]}, visible=True,
                      spatial_model=PlummerSpatial(12.0, 12000.0, center=(0, 0)), kinematic_model=IsotropicKinematics())
        ]
    ))

ASTEROID = 4

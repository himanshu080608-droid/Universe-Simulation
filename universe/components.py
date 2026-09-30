import numpy as np

class Component:
    def __init__(self, name, particle_fraction, total_mass, particle_types, visible=True, spatial_model=None, kinematic_model=None, metadata=None):
        self.name = name
        self.particle_fraction = particle_fraction
        self.total_mass = total_mass
        self.particle_types = particle_types
        self.visible = visible
        self.spatial_model = spatial_model
        self.kinematic_model = kinematic_model
        self.metadata = metadata or {}

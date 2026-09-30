from universe.registry import PresetRegistry

class UnitSystem:
    def __init__(self, length="1 normalized", mass="1 normalized", time="1 normalized", G=1.0, eps=1.0):
        self.length = length
        self.mass = mass
        self.time = time
        self.G = G
        self.eps = eps

class Preset:
    def __init__(self, name, description, unit_system=None, components=None):
        self.name = name
        self.description = description
        self.unit_system = unit_system or UnitSystem()
        self.components = components or []

    def get_render_semantics(self):
        invisible = set()
        auto_fit_exclude = set()
        for c in self.components:
            for t in c.particle_types['types']:
                if not c.visible:
                    invisible.add(t)
                if not c.visible or c.metadata.get('auto_fit_exclude', False):
                    auto_fit_exclude.add(t)
        return {
            'invisible_types': list(invisible),
            'auto_fit_exclude_types': list(auto_fit_exclude)
        }

    def generate(self, N_total, seed=42):
        import numpy as np
        rng = np.random.default_rng(seed)
        
        # Determine N per component
        total_fraction = sum(c.particle_fraction for c in self.components)
        all_pos, all_vel, all_mass, all_type = [], [], [], []
        
        # 1. Spatial Initialization
        spatial_results = []
        for c in self.components:
            N_c = max(1, int(N_total * (c.particle_fraction / total_fraction)))
            if N_c > 0:
                pos, r = c.spatial_model.sample_positions(N_c, rng)
                types = rng.choice(c.particle_types['types'], p=c.particle_types.get('probs'), size=N_c)
                masses = np.full(N_c, c.total_mass / N_c)
                spatial_results.append((c, pos, r, types, masses, N_c))
        
        # 2. Kinematic Initialization (Global solver can access all spatial models)
        for c, pos, r, types, masses, N_c in spatial_results:
            vel = c.kinematic_model.solve(pos, r, self.components, self.unit_system, rng)
            all_pos.append(pos)
            all_vel.append(vel)
            all_mass.append(masses)
            all_type.append(types)
            
        return (np.vstack(all_pos), np.vstack(all_vel),
                np.concatenate(all_mass), np.concatenate(all_type))

class LegacyAdapterPreset(Preset):
    def __init__(self, name, description, legacy_func):
        super().__init__(name, description)
        self.legacy_func = legacy_func

    def generate(self, N_total, seed=42):
        return self.legacy_func(self.name, N_total, seed)

    def get_render_semantics(self):
        return {
            'invisible_types': [14], # DARK_MATTER
            'auto_fit_exclude_types': [2, 14] # COMET, DARK_MATTER
        }

from universe.registry import PresetRegistry

class UnitSystem:
    def __init__(self, L0=1.0, M0=1.0, T0=1.0, G_sim=1.0, eps_sim=1.0,
                 L_unit="normalized", M_unit="normalized", T_unit="normalized"):
        self.L0 = L0
        self.M0 = M0
        self.T0 = T0
        self.G = G_sim
        self.eps = eps_sim
        self.L_unit = L_unit
        self.M_unit = M_unit
        self.T_unit = T_unit
        self.V0 = L0 / T0
        
    def length_to_sim(self, physical_length):
        return physical_length / self.L0
        
    def length_to_physical(self, sim_length):
        return sim_length * self.L0
        
    def mass_to_sim(self, physical_mass):
        return physical_mass / self.M0
        
    def mass_to_physical(self, sim_mass):
        return sim_mass * self.M0
        
    def velocity_to_sim(self, physical_vel):
        return physical_vel / self.V0
        
    def velocity_to_physical(self, sim_vel):
        return sim_vel * self.V0

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
        
        # Use SeedSequence for stable independent per-component streams
        sq = np.random.SeedSequence(seed)
        
        # Determine exact N per component to avoid exceeding N_total
        total_fraction = sum(c.particle_fraction for c in self.components)
        if total_fraction == 0:
            total_fraction = 1.0 # fallback
        
        N_assigned = []
        remainder = N_total
        for c in self.components[:-1]:
            n = int(N_total * (c.particle_fraction / total_fraction))
            N_assigned.append(n)
            remainder -= n
        if self.components:
            N_assigned.append(max(0, remainder))
            
        all_pos, all_vel, all_mass, all_type = [], [], [], []
        
        # 1. Spatial Initialization
        spatial_results = []
        for i, c in enumerate(self.components):
            # Component-local RNG derived stably from its name hash and base seed
            c_seed = int(sq.generate_state(1)[0]) ^ hash(c.name)
            c_rng = np.random.default_rng(abs(c_seed) % (2**31 - 1))
            
            N_c = N_assigned[i]
            if N_c > 0:
                pos, r = c.spatial_model.sample_positions(N_c, c_rng)
                types = c_rng.choice(c.particle_types['types'], p=c.particle_types.get('probs'), size=N_c)
                masses = np.full(N_c, c.total_mass / N_c)
                spatial_results.append((c, pos, r, types, masses, N_c, c_rng))
        
        # 2. Kinematic Initialization (Global solver can access all spatial models)
        self._last_slices = {}
        current_idx = 0
        for c, pos, r, types, masses, N_c, c_rng in spatial_results:
            vel = c.kinematic_model.solve(pos, r, self.components, self.unit_system, c_rng)
            all_pos.append(pos)
            all_vel.append(vel)
            all_mass.append(masses)
            all_type.append(types)
            
            self._last_slices[c.name] = (current_idx, current_idx + N_c)
            current_idx += N_c
            
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

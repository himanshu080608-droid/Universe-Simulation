from universe.registry import PresetRegistry

class UnitSystem:
    def __init__(self, L0=1.0, M0=1.0, T0=1.0, G_phys=1.0, eps_sim=1.0,
                 L_unit="normalized", M_unit="normalized", T_unit="normalized",
                 G_sim=None):
        self.L0 = L0
        self.M0 = M0
        self.T0 = T0
        self.G_phys = G_phys
        self.eps = eps_sim
        self.L_unit = L_unit
        self.M_unit = M_unit
        self.T_unit = T_unit
        self.V0 = L0 / T0
        
        # Verify dimensionally coherent G_sim
        expected_G_sim = G_phys * M0 * (T0**2) / (L0**3)
        if G_sim is not None and abs(G_sim - expected_G_sim) > 1e-7:
            raise ValueError(f"Inconsistent G_sim: {G_sim} != expected {expected_G_sim}")
        self.G = expected_G_sim
        
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
    def __init__(self, name, description, unit_system=None, components=None, visual_metadata=None):
        self.name = name
        self.description = description
        self.unit_system = unit_system or UnitSystem()
        self.components = components or []
        self.visual_metadata = visual_metadata or {}

    def get_render_semantics(self):
        invisible = set()
        auto_fit_exclude = set()
        for c in self.components:
            for t in c.particle_types['types']:
                if not c.visible:
                    invisible.add(t)
                if not c.visible or c.metadata.get('auto_fit_exclude', False):
                    auto_fit_exclude.add(t)
        semantics = {
            'invisible_types': list(invisible),
            'auto_fit_exclude_types': list(auto_fit_exclude)
        }
        semantics.update(self.visual_metadata)
        return semantics

    def generate(self, N_total, seed=42):
        import numpy as np
        
        import hashlib
        
        # Use SeedSequence for stable independent per-component streams
        sq = np.random.SeedSequence(seed)
        
        # Determine exact N per component to avoid exceeding N_total
        total_fraction = sum(c.particle_fraction for c in self.components)
        if total_fraction == 0:
            total_fraction = 1.0 # fallback
        
        N_assigned = [0] * len(self.components)
        if self.components:
            ideal_counts = [N_total * (c.particle_fraction / total_fraction) for c in self.components]
            floor_counts = [int(np.floor(ic)) for ic in ideal_counts]
            remainder = N_total - sum(floor_counts)
            
            remainders = [(ic - fc, i) for i, (ic, fc) in enumerate(zip(ideal_counts, floor_counts))]
            remainders.sort(key=lambda x: (-x[0], x[1]))
            
            N_assigned = floor_counts
            for idx in range(remainder):
                N_assigned[remainders[idx][1]] += 1
                
        all_pos, all_vel, all_mass, all_type, all_ids = [], [], [], [], []
        
        # 1. Spatial Initialization
        spatial_results = []
        for i, c in enumerate(self.components):
            # Component-local RNG derived stably from its name hash and base seed
            name_hash = int(hashlib.sha256(c.name.encode('utf-8')).hexdigest()[:15], 16)
            c_seed = int(sq.generate_state(1)[0]) ^ name_hash
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
            all_ids.append(np.full(N_c, i, dtype=np.int32))
            
            self._last_slices[c.name] = (current_idx, current_idx + N_c)
            current_idx += N_c
            
        self.last_identities = np.concatenate(all_ids) if all_ids else np.array([], dtype=np.int32)
            
        return (np.vstack(all_pos) if all_pos else np.zeros((0,2), dtype=np.float64),
                np.vstack(all_vel) if all_vel else np.zeros((0,2), dtype=np.float64),
                np.concatenate(all_mass) if all_mass else np.array([], dtype=np.float64),
                np.concatenate(all_type) if all_type else np.array([], dtype=np.int32))


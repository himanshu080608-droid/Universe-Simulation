"""
Universe particle generator.
Supports three particle archetypes:
  STAR   — massive, hot, clustered in dense stellar cores
  PLANET — medium mass, in stable circular orbits around stars
  COMET  — lightweight, eccentric hyperbolic/parabolic trajectories

G = 1 in normalized units. Mass hierarchy (realistic ratios):
  M_star >> M_planet >> M_comet
  e.g. star 200–3000 : planet 0.5–5 : comet 0.001

Particle types encoded as integers:
  0 = STAR, 1 = PLANET, 2 = COMET
"""

import numpy as np

STAR   = np.int32(0)
PLANET = np.int32(1)
COMET  = np.int32(2)

PIXEL_COLORS = {
    STAR:   (255, 220, 60),    # warm gold
    PLANET: (50, 165, 255),    # cool blue
    COMET:  (178, 255, 215),   # icy cyan-green
}


class StellarCluster:
    """
    Stellar cluster using Hernquist radial profile.
    Velocities computed from the ACTUAL Hernquist circular velocity:
        v_circ(r) = sqrt(G * M_total * r) / (r + a)
    with G = 1, giving properly bound, non-escaping orbits.
    """
    def __init__(self, center=(0.0, 0.0), drift=(0.0, 0.0),
                 num_stars=200, core_radius=4.0, total_mass=1500.0,
                 rng=None):
        self.center    = np.array(center, dtype=np.float64)
        self.drift     = np.array(drift,  dtype=np.float64)
        self.N         = num_stars
        self.radius    = core_radius
        self.mass      = total_mass / max(num_stars, 1)
        self.rng       = rng or np.random.default_rng()

    def generate(self):
        rng = self.rng
        total_mass = self.mass * self.N

        # Hernquist scale radius
        a = self.radius * 0.2

        # Radial sampling: r ~ Hernquist CDF
        # To keep stars within self.radius naturally without a hard ring, we limit u.
        # r_max = self.radius = 5*a -> sqrt(u) = 5/6 -> u = 25/36 = 0.694
        u = rng.uniform(0.001, 0.69, self.N)
        r = a * np.sqrt(u) / (1.0 - np.sqrt(u) + 1e-9)

        # Clamp: r >= 0.5 (avoid singularity), no upper bound needed
        r = np.clip(r, 0.5, None)


        theta = rng.uniform(0, 2 * np.pi, self.N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center

        # Hernquist circular velocity (exact, G=1)
        v_circ = np.sqrt(total_mass * r) / (r + a + 1e-9)

        # Tiny thermal dispersion: 2% of v_circ
        sigma  = 0.02 * v_circ
        v_disp = rng.normal(0, sigma[:, None], (self.N, 2))

        vel = np.column_stack((-v_circ * np.sin(theta),
                                v_circ * np.cos(theta))) + v_disp + self.drift

        masses = np.full(self.N, self.mass)
        types  = np.full(self.N, STAR, dtype=np.int32)
        return pos, vel, masses, types


class OrbitalPlanetBelt:
    """
    Ring of PLANET particles in near-circular orbits.
    Vis-viva: v_orb = sqrt(G * M_central / r), G = 1.
    """
    def __init__(self, center=(0.0, 0.0), drift=(0.0, 0.0),
                 num_planets=50, r_min=3.0, r_max=6.5,
                 planet_mass=1.0, central_mass=1000.0, G=1.0, rng=None):
        self.center = np.array(center, dtype=np.float64)
        self.drift  = np.array(drift,  dtype=np.float64)
        self.N      = num_planets
        self.r_min  = r_min
        self.r_max  = r_max
        self.mass   = planet_mass
        self.M      = central_mass
        self.G      = G
        self.rng    = rng or np.random.default_rng()

    def generate(self):
        rng = self.rng
        r     = rng.uniform(self.r_min, self.r_max, self.N)
        theta = rng.uniform(0, 2 * np.pi, self.N)

        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center

        v_orb = np.sqrt(self.G * self.M / r)
        # Tiny eccentricity: ±2% random perturbation
        v_eps = rng.normal(0, 0.02 * v_orb)
        v_orb = v_orb + v_eps

        vel = np.column_stack((-v_orb * np.sin(theta),
                                v_orb * np.cos(theta))) + self.drift

        masses = np.full(self.N, self.mass)
        types  = np.full(self.N, PLANET, dtype=np.int32)
        return pos, vel, masses, types


class CometCloud:
    """
    Oort-cloud-like distribution of COMET particles.
    Random-direction velocities at 30–80% of local escape speed → eccentric orbits.
    """
    def __init__(self, center=(0.0, 0.0), drift=(0.0, 0.0),
                 num_comets=300, cloud_radius=18.0,
                 comet_mass=0.001, central_mass=1000.0, G=1.0, rng=None):
        self.center = np.array(center, dtype=np.float64)
        self.drift  = np.array(drift,  dtype=np.float64)
        self.N      = num_comets
        self.R      = cloud_radius
        self.mass   = comet_mass
        self.M      = central_mass
        self.G      = G
        self.rng    = rng or np.random.default_rng()

    def generate(self):
        rng = self.rng
        # Exponential falloff: r ~ -R/2 * ln(u), capped at 2*R.
        # This gives high density near center, smooth tail — NO hard outer ring.
        u = rng.uniform(1e-4, 1.0, self.N)
        r = np.clip(-self.R * 0.5 * np.log(u), self.R * 0.05, self.R * 2.0)

        theta = rng.uniform(0, 2 * np.pi, self.N)

        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center

        # Circular orbital velocity
        v_circ = np.sqrt(self.G * self.M / r)

        # Give each comet a random eccentricity: v = v_circ * sqrt(1 + e)
        # where e in [-0.6, +0.6]. Positive e = hyperbolic boost, negative = elliptical.
        # Tangential only: this ensures perihelion = r*(1-|e|) >= r*0.4 >= 0.4*r_min.
        # This PREVENTS comets from falling radially into the star.
        ecc = rng.uniform(-0.6, 0.6, self.N)        # eccentricity factor
        v_tang = v_circ * np.sqrt(np.clip(1.0 + ecc, 0.1, 1.7))

        vel = np.column_stack((-v_tang * np.sin(theta),
                                v_tang * np.cos(theta))) + self.drift

        masses = np.full(self.N, self.mass)
        types  = np.full(self.N, COMET, dtype=np.int32)
        return pos, vel, masses, types


class GalacticDisk:
    """
    Flat rotating disk: stellar bulge (StellarCluster) + planet belt + comet cloud.
    Mass hierarchy: stars carry 80%, planets 18%, comets 2% of total disk mass.
    """
    def __init__(self, center=(0.0, 0.0), drift=(0.0, 0.0),
                 n_stars=200, n_planets=150, n_comets=400,
                 disk_radius=14.0, total_mass=3000.0,
                 G=1.0, rng=None):
        self.center    = np.array(center, dtype=np.float64)
        self.drift     = np.array(drift,  dtype=np.float64)
        self.n_stars   = n_stars
        self.n_planets = n_planets
        self.n_comets  = n_comets
        self.R         = disk_radius
        self.total_mass = total_mass
        self.G         = G
        self.rng       = rng or np.random.default_rng()

    def generate(self):
        M = self.total_mass
        smbh_mass = M * 0.15   # central SMBH = 15% of galaxy mass

        # ── Mass fractions (remaining 85% split among disk particles) ────────
        disk_M = M - smbh_mass
        star_mass_total   = disk_M * 0.75
        comet_mass_total  = disk_M * 0.25

        m_star  = star_mass_total  / max(self.n_stars + self.n_planets, 1)
        m_comet = comet_mass_total / max(self.n_comets, 1)

        central_M = smbh_mass + star_mass_total * 0.5  # enclosed mass for v_orb

        core_radius = self.R * 0.30

        # ── Central SMBH ─────────────────────────────────────────────────────
        all_pos  = [self.center[None, :].copy()]
        all_vel  = [self.drift[None, :].copy()]
        all_mass = [np.array([smbh_mass])]
        all_type = [np.array([STAR], dtype=np.int32)]

        # ── Stellar disk: bulge (Hernquist) + extended disk stars ─────────────
        # Use ALL of n_stars + n_planets as disk stars for a smooth distribution
        n_disk = self.n_stars + self.n_planets
        bulge = StellarCluster(
            center=self.center, drift=self.drift,
            num_stars=n_disk,
            core_radius=self.R,  # Use full disk radius so stars fill the galaxy
            total_mass=star_mass_total,
            rng=self.rng
        )
        sp, sv, sm, st = bulge.generate()
        
        # Color a portion of the disk particles as PLANETs
        if self.n_planets > 0:
            st[-self.n_planets:] = PLANET

        all_pos.append(sp);  all_vel.append(sv)
        all_mass.append(sm); all_type.append(st)

        # ── Comet halo ────────────────────────────────────────────────────────
        comets = CometCloud(
            center=self.center, drift=self.drift,
            num_comets=self.n_comets,
            cloud_radius=self.R * 1.2,   # slightly extended beyond stellar disk
            comet_mass=m_comet, central_mass=central_M,

            G=self.G, rng=self.rng
        )
        cp, cv, cm, ct = comets.generate()
        all_pos.append(cp);  all_vel.append(cv)
        all_mass.append(cm); all_type.append(ct)

        return (np.vstack(all_pos), np.vstack(all_vel),
                np.concatenate(all_mass), np.concatenate(all_type))


# ─────────────────────────────────────────────────────────────────────────────
def build_universe(preset="twin_galaxies", N_total=5000, seed=42):
    """
    High-level factory. Returns (pos, vel, mass, types).

    G = 1 (normalized). Typical mass scale:
      Star particle  : 3–30 mass units
      Planet particle: 0.5–5 mass units
      Comet particle : 0.001–0.01 mass units

    Presets:
      'twin_galaxies' — two colliding galactic disks (default)
      'solar_system'  — single star + planet belt + Oort cloud
      'chaos'         — 5 random clusters on collision course
      'laplace'       — ordered concentric rings (determinism demo)
    """
    rng = np.random.default_rng(seed)

    if preset == "twin_galaxies":
        # Two colliding galactic disks.
        # total_mass=600 per galaxy keeps stellar circular velocities compatible
        # with dt=0.001, eps=1.0 (max v_circ ~8 at r=3, T_min ~2.4 >> 0.1)
        n_each    = N_total // 2
        n_stars   = n_each * 20 // 100
        n_planets = n_each * 30 // 100
        n_comets  = n_each - n_stars - n_planets

        g1 = GalacticDisk(center=(-38, 3), drift=(0.18, 0.0),
                          n_stars=n_stars, n_planets=n_planets, n_comets=n_comets,
                          disk_radius=28, total_mass=600, G=1.0, rng=rng)
        g2 = GalacticDisk(center=(38, -3), drift=(-0.18, 0.0),
                          n_stars=n_stars, n_planets=n_planets, n_comets=n_comets,
                          disk_radius=28, total_mass=600, G=1.0, rng=rng)

        parts = [g1.generate(), g2.generate()]

    elif preset == "solar_system":
        # ── Accurate Solar System ────────────────────────────────────────────
        # G=1 normalized units. AU=12.8 sim units → Mercury at r≈5 (well
        # outside softening eps=1), Neptune at r≈385.
        # star_mass=500 → Mercury v_orb≈10, T≈3.1 (3100 steps, well resolved).

        star_mass = 500.0    # Sun (normalized)
        AU        = 12.8     # sim units per AU (Mercury → r≈5, Neptune → r≈385)

        # Earth mass in sim units = star_mass / 333000 ≈ 0.0015
        M_earth = star_mass / 333_000.0

        # ── 8 planets: (name, r_AU, mass_in_earth_masses, n_ring_particles) ─
        planets_data = [
            ("Mercury",  0.39,   0.055,  25),
            ("Venus",    0.72,   0.815,  30),
            ("Earth",    1.00,   1.000,  35),
            ("Mars",     1.52,   0.107,  25),
            ("Jupiter",  5.20, 317.8,    60),
            ("Saturn",   9.54,  95.2,    55),
            ("Uranus",  19.18,  14.5,    40),
            ("Neptune", 30.07,  17.1,    35),
        ]

        all_pos, all_vel, all_mass, all_type = [], [], [], []

        # ── Central star ─────────────────────────────────────────────────────
        all_pos.append(np.zeros((1, 2), dtype=np.float64))
        all_vel.append(np.zeros((1, 2), dtype=np.float64))
        all_mass.append(np.array([star_mass], dtype=np.float64))
        all_type.append(np.array([STAR], dtype=np.int32))

        n_used = 1

        # ── Place each planet + its orbital ring ────────────────────────────
        for name, r_au, me, n_ring in planets_data:
            r_sim  = r_au * AU
            m_body = M_earth * me

            v_orb = np.sqrt(star_mass / max(r_sim, 0.5))

            theta0 = rng.uniform(0, 2 * np.pi)
            px = r_sim * np.cos(theta0)
            py = r_sim * np.sin(theta0)
            vx = -v_orb * np.sin(theta0)
            vy =  v_orb * np.cos(theta0)

            all_pos.append(np.array([[px, py]]))
            all_vel.append(np.array([[vx, vy]]))
            all_mass.append(np.array([m_body]))
            ptype = STAR if me > 10 else PLANET
            all_type.append(np.array([ptype], dtype=np.int32))
            n_used += 1

            # Ring of test-particles co-orbiting in the same lane (±2% width)
            dr     = r_sim * 0.02
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            r_ring = rng.uniform(max(r_sim - dr, 0.3), r_sim + dr, n_ring)
            ring_pos = np.column_stack((r_ring * np.cos(angles),
                                        r_ring * np.sin(angles)))
            v_ring = np.sqrt(star_mass / np.clip(r_ring, 0.3, None))
            ring_vel = np.column_stack((-v_ring * np.sin(angles),
                                         v_ring * np.cos(angles)))
            all_pos.append(ring_pos)
            all_vel.append(ring_vel)
            all_mass.append(np.full(n_ring, m_body * 0.001))
            all_type.append(np.full(n_ring, PLANET, dtype=np.int32))
            n_used += n_ring

        # ── Asteroid belt (2.2–3.2 AU) ───────────────────────────────────────
        n_asteroids = 150
        belt = OrbitalPlanetBelt(
            num_planets=n_asteroids,
            r_min=2.2 * AU, r_max=3.2 * AU,
            planet_mass=M_earth * 0.0001,
            central_mass=star_mass, G=1.0, rng=rng
        )
        bp, bv, bm, bt = belt.generate()
        all_pos.append(bp); all_vel.append(bv)
        all_mass.append(bm); all_type.append(bt)
        n_used += n_asteroids

        # ── Kuiper Belt / short-period comets (35–90 AU) ─────────────────────
        # Kuiper Belt / short-period comets: 35–55 AU from Sun.
        # At AU=12.8, that is 448–704 sim units (~1.8× Neptune orbit).
        # User can drag/zoom to explore the outer system freely.
        n_comets = max(N_total - n_used, 50)
        oort = CometCloud(
            num_comets=n_comets,
            cloud_radius=55.0 * AU,   # 704 sim units ≈ 1.8× Neptune
            comet_mass=M_earth * 1e-5,
            central_mass=star_mass, G=1.0, rng=rng
        )

        cp, cv, cm, ct = oort.generate()
        all_pos.append(cp); all_vel.append(cv)
        all_mass.append(cm); all_type.append(ct)

        parts = [(np.vstack(all_pos), np.vstack(all_vel),
                  np.concatenate(all_mass), np.concatenate(all_type))]




    elif preset == "chaos":
        # 5 galaxy clusters arranged on a pentagon, all falling toward center.
        # Fixed geometry ensures clusters are always visually distinct and
        # guaranteed to collide in the middle.
        n_clusters = 5
        n_each     = N_total // n_clusters
        n_s = n_each * 25 // 100
        n_p = n_each * 25 // 100
        n_c = n_each - n_s - n_p

        # Pentagon: 5 equally-spaced points at r=28, rotated 18° so no cluster
        # sits exactly on the x-axis (makes the collision more visually interesting)
        ring_r = 28.0
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + np.radians(18)
        positions = np.column_stack((ring_r * np.cos(angles), ring_r * np.sin(angles)))

        # Each cluster drifts toward the origin at speed 0.12, giving a clean merge
        inward_speed = 0.12
        drifts = -positions / np.linalg.norm(positions, axis=1, keepdims=True) * inward_speed

        parts = []
        for k in range(n_clusters):
            g = GalacticDisk(
                center=positions[k], drift=drifts[k],
                n_stars=n_s, n_planets=n_p, n_comets=n_c,
                disk_radius=8, total_mass=250, G=1.0, rng=rng
            )
            parts.append(g.generate())

    elif preset == "laplace":
        # Ordered concentric rings: perfect determinism demonstration.
        # All particles on exact circular orbits → zero chaos by design.
        rings   = 8
        n_per   = N_total // rings
        central_M = 2000.0

        all_pos, all_vel, all_mass, all_type = [], [], [], []

        # Central massive "black hole"
        all_pos.append(np.array([[0.0, 0.0]]))
        all_vel.append(np.array([[0.0, 0.0]]))
        all_mass.append(np.array([central_M]))
        all_type.append(np.array([STAR], dtype=np.int32))

        ring_types = [STAR, PLANET, COMET, PLANET, STAR, COMET, PLANET, STAR]
        ring_masses = [20.0, 2.0, 0.01, 2.0, 20.0, 0.01, 2.0, 20.0]

        for k in range(rings):
            r      = 4.0 + k * 4.0
            angles = np.linspace(0, 2 * np.pi, n_per, endpoint=False)
            pos    = np.column_stack((r * np.cos(angles), r * np.sin(angles)))
            v_orb  = np.sqrt(1.0 * central_M / r)
            vel    = np.column_stack((-v_orb * np.sin(angles), v_orb * np.cos(angles)))
            mass   = np.full(n_per, ring_masses[k])
            types  = np.full(n_per, ring_types[k], dtype=np.int32)

            all_pos.append(pos);  all_vel.append(vel)
            all_mass.append(mass); all_type.append(types)

        parts = [(np.vstack(all_pos), np.vstack(all_vel),
                  np.concatenate(all_mass), np.concatenate(all_type))]

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    pos   = np.vstack([p[0] for p in parts]).astype(np.float64)
    vel   = np.vstack([p[1] for p in parts]).astype(np.float64)
    mass  = np.concatenate([p[2] for p in parts]).astype(np.float64)
    types = np.concatenate([p[3] for p in parts]).astype(np.int32)

    return pos, vel, mass, types

"""
Universe particle generator.
Supports five particle archetypes:
  STAR      — massive, hot, clustered in dense stellar cores
  PLANET    — generic orbital particle / accretion disk filler
  COMET     — lightweight, eccentric hyperbolic/parabolic trajectories
  ROCKY     — small rocky inner planets (Mercury, Venus, Earth, Mars)
  GAS_GIANT — large gas/ice giants (Jupiter, Saturn, Uranus, Neptune)

G = 1 in normalized units. Mass hierarchy (realistic ratios):
  M_star >> M_gas_giant >> M_rocky >> M_comet
  e.g. star 500 : gas_giant 0.5 : rocky 0.0015 : comet 1e-5

Particle types encoded as integers:
  0 = STAR, 1 = PLANET, 2 = COMET, 3 = ROCKY, 4 = GAS_GIANT
"""

import numpy as np

# Softening length matching the engine's DEFAULT_EPS = 1.0
_EPS = 1.0

def softened_v_circ(G, M_enc, r, eps=_EPS):
    """
    Circular speed in the EXACT softened potential used by Barnes-Hut engine.
    Force: F = G*M*m * r_vec / (r^2 + eps^2)^(3/2)
    => v_circ = sqrt(G * M_enc * r^2 / (r^2 + eps^2)^(3/2))
    This ensures generated velocities perfectly balance the engine forces.
    """
    r2   = r * r + eps * eps
    return np.sqrt(G * M_enc * r * r / (r2 ** 1.5))

STAR           = np.int32(0)
PLANET         = np.int32(1)
COMET          = np.int32(2)
ROCKY          = np.int32(3)
GAS_GIANT      = np.int32(4)
BLACK_HOLE     = np.int32(5)
RED_GIANT      = np.int32(6)
BLUE_STRAGGLER = np.int32(7)
WHITE_DWARF    = np.int32(8)
NEUTRON_STAR   = np.int32(9)
EMISSION_STAR  = np.int32(10)
RED_DWARF      = np.int32(11)
ASTEROID       = np.int32(12)

PIXEL_COLORS = {
    STAR:           (255, 210, 100),   # 0  Warm Honey Gold (G-type Sun-like main sequence star)
    PLANET:         ( 90, 175, 235),   # 1  Soft Sky Blue (orbital disk / asteroid dot)
    COMET:          (140, 245, 210),   # 2  Icy Aquamarine (comet & polar jet plasma)
    ROCKY:          (240, 110,  50),   # 3  Warm Terracotta Rust (terrestrial rocky planet)
    GAS_GIANT:      (210, 150,  90),   # 4  Jovian Bands (rich amber/gold)
    BLACK_HOLE:     (255,  45,  65),   # 5  Radiant Crimson Red (accreting black hole core)
    RED_GIANT:      (255, 110,  75),   # 6  Warm Amber Coral (evolved Red Giant & Supergiant)
    BLUE_STRAGGLER: (120, 180, 255),   # 7  Luminous Sapphire Ice Blue (hot O/B star & Blue Straggler)
    WHITE_DWARF:    (240, 245, 255),   # 8  Soft Pearl White (compact White Dwarf remnant)
    NEUTRON_STAR:   (255,  50, 255),   # 9  Neon Magenta (Pulsar / Magnetar - easily distinguishable)
    EMISSION_STAR:  ( 90, 220, 170),   # 10 Soft Emerald Teal (Wolf-Rayet O-III emission star)
    RED_DWARF:      (235,  70,  60),   # 11 Deep Crimson / Coral (M-type Red Dwarf)
    ASTEROID:       (130, 130, 130),   # 12 Dusty Grey (Asteroids / Debris)
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
        u = rng.uniform(0.001, 0.69, self.N)
        r = a * np.sqrt(u) / (1.0 - np.sqrt(u) + 1e-9)
        r = np.clip(r, self.radius * 0.15, None)

        theta = rng.uniform(0, 2 * np.pi, self.N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center

        # Hernquist circular velocity (exact, G=1)
        v_circ = np.sqrt(total_mass * r) / (r + a + 1e-9)

        # Tiny thermal dispersion: 2% of v_circ
        sigma  = 0.02 * v_circ
        v_disp = rng.normal(0, sigma[:, None], (self.N, 2))

        vel = np.column_stack((-v_circ * np.sin(theta),
                                v_circ * np.cos(theta))) + v_disp + self.drift

        # Sample realistic stellar population (HR-Diagram color balance including Red Dwarfs)
        star_types   = np.array([BLUE_STRAGGLER, STAR, RED_GIANT, WHITE_DWARF, NEUTRON_STAR, EMISSION_STAR, RED_DWARF], dtype=np.int32)
        type_weights = [0.10, 0.35, 0.15, 0.05, 0.02, 0.01, 0.32]
        types  = rng.choice(star_types, size=self.N, p=type_weights)
        masses = np.full(self.N, self.mass)
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
        types  = np.full(self.N, ASTEROID, dtype=np.int32)
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
        u = rng.uniform(1e-4, 1.0, self.N)
        r = np.clip(-self.R * 0.5 * np.log(u), self.R * 0.05, self.R * 2.0)

        theta = rng.uniform(0, 2 * np.pi, self.N)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center

        # Softened circular velocity — matches Barnes-Hut engine exactly
        v_circ = softened_v_circ(self.G, self.M, r)

        # Give each comet a random eccentricity: v = v_circ * sqrt(1 + e)
        # where e in [-0.6, +0.6]. Elliptical to parabolic, never fully hyperbolic.
        ecc    = rng.uniform(-0.6, 0.5, self.N)
        v_tang = v_circ * np.sqrt(np.clip(1.0 + ecc, 0.1, 1.5))

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
        smbh_mass = M * 0.45   # Massive central core prevents bar instability (flat rotation curve)
        disk_M    = M - smbh_mass
        star_mass_total   = disk_M * 0.90
        planet_mass_total = disk_M * 0.08
        comet_mass_total  = disk_M * 0.02

        m_star   = star_mass_total   / max(self.n_stars,   1)
        m_planet = planet_mass_total / max(self.n_planets, 1)
        m_comet  = comet_mass_total  / max(self.n_comets,  1)

        # ── Central SMBH ─────────────────────────────────────────────────────
        all_pos  = [self.center[None, :].copy()]
        all_vel  = [self.drift[None, :].copy()]
        all_mass = [np.array([smbh_mass])]
        all_type = [np.array([BLACK_HOLE], dtype=np.int32)]

        # ── Stellar disk (Spiral Arms) ────────────────────────────────────────
        if self.n_stars > 0:
            a = self.R * 0.2
            # Use u > 0.05 to naturally create an inner hole without breaking density profile
            u = self.rng.uniform(0.05, 0.69, self.n_stars)
            r = a * np.sqrt(u) / (1.0 - np.sqrt(u) + 1e-9)

            # Generate Spiral Arms (4 arms gives a fuller, 'milkier' look)
            n_arms = 4
            arm_offset = self.rng.integers(0, n_arms, self.n_stars) * (2 * np.pi / n_arms)
            winding = 2.0
            theta_spiral = (r / self.R) * np.pi * winding + arm_offset
            theta_scatter = self.rng.normal(0, 0.20, self.n_stars) # Tighter organic scatter for sharper arms
            theta = theta_spiral + theta_scatter

            sp = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center
            
            # Softened circular velocity — matches Barnes-Hut engine exactly
            star_M_enc = star_mass_total * (r**2) / ((r + a)**2)
            M_enc  = smbh_mass + star_M_enc
            v_circ = softened_v_circ(self.G, M_enc, r)
            
            # A massive central SMBH stabilizes the disk, so we can use a cold disk (sigma=0.03).
            # This perfectly preserves the sharp, beautiful spiral arms for a very long time!
            sigma  = 0.03 * v_circ
            sv_disp = self.rng.normal(0, sigma[:, None], (self.n_stars, 2))
            sv = np.column_stack((-v_circ * np.sin(theta), v_circ * np.cos(theta))) + sv_disp + self.drift

            sm = np.full(self.n_stars, m_star)
            star_types = np.array([BLUE_STRAGGLER, STAR, RED_GIANT, WHITE_DWARF, NEUTRON_STAR, EMISSION_STAR, RED_DWARF], dtype=np.int32)
            type_weights = [0.10, 0.35, 0.15, 0.05, 0.02, 0.01, 0.32]
            st = self.rng.choice(star_types, size=self.n_stars, p=type_weights)

            all_pos.append(sp);  all_vel.append(sv)
            all_mass.append(sm); all_type.append(st)

        if self.n_planets > 0:
            a_p = self.R * 0.3
            # Use u > 0.08 for planets
            u_p = self.rng.uniform(0.08, 0.69, self.n_planets)
            r = a_p * np.sqrt(u_p) / (1.0 - np.sqrt(u_p) + 1e-9)
            
            # Planets follow the same spiral arms
            arm_offset = self.rng.integers(0, n_arms, self.n_planets) * (2 * np.pi / n_arms)
            theta = (r / self.R) * np.pi * winding + arm_offset + self.rng.normal(0, 0.4, self.n_planets)
            
            pp = np.column_stack((r * np.cos(theta), r * np.sin(theta))) + self.center
            
            # Softened circular velocity — matches Barnes-Hut engine exactly
            a_star     = self.R * 0.2
            star_M_enc = star_mass_total * (r**2) / ((r + a_star)**2)
            M_enc  = smbh_mass + star_M_enc
            v_orb  = softened_v_circ(self.G, M_enc, r)
            v_orb += self.rng.normal(0, 0.015 * v_orb)
            
            pv = np.column_stack((-v_orb * np.sin(theta), v_orb * np.cos(theta))) + self.drift
            
            pm = np.full(self.n_planets, m_planet)
            pt = np.full(self.n_planets, PLANET, dtype=np.int32)

            all_pos.append(pp);  all_vel.append(pv)
            all_mass.append(pm); all_type.append(pt)

        # ── Comet halo (Outer Galactic Halo) ──────────────────────────────────
        if self.n_comets > 0:
            # Comets form a spherical halo mostly outside the dense disk
            comets = CometCloud(
                center=self.center, drift=self.drift,
                num_comets=self.n_comets, cloud_radius=self.R * 1.5,
                comet_mass=m_comet, central_mass=self.total_mass,
                G=self.G, rng=self.rng
            )
            cp, cv, cm, ct = comets.generate()
            
            # Filter out any comets that randomly spawned too deep in the core 
            # where the point-mass approximation fails.
            distances = np.linalg.norm(cp - self.center, axis=1)
            valid = distances > (self.R * 0.5)
            
            all_pos.append(cp[valid]);  all_vel.append(cv[valid])
            all_mass.append(cm[valid]); all_type.append(ct[valid])

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
        n_each    = N_total // 2
        n_stars   = n_each * 20 // 100
        n_planets = n_each * 30 // 100
        n_comets  = n_each - n_stars - n_planets

        # Galaxies separated by ~8 disk_radii so they are visually distinct
        # before collision.  total_mass=600 keeps v_circ < 10 at r=3, stable
        # at dt=0.001 with eps=1.0.
        g1 = GalacticDisk(center=(-38.0, 3.0), drift=( 0.18, 0.0),
                          n_stars=n_stars, n_planets=n_planets, n_comets=n_comets,
                          disk_radius=28, total_mass=600, G=1.0, rng=rng)
        g2 = GalacticDisk(center=( 38.0,-3.0), drift=(-0.18, 0.0),
                          n_stars=n_stars, n_planets=n_planets, n_comets=n_comets,
                          disk_radius=28, total_mass=600, G=1.0, rng=rng)

        parts = [g1.generate(), g2.generate()]

    elif preset == "solar_system":
        # ── Pristine Real-World Solar System Model ────────────────────────────
        # 1. Central Sun (STAR, Warm Gold, M=500)
        # 2. 8 Major Planets (Mercury..Neptune) orbiting in empty, pristine space (NO co-orbiting comet/debris clutter!)
        # 3. Dense, rich Main Asteroid Belt (2.1–3.3 AU) between Mars & Jupiter (1,200 Sky-Blue asteroids)
        # 4. Vibrant Kuiper Belt & Oort Cloud (30–65 AU) encircling the outer system (2,500 Mint-Green comets)

        star_mass = 500.0    # Sun (normalized)
        AU        = 12.8     # sim units per AU (Mercury → r≈5, Neptune → r≈385)

        M_earth = star_mass / 333_000.0

        # ── 8 Major Planets ──────────────────────────────────────────────────
        planets_data = [
            ("Mercury",  0.39,   0.055, ROCKY),        # Orange/Rust
            ("Venus",    0.72,   0.815, ROCKY),        # Orange/Rust
            ("Earth",    1.00,   1.000, ROCKY),        # Orange/Rust
            ("Mars",     1.52,   0.107, ROCKY),        # Orange/Rust
            ("Jupiter",  5.20, 317.8,   GAS_GIANT),    # Amber/Gold
            ("Saturn",   9.54,  95.2,   GAS_GIANT),    # Amber/Gold
            ("Uranus",  19.18,  14.5,   GAS_GIANT),    # Amber/Gold
            ("Neptune", 30.07,  17.1,   GAS_GIANT),    # Amber/Gold
        ]

        all_pos, all_vel, all_mass, all_type = [], [], [], []

        # ── Central Sun ──────────────────────────────────────────────────────
        all_pos.append(np.zeros((1, 2), dtype=np.float64))
        all_vel.append(np.zeros((1, 2), dtype=np.float64))
        all_mass.append(np.array([star_mass], dtype=np.float64))
        all_type.append(np.array([STAR], dtype=np.int32))

        # ── Place 8 Major Planets (Pristine, Clean Space) ───────────────────
        for name, r_au, me, ptype in planets_data:
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
            all_type.append(np.array([ptype], dtype=np.int32))

            # Add Saturn's ring system (planetary ring around Saturn specifically)
            if name == "Saturn":
                n_ring = 60
                r_ring_sim = rng.uniform(r_sim * 1.03, r_sim * 1.08, n_ring)
                angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
                ring_pos = np.column_stack((r_ring_sim * np.cos(angles), r_ring_sim * np.sin(angles)))
                v_ring = np.sqrt(star_mass / r_ring_sim)
                ring_vel = np.column_stack((-v_ring * np.sin(angles), v_ring * np.cos(angles)))
                all_pos.append(ring_pos); all_vel.append(ring_vel)
                all_mass.append(np.full(n_ring, m_body * 0.0001))
                all_type.append(np.full(n_ring, PLANET, dtype=np.int32))

        # ── Main Asteroid Belt (Mars to Jupiter: 2.1–3.3 AU, 1,200 Asteroids) ─
        n_asteroids = min(N_total * 30 // 100, 1200)
        belt = OrbitalPlanetBelt(
            num_planets=n_asteroids,
            r_min=2.0 * AU, r_max=4.5 * AU,  # Widened to look physically thicker
            planet_mass=M_earth * 0.0001,
            central_mass=star_mass, G=1.0, rng=rng
        )
        bp, bv, bm, bt = belt.generate()
        all_pos.append(bp); all_vel.append(bv)
        all_mass.append(bm); all_type.append(bt)

        # ── Vibrant Kuiper Belt & Oort Cloud (30–65 AU, 2,500 Comets) ──────────
        n_comets = min(N_total * 60 // 100, 2500)
        oort = CometCloud(
            num_comets=n_comets,
            cloud_radius=50.0 * AU,   # Expanded to give a thicker, more expansive Oort cloud/Kuiper belt
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

        # Pentagon: 5 equally-spaced points at r=140, rotated 18° so no cluster
        # sits exactly on the x-axis (makes the collision more visually interesting)
        ring_r = 32.0
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + np.radians(18)
        positions = np.column_stack((ring_r * np.cos(angles), ring_r * np.sin(angles)))

        inward_speed = 0.12
        drifts = -positions / np.linalg.norm(positions, axis=1, keepdims=True) * inward_speed

        parts = []
        for k in range(n_clusters):
            g = GalacticDisk(
                center=positions[k], drift=drifts[k],
                n_stars=n_s, n_planets=n_p, n_comets=n_c,
                disk_radius=10, total_mass=300, G=1.0, rng=rng
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
        all_type.append(np.array([BLACK_HOLE], dtype=np.int32))

        # Ring types and strict physical mass hierarchy:
        # Central BH (2000.0) >> Gas Giants (2.0) >> Rocky Planets (0.005) >> Disk Debris (0.0005) >> Comets (0.00001)
        ring_types  = [PLANET, ROCKY, COMET, GAS_GIANT, PLANET, COMET, ROCKY, GAS_GIANT]
        ring_masses = [0.0005, 0.005, 0.00001,    2.0,  0.0005, 0.00001, 0.005,     2.0]

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

    elif preset == "alpha_centauri":
        # ── Alpha Centauri Triple Star & Exoplanet System ──────────────────────
        # Accurate real-world celestial layout:
        # 1. Rigil Kentaurus (Alpha Centauri A - G2V, m=300) & Toliman (Alpha Centauri B - K1V, m=240)
        # 2. Proxima Centauri (Red Dwarf M5.5V, m=40) orbiting at r=38.0
        # 3. Proxima b (Habitable Zone Earth-mass Rocky Planet) & Proxima c (Super-Earth / Gas Giant)
        # 4. Circumstellar Debris Belt & Clean Outer Oort Cloud
        all_pos, all_vel, all_mass, all_type = [], [], [], []

        m_A, m_B = 300.0, 240.0
        m_binary = m_A + m_B
        a_bin = 14.0  # binary separation

        # Alpha Centauri A & B orbit their center of mass
        r_A = a_bin * (m_B / m_binary)
        r_B = a_bin * (m_A / m_binary)
        v_bin = np.sqrt(m_binary / a_bin)

        pos_A = np.array([[-r_A, 0.0]])
        vel_A = np.array([[0.0, -v_bin * (m_B / m_binary)]])
        pos_B = np.array([[r_B, 0.0]])
        vel_B = np.array([[0.0, v_bin * (m_A / m_binary)]])

        all_pos.extend([pos_A, pos_B])
        all_vel.extend([vel_A, vel_B])
        all_mass.extend([np.array([m_A]), np.array([m_B])])
        all_type.extend([np.array([STAR], dtype=np.int32), np.array([STAR], dtype=np.int32)])

        # Proxima Centauri (Red Dwarf M5.5V, m=40.0, orbiting A-B binary at r=80.0)
        r_prox = 80.0
        m_prox = 40.0
        v_prox = np.sqrt(m_binary / r_prox)
        theta_p = np.radians(45.0)  # fixed angle for clean, beautiful framing
        pos_P = np.array([[r_prox * np.cos(theta_p), r_prox * np.sin(theta_p)]])
        vel_P = np.array([[-v_prox * np.sin(theta_p), v_prox * np.cos(theta_p)]])

        all_pos.append(pos_P); all_vel.append(vel_P)
        all_mass.append(np.array([m_prox]))
        all_type.append(np.array([RED_DWARF], dtype=np.int32))   # M-dwarf Red Star (Coral Red)

        # ── Proxima b & Proxima c Exoplanets (Clean, Pristine Orbits) ────────
        m_earth = 0.0015
        exoplanets_proxima = [
            ("Proxima b", 1.5, 1.17, ROCKY),     # Earth-mass rocky world in habitable zone (Terracotta Orange)
            ("Proxima c", 3.5, 7.00, GAS_GIANT), # Super-Earth / Gas Giant (Electric Violet)
        ]

        for pname, r_p, me_p, ptype in exoplanets_proxima:
            m_body = m_earth * me_p
            # Calculate precise orbital velocity considering the simulation's gravitational softening (eps=1.0).
            # Unsoftened v = sqrt(GM/r). Softened v = r * sqrt(GM / (r^2 + eps^2)^1.5)
            eps = 1.0
            v_p_orb = r_p * np.sqrt(m_prox / (r_p**2 + eps**2)**1.5)
            ang = rng.uniform(0, 2 * np.pi)

            # Planet position & velocity relative to Proxima Centauri
            p_pos = pos_P + np.array([[r_p * np.cos(ang), r_p * np.sin(ang)]])
            p_vel = vel_P + np.array([[-v_p_orb * np.sin(ang), v_p_orb * np.cos(ang)]])

            all_pos.append(p_pos); all_vel.append(p_vel)
            all_mass.append(np.array([m_body])); all_type.append(np.array([ptype], dtype=np.int32))

        # ── Circumstellar Debris Belt around A-B Binary (Clean & Realistic) ───
        n_belt = 20
        belt = OrbitalPlanetBelt(num_planets=n_belt, r_min=18.0, r_max=30.0,
                                 planet_mass=0.005, central_mass=m_binary, G=1.0, rng=rng)
        bp, bv, bm, bt = belt.generate()
        all_pos.append(bp); all_vel.append(bv)
        all_mass.append(bm); all_type.append(bt)

        # ── Sparse Outer Oort Cloud (50 comets in outer space) ─────────────
        n_comets = 50
        comets = CometCloud(num_comets=n_comets, cloud_radius=18.0,
                            comet_mass=0.0001, central_mass=m_binary + m_prox, G=1.0, rng=rng)
        cp, cv, cm, ct = comets.generate()
        all_pos.append(cp); all_vel.append(cv)
        all_mass.append(cm); all_type.append(ct)

        parts = [(np.vstack(all_pos), np.vstack(all_vel),
                  np.concatenate(all_mass), np.concatenate(all_type))]

    elif preset == "milkomeda":
        # ── Milkomeda: Milky Way & Andromeda (M31) Galactic Merger ──────────────
        # 4.5 billion year future galactic merger of Andromeda (M31) + Milky Way + Triangulum (M33)
        n_m31 = N_total * 45 // 100
        n_mw  = N_total * 35 // 100
        n_m33 = N_total - n_m31 - n_mw

        # Andromeda (M31): Massive spiral galaxy drifting southeast
        g_m31 = GalacticDisk(center=(-45.0, 20.0), drift=(0.12, -0.05),
                             n_stars=n_m31 * 25 // 100, n_planets=n_m31 * 35 // 100, n_comets=n_m31 * 40 // 100,
                             disk_radius=20, total_mass=1200.0, G=1.0, rng=rng)

        # Milky Way: Intermediate spiral galaxy drifting northwest
        g_mw = GalacticDisk(center=(45.0, -20.0), drift=(-0.12, 0.05),
                            n_stars=n_mw * 25 // 100, n_planets=n_mw * 35 // 100, n_comets=n_mw * 40 // 100,
                            disk_radius=16, total_mass=800.0, G=1.0, rng=rng)

        # Triangulum (M33): Dwarf satellite galaxy orbiting M31
        g_m33 = GalacticDisk(center=(-75.0, 40.0), drift=(0.18, -0.10),
                             n_stars=n_m33 * 20 // 100, n_planets=n_m33 * 30 // 100, n_comets=n_m33 * 50 // 100,
                             disk_radius=9, total_mass=160.0, G=1.0, rng=rng)

        parts = [g_m31.generate(), g_mw.generate(), g_m33.generate()]

    elif preset == "cygnus_x1":
        # ── Cygnus X-1: Black Hole Binary, Accretion Disk & Relativistic Bipolar Jets ─
        # Central SMBH (m=3500) + Blue Supergiant Companion (HDE 226868, m=250) + 15 swirling accretion disk rings + jets
        all_pos, all_vel, all_mass, all_type = [], [], [], []

        bh_mass = 3500.0
        all_pos.append(np.array([[0.0, 0.0]]))
        all_vel.append(np.array([[0.0, 0.0]]))
        all_mass.append(np.array([bh_mass]))
        all_type.append(np.array([BLACK_HOLE], dtype=np.int32))

        # Blue Supergiant companion star HDE 226868 (O-type massive star, m=250.0, Electric Sapphire Blue)
        r_comp = 18.0
        v_comp = np.sqrt(bh_mass / r_comp)
        all_pos.append(np.array([[r_comp, 0.0]]))
        all_vel.append(np.array([[0.0, v_comp]]))
        all_mass.append(np.array([250.0]))
        all_type.append(np.array([BLUE_STRAGGLER], dtype=np.int32))

        # Accretion disk: 15 swirling concentric gas/plasma rings
        n_disk = N_total * 70 // 100
        r_inner, r_outer = 4.0, 32.0
        r_vals = rng.uniform(r_inner, r_outer, n_disk)
        angles = rng.uniform(0, 2 * np.pi, n_disk)

        pos_disk = np.column_stack((r_vals * np.cos(angles), r_vals * np.sin(angles)))
        v_orb = np.sqrt(bh_mass / r_vals)
        # Viscous spiraling drift toward event horizon
        v_radial = -0.015 * v_orb
        vel_disk = np.column_stack((-v_orb * np.sin(angles) + v_radial * np.cos(angles),
                                     v_orb * np.cos(angles) + v_radial * np.sin(angles)))

        # Inner hot disk is GAS_GIANT (violet), outer disk is PLANET (sky blue)
        types_disk = np.where(r_vals < 14.0, GAS_GIANT, PLANET)
        masses_disk = np.where(r_vals < 14.0, 0.05, 0.005)

        all_pos.append(pos_disk); all_vel.append(vel_disk)
        all_mass.append(masses_disk); all_type.append(types_disk)

        # Polar Relativistic Jets (high-speed collimated plasma beam escaping perpendicular along y-axis)
        n_jets = N_total - len(np.vstack(all_pos))
        n_half = n_jets // 2

        # Top Jet (+y direction): uniform spacing & hyperbolic escape velocity (1.12 * v_esc)
        r_jet_top = np.linspace(1.5, 35.0, n_half) + rng.uniform(-0.08, 0.08, n_half)
        theta_jet = rng.normal(np.pi / 2, 0.05, n_half)
        pos_jet_top = np.column_stack((r_jet_top * np.cos(theta_jet), r_jet_top * np.sin(theta_jet)))
        v_jet_top = 1.12 * np.sqrt(2.0 * bh_mass / np.clip(r_jet_top, 1.5, None))
        vel_jet_top = np.column_stack((v_jet_top * np.cos(theta_jet), v_jet_top * np.sin(theta_jet)))

        # Bottom Jet (-y direction): uniform spacing & hyperbolic escape velocity (1.12 * v_esc)
        r_jet_bot = np.linspace(1.5, 35.0, n_half) + rng.uniform(-0.08, 0.08, n_half)
        theta_bot = rng.normal(-np.pi / 2, 0.05, n_half)
        pos_jet_bot = np.column_stack((r_jet_bot * np.cos(theta_bot), r_jet_bot * np.sin(theta_bot)))
        v_jet_bot = 1.12 * np.sqrt(2.0 * bh_mass / np.clip(r_jet_bot, 1.5, None))
        vel_jet_bot = np.column_stack((v_jet_bot * np.cos(theta_bot), v_jet_bot * np.sin(theta_bot)))

        pos_jets = np.vstack((pos_jet_top, pos_jet_bot))
        vel_jets = np.vstack((vel_jet_top, vel_jet_bot))
        masses_jets = np.full(len(pos_jets), 0.0001)
        types_jets = np.full(len(pos_jets), COMET, dtype=np.int32)

        all_pos.append(pos_jets); all_vel.append(vel_jets)
        all_mass.append(masses_jets); all_type.append(types_jets)

        parts = [(np.vstack(all_pos), np.vstack(all_vel),
                  np.concatenate(all_mass), np.concatenate(all_type))]

    elif preset == "messier13":
        # ── Messier 13 (M13 Hercules Globular Cluster) ─────────────────────────
        # Real globular clusters are PRESSURE-SUPPORTED (like elliptical galaxies):
        # stars move in random orbits, NOT circular orbits.
        # The Plummer model gives the exact equilibrium velocity dispersion:
        #   sigma_1D^2(r) = G * M / (6 * sqrt(r^2 + a^2))
        # Each component of velocity is independently sampled from N(0, sigma_1D).
        all_pos, all_vel, all_mass, all_type = [], [], [], []

        N_stars   = N_total
        a_scale   = 3.0      # Plummer scale radius — compact like a real globular cluster
        cluster_M = 2000.0   # Total cluster gravitational mass

        # Plummer CDF: r = a * sqrt(u / (1-u)), u in [0,1)
        u = rng.uniform(0.001, 0.90, N_stars)
        r = a_scale * np.sqrt(u) / np.sqrt(1.0 - u + 1e-9)
        r = np.clip(r, 0.2, 22.0)

        theta = rng.uniform(0, 2 * np.pi, N_stars)
        pos   = np.column_stack((r * np.cos(theta), r * np.sin(theta)))

        # Plummer velocity dispersion (ISOTROPIC — not circular orbits):
        # sigma_1D^2(r) = G*M / (6 * sqrt(r^2 + a^2))
        # This is the EXACT virial-equilibrium velocity for the Plummer model.
        sigma_1d = np.sqrt(cluster_M / (6.0 * np.sqrt(r**2 + a_scale**2) + 1e-9))
        sigma_1d = np.sqrt(cluster_M / (6.0 * np.sqrt(r**2 + a_scale**2) + 1e-9))
        vx = rng.normal(0.0, sigma_1d)
        vy = rng.normal(0.0, sigma_1d)
        # Clamp to 85% of local escape speed to prevent the Plummer formula
        # from launching inner-core stars out of the cluster (sigma -> inf as r->0)
        v_esc  = np.sqrt(2.0 * cluster_M / np.sqrt(r**2 + a_scale**2 + 1e-9))
        speeds = np.sqrt(vx**2 + vy**2)
        clamp  = np.minimum(1.0, 0.85 * v_esc / (speeds + 1e-9))
        vx    *= clamp
        vy    *= clamp
        vel = np.column_stack((vx, vy))
        # Population II mass segregation
        w_star   = np.full(N_stars, 1.0)
        w_redg   = np.exp(-r / 10.0) * 0.5
        w_white  = np.exp(-r /  7.0) * 0.3
        w_blue   = np.exp(-r /  3.5) * 0.6
        w_pulsar = np.exp(-r /  2.5) * 0.3
        w_redd   = np.full(N_stars, 1.2)

        weights = np.column_stack((w_star, w_redg, w_white, w_blue, w_pulsar, w_redd))
        weights /= weights.sum(axis=1, keepdims=True)

        types  = np.zeros(N_stars, dtype=np.int32)
        masses = np.zeros(N_stars, dtype=np.float64)
        archetypes = np.array([STAR, RED_GIANT, WHITE_DWARF, BLUE_STRAGGLER, NEUTRON_STAR, RED_DWARF], dtype=np.int32)
        mass_map   = {STAR: 1.0, RED_GIANT: 3.0, WHITE_DWARF: 1.4, BLUE_STRAGGLER: 4.0, NEUTRON_STAR: 1.8, RED_DWARF: 0.3}
        for i in range(N_stars):
            t = rng.choice(archetypes, p=weights[i])
            types[i]  = t
            masses[i] = mass_map.get(int(t), 1.0)

        # Normalise total particle mass to exactly cluster_M so that the
        # Plummer sigma formula and actual N-body gravity are perfectly consistent.
        masses *= (cluster_M / masses.sum())

        all_pos.append(pos); all_vel.append(vel)
        all_mass.append(masses); all_type.append(types)

        parts = [(np.vstack(all_pos), np.vstack(all_vel),
                  np.concatenate(all_mass), np.concatenate(all_type))]

    elif preset == "messier31":
        # ── Messier 31 (M31 Andromeda Galaxy) ────────────────────────────────────
        # Standalone Andromeda Galaxy: Central SMBH + 2 logarithmic spiral arms + bulge
        g_m31 = GalacticDisk(center=(0.0, 0.0), drift=(0.0, 0.0),
                             n_stars=N_total * 25 // 100,
                             n_planets=N_total * 45 // 100,
                             n_comets=N_total * 30 // 100,
                             disk_radius=30.0, total_mass=1200.0, G=1.0, rng=rng)
        parts = [g_m31.generate()]

    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    pos   = np.vstack([p[0] for p in parts]).astype(np.float64)
    vel   = np.vstack([p[1] for p in parts]).astype(np.float64)
    mass  = np.concatenate([p[2] for p in parts]).astype(np.float64)
    types = np.concatenate([p[3] for p in parts]).astype(np.int32)

    return pos, vel, mass, types

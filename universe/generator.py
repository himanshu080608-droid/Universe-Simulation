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
DUST           = np.int32(13)
DARK_MATTER    = np.int32(14)

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
    DUST:           ( 90, 160, 255),   # 13 Sapphire Blue / Periwinkle (Reflection Nebula Gas)
    DARK_MATTER:    (128,  30, 255),   # 14 Invisible/Debug Violet (Dark Matter)
}


from universe.registry import PresetRegistry

def build_universe(preset="twin_galaxies", N_total=5000, seed=42):
    return PresetRegistry.get(preset).generate(N_total, seed)

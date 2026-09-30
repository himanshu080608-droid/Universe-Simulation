# Preset Parameter Ledger

## 1. twin_galaxies
- **Classification:** Synthetic Numerical Benchmark / Physical Scenario
- **Parameters:**
  - `M_disk`: 1.0 (Synthetic normalization)
  - `M_halo`: 5.0 (Standard disk-to-halo mass ratio, synthetic)
  - `R_d`: 1.0 (Scale length, normalized)
  - `R_s_halo`: 10.0 (Halo scale radius, synthetic)
  - `C_halo`: 10.0 (Halo concentration, synthetic)
  - `Galaxy Separation`: 25.0 (Synthetic for visualization)
- **Model Equations:** Exponential Disk (Surface density), NFW Halo (Enclosed Mass)
- **Visual:** Spiral structure enhanced for visual clarity, DM is invisible.

## 2. solar_system
- **Classification:** Real Observational System
- **Parameters:**
  - `G`: 1.0 (Normalized mapping to AU, Solar Masses, Years / 2pi)
  - `Sun Mass`: 1.0 
  - `Jupiter Mass`: ~0.001 (Observed, NASA JPL)
  - `Semimajor Axes`: Mercury 0.387, Venus 0.723, Earth 1.0, Mars 1.524, Jupiter 5.204, Saturn 9.582, Uranus 19.201, Neptune 30.047 (JPL DE440/DE441)
- **Model:** Keplerian initialization around a central point mass.
- **Visual:** Exaggerated planetary radii for visibility.

## 3. chaos
- **Classification:** Synthetic Numerical Benchmark
- **Parameters:** 5 interacting subclusters of equal mass 200, isotropic Jeans velocity dispersion.

## 4. laplace
- **Classification:** Synthetic Numerical Benchmark
- **Parameters:** Central mass 1000, uniform concentric rings with exact softened circular velocity.

## 5. alpha_centauri
- **Classification:** Real Observational System
- **Parameters:** 
  - `A Mass`: 1.1 M_sun
  - `B Mass`: 0.9 M_sun
  - `Proxima Mass`: 0.12 M_sun
- **Model:** Hierarchical Keplerian orbits (NASA Exoplanet Archive).

## 6. milkomeda
- **Classification:** Physical Scenario (Local Group Future Evolution)
- **Parameters:** MW and M31 mass models (NFW + Disk), separated by 780 kpc (scaled).

## 7. stephans_quintet
- **Classification:** Physical Scenario
- **Parameters:** 4 interacting bodies, 1 foreground body.

## 8. messier13
- **Classification:** Real Observational System
- **Parameters:** Plummer pressure-supported spherical system. No disk rotation.

## 9. messier31
- **Classification:** Real Observational System
- **Parameters:**
  - `R_d`: 5.4 kpc (Observed)
  - `M_d`: 3.66e10 M_sun (Observed)
  - `R_b`: 0.61 kpc, `M_b`: 3.24e10 M_sun (Observed Hernquist Bulge)
  - `R_s`: 7.63 kpc, `M_200`: 8.8e11 M_sun (NFW Halo)

## 10. ngc1052_df2
- **Classification:** Physical Scenario
- **Parameters:** Diffuse pressure-supported stellar system, extremely low dark matter mass fraction.

## 11. castor_sextuple
- **Classification:** Real Observational System
- **Parameters:** 3 tight binaries in a hierarchical arrangement.

## 12. hd98800_polar
- **Classification:** Real Observational System
- **Parameters:** Quadruple system with polar circumbinary disk.

## 13. trappist_1
- **Classification:** Real Observational System
- **Parameters:** 7 confirmed planets (NASA Exoplanet Archive).

## 14. gravothermal_catastrophe
- **Classification:** Synthetic Numerical Benchmark
- **Parameters:** Plummer sphere undergoing relaxation.

## 15. wr104_pinwheel
- **Classification:** Physical Scenario
- **Parameters:** Colliding-wind binary proxy with Archimedean spiral.

## 16. omega_centauri
- **Classification:** Physical Scenario
- **Parameters:** IMBH candidate embedded in a Plummer cluster.

## 17. pleiades_m45
- **Classification:** Real Observational System
- **Parameters:** Sparse open cluster.

## 18. hirayama_family
- **Classification:** Physical Scenario
- **Parameters:** Asteroid disruption benchmark.

## 19. dark_matter_halo_merger
- **Classification:** Synthetic Numerical Benchmark
- **Parameters:** 2 interacting NFW dark matter halos without stellar disks.

## 20. great_attractor
- **Classification:** Physical Scenario / Visual Proxy
- **Parameters:** Large scale structure visualization, not a literal N-body reconstruction.

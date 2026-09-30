# Preset Authoring Standard

## 1. What is a "Preset"?

A **Preset** in the Laplace's Demon simulation is a purely descriptive data structure that composes reusable N-body physical components (`Component`) using a shared `UnitSystem`.

A Preset is **NOT**:
- A subclass with complex hardcoded mathematical logic in `generate()`.
- An if/else branch in a monolithic generator function.
- A loose collection of particles returned as raw numpy arrays.

A Preset **IS**:
- Registered declaratively with a name and a description via `PresetRegistry.register()`.
- Composed of typed `Component` instances.
- Strictly isolated, producing deterministic output arrays when evaluated with a given random seed and total particle count $N_{total}$.
- Unit-aware, carrying explicit length ($L_0$), mass ($M_0$), and time ($T_0$) scales.

## 2. The 15-Step Authoring Pipeline

When correcting an existing preset or adding a new one, the following mandatory pipeline must be followed:

### RESEARCH & MODELING
1. **Goal Specification:** Identify the target system (synthetic vs. real). Define its visual and physical purpose.
2. **Parameter Ledgering:** Record all source parameters (masses, radii, distances) in `docs/parameter_ledger.md` with explicit provenance.
3. **Model Selection:** Choose appropriate spatial (`SpatialModel`) and kinematic (`KinematicModel`) models (e.g., `ExponentialDiskSpatial`, `NFWSpatial`).
4. **Unit System Calibration:** Define the global $G_{sim}$ and $\epsilon_{sim}$ appropriate for the scale of the system.
5. **Component Decomposition:** Deconstruct the target into discrete `Component` instances with explicit mass fractions and generation probabilities.

### INITIAL CONDITIONS (KINEMATICS)
6. **Spatial Density Verification:** Verify that the chosen spatial surrogate produces the expected mass enclosure $M_{enc}(<r)$.
7. **Cross-Component Kinematics:** Ensure that the kinematics model derives velocity $v_c = \sqrt{r \sum F_{radial}}$ using the forces exerted by *all* spatial models in the system.
8. **Particle Assignment Allocation:** Ensure the architecture allocates exact integer particle counts per component, resolving rounding remainders strictly (no $+1$ or $-1$ particle drift).
9. **Deterministic RNG Binding:** Ensure each component uses a stably seeded random number generator tied to its name.

### VALIDATION & SIMULATION
10. **Sanity Evaluation (N=1k):** Run the generator on a small particle count and visualize output spatial extents and density profiles.
11. **Short N-Body Run:** Run the simulation dynamically to verify stable kinematics (no immediate non-physical collapse or explosion unless expected).
12. **Visual Inspection:** Verify visual correctness in the Pygame/Advanced engine (scale, bloom, rendering semantics).

### DOCUMENTATION & REGRESSION
13. **Documentation Sync:** Update module docstrings and the parameter ledger to reflect finalized parameters.
14. **Visual Baseline Regeneration:** Update `tests/baselines/*.png` for the affected system to lock in the new visual contract.
15. **Integration Test Suite Verification:** Ensure the full regression suite (`python3 -m unittest`) passes unconditionally.

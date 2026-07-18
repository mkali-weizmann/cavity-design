---
name: cavity-calculations
description: How to use the cavity_design library for optical calculations — building cavities and optical systems, Gaussian mode propagation, perturbation/tolerance analysis, plotting. Use whenever asked to perform a calculation, simulation, or analysis with this library, instead of reading the whole cavity_design package.
---

# cavity_design — calculation cookbook

`cavity_design` is a 3D Gaussian-beam / ray-tracing library for designing the Laser Phase Plate
optical cavity. This skill is a seed: it covers the core conventions plus a few worked operations.
Each operation has a short snippet here and a complete, runnable script in `examples/`.
Read the example script before writing a new calculation of the same kind — adapt it, don't
re-derive from library source.

## Conventions and core objects

- Import style (used across the whole repo): `from cavity_design import *`.
  This star import also brings in `np`, `plt`, and `copy`.
- All lengths are in **meters** (SI throughout). Default wavelength: `LAMBDA_0_LASER = 1064e-9`.
- The optical axis usually runs along **x**. Direction/position constants:
  `LEFT, RIGHT, UP, DOWN, INWARD, OUTWARD, ORIGIN, INCH` (defined in `cavity_design/_utils.py`).
- **`Cavity`** — a resonator: a list of elements + `standing_wave=True` finds the self-consistent
  cavity mode (`cavity.arms[i].mode_parameters` → `w_0`, `z_R`, `NA`, ...).
- **`OpticalSystem`** — a one-pass chain of surfaces (no round-trip condition); also used to model
  compound elements (a lens = two refractive surfaces).
- **Surfaces** (in `_surfaces.py`): `CurvedMirror`, `FlatMirror`, `CurvedRefractiveSurface`,
  `FlatRefractiveSurface`, `AsphericRefractiveSurface`, ... plus `MaterialProperties`.
- **Catalog elements** (in `_existing_elements.py`): pre-built real components, e.g.
  `LASER_OPTIK_MIRROR`, `LASER_OPTIK_MIRROR_REFRACTIVE` (transmissive version),
  `EKSMA_LENS_20MM_ASPHERIC`, `THORLABS_8MM_ASPHERIC`, `DUMMY_LENS`, `COASTLINE_20CM_MIRROR`, ...
  Catalog elements are **floating**: their absolute positions are undefined (nan) until placed;
  only their internal geometry is encoded, as relative offsets. Always place one before using it.
- Undefined positions: a floating surface stores a size-3 nan `center` (constructed with
  `center=None` or omitted); an **imaginary** center component is a relative offset from the
  previous surface, resolved when the element/system is placed. Check `element.positions_defined`;
  geometry computations on unplaced elements raise.
- Placing elements: `element.to_position(p)` returns a **copy** placed with its first surface at
  `p` (non-mutating, chainable with `.to_orientation(n)`). `set_element_position(element, p)`
  moves an element **in place** — deepcopy a catalog element first if you use it, so the shared
  catalog object stays pristine.
- Common flags: `t_is_trivial` / `p_is_trivial` (system confined to a plane — no astigmatism
  mixing), `use_paraxial_ray_tracing` (fast ABCD vs. exact ray tracing).
- Live objects are the source of truth: `repr()` of elements/systems prints init-syntax you can
  paste back into code.

## Operation: cavity perturbation / tolerance analysis

How much can each element move (x, y shift, phi tilt, ...) before the cavity mode degrades?
Full runnable version: [examples/perturbation_tolerance_analysis.py](examples/perturbation_tolerance_analysis.py)

```python
from cavity_design import *

cavity = Cavity(
    elements=[...],           # mirrors / lenses placed with .to_position(...)
    standing_wave=True,
    lambda_0_laser=1064e-9,
    t_is_trivial=True, p_is_trivial=True,
)
plot_mirror_lens_mirror_cavity_analysis(cavity)  # overview plot of geometry + mode

perturbable_params_names = ["x", "y", "phi"]
# Per element & parameter: the shift that degrades the mode overlap to the threshold (default 0.9)
tolerance_df = cavity.generate_tolerance_dataframe(perturbable_params_names=perturbable_params_names)
# Overlap vs. shift curves around the working point, and their summary plot:
overlaps_series = cavity.generate_overlap_series(
    shifts=2 * np.abs(tolerance_df.to_numpy()), shift_numel=30,
    perturbable_params_names=perturbable_params_names,
)
cavity.generate_overlaps_graphs(
    arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
    overlaps_series=overlaps_series, perturbable_params_names=perturbable_params_names,
)
```

Related API for finer control: `perturb_cavity`, `PerturbationPointer`,
`gaussians_overlap_integral` (see `tests/test_cavity.py` for usage).

## Operation: free-space mode propagation (no cavity)

Launch a Gaussian mode at an element and propagate it through an `OpticalSystem`, then query the
beam anywhere along the way. Full runnable version:
[examples/mode_propagation_free_space.py](examples/mode_propagation_free_space.py)

```python
from cavity_design import *

mirror = LASER_OPTIK_MIRROR_REFRACTIVE.to_position(5e-3 * LEFT)  # catalog elements are floating - place first
lens = EKSMA_LENS_20MM_ASPHERIC.to_position(mirror.surfaces[1].center + 5.5e-3 * LEFT)
system = OpticalSystem(elements=[mirror, lens],
                       use_paraxial_ray_tracing=True, t_is_trivial=True, p_is_trivial=True)

# A mode that matches the mirror's curvature at a given NA:
mode_0 = match_a_mode_to_mirror(lambda_0_laser=LAMBDA_0_LASER, mirror=mirror.surfaces[0],
                                NA=0.02, mode_going_away_from_mirror=False)
# One ModeParameters per region (before/inside/after each surface), in global coordinates:
modes = system.propagate_mode_parameters_return_global(mode_parameters_before_first_surface=mode_0)

# Query the beam at any point, e.g. spot size 2 cm past the last surface:
camera_plane = system.surfaces[-1].center + 0.02 * LEFT
spot_size = modes[-1].local_mode_parameters_at_a_point(camera_plane).spot_size[0]

ax = system.plot()                                          # elements + modes[i].plot(...) overlay
```

## Where to look for more

- `cavity_design/_cavity.py` — `Cavity`, `OpticalSystem`, perturbation & tolerance methods, plotting helpers.
- `cavity_design/_surfaces.py` — all surface types.
- `cavity_design/_modes.py` — `ModeParameters`, `LocalModeParameters`, mode matching.
- `cavity_design/_rays.py` — `Ray`, ray tracing.
- `cavity_design/_potential.py` — potential/energy-level analysis of the trapped electron.
- `cavity_design/_existing_elements.py` — the element catalog.
- `tests/test_cavity.py` — many short, verified usage examples of the API.

## Maintaining this skill

- Every `examples/*.py` script is smoke-tested by `tests/test_skill_examples.py` (auto-discovered,
  so a newly added example is tested with no extra wiring). Keep examples self-checking: end with
  `assert`s on physical quantities.
- To add an operation: short snippet + explanation here, complete runnable script in `examples/`.
- Keep snippets in this file short (5–20 lines); anything longer belongs in `examples/`.

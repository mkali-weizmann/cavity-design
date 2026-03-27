# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python codebase for designing and analyzing stable optical cavities (resonators) used in physics experiments. It models Gaussian beam propagation, ray tracing, thermal effects, and tolerances for various cavity configurations (Fabry-Perot, mirror-lens-mirror, etc.).

## Commands

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_cavity.py::test_fabry_perot_mode_finding

# Lint/format
python -m black -l 120 cavity.py
.venv/bin/ruff check .

# Activate venv
source .venv/bin/activate
```

Dependencies are managed via `.venv/` and `requirements.txt`.

## Code Architecture

### Core Modules

**`cavity.py`** — Main module (~5700 lines). Contains:
- **Surface hierarchy**: `Surface` → `PhysicalSurface` → `ReflectiveSurface`/`RefractiveSurface` → concrete types (`CurvedMirror`, `FlatMirror`, `FlatRefractiveSurface`, `AsphericRefractiveSurface`, `IdealLens`, etc.)
- **`Arm`**: Represents one segment between two surfaces; handles ABCD matrix propagation and ray tracing between them.
- **`OpticalSystem`**: Base class owning a list of `Arm`s. Handles central line finding and mode propagation.
- **`Cavity(OpticalSystem)`**: The main class for a resonant cavity. Adds standing-wave logic, thermal transformations, and tolerance analysis.
- **Mode classes**: `LocalModeParameters` (Gaussian q-parameter at a point) and `ModeParameters` (mode propagated globally through all arms).
- **`Ray`/`RaySequence`**: Ray tracing primitives. Tensors can have arbitrary batch dimensions; the last dimension is always the spatial coordinate (size 3).
- **Generator functions**: `mirror_lens_mirror_cavity_generator`, `fabry_perot_generator`, `fixed_NA_cavity_generator` — build configured `Cavity` instances from high-level parameters.
- **Tolerance functions**: `generate_tolerance_of_NA`, `perturb_cavity`, `calculate_cavities_overlap` — compute sensitivity of mode overlap to element perturbations.

**`utils.py`** — Shared types and math utilities:
- `INDICES_DICT` / `PRETTY_INDICES_NAMES`: Defines the column layout of the surface parameter array (x, y, theta, phi, r_1, r_2, n, T_c, z, curvature_sign, thermal coefficients, surface_type).
- `OpticalElementParams`: Dataclass representing a single surface's full parameter set.
- `MaterialProperties`: Thermal/optical material constants.
- `PerturbationPointer`: Identifies which parameter of which surface to perturb.
- ABCD matrix helpers, Gaussian beam math (w_0, z_R, NA conversions, overlap integrals).
- Direction constants: `LEFT`, `RIGHT`, `UP`, `DOWN`, `INWARD`, `OUTWARD`.

### Coordinate Conventions (from README)
- **`t`/`theta`**: polar angle from z-axis (elevation).
- **`p`/`phi`**: azimuthal angle from x-axis.
- **Curvature sign**: `+1` means ray hits from inside the sphere; `-1` means from outside.
- `thermal_transformation` cools a heated cavity back to reference state.
- Ray tracing always starts from the *last* surface and traces toward the first.

### `simple_analysis_scripts/`
Standalone analysis scripts and Jupyter notebooks for specific sub-problems (NA vs mirror size, tolerance comparison plots, potential analysis with ray tracing, etc.). These import from `cavity` and `utils` directly.

### `tests/`
`tests/test_cavity.py` — Integration-style tests that construct real cavity objects and validate physics (e.g., Fabry-Perot mode against analytic formula).

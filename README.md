# cavity-design
Calculations to design a stable thin cavity

Code conventions:
- The angle "t" represents the theta angles of spherical coordinates (the angle from the z-axis)
- The angle "p" represents the phi angles of spherical coordinates (the angle from the x-axis of the projection of a vector onto the xy plane)
- When the curvature sign of a spherical surface is +1 then the ray hits it from the inside, and when it is -1 then the ray hits the surface from the inside.
- Given a cavity, the thermal_transformation method assumes the cavity is heated, and cools it down.
- The geometry and thermal properties of a cavity is fully defined by a matrix params, where each row specifies the parameters of a surface, weather it is a mirror or a lens's interface.
  - each column of the row specifies a different parameter of the surface, where the index is matched to the parameter according to this dictionary:

```
INDICES_DICT = {'x': 0, 'y': 1, 't': 2, 'p': 3, 'r': 4, 'n_1': 5, 'w': 6, 'n_2': 7, 'z': 8,
'curvature_sign': 9, 'alpha_thermal_expansion': 10, 'beta_power_absorption': 11,
'kappa_thermal_conductivity': 12, 'dn_dT': 13, 'nu_poisson_ratio': 14,
'alpha_volume_absorption': 15, 'surface_type': 16}
```
  - Where n_1 is the refractive index of the medium before the ray is crossing the surface, and n_2 is the refractive index of the medium after the ray is crossing the surface.


## Curved-surface intersection method

The ray–sphere intersection in `SphericalSurface.find_intersection_with_ray_exact` selects which of the two roots to keep using the surface's given `curvature_sign`. This is fast but assumes the curvature is known and returns the wrong root in extreme, very-non-paraxial cases (e.g. a ray that clips the cap twice near its rim, or approaches from behind). We tried a more robust variant that instead picks the nearest forward intersection lying on the physical cap (effectively "first valid positive intersection"), but it ran ~8% slower on our mixed paraxial/exact workloads. Since those extreme cases do not arise in our paraxial-dominated use, we reverted to the original method for efficiency. The unused robust implementation (plus its tests) lives on the `exact-intersection-slower-method` branch; the params-only `main` keeps the lighter method.


## Cartesian ovals, and where to place a lens made of them

A `RefractiveCartesianOval` images one conjugate pair with *zero* spherical aberration, unlike the polynomial fit of an `AsphericRefractiveSurface`, whose residual is a floor on the achievable NA. Its shape follows from the signed focal distances `E_1` (object) and `E_2` (image) plus `n_1`/`n_2`, so its `radius` and `curvature_sign` are derived, not chosen. `generate_cartesian_oval_lens` builds a thick lens out of two of them. Where the intermediate image between the faces goes is a **free parameter that costs nothing in image quality** — both halves are exact, so the composite is exact wherever it is put — but it does set the angles of incidence, and with them the Fresnel loss and the TIR margin. The default `split="equal_deviation"` balances the two angles and minimises the worst one; paraxially the balance holds for every ray at once, and even at NA 0.23 a full numerical minimax over the fan buys under 2%. The derivation of all three splits is in `theory/cartesian_oval_lens_power_split.md`, worked through in `simple_analysis_scripts/small_debugging_scripts/thick_oval_lens.py`.

The trap is placement. `E_1` is the conjugate of `E_2`; a lens's *focal distance* is the conjugate of infinity. These are different distances, and putting the object at the focal distance collimates the output — sending the image to infinity no matter what `E_2` says. They are separated by Newton's relation `x·x' = f²`, so a short-focus lens throwing its image out to 0.2 m needs the object only `≈ f²/E_2` beyond the focal point (0.66 mm for an 11 mm lens) — a hair, but the difference between 0.2 m and infinity. Place an oval lens at `E_1`, not at its focal distance. One naming trap feeds this: `back_focal_length_of_lens_formula` returns `f - h_1`, which is the **front** focal distance measured from the first vertex — the right quantity for collimation, wrong for a finite conjugate. Relatedly, `back_focal_length_of_lens_object` and `focal_length_of_lens_object` take each radius as `radius * curvature_sign` (`lensmaker_radius_of_a_surface`) rather than assuming `+R_1`/`-R_2`; the old hard-coding was right for every biconvex or plano lens but turned a meniscus into a biconvex one, which two-oval lenses become for some conjugate pairs.

## Environment setup (uv)

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`, hatchling backend). The lock file is committed, so the environment is reproducible: `uv sync` installs exactly the versions recorded there.

```bash
uv sync                       # create/update the environment from uv.lock
uv run python -m pytest       # run the test suite
```

Launch the notebooks with `cavity_jupyter_lab.bat`, which wraps `uv run python -m jupyterlab`.

Three Windows-specific settings are baked in, each of which broke the first migration attempt:

- **Python 3.11, pinned in `.python-version`.** The pinned scientific stack (numpy/scipy/matplotlib 3.8.4) has no wheels for 3.14, which uv otherwise picks as the newest available interpreter. Note that `.python-version` is *deliberately un-ignored* in `.gitignore` — the stock Python `.gitignore` template ignores it under "pyenv", which is exactly why the pin never made it into the first attempt's commits.
- **The environment lives outside the repo**, at `C:\venvs\cavity-design-uv2`, set explicitly by `cavity_jupyter_lab.bat`. uv's default `.venv` would sit inside the project, which is on a Dropbox-synced path and produces "file in use" lock errors during `uv sync`. The `.bat` sets `UV_PROJECT_ENVIRONMENT` itself rather than relying on the user-level variable, because `uv run` syncs whatever environment it is pointed at — an inherited stale value would silently rewrite a different env.
- **`[tool.uv] required-environments`** declares `sys_platform == 'win32' and platform_machine == 'AMD64'`, so universal resolution cannot lock a distribution that ships only Linux/macOS wheels.

### Qt binding: exactly one, on purpose

The notebooks use `%matplotlib qt` for a live figure window. Nothing in this repo imports a Qt binding directly — every use goes through `matplotlib.backends.qt_compat` — so **PySide6 is the only binding declared**, which makes the choice deterministic.

This matters because `qt_compat`'s preference order is PyQt6 → PySide6 → PyQt5 → PySide2, and it picks the first one installed *regardless of what you think you configured*. The old hand-built env had both PyQt5 and PySide6 present, so it silently ran PySide6 while `requirements.txt` pinned PyQt5, and the `QT_API=pyside6` environment variable people kept setting was a no-op. Declaring one binding removes the ambiguity; dropping PyQt5 also removes the `pyqt5-qt5` Windows-wheel problem entirely. If you ever need to check what is actually in use: `python -c "from matplotlib.backends import qt_compat; print(qt_compat.QT_API)"`.

# The tolerances pseudo code:

For each NA:
- Generate two cavities with this NA - mirror-lens-mirror and regular Fabry-Perot.
- For each cavity:
  - Calculate tolerances:
  - For each optical element and geometrical parameter (e.g., Lens, y lateral shift):
    - Find the value of shift such that the modes overlap between the unperturbed and perturbed cavities is 0.9:
      - Define function `f(shift)`:
        - Generates an identical cavity, with a shifted parameter (shifted by `shift`) for this optical element.
        - Finds the mode in the new, perturbed cavity:
          - Find the central line (the line that retraces itself after one roundtrip).
          - If the cavity is a standing wave cavity (which is always the case for us):
            - Define a function `g(direction)`:
              - Propagate the ray starting at the origin of the first mirror.
              - Calculate the intersection point with a plane that contains the origin of the last mirror.
              - Return the distance from the origin (of the last mirror).
            - Find numerically the root of `g`.
          - Calculate the ABCD matrix of one roundtrip: `M`:
            - Calculate each incidence angle between the central line and each of the faces of the optical elements.
            - Calculate the ABCD matrix of this optical element.
          - Calculate the mode parameter `v` for which `M(v) = v`.
          - Propagate this mode `v` to all the other arms using the ABCD matrices.
        - Define a plane (usually the first waist) on which to evaluate both modes.
        - Calculate their overlap integral.
      - Find `shift` such that `f(shift) = 0.9`.

Formatting: ```python -m black -l 120 cavity.py```



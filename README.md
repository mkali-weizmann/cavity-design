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


## uv migration attempt (reverted; kept on the `migrate-to-uv` branch)

We tried migrating dependency management from `requirements.txt` to **uv** (`pyproject.toml` + `uv.lock`, hatchling backend). On Windows this surfaced a chain of environment problems, so **`main` was reverted to the pre-migration, `requirements.txt`-based setup**, which at the time appeared to run the analysis notebooks cleanly. (It didn't, quite — see the root-cause section at the end of this section.) The uv work is preserved on the **`migrate-to-uv`** branch (pushed to GitHub) for anyone who wants to finish it.

Problems hit during the migration — all traceable to moving from a hand-curated Windows venv to a freshly-resolved, cross-platform, mostly-unpinned uv environment:

- **`pyqt5-qt5` had no Windows wheel.** uv's universal resolution (done with Ubuntu support in mind) locked a Linux/macOS-only version. Fixed by declaring `[tool.uv] required-environments` for `sys_platform == 'win32' and platform_machine == 'AMD64'`.
- **Wrong Python.** With no `.python-version`, `uv run` grabbed the newest interpreter (3.14), for which the pinned scientific stack (numpy/scipy/matplotlib 3.8.4) has no wheels. The pre-migration env was Python **3.11**.
- **ipywidgets rendered as plain text.** `jupyterlab` was left unpinned and resolved to 4.6.x, which does not render the `jupyterlab_widgets` frontend extension. Pinned back to `jupyterlab==4.5.3`.
- **`.venv` under Dropbox.** uv's default `.venv` lives inside the project, which sits on a Dropbox-synced path, causing "file in use" lock errors during `uv sync`. Worked around by putting the env at `C:\venvs\cavity-design` via the `UV_PROJECT_ENVIRONMENT` variable.
- **Interactive Qt backend broke.** The clean uv env installed only *declared* deps, so **PySide6** — present ad-hoc in the old venv and selected via a global `QT_API=pyside6` env var — was missing. Standardizing on **PyQt5** instead made the `%matplotlib qt` figure window freeze ("Not Responding") and surfaced a matplotlib async draw-race (`'NoneType' object has no attribute 'canvas' / 'dpi_scale_trans'`). Installing PySide6 fixed the freeze, but the draw-race persisted even after matching Python 3.11 / ipython 9.9.0 / ipykernel 7.1.0 — it is a timing-sensitive interaction with the notebook's live-figure clearing code, not a packaging issue. Reverting `main` appeared to clear it, but that was coincidence — the race is latent in the notebook code on `main` too, and resurfaced there on 2026-08-15. See the root-cause section below.

What the **`migrate-to-uv`** branch contains (a near-working state; the notebook still shows the transient draw-race):

- `pyproject.toml` (hatchling, `requires-python >= 3.11`) + `uv.lock`, with `[tool.uv] required-environments` for Windows.
- Dependency pins matched to the pre-migration versions: `jupyterlab==4.5.3`, `ipython==9.9.0`, `ipykernel==7.1.0`, `matplotlib==3.8.4`, `PyQt5==5.15.11`, plus `PySide6>=6.5`.
- `analyze_potential.ipynb`: an explicit `os.environ["QT_API"] = "pyside6"` pin (replacing the old global env var) and a `_drawing_suspended` guard around `clear_figure_extra_axes` to suppress the draw-race.
- Not on the branch: the Python-3.11 pin was set locally via `.python-version` (never committed); the env lived at `C:\venvs\cavity-design`.

### Draw-race root cause — resolved 2026-08-15 (it was never a packaging problem)

The paragraphs above blamed the draw-race on the Qt binding, and guessed that the **original venv's PySide6 version** was the unrecoverable unknown behind it. Both are wrong, and neither is worth chasing again:

- **`QT_API` never mattered here.** In `C:\venvs\cavity-design`, with `QT_API` unset, `matplotlib.backends.qt_compat` already resolves to **PySide6 6.11.1** — its preference order is PyQt6 → PySide6 → PyQt5 → PySide2, and PyQt6 isn't installed. So the global `QT_API=pyside6` var, and its later deletion, changed nothing; the binding was PySide6 throughout. Check `qt_compat.QT_API` before theorising about bindings.
- **`requirements.txt` is a curated subset, not a `pip freeze`** (it omits jupyterlab, ipython and ipykernel), so PySide6's absence from it says nothing about what the original venv contained.

The real cause is in the notebook's own live-figure code. `plot_results` adds a twin axes each call (`ax2 = ax[0].twinx()`, `cavity_design/_potential.py:1192`), and `clear_figure_extra_axes` removes it at the top of every widget callback. Removing a twin axes is itself clean — verified against matplotlib 3.8.4: `twinx()` → `remove()` → `draw()` does not raise, and the axes is properly gone from `fig.axes`, `fig._localaxes` and the twin grouper. The crash was a **race**: each callback ended with `plt.show()`, which only *schedules* a `draw_idle()`. That queued draw snapshots the artist list (twin included) and runs later; if the next callback fires first, the twin is removed mid-pass and the draw walks an axes whose `.figure` is `None` → `matplotlib/axes/_base.py:3040`, `'NoneType' object has no attribute 'canvas'`. The aborted draw is also why the window sat half-painted until it regained focus and Windows sent a fresh expose event.

Fixed in `analyze_potential.ipynb` by making each callback own its drawing: `fig.canvas.flush_events()` at the top of `clear_figure_extra_axes` (drain queued draws before mutating the figure), and a `refresh(fig)` helper — `fig.canvas.draw()` + `fig.canvas.flush_events()` — replacing the in-callback `plt.show()` calls, so nothing is left pending for the next callback to race. This is the opposite of the `_drawing_suspended` guard on `migrate-to-uv`, which suppressed draws and caused blank-frame flicker.

This removes the last blocker attributed to uv: the migration's genuine problems were the four packaging ones listed above, all of which have fixes on the branch.

Local dev note: the `.git/hooks/pre-commit` hook runs the tests and must invoke a *working* Python — it calls the project venv's Python directly, because the system Python 3.11 on this machine has a broken `pyreadline` that crashes pytest at startup. It also skips `tests/test_skill_examples.py` (slow — it runs every example script end-to-end).


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



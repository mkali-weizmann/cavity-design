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

### History: the first migration attempt (2026-07, reverted)

An earlier attempt is preserved on the **`migrate-to-uv`** branch. It was reverted after a chain of Windows environment problems, all of which are addressed above: the `pyqt5-qt5` wheel (now moot — PyQt5 is gone), the wrong interpreter (now pinned and committed), ipywidgets rendering as plain text under an unpinned JupyterLab 4.6.x (now `jupyterlab==4.5.3`), and `.venv` under Dropbox (now an explicit out-of-tree path).

The fifth and decisive problem was blamed on uv but wasn't uv's fault at all — see below. Note also that that branch's `analyze_potential.ipynb` carries a `QT_API` pin and a `_drawing_suspended` guard that are both obsolete; do not port them forward.

### Draw-race root cause — resolved 2026-08-15 (it was never a packaging problem)

The first attempt blamed a live-figure draw-race on the Qt binding, and guessed that the **original venv's PySide6 version** was the unrecoverable unknown behind it. Both are wrong, and neither is worth chasing again:

- **`QT_API` never mattered here.** In `C:\venvs\cavity-design`, with `QT_API` unset, `matplotlib.backends.qt_compat` already resolves to **PySide6 6.11.1** — its preference order is PyQt6 → PySide6 → PyQt5 → PySide2, and PyQt6 isn't installed. So the global `QT_API=pyside6` var, and its later deletion, changed nothing; the binding was PySide6 throughout. Check `qt_compat.QT_API` before theorising about bindings.
- **The old `requirements.txt` was a curated subset, not a `pip freeze`** (it omitted jupyterlab, ipython and ipykernel), so PySide6's absence from it said nothing about what the original venv contained. It has since been replaced by `pyproject.toml` + `uv.lock`.

The real cause is in the notebook's own live-figure code. `plot_results` adds a twin axes each call (`ax2 = ax[0].twinx()`, `cavity_design/_potential.py:1192`), and `clear_figure_extra_axes` removes it at the top of every widget callback. Removing a twin axes is itself clean — verified against matplotlib 3.8.4: `twinx()` → `remove()` → `draw()` does not raise, and the axes is properly gone from `fig.axes`, `fig._localaxes` and the twin grouper. The crash was a **race**: each callback ended with `plt.show()`, which only *schedules* a `draw_idle()`. That queued draw snapshots the artist list (twin included) and runs later; if the next callback fires first, the twin is removed mid-pass and the draw walks an axes whose `.figure` is `None` → `matplotlib/axes/_base.py:3040`, `'NoneType' object has no attribute 'canvas'`. The aborted draw is also why the window sat half-painted until it regained focus and Windows sent a fresh expose event.

Fixed in `analyze_potential.ipynb` by making each callback own its drawing: `fig.canvas.flush_events()` at the top of `clear_figure_extra_axes` (drain queued draws before mutating the figure), and a `refresh(fig)` helper — `fig.canvas.draw()` + `fig.canvas.flush_events()` — replacing the in-callback `plt.show()` calls, so nothing is left pending for the next callback to race. This is the opposite of the `_drawing_suspended` guard on `migrate-to-uv`, which suppressed draws and caused blank-frame flicker.

This removed the last blocker attributed to uv, and is what made the second migration possible: the genuine problems were the four packaging ones, all fixed above.

Local dev note: the `.git/hooks/pre-commit` hook runs the tests and must invoke a *working* Python — it calls a project venv's Python directly (trying the uv environment first, then the older hand-built one), rather than bare `python`, because the system Python 3.11 on this machine has a broken `pyreadline` that crashes pytest at startup. It avoids `uv run` so that it still works on branches that predate the migration and have no `pyproject.toml`. It also skips `tests/test_skill_examples.py` (slow — it runs every example script end-to-end).


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



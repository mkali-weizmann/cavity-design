# cavity-design
Calculations to design a stable thin cavity

Code conventions:
- The angle "t" represents the theta angles of spherical coordinates (the angle from the z-axis)
- The angle "p" represents the phi angles of spherical coordinates (the angle from the x-axis of the projection of a vector onto the xy plane)
- When the curvature sign of a spherical surface is +1 then the ray hits it from the inside, and when it is -1 then the ray hits the surface from the inside.
- Given a cavity, the thermal_transformation method assumes the cavity is heated, and cools it down.
- The geometry and thermal properties of a cavity is fully defined by a matrix params, where each row specifies the parameters of a surface, weather it is a mirror or a lens's inteface.
  - each column of the row specifies a different parameter of the surface, where the index is matched to the parameter according to this dictionary:

```
INDICES_DICT = {'x': 0, 'y': 1, 't': 2, 'p': 3, 'r': 4, 'n_1': 5, 'w': 6, 'n_2': 7, 'z': 8,
'curvature_sign': 9, 'alpha_thermal_expansion': 10, 'beta_power_absorption': 11,
'kappa_thermal_conductivity': 12, 'dn_dT': 13, 'nu_poisson_ratio': 14,
'alpha_volume_absorption': 15, 'surface_type': 16}
```
  - Where n_1 is the refractive index of the medium before the ray is crossing the surface, and n_2 is the refractive index of the medium after the ray is crossing the surface.



The tolerances pseudo code:

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

Formatting: `python -m black -l 120 cavity.py`

Formatting: ```python -m black -l 120 cavity.py```



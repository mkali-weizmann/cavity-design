import matplotlib.pyplot as plt
import numpy as np

from cavity import *

lambda_0_laser = 1064e-9
from matplotlib import use
from matplotlib.lines import Line2D
use('TkAgg')
# %%
params = [OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=3.080114516e-01      , y=0                    , z=0                    , t=0                    , p=0                    , r_1=1.5039504639e-01     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.thick_lens,    x=6.5057257992e-03     , y=0                    , z=0                    , t=0                    , p=1e+00 * np.pi        , r_1=7.967931913299999e-03, r_2=7.967931913299999e-03, curvature_sign=1.0, T_c=3.0114515984e-03     , n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(surface_type=SurfacesTypes.curved_mirror, x=-5e-03               , y=0                    , z=0                    , t=0                    , p=-1e+00 * np.pi       , r_1=5.0000281233e-03     , r_2=np.nan               , curvature_sign=1.0, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=np.nan               , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=np.nan               , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=np.nan               , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

use_paraxial_ray_tracing = True
cavity_paraxial = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4, use_brute_force_for_central_line=True,
                            use_paraxial_ray_tracing=True)

cavity_exact = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4, use_brute_force_for_central_line=True,
                            use_paraxial_ray_tracing=False)

origin = cavity_paraxial.physical_surfaces[0].center
k_vector = unit_vector_of_angles(0, np.pi - 1e-4)
ray = Ray(origin=origin, k_vector=k_vector)
ray_history_paraxial = cavity_paraxial.trace_ray(ray)
ray_history_exact = cavity_exact.trace_ray(ray)
fig, ax = plt.subplots()
cavity_paraxial.plot(ax=ax)

for i in range(len(ray_history_paraxial)):
    ray_history_exact[i].plot(ax=ax, color='b')
    ray_history_paraxial[i].plot(ax=ax, color='g', linestyle='--')

# Add legend for one ray_history_paraxial element and one ray_history_exact element:
legend_elements = [Line2D([0], [0], color='g', linestyle='--', label='Paraxial'),
                   Line2D([0], [0], color='b', linestyle='-', label='Exact')]
ax.legend(handles=legend_elements)

ax.set_ylim(-1e-4, 1e-4)
plt.show()


from cavity import *

np.seterr(all="raise")
import numpy as np
params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Lens aspherical back'   ,surface_type='aspheric_surface'               , x=5e-03                   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=0                       , r_2=0                       , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=np.array([ 0.00000000e+00,  6.91477693e+01, -1.44376098e+06,  3.18562044e+09, 3.81943965e+13,  5.12888896e+17])),
          OpticalElementParams(name='Lens aspherical front'  ,surface_type='aspheric_surface'               , x=9.430000000000001e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=0                       , r_2=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.45e+00                , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=np.array([ 0.00000000e+00,  1.13274377e+02, -3.18431627e+05,  3.74702914e+10, 1.20417584e+15,  4.33421818e+19])),
          OpticalElementParams(name='None'                   ,surface_type='curved_mirror'                  , x=4.087324973634464e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.003640724994363e-01   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=False,
                            debug_printing_level=1,
                            )
from simple_analysis_scripts.potential_analysis.analyze_potential import energy_level
print(2*energy_level(cavity))
# plot_mirror_lens_mirror_cavity_analysis(cavity, CA=5e-3, diameters=[7.75e-3, 7.75e-3, 7.75e-3, 0.0254])
cavity.plot()
# plt.xlim(0.34, 0.37)
plt.show()

# %%
perturbable_params_names=['x', 'y', 'phi']
tolerance_df = cavity.generate_tolerance_dataframe(perturbable_params_names=perturbable_params_names)
tolerance_matrix = tolerance_df.to_numpy()
## %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix),
                                                 shift_numel=50,
                                                 perturbable_params_names=perturbable_params_names,)
## %%
cavity.generate_overlaps_graphs(arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
                                overlaps_series=overlaps_series, perturbable_params_names=perturbable_params_names)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
# cavity.specs(tolerance_dataframe=tolerance_matrix, print_specs=True, contracted=False)


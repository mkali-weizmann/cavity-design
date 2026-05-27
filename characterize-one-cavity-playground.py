from cavity_design import *
# from simple_analysis_scripts.potential_analysis.analyze_potential import energy_level, analyze_potential_given_cavity, plot_results
# np.seterr(all="raise")
# %%
params = [
          OpticalSurfaceParams(name='Laser Optik Mirror'     ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=5e-03                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None), [
          OpticalSurfaceParams(name='aspheric_lens_automatic - flat side',surface_type='flat_refractive_surface'        , x=1.709663342810191e-02   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , radius=0                       , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=1.27e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='aspheric_lens_automatic - curved side',surface_type='aspheric_surface'               , x=2.052663342810191e-02   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=8.77035305937277e-03    , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.45e+00                , diameter=1.27e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=np.array([-2.31707219e-11,  5.70102477e+01,  7.99731802e+04,  1.44188815e+08, 1.61746471e+13, -2.81787362e+18,  2.99285435e+23, -2.01892830e+28, 8.88794116e+32, -2.54430957e+37,  4.56337202e+41, -4.65571842e+45, 2.06104346e+49]))
         ], [
          OpticalSurfaceParams(name='spherical_0'            ,surface_type='curved_refractive_surface'      , x=2.552663342810191e-02   , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.convex, T_c=np.nan                  , n_inside_or_after=1.51e+00                , n_outside_or_before=1e+00                   , diameter=1.27e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalSurfaceParams(name='spherical_1'            ,surface_type='curved_refractive_surface'      , x=2.98766334281019e-02    , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2.542632688301439e-01   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1.51e+00                , diameter=1.27e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)],
          OpticalSurfaceParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=4.776009036530685e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , radius=2e-01                   , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=None                    , beta_surface_absorption=None                    , kappa_conductivity=None                    , dn_dT=None                    , nu_poisson_ratio=None                    , alpha_volume_absorption=None                    , intensity_reflectivity=None                    , intensity_transmittance=None                    , temperature=np.nan                  ), polynomial_coefficients=None)]
# %%
cavity_paraxial = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=True,
                            debug_printing_level=1,
                            )

perturbable_params_names=['x', 'y', 'phi']
tolerance_df = cavity_paraxial.generate_tolerance_dataframe(perturbable_params_names=perturbable_params_names)
tolerance_matrix = tolerance_df.to_numpy()
## %%
overlaps_series = cavity_paraxial.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix),
                                                 shift_numel=50,
                                                 perturbable_params_names=perturbable_params_names,)
## %%
cavity_paraxial.generate_overlaps_graphs(arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
                                overlaps_series=overlaps_series, perturbable_params_names=perturbable_params_names)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
cavity_non_paraxial = Cavity.from_params(params=params,
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

print(2*energy_level(cavity_non_paraxial))
results_dict = analyze_potential_given_cavity(cavity=cavity_non_paraxial, n_rays=30, phi_max=0.2)
plot_results(results_dict=results_dict, far_away_plane=True)
plt.show()




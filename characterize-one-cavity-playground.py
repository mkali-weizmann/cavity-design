from cavity import *

lambda_0_laser = 1064e-9

params = [OpticalElementParams(surface_type='curved_mirror'            , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000038736030386e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.convex, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(surface_type='thick_lens'               , x=6.456799364770933e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.422461588676613e-02, r_2=5.488336266084286e-03, curvature_sign=CurvatureSigns.convex, T_c=2.913598729541864e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=None                 , beta_surface_absorption=None                 , kappa_conductivity=None                 , dn_dT=None                 , nu_poisson_ratio=None                 , alpha_volume_absorption=None                 , intensity_reflectivity=None                 , intensity_transmittance=None                 , temperature=np.nan               )),
          OpticalElementParams(surface_type='curved_mirror'            , x=3.079135987295418e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.503926943256874e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.convex, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=True,
                            )


cavity.plot()
# plot_mirror_lens_mirror_cavity_analysis(cavity)
plt.show()

# %%
tolerance_matrix = cavity.generate_tolerance_matrix()

# %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                 shift_size=50,
                                                 )
# # %%
cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                arm_index_for_NA=2)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
cavity.specs(print_specs=True, contracted=False)


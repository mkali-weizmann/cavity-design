from copy import deepcopy
from cavity import *



params = [OpticalElementParams(name='Small Mirror', surface_type=SurfacesTypes.curved_mirror , x=x_1         , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=12e-03           , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(name='Lens',         surface_type=SurfacesTypes.ideal_lens               , x=x_1 + 39e-3 , y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=25.4e-3          , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=PHYSICAL_SIZES_DICT['thermal_properties_bk7']),
          OpticalElementParams(name='Big Mirror',   surface_type=SurfacesTypes.curved_mirror            , x=x_2         , y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=0.5              , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]


cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER,
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4,
                            use_paraxial_ray_tracing=True,
                            debug_printing_level=1,
                            )

plot_mirror_lens_mirror_cavity_analysis(cavity, CA=12.5e-3)
plt.show()
# # %%


cavity.specs(print_specs=True, contracted=False, save_specs_name='classic_cavity_specs')

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


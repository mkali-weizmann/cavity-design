from cavity import *

params_original = [OpticalElementParams(name='Small Mirror', surface_type='curved_mirror'            , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000045315676729e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
                   OpticalElementParams(name='Lens',         surface_type='thick_lens'               , x=6.456776823267892e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.424176903520436e-02, r_2=5.487903137228402e-03, curvature_sign=CurvatureSigns.concave, T_c=2.913553646535783e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=PHYSICAL_SIZES_DICT['thermal_properties_sapphire']),
                   OpticalElementParams(name='Big Mirror',   surface_type='curved_mirror'            , x=3.079135536465358e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.504597593390832e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

params_org_2 = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000045315676729e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
                OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.072035493526157e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.424460387342245e-02, r_2=5.487594539969197e-03, curvature_sign=CurvatureSigns.concave, T_c=2.937008387052313e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=None                 , beta_surface_absorption=None                 , kappa_conductivity=None                 , dn_dT=None                 , nu_poisson_ratio=None                 , alpha_volume_absorption=None                 , intensity_reflectivity=None                 , intensity_transmittance=None                 , temperature=np.nan               )),
                OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=3.071420709870524e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.505832163042491e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

params_new = [OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'        , x=-5e-03               , y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=5.000045315676729e-03, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
              OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'           , x=6.071035493526157e-03, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.461082287342245e-02, r_2=5.173908539969197e-03, curvature_sign=CurvatureSigns.concave, T_c=2.142070987052313e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=None                 , beta_surface_absorption=None                 , kappa_conductivity=None                 , dn_dT=None                 , nu_poisson_ratio=None                 , alpha_volume_absorption=None                 , intensity_reflectivity=None                 , intensity_transmittance=None                 , temperature=np.nan               )),
              OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'        , x=3.071420709870524e-01, y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=1.505832163042491e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

params = params_new

cavity_original = Cavity.from_params(params=params_original,
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

cavity_org_2 = Cavity.from_params(params=params_org_2,
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

caivty_new = Cavity.from_params(params=params,
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

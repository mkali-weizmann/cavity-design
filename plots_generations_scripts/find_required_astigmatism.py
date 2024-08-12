import matplotlib.pyplot as plt

from cavity import *

params = [OpticalElementParams(name='Small Mirror',      surface_type='curved_mirror'               , x=-4.891347000000001e-03, y=0                    , z=0                    , theta=0                    , phi=-1e+00 * np.pi       , r_1=4.89138659648578e-03 , r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               )),
          OpticalElementParams(name='Lens',              surface_type='thick_lens'                  , x=6.348278185000162e-03,  y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.4223277e-02        , r_2=5.487739e-03         , curvature_sign=CurvatureSigns.concave, T_c=2.913862370000322e-03, n_inside_or_after=1.76e+00             , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=1.76e+00             , alpha_expansion=5.5e-06              , beta_surface_absorption=1e-06                , kappa_conductivity=4.606e+01            , dn_dT=1.17e-05             , nu_poisson_ratio=3e-01                , alpha_volume_absorption=1e-02                , intensity_reflectivity=1e-04                , intensity_transmittance=9.99899e-01          , temperature=np.nan               )),
          OpticalElementParams(name='Big Mirror',        surface_type='curved_mirror'               , x=5.078052093700003e-01,  y=0                    , z=0                    , theta=0                    , phi=0                    , r_1=2.519358017389886e-01, r_2=np.nan               , curvature_sign=CurvatureSigns.concave, T_c=np.nan               , n_inside_or_after=1e+00                , n_outside_or_before=1e+00                , material_properties=MaterialProperties(refractive_index=None                 , alpha_expansion=7.5e-08              , beta_surface_absorption=1e-06                , kappa_conductivity=1.31e+00             , dn_dT=None                 , nu_poisson_ratio=1.7e-01              , alpha_volume_absorption=None                 , intensity_reflectivity=9.99889e-01          , intensity_transmittance=1e-04                , temperature=np.nan               ))]

cavity = Cavity.from_params(params=params, standing_wave=True, lambda_0_laser=LAMBDA_0_LASER, power=3e4, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True)
# %%
astigmatic_cavity, _ = find_required_perturbation_for_desired_change(cavity=cavity,
                                                                  perturbation_pointer=PerturbationPointer(element_index=2,
                                                                                                           parameter_name=ParamsNames.r_1),
                                                                  desired_parameter=lambda cavity: 2 * cavity.arms[1].mode_parameters_on_surface_1.spot_size[0],
                                                                  x0=cavity.params[2].r_1/10,
                                                                  desired_value=1e-3 / 2.5,
                                                                  print_progress=True)
fig, ax = plt.subplots(2, 1, figsize=(16, 14))
plot_mirror_lens_mirror_cavity_analysis(cavity, ax=ax[0])
plot_mirror_lens_mirror_cavity_analysis(astigmatic_cavity, CA=1e-3, ax=ax[1])
plt.scatter(astigmatic_cavity.arms[2].mode_parameters.center[0, 0], astigmatic_cavity.arms[2].mode_parameters.center[0, 1], color='red', label='Center of the new mode')
plt.legend()
plt.savefig('figures/astigmatism-calculations/50cm long arm.png')
plt.show()

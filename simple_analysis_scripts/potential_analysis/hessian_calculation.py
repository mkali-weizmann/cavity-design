from matplotlib import use
use('TkAgg')
from cavity_design import *

# %% compare one lens and two lenses system energy level for the same spot_size:
params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.776592092031389e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=4.190155837768429e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=-3.561073324536594e-02  , r_2=1.732916608114493e-01   , curvature_sign=CurvatureSigns.concave, T_c=4.350000000000001e-03   , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='big mirror'             ,surface_type='curved_mirror'                  , x=4.330033837721835e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=6.896851719696528e-02   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

R = params[-1].r_1
NA = 0.15

cavity_two_lenses = Cavity.from_params(params=params, lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)

optical_system_single_lens = OpticalSystem.from_params(params[:2], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
optical_system_two_lenses = OpticalSystem.from_params(params[:-1], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)


cavity_single_lens = optical_system_to_cavity_completion(optical_system=optical_system_single_lens,
                                                         NA=cavity_two_lenses.arms[0].mode_parameters.NA[0],
                                                         end_mirror_distance_to_last_element=cavity_two_lenses.surfaces[3].center[0] - cavity_two_lenses.surfaces[2].center[0])

optical_system_mirror_right = OpticalSystem(surfaces=[cavity_single_lens.surfaces[-1]], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
cavity_fabry_perot = optical_system_to_cavity_completion(optical_system=optical_system_mirror_right,
                                                         NA=cavity_two_lenses.arms[2].mode_parameters.NA[0],
                                                         end_mirror_distance_to_last_element=cavity_two_lenses.surfaces[3].center[0] - cavity_two_lenses.surfaces[2].center[0])

cavity_single_lens_shortened = optical_system_to_cavity_completion(optical_system=optical_system_single_lens,
                                                 NA=cavity_two_lenses.arms[0].mode_parameters.NA[0],
                                                 end_mirror_distance_to_last_element=cavity_two_lenses.surfaces[3].center[0] - cavity_two_lenses.surfaces[2].center[0] - 0.1)
params_symmetric_long_arm = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.387481683531355e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=4.786693804540247e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=-6.059380397196618e-02  , r_2=4.761904761904711e-01   , curvature_sign=CurvatureSigns.concave, T_c=3.45e-03                , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Surface_3'              ,surface_type='curved_mirror'                  , x=4.860510834981394e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=9.040000000000001e-02   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]
cavity_symmetric_right_arm = Cavity.from_params(params=params_symmetric_long_arm, lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)


energy_level_single_lens = energy_level(cavity=cavity_single_lens)
energy_level_two_lenses = energy_level(cavity=cavity_two_lenses)
energy_level_fabry_perot = energy_level(cavity=cavity_fabry_perot)
energy_level_single_lens_shortened = energy_level(cavity=cavity_single_lens_shortened)
energy_level_symmetric_right_arm = energy_level(cavity=cavity_symmetric_right_arm)

print(f'V(w) cavity_original: {energy_level_two_lenses} m')
print(f'V(w) single lens: {energy_level_single_lens} m')
print(f'V(w) fabry perot: {energy_level_fabry_perot} m')
print(f'V(w) single lens shortened: {energy_level_single_lens_shortened} m')
print(f'V(w) symmetric right arm: {energy_level_symmetric_right_arm} m')
# %%
cavity_single_lens.plot()
plt.show()

cavity_two_lenses.plot()
plt.show()

cavity_fabry_perot.plot()
plt.show()

cavity_single_lens_shortened.plot()
plt.show()

cavity_symmetric_right_arm.plot()
plt.show()

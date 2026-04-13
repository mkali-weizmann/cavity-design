from simple_analysis_scripts.potential_analysis.analyze_potential import *
# Double lens system:
params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.776592092031389e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=4.190164703571147e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=-3.561084685817112e-02  , r_2=1.732922172776388e-01   , curvature_sign=CurvatureSigns.concave, T_c=4.350000000000001e-03   , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='big mirror'             ,surface_type='curved_mirror'                  , x=4.330042644697557e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=6.896719562240133e-02   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

optical_system_small_elements = OpticalSystem.from_params(params[:-1], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
R = params[-1].r_1

# Single lens system:
base_params = params = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-5e-3                   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.074677357638641e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None)]
optical_system_small_elements = OpticalSystem.from_params(base_params[:-1], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)


# %% General cavity analysis:
R = params[-1].r_1
u = 5e-6
cavity = optical_system_to_cavity_completion(optical_system=optical_system_small_elements, unconcentricity=u, end_mirror_ROC=R)
hessian_ray_tracing_value = hessian_ray_tracing(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]
hessian_ABCD_matrices_value = hessian_ABCD_matrices(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]

dp = 1e-8
slightly_shifted_ray = Ray(origin=cavity.surfaces[0].parameterization(0, dp),
                           k_vector=-cavity.surfaces[0].normal_at_a_point(cavity.surfaces[0].parameterization(0, dp)))
slightly_shifted_ray_propagated = cavity.propagate_ray(ray=slightly_shifted_ray, n_arms = len(cavity.arms) // 2)
landing_point = slightly_shifted_ray_propagated[-1].origin
landing_point_parameterization = cavity.surfaces[-1].get_parameterization(landing_point)[1]
jacobian = dp / landing_point_parameterization

results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays = 10, phi_max = 0.01, print_tests=False)
a_2_numerical = results_dict['polynomial_residuals_mirror'].coef[1]
a_2_analytical = u / (2 * R ** 2)

hessian_normalized = hessian_ABCD_matrices_value * jacobian
a_2_normalized = a_2_analytical * jacobian

w_squared_numerical = cavity.arms[len(cavity.arms) // 2-1].mode_parameters_on_surface_1.spot_size[0] ** 2
w_squared_analytical_potential = cavity.lambda_0_laser / (np.pi * np.sqrt(-2 * hessian_ABCD_matrices_value * a_2_analytical))
w_squared_analytical_potential_normalized = cavity.lambda_0_laser / (np.pi * np.sqrt(-2 * hessian_normalized * a_2_normalized))

energy_level_hessian_only = np.sqrt(a_2_analytical / (-2 * hessian_ABCD_matrices_value)) * cavity.lambda_0_laser / np.pi
# energy_level_hessian_and_potential = -cavity.lambda_0_laser ** 2 / (2 * np.pi ** 2 * w_squared_numerical * hessian_ABCD_matrices_value)
energy_level_hessian_and_potential_normalized = cavity.lambda_0_laser ** 2 / (2 * np.pi ** 2 * w_squared_analytical_potential_normalized * hessian_normalized)

print(f'Potential quadratic coefficient: {a_2_numerical:.3e} m^-1')
print(f'Analytical potential quadratic coefficient: {a_2_analytical:.3e} m^-1')
print(f'Hessian ray tracing: {hessian_ray_tracing_value}')
print(f'Hessian ABCD matrices: {hessian_ABCD_matrices_value}')
print(f'Numerical spot size squared: {w_squared_numerical:.3e} m^2')
print(f'Analytical spot size potential squared: {w_squared_analytical_potential:.3e} m^2')
print(f'Analytical spot size potential squared normalized: {w_squared_analytical_potential_normalized:.3e} m^2')
print(f'Spot sizes squared ratio: {w_squared_numerical / w_squared_analytical_potential_normalized:.5f}')
print(f'jacobian: {jacobian:.3e}')
print(f'Energy level from Hessian only: {energy_level_hessian_only:.3e} m')
print(f'Energy level from Hessian and potential normalized: {energy_level_hessian_and_potential_normalized:.3e} m')
# print(f'Energy level from Hessian and potential: {energy_level_hessian_and_potential:.3e} m')

# %%  Fabry-perot cavity analysis:
u = 1e-6
R_0 = 5e-3
R_1 = 15e-3
cavity = fabry_perot_generator((R_0, R_1), unconcentricity=u, lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False)
hessian_ray_tracing_value = hessian_ray_tracing(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]
hessian_ABCD_matrices_value = hessian_ABCD_matrices(cavity=cavity, n_rays=1, phi_max=0.1)[0, 0]
hessian_analytical = -R_1 / ((R_0 + R_1) * R_0)

dp = 1e-3
slightly_shifted_ray = Ray(origin=cavity.surfaces[0].parameterization(0, dp),
                           k_vector=-cavity.surfaces[0].normal_at_a_point(cavity.surfaces[0].parameterization(0, dp)))
slightly_shifted_ray_propagated = cavity.propagate_ray(ray=slightly_shifted_ray, n_arms = len(cavity.arms) // 2)
landing_point = slightly_shifted_ray_propagated[-1].origin
landing_point_parameterization = cavity.surfaces[-1].get_parameterization(landing_point)[1]
jacobian = dp / landing_point_parameterization

results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays = 10, phi_max = 0.01, print_tests=False)
a_2_numerical = results_dict['polynomial_residuals_mirror'].coef[1]
a_2_analytical = u / (2 * R_1 ** 2)

hessian_normalized = hessian_ABCD_matrices_value * jacobian
a_2_normalized = a_2_analytical * jacobian

w_squared_numerical = cavity.arms[0].mode_parameters_on_surface_1.spot_size[0] ** 2
w_squared_analytical_potential = cavity.lambda_0_laser / (np.pi * np.sqrt(-2 * hessian_ABCD_matrices_value * a_2_analytical))
w_squared_analytical_potential_normalized = cavity.lambda_0_laser / (np.pi * np.sqrt(-2 * hessian_normalized * a_2_normalized))
# w_squared_analytical_optics = R * cavity.lambda_0_laser / np.pi * np.sqrt(2 * R / u)

energy_level_hessian_and_potential = np.sqrt(a_2_numerical / (-2 * hessian_analytical)) * cavity.lambda_0_laser / np.pi
energy_level_hessian_only = cavity.lambda_0_laser ** 2 / (2 * np.pi ** 2 * w_squared_analytical_potential_normalized * hessian_normalized)

# plot_results(results_dict)
# plt.show()
print(f'Potential quadratic coefficient: {a_2_numerical:.3e} m^-1')
print(f'Analytical potential quadratic coefficient: {a_2_analytical:.3e} m^-1')
print(f'Hessian ray tracing: {hessian_ray_tracing_value}')
print(f'Hessian ABCD matrices: {hessian_ABCD_matrices_value}')
print(f'Analytical Hessian: {hessian_analytical}')
print(f'Numerical spot size squared: {w_squared_numerical:.3e} m^2')
print(f'Analytical spot size potential squared: {w_squared_analytical_potential:.3e} m^2')
print(f'Analytical spot size potential squared normalized: {w_squared_analytical_potential_normalized:.3e} m^2')
print(f'Spot sizes squared ratio: {w_squared_numerical / w_squared_analytical_potential:.5f}')
print(f'Energy level from Hessian only: {energy_level_hessian_only:.3e} m')
print(f'Energy level from Hessian and potential: {energy_level_hessian_and_potential:.3e} m')

energy_level_ABCD_matrices_value = energy_level(cavity=cavity, hessian_method = 'ABCD_matrices')
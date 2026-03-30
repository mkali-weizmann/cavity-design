from simple_analysis_scripts.potential_analysis.analyze_potential import *


params = params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.776592092031389e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=4.190156306096804e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=-3.561074091243219e-02  , r_2=1.732916996347667e-01   , curvature_sign=CurvatureSigns.concave, T_c=4.350000000000001e-03   , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='big mirror'             ,surface_type='curved_mirror'                  , x=4.330034301945979e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=6.896846210837621e-02   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

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

fake_mirror_big = match_a_mirror_to_mode(mode=cavity.arms[2].mode_parameters, z = np.linalg.norm(cavity.surfaces[3].center - cavity.arms[2].mode_parameters.center[0, :]), diameter=0.05)
fake_mirror_small = match_a_mirror_to_mode(mode=cavity.arms[2].mode_parameters, z =-np.linalg.norm(cavity.arms[2].mode_parameters.center[0, :] - cavity.surfaces[2].center), diameter=0.02)

cavity_fake = Cavity.from_params(params=[params[0], params[1], fake_mirror_big.to_params],
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
fake_mirror_small_shifted = fake_mirror_small.to_params
fake_mirror_big_shifted = fake_mirror_big.to_params
small_mirrors_center_current = fake_mirror_small_shifted.x + fake_mirror_small_shifted.r_1
fake_mirror_small_shifted.x -= small_mirrors_center_current
fake_mirror_big_shifted.x -= small_mirrors_center_current

cavity_fabry_perot_fake = Cavity.from_params(params=[fake_mirror_small_shifted, fake_mirror_big_shifted],
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
# %%
rays_0 = initialize_rays(n_rays=100, phi_max=0.02, starting_mirror=cavity_fabry_perot_fake.surfaces[0])
base_length = np.linalg.norm(cavity_fabry_perot_fake.surfaces[1].center - rays_0.origin[0, :])
wavefront_points = rays_0.parameterization(base_length)
propagated_ray = cavity_fabry_perot_fake.propagate_ray(ray=rays_0, n_arms=1, propagate_with_first_surface_first=False)
mirror_intersection_points = propagated_ray[-1].origin
center_of_curvature = cavity_fabry_perot_fake.surfaces[0].origin
R_opposite = np.linalg.norm(mirror_intersection_points[0, :] - center_of_curvature)
end_mirror_ROC = cavity_fabry_perot_fake.surfaces[1].radius
end_mirror_origin = cavity_fabry_perot_fake.surfaces[1].origin

residual_distances_opposite = np.abs(R_opposite) - np.linalg.norm(wavefront_points - center_of_curvature, axis=-1)
polynomial_residuals_opposite = Polynomial.fit(
    wavefront_points[:, 1] ** 2, residual_distances_opposite, 6
).convert()

# Analyze unconcentric mirror case:
residual_distances_mirror = end_mirror_ROC - np.linalg.norm(
    wavefront_points - end_mirror_origin, axis=-1
)  # Mirror has a radius of R_output, not R_opposite.
polynomial_residuals_mirror = Polynomial.fit(
    wavefront_points[:, 1] ** 2, residual_distances_mirror, 6
).convert()

results_dict = {
        "NA_paraxial": cavity_fabry_perot_fake.arms[0].mode_parameters.NA[0],
        "spot_size_paraxial": cavity_fabry_perot_fake.arms[0].mode_parameters_on_surface_1.spot_size[0],
        "ray_sequence": RaySequence([rays_0]),
        "R_output": cavity_fabry_perot_fake.surfaces[0].radius,
        "center_of_curvature": center_of_curvature,
        "R_opposite": R_opposite,
        "wavefront_points_opposite": wavefront_points,
        "residual_distances_opposite": residual_distances_opposite,
        "residual_distances_mirror": residual_distances_mirror,
        "polynomial_residuals_opposite": polynomial_residuals_opposite,
        "polynomial_residuals_mirror": polynomial_residuals_mirror,
        "end_mirror_object": cavity_fabry_perot_fake.surfaces[1],
        "zero_derivative_points": None,
        "cavity": cavity_fabry_perot_fake,
    }

fig, ax = plot_results(results_dict=results_dict, far_away_plane=True,
             unconcentricity=np.linalg.norm(cavity_fabry_perot_fake.surfaces[1].center - cavity_fabry_perot_fake.surfaces[0].center),
             potential_x_axis_angles=False,)
ax[1].set_title(f"NA middle arm = {cavity_fabry_perot_fake.arms[0].mode_parameters.NA[0]:.4f}, length middle arm = {cavity_fabry_perot_fake.arms[0].central_line.length:.2f}")

plt.show()


# %%
# cavity_fabry_perot_fake.plot()
# plt.show()
# %%
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=30, phi_max=0.15)
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True)
ax[1].set_title(f"NA middle arm = {cavity.arms[2].mode_parameters.NA[0]:.4f}, length middle arm = {cavity.arms[2].central_line.length:.2f}")
ax[1].set_ylim(-0.007, 0.007)
ax[1].grid()
plt.show()
# %%

# from matplotlib import use

from utils import angles_of_unit_vector

# use('TkAgg')
from simple_analysis_scripts.potential_analysis.analyze_potential import *
from scipy.optimize import minimize_scalar



params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.593276315694677e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=3.604751750861879e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.793113337652691e-02   , r_2=1.501501501501501e-02   , curvature_sign=CurvatureSigns.concave, T_c=3.45e-03                , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='End mirror'             ,surface_type='curved_mirror'                  , x=4.050459581077056e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=1.000000000464597e-01   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=np.nan                  , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]

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
# %%
rays_initial = initialize_rays(starting_mirror=cavity.surfaces[0], phi_max=0.1, n_rays=100)
propagated_ray = cavity.propagate_ray(ray=rays_initial, n_arms=len(cavity.arms) // 2, propagate_with_first_surface_first=False)
ax = cavity.plot()
propagated_ray[:-1].plot(ax=ax, linewidth=0.5)
end_points = propagated_ray[-1].origin
end_directions_inverted = -propagated_ray[-2].k_vector
plt.show()
# %%
optical_system_inverted_reduced = OpticalSystem(surfaces=cavity.surfaces_ordered[len(cavity.surfaces_ordered) // 2:],
                                       use_paraxial_ray_tracing=False,
                                       p_is_trivial=True,
                                       t_is_trivial=True,)

fig, ax = plt.subplots()

def optical_path_length_trip(p_1, theta, phi, plot=False):
    k_vector = unit_vector_of_angles(theta=theta, phi=phi)
    ray = Ray(origin=p_1, k_vector=k_vector)
    propagated_ray = optical_system_inverted_reduced.propagate_ray(ray=ray, propagate_with_first_surface_first=False)
    total_path_length = propagated_ray.cumulative_optical_path_length[-2]
    last_inner_product = propagated_ray.k_vector[-2] @ propagated_ray.k_vector[-1]
    if np.isnan(total_path_length):
        total_path_length = np.inf
        last_inner_product = np.inf
    if plot:
        ax.clear()
        optical_system_inverted_reduced.plot(ax=ax)
        propagated_ray.plot(ax=ax)
        ax.set_title(f'$\phi={phi:.10f}$\nOptical path length: {total_path_length:.10f} m\n last_inner product: {last_inner_product:.10f}')
        fig.canvas.draw_idle()
        plt.pause(0.001)
    return total_path_length

def f_for_extremum_1d(phi):
    return optical_path_length_trip(p_1=optical_system_inverted_reduced.surfaces[0].center, theta=0, phi=phi, plot=True)
# %%
d_angle = 1e-6
initial_angles = angles_of_unit_vector(end_directions_inverted)
initial_angles_plus_dtheta = (initial_angles[0] + d_angle, initial_angles[1])
initial_angles_plus_dphi = (initial_angles[0], initial_angles[1] + d_angle)
k_vector_0 = end_directions_inverted
k_vector_dtheta = unit_vector_of_angles(theta=initial_angles_plus_dtheta[0], phi=initial_angles_plus_dtheta[1])
k_vector_dphi = unit_vector_of_angles(theta=initial_angles_plus_dphi[0], phi=initial_angles_plus_dphi[1])
k_vectors_tilted = np.stack([k_vector_0, k_vector_dtheta, k_vector_dphi], axis=1)
initial_starting_points = np.stack([end_points, end_points, end_points], axis=1)
print(f"{k_vectors_tilted.shape=}, {initial_starting_points.shape=}")
initial_rays = Ray(origin=initial_starting_points, k_vector=k_vectors_tilted)
propagated_ray_backwards = optical_system_inverted_reduced.propagate_ray(ray=initial_rays, propagate_with_first_surface_first=False)
optical_path_lengths_backwards = propagated_ray_backwards.cumulative_optical_path_length[-2]
optical_path_lengths_backwards_minus_trivial = optical_path_lengths_backwards[:, 1:] - optical_path_lengths_backwards[:, 0:1]
final_points_backwards = propagated_ray_backwards[-1].origin
final_points_backwards_minus_trivial = final_points_backwards[:, 1:, :] - final_points_backwards[:, 0:1, :]
final_points_distances_to_trivial = np.linalg.norm(final_points_backwards_minus_trivial, axis=-1)[:, 1:]
quadratic_coefficients = (optical_path_lengths_backwards_minus_trivial / final_points_distances_to_trivial**2)
# %%
final_points_backwards_minus_trivial_normalized = normalize_vector(final_points_backwards_minus_trivial)
final_points_spanning_vectors_inner_product = np.einsum('ij,ij->i', final_points_backwards_minus_trivial_normalized[:, 0, :], final_points_backwards_minus_trivial_normalized[:, 1, :])
# %%

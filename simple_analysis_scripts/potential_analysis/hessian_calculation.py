# from matplotlib import use
# use('TkAgg')
from functools import reduce
from simple_analysis_scripts.potential_analysis.analyze_potential import *

# %%
def orthonormal_rays_end_points(cavity: Cavity, n_rays: int = 30, phi_max: float = 0.02):
    rays_initial = initialize_rays(starting_mirror=cavity.surfaces[0], phi_max=phi_max, n_rays=n_rays)
    propagated_ray = cavity.propagate_ray(ray=rays_initial, n_arms=len(cavity.arms) // 2,
                                          propagate_with_first_surface_first=False)
    end_points = propagated_ray[-1].origin
    end_directions_inverted = -propagated_ray[-2].k_vector
    optical_system_inverted_reduced = OpticalSystem(
        surfaces=cavity.surfaces_ordered[len(cavity.surfaces_ordered) // 2:],
        use_paraxial_ray_tracing=False,
        p_is_trivial=True,
        t_is_trivial=True, )
    return end_points, end_directions_inverted, optical_system_inverted_reduced

def hessian_ray_tracing(cavity: Cavity, n_rays: int = 30, phi_max: float = 0.02):
    end_points, end_directions_inverted, optical_system_inverted_reduced = orthonormal_rays_end_points(cavity=cavity, n_rays=n_rays, phi_max=phi_max)
    d_angle = 1e-5
    initial_angles = angles_of_unit_vector(end_directions_inverted)  # (n_rays, n_rays)
    initial_angles_plus_dtheta = (initial_angles[0] + d_angle, initial_angles[1])  # (n_rays, n_rays)
    initial_angles_plus_dphi = (initial_angles[0], initial_angles[1] + d_angle)  # (n_rays, n_rays)
    k_vector_0 = end_directions_inverted  # n_rays | 3
    k_vector_dtheta = unit_vector_of_angles(theta=initial_angles_plus_dtheta[0],
                                            phi=initial_angles_plus_dtheta[1])  # n_rays | 3
    k_vector_dphi = unit_vector_of_angles(theta=initial_angles_plus_dphi[0],
                                          phi=initial_angles_plus_dphi[1])  # n_rays | 3
    k_vectors_tilted = np.stack([k_vector_0, k_vector_dtheta, k_vector_dphi],
                                axis=1)  # n_rays | 3 (0, dtheta, dphi) | 3 (xyz)
    initial_starting_points = np.stack([end_points, end_points, end_points],
                                       axis=1)  # n_rays | 3 (0, dtheta, dphi) | 3 (xyz)
    initial_rays_backwards = Ray(origin=initial_starting_points,
                                 k_vector=k_vectors_tilted)  # origin.shape = n_rays | 3 (0, dtheta, dphi) | 3 (xyz)
    propagated_ray_backwards = optical_system_inverted_reduced.propagate_ray(ray=initial_rays_backwards,
                                                                             propagate_with_first_surface_first=False)  # origin.shape = n_arms (one way) | n_rays | 3 (0, dtheta, dphi) | 3 (xyz)
    optical_path_lengths_backwards = propagated_ray_backwards.cumulative_optical_path_length[
        -2]  # n_rays | 3 (0, dtheta, dphi)
    # DELETE ME
    M_1_prime_points = propagated_ray_backwards.parameterization(t=optical_path_lengths_backwards[0, 0], optical_path_length=True)
    p_1 = M_1_prime_points[0, 0, :]
    p_2 = M_1_prime_points[0, 2, :]
    k_1 = -propagated_ray_backwards[-2].k_vector[0, 0, :]
    R, center = extract_matching_sphere(p_1=p_1, p_2=p_2, k_1=k_1)
    # DELETE ME
    optical_path_lengths_backwards_minus_trivial = optical_path_lengths_backwards[:, 1:] - \
                                                   optical_path_lengths_backwards[:, 0:1]  # n_rays | 2 (dtheta, dphi)
    final_points_backwards = propagated_ray_backwards[-1].origin  # n_rays | 3 (0,dtheta,dphi) | 3 (xyz)
    final_points_backwards_minus_trivial = final_points_backwards[:, 1:, :] - final_points_backwards[
        :, 0:1, :]  # n_rays | 2 (dtheta, dphi) | 3 (xyz)
    final_points_distances_to_trivial = np.linalg.norm(final_points_backwards_minus_trivial,
                                                       axis=-1)  # n_rays | 2 (dtheta, dphi)
    # Factor of two is because y=(1/2) * y'' * x^2 is the same as y'' = 2 * y / x^2.
    hessian = 2 * (
                optical_path_lengths_backwards_minus_trivial / final_points_distances_to_trivial ** 2)  # n_rays | 2 (dtheta, dphi)

    # To see if the resulted displacement vectors are orthogonal, we can check the inner product of the normalized vectors:
    # final_points_backwards_minus_trivial_normalized = normalize_vector(final_points_backwards_minus_trivial)
    # final_points_spanning_vectors_inner_product = np.einsum('ij,ij->i', final_points_backwards_minus_trivial_normalized[:, 0, :], final_points_backwards_minus_trivial_normalized[:, 1, :])
    return hessian

def hessian_ABCD_matrices(cavity: Cavity, n_rays: int = 30, phi_max: float = 0.02):
    end_points, end_directions_inverted, optical_system_inverted_reduced = orthonormal_rays_end_points(cavity=cavity,
                                                                                                       n_rays=n_rays,
                                                                                                       phi_max=phi_max)
    initial_rays_backwards = Ray(origin=end_points, k_vector=end_directions_inverted)
    propagated_ray_backwards = optical_system_inverted_reduced.propagate_ray(ray=initial_rays_backwards, propagate_with_first_surface_first=False)
    ABCD_matrices_optical_system = optical_system_inverted_reduced.ABCD_matrices(ray_sequence=propagated_ray_backwards[:-1])  # n_arms | *n_rays | 4 | 4
    one_way_ABCD_matrix = reduce(np.matmul, ABCD_matrices_optical_system[:-1][::-1])  # *n_rays | 4 | 4, [:-1] because the last ABCD matrix corresponds to the last surface, but we want the propagation up to the last surface.
    A, B, C, D = decompose_ABCD_matrix(one_way_ABCD_matrix)
    output_ROC = B / D
    hessian = -(output_ROC - optical_system_inverted_reduced.surfaces[-1].radius) / (output_ROC * optical_system_inverted_reduced.surfaces[-1].radius)
    return hessian


# %% Cavity does not need to be concentric for the analysis.
params = [
          OpticalElementParams(name='LaserOptik mirror'      ,surface_type='curved_mirror'                  , x=-5e-03                  , y=0                       , z=0                       , theta=0                       , phi=1e+00 * np.pi           , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='spherical_lens'         ,surface_type='thick_lens'                     , x=6.776592092031389e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.convex, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='Negative Lens'          ,surface_type='thick_lens'                     , x=4.190164703571147e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=-3.561084685817112e-02  , r_2=1.732922172776388e-01   , curvature_sign=CurvatureSigns.concave, T_c=4.350000000000001e-03   , n_inside_or_after=1.45e+00                , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
          OpticalElementParams(name='big mirror'             ,surface_type='curved_mirror'                  , x=4.330042644697557e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=6.896719562240133e-02   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=5e-02                   , material_properties=MaterialProperties(refractive_index=1.45e+00                , alpha_expansion=5.2e-07                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.38e+00                , dn_dT=1.2e-05                 , nu_poisson_ratio=1.6e-01                 , alpha_volume_absorption=1e-03                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None)]
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
# base_params = params = [
#           OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-5e-3                   , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=-5.488e-03              , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , diameter=7.75e-03                , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  ), polynomial_coefficients=None),
#           OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=4.074677357638641e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , diameter=2.54e-02                , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ), polynomial_coefficients=None)]
#
#
# optical_system_small_elements = OpticalSystem.from_params(base_params[:2], lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True)
#
# cavity = optical_system_to_cavity_completion(optical_system=optical_system_small_elements, unconcentricity=0, end_mirror_ROC=0.2)

# %%
hessian_ray_tracing_value = hessian_ray_tracing(cavity=cavity, n_rays=1, phi_max=0.1)
hessian_ABCD_matrices_value = hessian_ABCD_matrices(cavity=cavity, n_rays=1, phi_max=0.1)
print(hessian_ray_tracing_value)
print(hessian_ABCD_matrices_value)
# %%
cavity_fabry_perot = fabry_perot_generator((0.005, 0.005), unconcentricity=0, lambda_0_laser=LAMBDA_0_LASER, use_paraxial_ray_tracing=False)
hessian_ray_tracing_value_fabry_perot = hessian_ray_tracing(cavity=cavity_fabry_perot, n_rays=1, phi_max=0.1)
hessian_ABCD_matrices_value_fabry_perot = hessian_ABCD_matrices(cavity=cavity_fabry_perot, n_rays=1, phi_max=0.1)
print(hessian_ray_tracing_value_fabry_perot)
print(hessian_ABCD_matrices_value_fabry_perot)

# %%
n = params[1].n_inside_or_after
R_1 = params[1].r_1
R_2 = params[1].r_2
lens = generate_lens_from_params(params[1])
lens_left_center = lens[0].center
T_c = params[1].T_c
f = focal_length_of_lens(R_1, -R_2, n, T_c)
h_2 = f * (n - 1) * T_c / (R_1 * n)
h_1 = f * (n - 1) * T_c / (R_2 * n)
d_1 = np.linalg.norm(lens_left_center)
lens_right_center = lens[1].center
d_2 = (1 / f - 1 / (d_1 + h_1)) ** -1 - h_2
right_mirror_coc = lens_right_center + np.array([d_2, 0, 0])
right_mirror_center = cavity.surfaces[3].center
right_mirror_radius = float(np.linalg.norm(right_mirror_center - right_mirror_coc))
right_mirror_fake = CurvedMirror(origin=right_mirror_center, outwards_normal=np.array([1, 0, 0]), radius=right_mirror_radius, curvature_sign=CurvatureSigns.concave, diameter=cavity.surfaces[-1].diameter)
cavity_fake = Cavity.from_params(params=[params[0], params[1], right_mirror_fake.to_params],
                                 standing_wave=True,
                                lambda_0_laser=LAMBDA_0_LASER,
                                set_central_line=True,
                                set_mode_parameters=True,
                                t_is_trivial=True,
                                p_is_trivial=True,
                                power=2e4,
                                use_paraxial_ray_tracing=False,
                                debug_printing_level=1,)
cavity_fake.plot()
plt.show()
quadratic_coefficients_fake = hessian_ray_tracing(cavity=cavity_fake, n_rays=100, phi_max=0.1)

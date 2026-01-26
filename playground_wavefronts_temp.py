from matplotlib import use
use('TkAgg')
from cavity import *

ALREADY_CHANGED_DIR = 0
try:
    from IPython import get_ipython
    shell = get_ipython()
    running_in_notebook = shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    if ALREADY_CHANGED_DIR == 0:
        import os
        os.chdir('..')
    ALREADY_CHANGED_DIR = 1
except Exception:
    running_in_notebook =  False
    from matplotlib import use
    use('TkAgg')


def point_of_equal_angles(ray_1: Ray, ray_2: Ray, p_1: np.ndarray):
    # projects point p_1 (assumed to be on ray_1) onto ray_2 such that the angles between the segment p_1 to p_2 and the rays are equal.
    p_1_minus_u_2 = p_1 - ray_2.origin
    k_1_plus_k_2 = ray_1.k_vector + ray_2.k_vector
    denominator = 1 + ray_1.k_vector @ ray_2.k_vector
    t = (p_1_minus_u_2 @ k_1_plus_k_2) / denominator
    p_2 = ray_2.origin + t * ray_2.k_vector
    # For debugging:
    # ray_1 = Ray(origin=np.array([1, 1, 0]), k_vector=normalize_vector(np.array([1, -1, 0])))
    # ray_2 = Ray(origin=np.array([0, 1, 0]), k_vector=normalize_vector(np.array([-21, -1, 0])))
    # p_1 = ray_1.parameterization(3)
    # p_2 = point_of_equal_angles(ray_1, ray_2, p_1)
    #
    # fig, ax = plt.subplots()
    # ax.plot([ray_1.origin[0], p_1[0]], [ray_1.origin[1], p_1[1]], color='blue', label='Ray 1')
    # ax.plot([ray_2.origin[0], p_2[0]], [ray_2.origin[1], p_2[1]], color='red', label='Ray 2')
    # ax.scatter(p_1[0], p_1[1], color='green', label='Point on Ray 1')
    # ax.scatter(p_2[0], p_2[1], color='orange', label='Point of Equal Angles')
    # ax.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color='black', linestyle='--', label='Connecting Line')
    # ax.set_aspect('equal', 'box')
    # ax.legend()
    # plt.show()
    return p_2


def points_of_equal_phase(ray: Ray, reference_point: np.ndarray):
    # Generate points of equal phase along an array of rays, starting from a reference point on the first ray.
    points = np.zeros_like(ray.origin)
    points[0, :] = reference_point
    for i in range(1, len(ray.origin)):  # Make sure it returns what I want:
        p_2 = point_of_equal_angles(ray_1=ray[i-1], ray_2=ray[i], p_1=points[i - 1, :])
        points[i, :] = p_2
    return points

def find_wavefront_deviation(cavity: Cavity,
                             max_initial_angle: float,
                             n_rays=100,
                             plot_potential_fits=True,
                             potential_fits_x_axis='arc_length',  # 'arc_length' or 'tilt_angle'
                             plot_wavefronts=True,
                             plot_first_mirror_arc=False,
                             secondary_axis_limits=None,  # (x_min, x_max, y_min, y_max)
                             print_summary=True,
                             suptitles: Optional[Union[List[str], str]]=None,
                             angles_parity_sign: int = -1  # 1 if ray that starts ascending from first mirror reaches
                            # last mirror while still ascending (like in Fabry-Perot), -1 otherwise (like in mirror-lens-mirror)
                             ):
    first_mirror = cavity.physical_surfaces[0]
    last_mirror = cavity.physical_surfaces[-1]
    tilt_angles = np.linspace(0, max_initial_angle, n_rays)
    initial_arc_lengths = tilt_angles * first_mirror.radius
    initial_rays_origin = first_mirror.parameterization(np.zeros_like(initial_arc_lengths), -initial_arc_lengths)
    orthonormal_direction = unit_vector_of_angles(theta=np.zeros_like(tilt_angles), phi=tilt_angles + np.pi * (1 - first_mirror.inwards_normal[0])/2)  # Assume system is alligned with x axis
    orthonormal_ray = Ray(origin=initial_rays_origin, k_vector=orthonormal_direction)
    ray_history = cavity.propagate_ray(orthonormal_ray, n_arms=len(cavity.physical_surfaces) - 1)
    intersection_points = ray_history[-1].origin
    intersection_directions = ray_history[-2].k_vector
    intersection_normals = normalize_vector(intersection_points - last_mirror.origin)
    sin_theta = np.cross(intersection_normals, intersection_directions)[:, 2] * angles_parity_sign
    incidence_angles = np.arcsin(sin_theta)
    incremental_arc_lengths = np.linalg.norm(intersection_points[1:, :] - intersection_points[:-1, :], axis=-1)
    incremental_arc_lengths = np.concatenate(([0], incremental_arc_lengths))
    integrated_arc_lengths = np.cumsum(incremental_arc_lengths)

    integrated_divergence = np.cumsum(incidence_angles * incremental_arc_lengths)
    points_of_equal_phase_values = points_of_equal_phase(ray=ray_history[-2],
                                                         reference_point=ray_history[-1].origin[0, :])
    points_of_equal_phase_values_distance_from_face = last_mirror.radius - np.linalg.norm(
        points_of_equal_phase_values - last_mirror.origin, axis=-1)

    integrated_arc_lengths_doubled = np.concatenate(
        (-integrated_arc_lengths[1:][::-1], integrated_arc_lengths))
    integrated_divergence_doubled = np.concatenate((integrated_divergence[1:][::-1], integrated_divergence))
    points_of_equal_phase_values_distance_from_face_doubled = np.concatenate((
        points_of_equal_phase_values_distance_from_face[1:][::-1], points_of_equal_phase_values_distance_from_face))
    tilt_angles_doubled = np.concatenate((-tilt_angles[1:][::-1], tilt_angles))

    if potential_fits_x_axis == 'arc_length':
        x_data = integrated_arc_lengths_doubled
    else:
        x_data = tilt_angles_doubled
    polynomial_coefficients_angles_doubled = np.polyfit(x_data, integrated_divergence_doubled, deg=4)
    polynomial_coefficients_wavefront_doubled = np.polyfit(x_data, points_of_equal_phase_values_distance_from_face_doubled, deg=4)

    results_dict = {
        'first_mirror': first_mirror,
        'last_mirror': last_mirror,
        'tilt_angles': tilt_angles,
        'ray_history': ray_history,
        'intersection_points': intersection_points,
        'intersection_directions': intersection_directions,
        'intersection_normals': intersection_normals,
        'incidence_angles': incidence_angles,
        'incremental_arc_lengths': incremental_arc_lengths,
        'integrated_arc_lengths': integrated_arc_lengths,
        'integrated_divergence': integrated_divergence,
        'points_of_equal_phase_values': points_of_equal_phase_values,
        'points_of_equal_phase_values_distance_from_face': points_of_equal_phase_values_distance_from_face,
        'polynomial_coefficients_angles_doubled': polynomial_coefficients_angles_doubled,
        'polynomial_coefficients_wavefront_doubled': polynomial_coefficients_wavefront_doubled
    }

    if isinstance(suptitles, str):
        suptitles = [suptitles, suptitles]
    if plot_potential_fits:
        x_fit = np.linspace(np.min(integrated_arc_lengths_doubled), np.max(integrated_arc_lengths_doubled), 100)
        y_fit_angles = np.polyval(polynomial_coefficients_angles_doubled, x_fit)
        y_fit_wavefront = np.polyval(polynomial_coefficients_wavefront_doubled, x_fit)
        fig_0, ax_0 = plt.subplots(2, 1, figsize=(10, 10))
        if potential_fits_x_axis == 'arc_length':
            x_axis_name = 'integrated arc length'
            x_axis_label = 'Integrated Arc Length (m)'
            x_data = integrated_arc_lengths_doubled
        else:
            x_axis_name = 'initial tilt angle'
            x_axis_label = 'Initial Tilt Angle (rad)'
            x_data = tilt_angles_doubled
        ax_0[0].scatter(x_data, integrated_divergence_doubled, color='blue', s=5, label='Numerical result')
        ax_0[0].plot(x_fit, y_fit_angles, color='red', label='Fitted Polynomial')
        ax_0[0].set_title(f'Integrated Incidence Angle vs {x_axis_name}\nQuadratic coefficient = {polynomial_coefficients_angles_doubled[-3]:.3e}, Quartic coefficient = {polynomial_coefficients_angles_doubled[-5]:.3e}')
        ax_0[0].set_ylabel('Integrated Incidence Angle (m)')
        ax_0[0].legend()
        ax_0[0].grid()
        ax_0[1].scatter(x_data, points_of_equal_phase_values_distance_from_face_doubled, color='green', s=5, label='Numerical result')
        ax_0[1].plot(x_fit, y_fit_wavefront, color='red', label='Fitted Polynomial')
        ax_0[1].set_title(f'Traced wavefront vs {x_axis_name}\nQuadratic coefficient = {polynomial_coefficients_wavefront_doubled[-3]:.3e}, Quartic coefficient = {polynomial_coefficients_wavefront_doubled[-5]:.3e}')
        ax_0[1].set_xlabel(x_axis_label)
        ax_0[1].set_ylabel('Extracted Wavefront Distance from Mirror (m)')
        ax_0[1].legend()
        ax_0[1].grid()

        if suptitles is not None:
            fig_0.suptitle(suptitles[0])
        fig_0.subplots_adjust(top=0.85, hspace=0.4)
        if not running_in_notebook:
            root = fig_0.canvas.manager.window
            root.geometry("+2000+200")
        results_dict['fig_potentials'] = fig_0
        results_dict['ax_potentials'] = ax_0

    if plot_wavefronts:
        fig_1, ax_1 = plt.subplots(2, 1, figsize=(8, 20))
        cavity.plot(ax=ax_1[0], fine_resolution=True, plot_mode_lines=False)
        ax_1[0].scatter(last_mirror.origin[0], last_mirror.origin[1], color='blue', s=10, label='Small Mirror Center')
        for i in np.arange(0, len(tilt_angles), 200):
            for ray in ray_history[:-1]:
                ray[i].plot(ax=ax_1[0], color='green', linewidth=0.5, alpha=0.3, label='Rays' if i == 0 else "")

        cavity.plot(ax=ax_1[1], label='mirror face', fine_resolution=True, plot_mode_lines=False)
        for i in np.arange(0, len(tilt_angles), 50):
            for ray in ray_history[:-1]:
                ray[i].plot(ax=ax_1[1], color='green', linewidth=0.5, alpha=0.3, label='Rays' if i == 0 else "")

        ax_1[0].set_title(f"initial tilt angle = {tilt_angles[-1]:.3e}")
        ax_1[0].plot(points_of_equal_phase_values[:, 0], points_of_equal_phase_values[:, 1], color='purple',
                     linestyle='-.', label='Wavefront')
        ax_1[1].plot(points_of_equal_phase_values[:, 0], points_of_equal_phase_values[:, 1], color='purple',
                     linestyle='-.', label='Wavefront')

        # plot a circle with radius equal to left mirror.radius and around left mirror origin:
        ax_1[1].scatter(intersection_points[::50, 0], intersection_points[::50, 1], color='black',
                     label='Intersection points', alpha=0.8, s=3)

        # plot an arc with the radius of the right arm, with angle between -0.156 to 0.156:
        if plot_first_mirror_arc:
            theta_min = np.min(tilt_angles)
            theta_max = np.max(tilt_angles)
            theta = np.linspace(theta_min, theta_max, 200)
            center = first_mirror.origin[:2]
            r = first_mirror.radius
            arc_x = center[0] + r * np.cos(theta)
            arc_y = center[1] + r * np.sin(theta)
            ax_1[0].plot(arc_x, arc_y, color='red', linewidth=1.5, linestyle='--', label='Left mirror arc', alpha=0.8)
            ax_1[1].plot(arc_x, arc_y, color='red', linewidth=1.5, linestyle='--', label='Left mirror arc', alpha=0.8)
        if secondary_axis_limits is not None:
            ax_1[1].set_xlim(secondary_axis_limits[0], secondary_axis_limits[1])
            ax_1[1].set_ylim(secondary_axis_limits[2], secondary_axis_limits[3])
        leg = ax_1[1].legend()
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.texts]
        # remove the second one (index 1)
        del handles[0:len(cavity.physical_surfaces)-1]
        del labels[00:len(cavity.physical_surfaces)-1]
        ax_1[1].legend(handles, labels)

        ax_1[1].set_title(f"Zoomed in view")
        if suptitles is not None:
            fig_1.suptitle(suptitles[1])
        fig_1.subplots_adjust(top=0.9, hspace=0.5)
        results_dict['fig_wavefronts'] = fig_1
        results_dict['ax_wavefronts'] = ax_1

    if print_summary:
        print(
            f"wavefront distance from mirror according to wave front tracing = {points_of_equal_phase_values_distance_from_face[-1]:.3e}m")
        print(
            f"wavefront distance from mirror according to angle of incidence integration = {integrated_divergence[-1]:.3e}m")
        print(
            f"polynomial quadratic and quartic coefficients for incidence angle integration method: {polynomial_coefficients_angles_doubled[-3]:.3e}, {polynomial_coefficients_angles_doubled[-5]:.3e}")

    return results_dict


base_params = [
    OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=-4.999961263669513e-03, y=0, z=0, theta=0,
                         phi=-1e+00 * np.pi, r_1=5e-03, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00, diameter=7.75e-3,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan)),
    OpticalElementParams(name='Lens', surface_type='thick_lens', x=6.387599281689135e-03, y=0, z=0, theta=0, phi=0,
                         r_1=2.422e-02, r_2=5.488e-03, curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03,
                         n_inside_or_after=1.76e+00, n_outside_or_before=1e+00, diameter=7.75e-3,
                         material_properties=MaterialProperties(refractive_index=1.76e+00, alpha_expansion=5.5e-06,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=4.606e+01, dn_dT=1.17e-05,
                                                                nu_poisson_ratio=3e-01, alpha_volume_absorption=1e-02,
                                                                intensity_reflectivity=1e-04,
                                                                intensity_transmittance=9.99899e-01,
                                                                temperature=np.nan)),
    OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=4.078081462362321e-01, y=0, z=0, theta=0,
                         phi=0, r_1=2e-01, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                         n_inside_or_after=1e+00, n_outside_or_before=1e+00, diameter=25.4e-3,
                         material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                beta_surface_absorption=1e-06,
                                                                kappa_conductivity=1.31e+00, dn_dT=None,
                                                                nu_poisson_ratio=1.7e-01, alpha_volume_absorption=None,
                                                                intensity_reflectivity=9.99889e-01,
                                                                intensity_transmittance=1e-04, temperature=np.nan))
]

def mirror_lens_mirror_generator_collimated_beam(base_params: list[OpticalElementParams]):
    # Small mirror parameters
    base_params[0].x = -base_params[0].r_1
    base_params[0].y = 0
    base_params[0].z = 0
    # Lens parameters
    n = base_params[1].n_inside_or_after
    R_1 = base_params[1].r_1
    R_2 = base_params[1].r_2
    T_c = base_params[1].T_c
    f = focal_length_of_lens(R_1, R_2, n, T_c)
    h_2 = f * (n-1) * T_c / (R_1 * n) # principle plane to lens surface 0 distance
    h_1 = f * (n-1) * T_c / (R_2 * n) # principle plane to lens surface 2 distance
    d_1 = f - h_1 # distance from small mirror origin to lens surface 0
    lens = generate_lens_from_params(base_params[1])
    base_params[1].x = d_1 + T_c / 2
    # Large mirror parameters:
    lens_right_center = lens[1].center
    right_mirror_coc = lens_right_center + np.array([0.2, 0, 0])  # This is arbitrary, the big mirror does not have a meaning for those
    # calculations anyway.
    base_params[2].x = right_mirror_coc[0] + base_params[2].r_1
    base_params[2].y = right_mirror_coc[1]
    base_params[2].z = right_mirror_coc[2]
    return Cavity.from_params(params=base_params,
                              standing_wave=True,
                              lambda_0_laser=LAMBDA_0_LASER,
                              set_central_line=False,
                              set_mode_parameters=False,
                              set_initial_surface=False,
                              t_is_trivial=True,
                              p_is_trivial=True,
                              use_paraxial_ray_tracing=False,
                              debug_printing_level=1,
                              )
plt.close('all')
max_initial_angle = 0.5
n_rays = 200
cavity = mirror_lens_mirror_generator_collimated_beam(base_params=base_params)
first_mirror = cavity.physical_surfaces[0]
last_mirror = cavity.physical_surfaces[-1]
tilt_angles = np.linspace(0, max_initial_angle, n_rays)
initial_arc_lengths = tilt_angles * first_mirror.radius
initial_rays_origin = first_mirror.parameterization(np.zeros_like(initial_arc_lengths), -initial_arc_lengths)
orthonormal_direction = unit_vector_of_angles(theta=np.zeros_like(tilt_angles), phi=tilt_angles + np.pi * (1 - first_mirror.inwards_normal[0])/2)  # Assume system is alligned with x axis
orthonormal_ray = Ray(origin=initial_rays_origin, k_vector=orthonormal_direction)
ray_history = cavity.propagate_ray(orthonormal_ray, n_arms=len(cavity.physical_surfaces) - 1)
dummy_plane = FlatSurface(outwards_normal=np.array([1, 0, 0]), center=cavity.physical_surfaces[2].center)
intersection_points = dummy_plane.find_intersection_with_ray_exact(ray=ray_history[2])
# Stupid, I know:
ray_history = cavity.propagate_ray(orthonormal_ray, n_arms=len(cavity.physical_surfaces) - 1)

fig, ax = plt.subplots()
cavity.plot(ax=ax, fine_resolution=True)
for rays in ray_history[:-1]:
    for i in range(len(tilt_angles)):
        rays[i].plot(ax=ax, color='green', linewidth=0.5, alpha=0.3)
ax.scatter(intersection_points[:, 0], intersection_points[:, 1], color='red', s=5)
plt.show()

output_angles = np.arcsin(ray_history[-2].k_vector[:, 1])

dys = np.concatenate((np.array([0]), intersection_points[1:, 1] - intersection_points[:-1, 1]))
integrated_angles = np.cumsum(output_angles * dys)
fig, ax = plt.subplots(2, 1)
ax[0].plot(intersection_points[:, 1], output_angles, 'o')
ax[0].set_title('Output Angles at Dummy Plane')
ax[0].set_xlabel('y position at Dummy Plane (m)')
ax[0].set_ylabel('Output Angle (rad)')
ax[1].plot(intersection_points[:, 1], integrated_angles, 'o')
ax[1].set_title('Integrated Output Angles at Dummy Plane')
ax[1].set_xlabel('y position at Dummy Plane (m)')
ax[1].set_ylabel('Integrated Angle (m)')
plt.show()
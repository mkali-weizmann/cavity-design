from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from cavity import *

# %%
def analyze_potential(R_1: Optional[float] = None, R_2: Optional[float] = None, back_focal_length: Optional[float] = None, defocus=0, T_c=3e-3, n_design=1.8, diameter=12.7e-3, unconcentricity: float = 0, n_actual = None,
                      n_rays=100, dphi: Optional[float] =  None, phi_max: Optional[float] = None,
                      extract_R_analytically: bool = False
                      ):
    optical_axis = np.array([1, 0, 0])
    # Enrich input arguments:
    if n_actual is None:
        n_actual = n_design
    if phi_max is None and dphi is not None:
        phi = np.arange(0, n_rays) * dphi
    elif phi_max is None and dphi is None:
        phi_max = np.arctan((diameter / 2) / (back_focal_length - defocus) / 2)
        phi = np.linspace(0, phi_max, n_rays)
    else:
        phi = np.linspace(0, phi_max, n_rays)
    if R_1 is not None and R_2 is not None:
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n_design, T_c=T_c)
        first_surface, second_surface = CurvedRefractiveSurface(radius=R_1, outwards_normal=-optical_axis, center = back_focal_length * optical_axis, n_1=1, n_2=n_actual, curvature_sign=CurvatureSigns.convex, name='first surface', thickness=T_c / 2, diameter=diameter), \
                                        CurvedRefractiveSurface(radius=R_2, outwards_normal=optical_axis, center=(back_focal_length + T_c) * optical_axis, n_1=n_actual, n_2=1, curvature_sign=CurvatureSigns.concave, name='second surface', thickness=T_c / 2, diameter=diameter)
    elif back_focal_length is not None:
        back_center = back_focal_length * optical_axis
        first_surface, second_surface = Surface.from_params(generate_aspheric_lens_params(f=back_focal_length,
                                                                                          T_c=T_c,
                                                                                          n=n_design,
                                                                                          forward_normal=optical_axis,
                                                                                          diameter=diameter,
                                                                                          polynomial_degree=8,
                                                                                          flat_faces_center=back_center,
                                                                                          name="aspheric_lens_automatic"))
    else:
        raise ValueError("Either R_1 and R_2, or back_focal_length must be provided.")

    # Generate objects:
    k_vectors = unit_vector_of_angles(theta=0, phi=phi)
    ray_origin = optical_axis * defocus
    rays_0 = Ray(origin=ray_origin, k_vector=k_vectors)
    first_surface.n_2 = n_actual
    second_surface.n_1 = n_actual
    # Trace rays through the lens, and add refractive index perturbation:
    rays_1 = first_surface.propagate_ray(rays_0)
    rays_2 = second_surface.propagate_ray(rays_1)
    rays_0.n = 1
    rays_1.n = n_actual
    rays_2.n = 1
    rays_history = [rays_0, rays_1, rays_2]
    ray_sequence = RaySequence(rays_history)

    # Extract all wavefront features at the output surface of the lens.
    d_0 = ray_sequence.cumulative_optical_path_length[1, 0]  # Assumes the first ray is the optical axis ray.
    wavefront_points = ray_sequence.parameterization(d_0, optical_path_length=True)
    if extract_R_analytically:
        R = image_of_a_point_with_thick_lens(distance_to_face_1=back_focal_length - defocus, R_1=first_surface.radius,
                                             R_2=second_surface.radius, n=n_actual,
                                             T_c=T_c)  # Assumes cylindrical symmetry.
        center_of_curvature = rays_2.parameterization(R, optical_path_length=False)[0, :]  # Along the optical axis.
    else:
        R, center_of_curvature = extract_matching_sphere(wavefront_points[..., 0, :], wavefront_points[..., 1, :], rays_0.k_vector[..., 0, :])
    points_rel = wavefront_points - center_of_curvature
    phi_dummy = np.linspace(0, np.arctan(points_rel[-1, 1] / points_rel[-1, 0]), 100)
    sphere_dummy_points = center_of_curvature - R * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                             axis=-1)
    points_residual_distances = R - np.linalg.norm(wavefront_points - center_of_curvature, axis=-1)
    wavefront_polynomial = Polynomial.fit(wavefront_points[:, 1]**2, points_residual_distances, 4).convert()

    # Extract wavefront features at a far away plane (2*ROC - u from the lens):
    wavefront_points_opposite = ray_sequence.parameterization(d_0 + 2 * R - unconcentricity, optical_path_length=True)
    if extract_R_analytically:
        R_opposite = R - unconcentricity
        center_of_curvature_opposite = center_of_curvature
    else:
        R_opposite, center_of_curvature_opposite = extract_matching_sphere(
            wavefront_points_opposite[..., 0, :], wavefront_points_opposite[..., 1, :], rays_0.k_vector[..., 0, :])  # Should be the same as the original center of curvature
    if not np.allclose(center_of_curvature_opposite, center_of_curvature):
        print(f"Warning: center_of_curvature_opposite {center_of_curvature_opposite} is not equal to center_of_curvature {center_of_curvature}")
    if not np.allclose(R_opposite, R - unconcentricity):
        print(f"Warning: R_opposite {R_opposite} is not equal to R - unconcentricity {R - unconcentricity}")
    sphere_dummy_points_opposite = center_of_curvature_opposite + R_opposite * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                             axis=-1)
    points_residual_distances_opposite = R_opposite - np.linalg.norm(wavefront_points_opposite - center_of_curvature_opposite, axis=-1)
    wavefront_polynomial_opposite = Polynomial.fit(wavefront_points_opposite[:, 1]**2, points_residual_distances_opposite, 4).convert()

    # Analyze unconcentric mirror case:
    center_of_mirror = center_of_curvature + np.array([-unconcentricity, 0, 0])
    sphere_dummy_points_mirror = center_of_mirror + R * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
        axis=-1)  # Mirror has the radius of the original wavefront sphere, but centered at the shifted center.
    points_residual_distances_mirror = R - np.linalg.norm(wavefront_points_opposite - center_of_mirror,
                                                      axis=-1)  # Mirror has a radius of R, not R_opposite.
    mirror_deviation_polynomial = Polynomial.fit(wavefront_points_opposite[:, 1] ** 2,
                                         points_residual_distances_mirror, 4).convert()
    expected_second_order_term = 1/2 * (1 / R_opposite - 1 / R)
    print(f"Expected second order term in mirror deviation polynomial due to unconcentricity: {expected_second_order_term:.3e}, actual: {mirror_deviation_polynomial.coef[1]:.3e}")


    results_dict = {
        'rays_0': rays_0,
        'rays_1': rays_1,
        'rays_2': rays_2,
        'first_surface': first_surface,
        'second_surface': second_surface,
        'rays_history': rays_history,
        'ray_sequence': ray_sequence,
        'wavefront_points': wavefront_points,
        'R': R,
        'center_of_curvature': center_of_curvature,
        'sphere_dummy_points': sphere_dummy_points,
        'points_residual_distances': points_residual_distances,
        'wavefront_polyomial': wavefront_polynomial,
        'R_opposite': R_opposite,
        'wavefront_points_opposite': wavefront_points_opposite,
        'center_of_curvature_opposite': center_of_curvature_opposite,
        'sphere_dummy_points_opposite': sphere_dummy_points_opposite,
        'sphere_dummy_points_mirror': sphere_dummy_points_mirror,
        'points_residual_distances_opposite': points_residual_distances_opposite,
        'points_residual_distances_mirror': points_residual_distances_mirror,
        'wavefront_polynomial_opposite': wavefront_polynomial_opposite,
        'mirror_deviation_polynomial': mirror_deviation_polynomial,
    }
    return results_dict

def plot_results(results_dict, far_away_plane: bool = False):
    (rays_0, rays_1, rays_2, first_surface, second_surface, rays_history, ray_sequence, R,
     center_of_curvature) = results_dict['rays_0'], results_dict['rays_1'], results_dict['rays_2'], results_dict['first_surface'], \
            results_dict['second_surface'], results_dict['rays_history'], results_dict['ray_sequence'], \
            results_dict['R'], results_dict['center_of_curvature']
    if far_away_plane:
        wavefront_points, points_residual_distances, sphere_dummy_points, polynomial, sphere_dummy_points_mirror, mirror_deviation_polynomial, points_residual_distances_mirror = results_dict['wavefront_points_opposite'], results_dict['points_residual_distances_opposite'], results_dict['sphere_dummy_points_opposite'], results_dict['wavefront_polynomial_opposite'], results_dict['sphere_dummy_points_mirror'], results_dict['mirror_deviation_polynomial'], results_dict['points_residual_distances_mirror']
    else:
        wavefront_points, points_residual_distances, sphere_dummy_points, polynomial, sphere_dummy_points_mirror, mirror_deviation_polynomial, points_residual_distances_mirror = results_dict['wavefront_points'], results_dict['points_residual_distances'], results_dict['sphere_dummy_points'], results_dict['wavefront_polyomial'], None, None, None
    # plot points:
    fig, ax = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)

    rays_0.plot(ax=ax[0, 0], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 0], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 0], label='After aspheric surface', color='red', linewidth=0.5)
    first_surface.plot(ax=ax[0, 0], color='green')
    second_surface.plot(ax=ax[0, 0], color='orange')
    ax[0, 0].set_xlim(rays_0.origin[0, 0] - 0.01, center_of_curvature[0] * 2)  # (-1e-3, 100e-3)
    ax[0, 0].set_ylim(-second_surface.diameter / 2, second_surface.diameter / 2)  # (-4.2e-3, 4.2e-3)
    ax[0, 0].grid()
    ax[0, 0].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 0].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    rays_0.plot(ax=ax[0, 1], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 1], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 1], label='After aspheric surface', color='red', linewidth=0.5)
    first_surface.plot(ax=ax[0, 1], color='green')
    second_surface.plot(ax=ax[0, 1], color='orange')
    ax[0, 1].set_xlim((second_surface.center[0] + center_of_curvature[0]) / 2 - 0.002, center_of_curvature[0] + 0.005)#(-1e-3, 100e-3)
    ax[0, 1].set_ylim(-rays_2.origin[1, 1]*0.5, 1*rays_2.origin[1, 1])#(-4.2e-3, 4.2e-3)
    ax[0, 1].grid()
    ax[0, 1].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 1].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    ax[1, 0].plot(wavefront_points[:, 0] * 1e3, wavefront_points[:, 1] * 1e3, label='wavefront points', color='black', marker='o', linestyle='', markersize=5, zorder=1, alpha=0.4,)
    ax[1, 0].plot(sphere_dummy_points[:, 0] * 1e3,
                  sphere_dummy_points[:, 1] * 1e3,
                  linestyle='dashdot',
                  label='Fitted sphere for wavefront',
                  color='C0',
                  linewidth=2.0,
                  alpha=0.95,
                  zorder=3)
    ax[1, 0].set_xlabel('x (mm)')
    ax[1, 0].set_ylabel('y (mm)')
    ax[1, 0].grid()
    ax[1, 0].set_title(f'wavefront and fitted sphere: {R*1e3:.2f} mm')
    if sphere_dummy_points_mirror is not None:
        ax[1, 0].plot(sphere_dummy_points_mirror[:, 0] * 1e3,
                      sphere_dummy_points_mirror[:, 1] * 1e3,
                      linestyle='dashed',
                      color='magenta',
                      label='Fitted sphere for unconcentric mirror',
                      linewidth=2.0,
                      alpha=0.5,
                      zorder=2)
        ax[1, 0].legend()

    ax[1, 1].plot(wavefront_points[:, 1] * 1e3, points_residual_distances * 1e6,
                  label='wavefront residual from matching sphere', marker='o', linestyle='', color='blue', markersize=5,
                  alpha=0.6)
    x_fit = np.linspace(np.min(wavefront_points[:, 1]), np.max(wavefront_points[:, 1]), 100)
    ax[1, 1].set_xlabel('y (mm)')
    ax[1, 1].set_ylabel('wavefront difference (µm)')
    ax[1, 1].grid()
    # build polynomial term string with ascending powers and .1e formatting, include x^{n} terms
    coeffs_asc = polynomial.coef
    terms_parts = []
    for i, c in enumerate(coeffs_asc):
        s = f"{c:.1e}"
        mant_str, exp_str = s.split('e')
        mant = float(mant_str)
        exp = int(exp_str)
        terms_parts.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
    terms = " + ".join(terms_parts)
    title = "residual distance between wavefront and fitted sphere. fit:\n" + terms
    # Build terms for mirror deviation polynomial (if present)
    terms_parts_mirror = []
    if mirror_deviation_polynomial is not None:
        coeffs_mirror_asc = mirror_deviation_polynomial.coef
        for i, c in enumerate(coeffs_mirror_asc):
            s = f"{c:.1e}"
            mant_str, exp_str = s.split('e')
            mant = float(mant_str)
            exp = int(exp_str)
            terms_parts_mirror.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
        terms_mirror = " + ".join(terms_parts_mirror)
        title += f"\nmirror deviation fit (unconcentricity = {unconcentricity*1e6:.1f} µm):\n" + terms_mirror
        ax[1, 1].plot(x_fit * 1e3, mirror_deviation_polynomial(x_fit ** 2) * 1e6, color='green', linestyle='dashed',
                      label='Mirror residuals Polynomial fit',linewidth=0.5)
        ax[1, 1].plot(wavefront_points[:, 1] * 1e3, points_residual_distances_mirror * 1e6, marker='x', linestyle='', color='magenta', label='Mirror deviation data', markersize=5, alpha=0.6)

    ax[1, 1].set_title(title)
    ax[1, 1].plot(x_fit * 1e3, polynomial(x_fit**2) * 1e6, color='red', linestyle='dashed', label='Matching sphere residuals Polynomial fit', linewidth=0.5)

    ax[1, 1].legend()
    return fig, ax


def choose_source_position_for_desired_focus_analytic(back_focal_length,
                                               desired_focus,
                                               T_c,
                                               n_design,
                                               diameter, R_1 = None, R_2 = None):
    if R_1 is None and R_2 is None:
        p = LensParams(n=n_design, f=back_focal_length, T_c=T_c)
        coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=8)
        R_2 = 1 / (2*coeffs[1])
        R_1 = np.inf
    elif R_1 is not None and R_2 is not None:
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n_design, T_c=T_c)
    else:
        raise ValueError("Either both R_1 and R_2 must be provided, or neither.")
    distance_to_flat_face = image_of_a_point_with_thick_lens(distance_to_face_1=desired_focus, R_1=R_2,
                                                             R_2=R_1, n=n_design, T_c=T_c)
    defocus = back_focal_length - distance_to_flat_face
    return defocus

# %%
# For aspheric lens:
aspheric = True
if aspheric:
    back_focal_length = 20e-3
    R_1 = None
    R_2 = None
    T_c = 4.35e-3
    diameter = 7.75e-3
else:
    R_1 = 24.22e-3
    R_2 = 5.49e-3
    T_c = 2.91e-3
    diameter = 7.75e-3
    back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=1.8, T_c=T_c)
# rest of parameters
n_actual = 1.8
dn=0
n_design = n_actual + dn
n_rays = 60
unconcentricity = 0e-6
phi_max = 0.06
desired_focus = 200e-3


defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter,
                                                            back_focal_length=back_focal_length,
                                                            R_1=R_1, R_2=R_2,
)


# %%
results_dict = analyze_potential(
                                 back_focal_length=back_focal_length, R_1=R_1, R_2=R_2,
                                 defocus=defocus, T_c=T_c,
                                 n_design=n_design, diameter=diameter,
                                 n_actual=n_actual, n_rays=n_rays,
                                 unconcentricity=unconcentricity, extract_R_analytically=True, phi_max=phi_max)
print(f"Defocus solution for 30 mm focus: {defocus*1e3:.3f} mm, focal point distance: {(results_dict['center_of_curvature'][0] - results_dict['second_surface'].center[0]) * 1e3:.2f} mm")
# plt.close('all')
fig, ax = plot_results(results_dict, far_away_plane=True)
center = results_dict['center_of_curvature']
ax[0, 1].set_xlim((center[0]-0.002, center[0]+0.002))
plt.suptitle(f"n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {back_focal_length * 1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus * 1e3:.1f} mm, T_c: {T_c * 1e3:.1f} mm, Diameter: {diameter * 1e3:.2f} mm")
# Save image with suptitle in name:
plt.savefig(f"outputs/figures/analyze_potential_n_design_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{back_focal_length * 1e3:.1f}mm_defocus_{defocus * 1e3:.1f}mm_Tc_{T_c * 1e3:.1f}mm_diameter_{diameter * 1e3:.2f}mm.svg", dpi=300)
plt.show()

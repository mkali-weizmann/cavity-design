from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from cavity import *

# %%
def analyze_potential(back_focal_length, defocus, T_c, n_design, diameter, unconcentricity: float = 0, n_actual = None, n_rays=100, dphi: Optional[float] =  0.01, phi_max: Optional[float] = None):
    if n_actual is None:
        n_actual = n_design
    if phi_max is None and dphi is not None:
        phi = np.arange(0, n_rays) * dphi
    elif phi_max is None and dphi is None:
        phi_max = np.arctan((diameter / 2) / (back_focal_length - defocus)) - 1e-2
        phi = np.linspace(0, phi_max, n_rays)
    else:
        phi = np.linspace(0, phi_max, n_rays)
    k_vectors = unit_vector_of_angles(theta=0, phi=phi)
    optical_axis = np.array([1, 0, 0])
    ray_origin = optical_axis * defocus
    back_center = back_focal_length * optical_axis
    # Objects:
    rays_0 = Ray(origin=ray_origin, k_vector=k_vectors)
    flat_surface, aspheric_surface = Surface.from_params(generate_aspheric_lens_params(f=back_focal_length,
                                                                                       T_c=T_c,
                                                                                       n=n_design,
                                                                                       forward_normal=optical_axis,
                                                                                       diameter=diameter,
                                                                                       polynomial_degree=8,
                                                                                       flat_faces_center=back_center,
                                                                                       name="aspheric_lens_automatic"))
    flat_surface.n_2 = n_actual
    aspheric_surface.n_1 = n_actual
    # Trace rays through the lens:
    rays_1 = flat_surface.propagate_ray(rays_0)
    rays_2 = aspheric_surface.propagate_ray(rays_1)
    rays_0.n = 1
    rays_1.n = n_actual
    rays_2.n = 1
    rays_history = [rays_0, rays_1, rays_2]

    ray_sequence = RaySequence(rays_history)
    d_0 = ray_sequence.cumulative_optical_path_length[1, 0]
    wavefront_points = ray_sequence.parameterization(d_0, optical_path_length=True)
    R, center_of_curvature = extract_matching_sphere(wavefront_points[..., 0, :], wavefront_points[..., 1, :], rays_0.k_vector[..., 0, :])

    points_rel = wavefront_points - center_of_curvature
    phi_dummy = np.linspace(0, np.arctan(points_rel[-1, 1] / points_rel[-1, 0]), 100)
    sphere_dummy_points = center_of_curvature - R * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                             axis=-1)
    points_residual_distances = np.linalg.norm(wavefront_points - center_of_curvature, axis=-1) - R
    # Fit a polynomial to the relative distances:
    wavefront_polynomial_coeffs = np.polyfit(wavefront_points[:, 1]**2, points_residual_distances, 4)
    wavefront_polynomial = np.poly1d(wavefront_polynomial_coeffs)

    wavefront_points_other_side = ray_sequence.parameterization(d_0 + 2 * R, optical_path_length=True)
    R_other_side, center_of_curvature_other_side = extract_matching_sphere(
        wavefront_points_other_side[..., 0, :], wavefront_points_other_side[..., 1, :], rays_0.k_vector[..., 0, :])  # Should be the same as the original center of curvature
    sphere_dummy_points_other_side = center_of_curvature_other_side + R_other_side * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                             axis=-1)
    center_of_mirror_other_side = center_of_curvature_other_side + np.array([-unconcentricity, 0, 0])
    sphere_dummy_points_mirror = center_of_mirror_other_side + R_other_side * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                                                       axis=-1)
    points_residual_distances_other_side = np.linalg.norm(wavefront_points_other_side - center_of_curvature_other_side, axis=-1) - R_other_side
    points_residual_distances_mirror = np.linalg.norm(wavefront_points_other_side - center_of_mirror_other_side, axis=-1) - R_other_side
    # Fit a polynomial to the relative distances:
    wavefront_polynomial_coeffs_other_side = np.polyfit(wavefront_points_other_side[:, 1]**2, points_residual_distances_other_side, 4)
    wavefront_polynomial_other_side = np.poly1d(wavefront_polynomial_coeffs_other_side)

    mirror_deviation_coeffs = np.polyfit(wavefront_points_other_side[:, 1] ** 2,
                                                        points_residual_distances_mirror, 4)
    mirror_deviation_polynomial = np.poly1d(mirror_deviation_coeffs)

    results_dict = {
        'rays_0': rays_0,
        'rays_1': rays_1,
        'rays_2': rays_2,
        'flat_surface': flat_surface,
        'aspheric_surface': aspheric_surface,
        'rays_history': rays_history,
        'ray_sequence': ray_sequence,
        'wavefront_points': wavefront_points,
        'R': R,
        'center_of_curvature': center_of_curvature,
        'sphere_dummy_points': sphere_dummy_points,
        'points_residual_distances': points_residual_distances,
        'wavefront_polyomial': wavefront_polynomial,
        'R_other_side': R_other_side,
        'wavefront_points_other_side': wavefront_points_other_side,
        'center_of_curvature_other_side': center_of_curvature_other_side,
        'sphere_dummy_points_other_side': sphere_dummy_points_other_side,
        'sphere_dummy_points_mirror': sphere_dummy_points_mirror,
        'points_residual_distances_other_side': points_residual_distances_other_side,
        'points_residual_distances_mirror': points_residual_distances_mirror,
        'wavefront_polynomial_other_side': wavefront_polynomial_other_side,
        'mirror_deviation_polynomial': mirror_deviation_polynomial,
    }
    return results_dict

def plot_results(results_dict, far_away_plane: bool = False):
    (rays_0, rays_1, rays_2, flat_surface, aspheric_surface, rays_history, ray_sequence, R,
     center_of_curvature) = results_dict['rays_0'], results_dict['rays_1'], results_dict['rays_2'], results_dict['flat_surface'], \
            results_dict['aspheric_surface'], results_dict['rays_history'], results_dict['ray_sequence'], \
            results_dict['R'], results_dict['center_of_curvature']
    if far_away_plane:
        wavefront_points, points_residual_distances, sphere_dummy_points, polynomial, sphere_dummy_points_mirror, mirror_deviation_polynomial = results_dict['wavefront_points_other_side'], results_dict['points_residual_distances_other_side'], results_dict['sphere_dummy_points_other_side'], results_dict['wavefront_polynomial_other_side'], results_dict['sphere_dummy_points_mirror'], results_dict['mirror_deviation_polynomial']
    else:
        wavefront_points, points_residual_distances, sphere_dummy_points, polynomial, sphere_dummy_points_mirror, mirror_deviation_polynomial= results_dict['wavefront_points'], results_dict['points_residual_distances'], results_dict['sphere_dummy_points'], results_dict['wavefront_polyomial'], None, None
    # plot points:
    fig, ax = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)

    rays_0.plot(ax=ax[0, 0], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 0], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 0], label='After aspheric surface', color='red', linewidth=0.5)
    flat_surface.plot(ax=ax[0, 0], color='green')
    aspheric_surface.plot(ax=ax[0, 0], color='orange')
    ax[0, 0].set_xlim(rays_0.origin[0, 0] - 0.01, center_of_curvature[0] * 2)  # (-1e-3, 100e-3)
    ax[0, 0].set_ylim(-aspheric_surface.diameter / 2, aspheric_surface.diameter / 2)  # (-4.2e-3, 4.2e-3)
    ax[0, 0].grid()
    ax[0, 0].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 0].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    rays_0.plot(ax=ax[0, 1], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 1], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 1], label='After aspheric surface', color='red', linewidth=0.5)
    flat_surface.plot(ax=ax[0, 1], color='green')
    aspheric_surface.plot(ax=ax[0, 1], color='orange')
    ax[0, 1].set_xlim((aspheric_surface.center[0] + center_of_curvature[0]) / 2 - 0.002, center_of_curvature[0] + 0.005)#(-1e-3, 100e-3)
    ax[0, 1].set_ylim(-rays_2.origin[1, 1]*0.5, 1*rays_2.origin[1, 1])#(-4.2e-3, 4.2e-3)
    ax[0, 1].grid()
    ax[0, 1].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 1].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    ax[1, 0].plot(wavefront_points[:, 0] * 1e3, wavefront_points[:, 1] * 1e3, label='wavefront points', color='black', marker='o', linestyle='', markersize=5, zorder=4)
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

    ax[1, 1].plot(wavefront_points[:, 1] * 1e3, points_residual_distances * 1e6)
    ax[1, 1].set_xlabel('y (mm)')
    ax[1, 1].set_ylabel('wavefront difference (µm)')
    ax[1, 1].grid()
    # build polynomial term string with ascending powers and .1e formatting, include x^{n} terms
    coeffs_asc = polynomial.coefficients[::-1]
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
        coeffs_mirror_asc = mirror_deviation_polynomial.coefficients[::-1]
        for i, c in enumerate(coeffs_mirror_asc):
            s = f"{c:.1e}"
            mant_str, exp_str = s.split('e')
            mant = float(mant_str)
            exp = int(exp_str)
            terms_parts_mirror.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
        terms_mirror = " + ".join(terms_parts_mirror)
        title += "\nmirror deviation fit:\n" + terms_mirror

    ax[1, 1].set_title(title)
    x_fit = np.linspace(np.min(wavefront_points[:, 1]), np.max(wavefront_points[:, 1]), 100)
    ax[1, 1].plot(x_fit * 1e3, polynomial(x_fit**2) * 1e6, color='red', linestyle='dashed', label='Polynomial fit', linewidth=0.5)
    ax[1, 1].legend()
    return fig, ax


def choose_source_position_for_desired_defocus(back_focal_length,
                                               desired_focus,
                                               T_c,
                                               n_design,
                                               diameter,
                                               unconcentricity: float = 0,
                                               n_actual = None,
                                               n_rays=100,
                                               dphi = 0.01):
    def f_roots(defocus):
        results = analyze_potential(back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design,
                                    diameter=diameter, unconcentricity=unconcentricity, n_actual=n_actual,
                                    n_rays=n_rays, dphi=dphi)
        if np.abs(results['center_of_curvature'][0]) > 1e2:
            R_inverse = 0
        else:
            R_inverse = 1 / (results['center_of_curvature'][0] - results['aspheric_surface'].center[0])
        residual = R_inverse - 1 / desired_focus
        return residual

    defocus_solution = optimize.brentq(f_roots, -back_focal_length, back_focal_length)
    results = analyze_potential(back_focal_length=back_focal_length, defocus=defocus_solution, T_c=T_c,
                                n_design=n_design, diameter=diameter, n_actual=n_actual, n_rays=n_rays, dphi=dphi)
    return defocus_solution, results


# %%
f = 5e-3
defocus = -1.905e-3
T_c = 3.0e-3
n_actual = 1.8
diameter = 7.75e-3
dn=0
n_design = n_actual + dn
n_rays = 10
unconcentricity = 5e-7
# defocus, results_dict = choose_source_position_for_desired_defocus(back_focal_length=f, desired_focus=30e-3, T_c=T_c,
#                                                                    n_design=n_design, diameter=diameter,
#                                                                    n_actual=n_actual, n_rays=n_rays)
results_dict = analyze_potential(back_focal_length=f, defocus=defocus, T_c=T_c,
                                                                   n_design=n_design, diameter=diameter,
                                                                   n_actual=n_actual, n_rays=n_rays,
                                 unconcentricity=unconcentricity)
print(f"Defocus solution for 30 mm focus: {defocus*1e3:.3f} mm, focal point distance: {(results_dict['center_of_curvature'][0] - results_dict['aspheric_surface'].center[0]) * 1e3:.2f} mm")
plt.close('all')
fig, ax = plot_results(results_dict, far_away_plane=True)
center = results_dict['center_of_curvature']
ax[0, 1].set_xlim((center[0]-0.002, center[0]+0.002))
plt.suptitle(f"n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {f*1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus*1e3:.1f} mm, T_c: {T_c*1e3:.1f} mm, Diameter: {diameter*1e3:.2f} mm")
# Save image with suptitle in name:
plt.savefig(f"outputs/figures/analyze_potential_n_design_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{f*1e3:.1f}mm_defocus_{defocus*1e3:.1f}mm_Tc_{T_c*1e3:.1f}mm_diameter_{diameter*1e3:.2f}mm.svg", dpi=300)
plt.show()

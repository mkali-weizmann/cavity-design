from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from cavity import *

# %%
def analyze_potential(f, defocus, T_c, n_design, diameter, n_actual = None, n_rays=100, dphi: Optional[float] =  0.01, phi_max: Optional[float] = None):
    if n_actual is None:
        n_actual = n_design
    if phi_max is None and dphi is not None:
        phi = np.arange(0, n_rays) * dphi
    elif phi_max is None and dphi is None:
        phi_max = np.arctan((diameter / 2) / (f - defocus)) - 1e-2
        phi = np.linspace(0, phi_max, n_rays)
    else:
        phi = np.linspace(0, phi_max, n_rays)
    k_vectors = unit_vector_of_angles(theta=0, phi=phi)
    optical_axis = np.array([1, 0, 0])
    ray_origin = optical_axis * defocus
    back_center = f * optical_axis
    # Objects:
    rays_0 = Ray(origin=ray_origin, k_vector=k_vectors)
    flat_surface, aspheric_surface = Surface.from_params(generate_aspheric_lens_params(f=f,
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
    # wf_poly_coeffs = np.polyfit(y_coords, points_relative_distances, deg)
    # wf_poly = np.poly1d(wf_poly_coeffs)
    # wf_poly_fitted = wf_poly(y_coords)
    # wf_poly_residuals = points_relative_distances - wf_poly_fitted
    # wf_poly_max_residual = float(np.max(np.abs(wf_poly_residuals)))

    wavefront_points_other_side = ray_sequence.parameterization(d_0 + 2 * R, optical_path_length=True)
    R_other_side, center_of_curvature_other_side = extract_matching_sphere(
        wavefront_points_other_side[..., 0, :], wavefront_points_other_side[..., 1, :], rays_0.k_vector[..., 0, :])
    sphere_dummy_points_other_side = center_of_curvature_other_side + R_other_side * np.stack((np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
                                                             axis=-1)
    points_residual_distances_other_side = np.linalg.norm(wavefront_points_other_side - center_of_curvature_other_side, axis=-1) - R_other_side

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
        'R_other_side': R_other_side,
        'wavefront_points_other_side': wavefront_points_other_side,
        'center_of_curvature_other_side': center_of_curvature_other_side,
        'sphere_dummy_points_other_side': sphere_dummy_points_other_side,
        'points_residual_distances_other_side': points_residual_distances_other_side,
    }
    return results_dict

def plot_results(results_dict, far_away_plane: bool = False):
    (rays_0, rays_1, rays_2, flat_surface, aspheric_surface, rays_history, ray_sequence, R,
     center_of_curvature) = results_dict['rays_0'], results_dict['rays_1'], results_dict['rays_2'], results_dict['flat_surface'], \
            results_dict['aspheric_surface'], results_dict['rays_history'], results_dict['ray_sequence'], \
            results_dict['R'], results_dict['center_of_curvature']
    if far_away_plane:
        wavefront_points, points_residual_distances, sphere_dummy_points = results_dict['wavefront_points_other_side'], results_dict['points_residual_distances_other_side'], results_dict['sphere_dummy_points_other_side']
    else:
        wavefront_points, points_residual_distances, sphere_dummy_points = results_dict['wavefront_points'], results_dict['points_residual_distances'], results_dict['sphere_dummy_points']
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

    ax[1, 0].plot(wavefront_points[:, 0] * 1e3, wavefront_points[:, 1] * 1e3)
    ax[1, 0].plot(sphere_dummy_points[:, 0] * 1e3, sphere_dummy_points[:, 1] * 1e3, linestyle='dashed')
    ax[1, 0].set_xlabel('x (mm)')
    ax[1, 0].set_ylabel('y (mm)')
    ax[1, 0].grid()
    ax[1, 0].set_title(f'wavefront and fitted sphere: {R*1e3:.2f} mm')

    ax[1, 1].plot(wavefront_points[:, 1] * 1e3, points_residual_distances * 1e6)
    ax[1, 1].set_xlabel('y (mm)')
    ax[1, 1].set_ylabel('phase front relative distance to sphere (µm)')
    ax[1, 1].grid()
    ax[1, 1].set_title('residual distance between wavefront and fitted sphere')
    return fig, ax


def choose_source_position_for_desired_defocus(desired_focus,
                                               back_focal_length,
                                               T_c,
                                               n_design,
                                               diameter,
                                               n_actual = None,
                                               n_rays=100,
                                               dphi = 0.01):
    def f_roots(defocus):
        results = analyze_potential(f=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design,
                                    diameter=diameter, n_actual=n_actual, n_rays=n_rays, dphi=dphi)
        if np.abs(results['center_of_curvature'][0]) > 1e2:
            R_inverse = 0
        else:
            R_inverse = 1 / (results['center_of_curvature'][0] - results['aspheric_surface'].center[0])
        residual = R_inverse - 1 / desired_focus
        return residual

    defocus_solution = optimize.brentq(f_roots, -back_focal_length, back_focal_length)
    results = analyze_potential(f=back_focal_length, defocus=defocus_solution, T_c=T_c, n_design=n_design,
                                diameter=diameter, n_actual=n_actual, n_rays=n_rays, dphi=dphi)
    return defocus_solution, results


# %%
f = 5e-3
defocus = 0
T_c = 3.0e-3
n = 1.8
diameter = 7.75e-3
dn=-0.02
n_actual = n + dn
n_rays = 10
defocus_solution, results_dict = choose_source_position_for_desired_defocus(desired_focus=30e-3,back_focal_length=f
                                           , T_c=T_c, n_design=n, diameter=diameter, n_rays=n_rays, n_actual=n_actual)
print(f"Defocus solution for 30 mm focus: {defocus_solution*1e3:.2f} mm, focal point distance: {(results_dict['center_of_curvature'][0] - results_dict['aspheric_surface'].center[0]) * 1e3:.2f} mm")
fig, ax = plot_results(results_dict, far_away_plane=False)
center = results_dict['center_of_curvature']
ax[0, 1].set_xlim((center[0]-0.002, center[0]+0.002))
plt.suptitle(f"n_design: {n:.3f}, n_actual: {n+dn:.3f}, Lens focal length: {f*1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus*1e3:.1f} mm, T_c: {T_c*1e3:.1f} mm, Diameter: {diameter*1e3:.2f} mm")
# Save image with suptitle in name:
plt.savefig(f"outputs/figures/analyze_potential_n_design_{n:.3f}_n_actual_{n+dn:.3f}_focal_length_{f*1e3:.1f}mm_defocus_{defocus*1e3:.1f}mm_Tc_{T_c*1e3:.1f}mm_diameter_{diameter*1e3:.2f}mm.svg", dpi=300)
plt.show()

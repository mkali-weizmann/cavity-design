from matplotlib import use
use('TkAgg')  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from cavity import *
from matplotlib.lines import Line2D

# %%

def initialize_rays(defocus: float=0, n_rays=100, dphi: Optional[float] =  None, phi_max: Optional[float] = None,):
    optical_axis = RIGHT
    if phi_max is None and dphi is not None:
        phi = np.arange(0, n_rays) * dphi
    elif phi_max is None and dphi is None:
        phi_max = np.arctan((diameter / 2) / (back_focal_length - defocus) / 2)
        phi = np.linspace(0, phi_max, n_rays)
    else:
        phi = np.linspace(0, phi_max, n_rays)
    ray_origin = optical_axis * defocus
    rays_0 = Ray(origin=ray_origin, k_vector=unit_vector_of_angles(theta=0, phi=phi))
    return rays_0

def analyze_output_wavefront(ray_sequence: RaySequence, R_analytical: Optional[float] = None):
    output_ray = ray_sequence[-1]
    # Extract all wavefront features at the output surface of the lens.
    d_0 = ray_sequence.cumulative_optical_path_length[1, 0]  # Assumes the first ray is the optical axis ray.
    wavefront_points_initial = ray_sequence.parameterization(d_0, optical_path_length=True)

    R_numerical, center_of_curvature_numerical = extract_matching_sphere(wavefront_points_initial[..., 0, :],
                                                                         wavefront_points_initial[..., 1, :],
                                                                         output_ray.k_vector[..., 0, :])
    if R_analytical is not None:
        R = R_analytical
        center_of_curvature = ray_sequence[-1].parameterization(R_analytical, optical_path_length=False)[
            0, :]  # Along the optical axis.
        print(
            f"Analytical/numerical output ROC:\n{R:.6e}\n{R_numerical:.6e} (inaccurate for large or extremeley small dphi)")
        print(
            f"Analytical/numerical center of curvature:\n{np.stack((center_of_curvature, center_of_curvature_numerical), axis=0)} (inacurate for large or extremeley small dphi)")
    else:
        R, center_of_curvature = R_numerical, center_of_curvature_numerical

    residual_distances_initial = R - np.linalg.norm(wavefront_points_initial - center_of_curvature, axis=-1)
    polynomial_residuals_initial = Polynomial.fit(wavefront_points_initial[:, 1] ** 2, residual_distances_initial,
                                                  4).convert()
    print(
        f"Initial wavefront residual from fitted sphere. 2nd order term should be singificantly smaller than 1/(2*R) = {1 / (2 * R):.3e}, actual: {polynomial_residuals_initial.coef[1]:.3e}")
    print(
        f" Fourth order term: {polynomial_residuals_initial.coef[2]:.3e}, should be significantly larger than y_max ** -2 * second order term = {wavefront_points_initial[-1, 1] ** -2 * polynomial_residuals_initial.coef[1]:.26e}")

    # Extract wavefront features at a far away plane (2*ROC - u from the lens):
    wavefront_points_opposite = ray_sequence.parameterization(d_0 + 2 * R - unconcentricity, optical_path_length=True)

    R_opposite = R - unconcentricity
    center_of_curvature_opposite = center_of_curvature
    R_opposite_numerical, center_of_curvature_opposite_numerical = extract_matching_sphere(
        wavefront_points_opposite[..., 0, :], wavefront_points_opposite[..., 1, :],
        ray_sequence[-1].k_vector[..., 0, :])  # Should be the same as the original center of curvature
    print(
        f"Far away plane analytical/numerical output ROC:\n{R_opposite:.6e}\n{R_opposite_numerical:.6e} (inaccurate for large or extremeley small dphi)")
    print(
        f"Far away plane analytical/numerical center of curvature:\n{np.stack((center_of_curvature_opposite, center_of_curvature_opposite_numerical), axis=0)} (inacurate for large or extremeley small dphi)")

    if R_analytical is None:
        R_opposite, center_of_curvature_opposite = R_opposite_numerical, center_of_curvature_opposite_numerical

    residual_distances_opposite = R_opposite - np.linalg.norm(wavefront_points_opposite - center_of_curvature_opposite,
                                                              axis=-1)
    polynomial_residuals_opposite = Polynomial.fit(wavefront_points_opposite[:, 1] ** 2, residual_distances_opposite,
                                                   4).convert()

    # Analyze unconcentric mirror case:
    center_of_mirror = center_of_curvature + np.array([-unconcentricity, 0, 0])
    residual_distances_mirror = R - np.linalg.norm(wavefront_points_opposite - center_of_mirror,
                                                   axis=-1)  # Mirror has a radius of R, not R_opposite.
    polynomial_residuals_mirror = Polynomial.fit(wavefront_points_opposite[:, 1] ** 2,
                                                 residual_distances_mirror, 4).convert()
    expected_second_order_term = 1 / 2 * (1 / R_opposite - 1 / R)
    print(
        f"Expected second order term in mirror deviation polynomial due to unconcentricity: {expected_second_order_term:.3e}, actual: {polynomial_residuals_mirror.coef[1]:.3e}")

    # Generate dummy points for fitted spheres:
    points_rel = wavefront_points_initial - center_of_curvature
    phi_dummy = np.linspace(0, np.arctan(points_rel[-1, 1] / points_rel[-1, 0]), 100)
    dummy_points_curvature_initial = center_of_curvature - R * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
        axis=-1)
    dummy_points_curvature_opposite = center_of_curvature_opposite + R_opposite * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
        axis=-1)
    dummy_points_mirror = center_of_mirror + R * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)),
        axis=-1)  # Mirror has the radius of the original wavefront sphere, but centered at the shifted center.
    L_long_arm = 2 * R - unconcentricity
    if unconcentricity > 0:
        NA_paraxial = np.sqrt(2 * LAMBDA_0_LASER / np.pi) * (L_long_arm * unconcentricity) ** (-1 / 4)
        spot_size_paraxial = NA_paraxial * (R - unconcentricity / 2)
    else:
        NA_paraxial, spot_size_paraxial = None, None

    # find point of 0 derivative (other than 0) in residual_distances_mirror:
    deriv_mirror = np.gradient(residual_distances_mirror, wavefront_points_opposite[:, 1])
    first_zero_crossings = np.where(np.diff(np.sign(deriv_mirror)))[0]
    if len(first_zero_crossings) > 0:
        zero_derivative_points = wavefront_points_opposite[first_zero_crossings[0], 1]
    else:
        zero_derivative_points = None

    results_dict = {
        'ray_sequence': ray_sequence,
        'wavefront_points_initial': wavefront_points_initial,
        'R': R,
        'center_of_curvature': center_of_curvature,
        'dummy_points_curvature_initial': dummy_points_curvature_initial,
        'residual_distances_initial': residual_distances_initial,
        'polynomial_residuals_initial': polynomial_residuals_initial,
        'R_opposite': R_opposite,
        'wavefront_points_opposite': wavefront_points_opposite,
        'center_of_curvature_opposite': center_of_curvature_opposite,
        'dummy_points_curvature_opposite': dummy_points_curvature_opposite,
        'dummy_points_mirror': dummy_points_mirror,
        'residual_distances_opposite': residual_distances_opposite,
        'residual_distances_mirror': residual_distances_mirror,
        'polynomial_residuals_opposite': polynomial_residuals_opposite,
        'polynomial_residuals_mirror': polynomial_residuals_mirror,
        'NA_paraxial': NA_paraxial,
        'spot_size_paraxial': spot_size_paraxial,
        'zero_derivative_points': zero_derivative_points,
    }

    return results_dict


def analyze_potential(R_1: Optional[float] = None, R_2: Optional[float] = None, back_focal_length: Optional[float] = None, defocus=0, T_c=3e-3, n_design=1.8, diameter=12.7e-3, unconcentricity: float = 0, n_actual = None,
                      n_rays=100, dphi: Optional[float] =  None, phi_max: Optional[float] = None,
                      extract_R_analytically: bool = False
                      ):

    optical_axis = RIGHT
    # Enrich input arguments:
    if n_actual is None:
        n_actual = n_design
    if R_1 is not None and R_2 is not None:
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n_design, T_c=T_c)
        surface_0, surface_1 = CurvedRefractiveSurface(radius=R_1, outwards_normal=-optical_axis, center=back_focal_length * optical_axis, n_1=1, n_2=n_actual, curvature_sign=CurvatureSigns.convex, name='first surface', thickness=T_c / 2, diameter=diameter), \
                                        CurvedRefractiveSurface(radius=R_2, outwards_normal=optical_axis, center=(back_focal_length + T_c) * optical_axis, n_1=n_actual, n_2=1, curvature_sign=CurvatureSigns.concave, name='second surface', thickness=T_c / 2, diameter=diameter)
    elif back_focal_length is not None:
        back_center = back_focal_length * optical_axis
        surface_0, surface_1 = Surface.from_params(generate_aspheric_lens_params(f=back_focal_length,
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
    rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, dphi=dphi, phi_max=phi_max)
    surface_0.n_2 = n_actual
    surface_1.n_1 = n_actual
    # Trace rays through the lens, and add refractive index perturbation:
    rays_1 = surface_0.propagate_ray(rays_0)
    rays_2 = surface_1.propagate_ray(rays_1)
    rays_0.n = 1
    rays_1.n = n_actual
    rays_2.n = 1
    rays_history = [rays_0, rays_1, rays_2]
    ray_sequence = RaySequence(rays_history)
    if extract_R_analytically:
        R_analytical = image_of_a_point_with_thick_lens(distance_to_face_1=back_focal_length - defocus, R_1=surface_0.radius,
                                             R_2=-surface_1.radius, n=n_actual,
                                             T_c=T_c)  # Assumes cylindrical symmetry.
    else:
        R_analytical = None
    results_dict = analyze_output_wavefront(ray_sequence, R_analytical=R_analytical)

    results_dict['surfaces'] = [surface_0, surface_1]

    return results_dict

def plot_results(results_dict, far_away_plane: bool = False):
    (surfaces, ray_sequence, R, center_of_curvature, NA_paraxial, spot_size_paraxial, zero_derivative_points) = \
        (results_dict['surfaces'], results_dict['ray_sequence'], results_dict['R'], results_dict['center_of_curvature'],
         results_dict['NA_paraxial'], results_dict['spot_size_paraxial'], results_dict['zero_derivative_points'])
    if far_away_plane:
        wavefront_points, residual_distances, dummy_points, polynomial, dummy_points_mirror, polynomial_residuals_mirror, residual_distances_mirror = results_dict['wavefront_points_opposite'], results_dict['residual_distances_opposite'], results_dict['dummy_points_curvature_opposite'], results_dict['polynomial_residuals_opposite'], results_dict['dummy_points_mirror'], results_dict['polynomial_residuals_mirror'], results_dict['residual_distances_mirror']
    else:
        wavefront_points, residual_distances, dummy_points, polynomial, dummy_points_mirror, polynomial_residuals_mirror, residual_distances_mirror = results_dict['wavefront_points_initial'], results_dict['residual_distances_initial'], results_dict['dummy_points_curvature_initial'], results_dict['polynomial_residuals_initial'], None, None, None
    # Truncate polynomial_residuals_mirror to first 3 orders only:
    # if polynomial_residuals_mirror is not None:
    #     coeffs_mirror_truncated = polynomial_residuals_mirror.coef[:3]
    #     polynomial_residuals_mirror = Polynomial(coeffs_mirror_truncated, domain=polynomial_residuals_mirror.domain, window=polynomial_residuals_mirror.window)
    # plot points:
    fig, ax = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    rays_0, rays_1, rays_2 = ray_sequence
    surface_0, surface_1 = surfaces
    rays_0.plot(ax=ax[0, 0], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 0], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 0], label='After aspheric surface', color='red', linewidth=0.5)
    surface_0.plot(ax=ax[0, 0], color='green')
    surface_1.plot(ax=ax[0, 0], color='orange')
    ax[0, 0].set_xlim(rays_0.origin[0, 0] - 0.01, center_of_curvature[0] * 2)  # (-1e-3, 100e-3)
    ax[0, 0].set_ylim(-surface_1.diameter / 2, surface_1.diameter / 2)  # (-4.2e-3, 4.2e-3)
    ax[0, 0].grid()
    ax[0, 0].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 0].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    rays_0.plot(ax=ax[0, 1], label='Before lens', color='black', linewidth=0.5)
    rays_1.plot(ax=ax[0, 1], label='After flat surface', color='blue', linewidth=0.5)
    rays_2.plot(ax=ax[0, 1], label='After aspheric surface', color='red', linewidth=0.5)
    surface_0.plot(ax=ax[0, 1], color='green')
    surface_1.plot(ax=ax[0, 1], color='orange')
    ax[0, 1].set_xlim((surface_1.center[0] + center_of_curvature[0]) / 2 - 0.002, center_of_curvature[0] + 0.005)#(-1e-3, 100e-3)
    ax[0, 1].set_ylim(-rays_2.origin[1, 1]*0.5, 1*rays_2.origin[1, 1])#(-4.2e-3, 4.2e-3)
    ax[0, 1].grid()
    ax[0, 1].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color='purple')
    ax[0, 1].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color='cyan', label='Center of curvature')

    ax[1, 0].plot(wavefront_points[:, 0] * 1e3, wavefront_points[:, 1] * 1e3, label='wavefront points', color='black', marker='o', linestyle='', markersize=5, zorder=1, alpha=0.4,)
    ax[1, 0].plot(dummy_points[:, 0] * 1e3,
                  dummy_points[:, 1] * 1e3,
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
    if dummy_points_mirror is not None:
        ax[1, 0].plot(dummy_points_mirror[:, 0] * 1e3,
                      dummy_points_mirror[:, 1] * 1e3,
                      linestyle='dashed',
                      color='magenta',
                      label='Fitted sphere for unconcentric mirror',
                      linewidth=2.0,
                      alpha=0.5,
                      zorder=2)
        ax[1, 0].legend()

    ax[1, 1].plot(wavefront_points[:, 1] * 1e3, residual_distances * 1e6,
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
    if polynomial_residuals_mirror is not None:
        coeffs_mirror_asc = polynomial_residuals_mirror.coef
        for i, c in enumerate(coeffs_mirror_asc):
            s = f"{c:.1e}"
            mant_str, exp_str = s.split('e')
            mant = float(mant_str)
            exp = int(exp_str)
            terms_parts_mirror.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
        terms_mirror = " + ".join(terms_parts_mirror)
        title += f"\nmirror deviation fit (unconcentricity = {unconcentricity*1e6:.1f} µm):\n" + terms_mirror
        ax[1, 1].plot(x_fit * 1e3, polynomial_residuals_mirror(x_fit ** 2) * 1e6, color='green', linestyle='dashed',
                      label='Mirror residuals Polynomial fit', linewidth=0.5)
        ax[1, 1].plot(wavefront_points[:, 1] * 1e3, residual_distances_mirror * 1e6, marker='x', linestyle='', color='magenta', label='Mirror deviation data', markersize=5, alpha=0.6)

    ax[1, 1].set_title(title)
    ax[1, 1].plot(x_fit * 1e3, polynomial(x_fit**2) * 1e6, color='red', linestyle='dashed', label='Matching sphere residuals Polynomial fit', linewidth=0.5)
    # Plot intensity profile on a new y axis for ax[1, 1] if NA_paraxial is not None: using the formula: e^{-y^{2} / (spot_size_paraxial / 2)^{2})}
    if NA_paraxial is not None and spot_size_paraxial is not None:
        ax2 = ax[1, 1].twinx()
        intensity_profile = np.exp(-x_fit**2 / (spot_size_paraxial / 2)**2)
        ax2.plot(x_fit * 1e3, intensity_profile, color='orange', linestyle='dashed', label='Paraxial Gaussian intensity profile', linewidth=1)
        legend_line = Line2D([], [], color='orange', linestyle='dashed', linewidth=1, label='Paraxial Gaussian intensity profile')
        handles, labels = ax[1, 1].get_legend_handles_labels()
        handles.append(legend_line)
        ax2.set_ylabel('Relative intensity (a.u.)')
        ax2.grid(False)

    ax[1, 1].axvline(zero_derivative_points * 1e3, label='2nd vs 4th order max', color='purple', linestyle='dotted')
    ax[1, 1].legend(handles=handles)
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
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n_design, T_c=T_c)
    else:
        raise ValueError("Either both R_1 and R_2 must be provided, or neither.")
    distance_to_flat_face = image_of_a_point_with_thick_lens(distance_to_face_1=desired_focus, R_1=R_2,
                                                             R_2=-R_1, n=n_design, T_c=T_c)
    defocus = back_focal_length - distance_to_flat_face
    return defocus

# %%
# For aspheric lens:
aspheric = True
if aspheric:
    back_focal_length = back_focal_length_of_lens(R_1=24.22e-3, R_2=-5.49e-3, n=1.8, T_c=2.91e-3)# 20e-3
    R_1 = None
    R_2 = None
    R_2_signed = None
    T_c = 4.35e-3
    diameter = 7.75e-3
else:
    R_1 = 24.22e-3
    R_2 = 5.49e-3
    R_2_signed = -R_2
    T_c = 2.91e-3
    diameter = 7.75e-3
    back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=1.8, T_c=T_c)
# rest of parameters
n_actual = 1.8
dn=0
n_design = n_actual + dn
n_rays = 50
unconcentricity = 5e-3
phi_max = 0.3
desired_focus = 200e-3

defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter,
                                                            back_focal_length=back_focal_length,
                                                            R_1=R_1, R_2=R_2_signed,
)


# %%
results_dict = analyze_potential(
                                 back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,
                                 defocus=defocus, T_c=T_c,
                                 n_design=n_design, diameter=diameter,
                                 n_actual=n_actual, n_rays=n_rays,
                                 unconcentricity=unconcentricity, extract_R_analytically=True, phi_max=phi_max)
print(f"Defocus solution for 30 mm focus: {defocus*1e3:.3f} mm, focal point distance: {(results_dict['center_of_curvature'][0] - results_dict['surfaces'][1].center[0]) * 1e3:.2f} mm")
# plt.close('all')
fig, ax = plot_results(results_dict, far_away_plane=True)
center = results_dict['center_of_curvature']
ax[0, 1].set_xlim((center[0]-0.002, center[0]+0.002))
plt.suptitle(f"aspheric={aspheric} n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {back_focal_length * 1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus * 1e3:.1f} mm, T_c: {T_c * 1e3:.1f} mm, Diameter: {diameter * 1e3:.2f} mm")
# Save image with suptitle in name:
plt.savefig(f"outputs/figures/analyze_potential_n_design_aspheric={aspheric}_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{back_focal_length * 1e3:.1f}mm_defocus_{defocus * 1e3:.1f}mm_Tc_{T_c * 1e3:.1f}mm_diameter_{diameter * 1e3:.2f}mm.svg", dpi=300)
plt.show()
print(f"Paraxial spot size for unconcentricity of {unconcentricity*1e3:.1f} µm: {results_dict['spot_size_paraxial']*1e6:.2f} µm")
print(f"Boundary of 2nd vs 4th order dominance for unconcentricity of {unconcentricity*1e3:.1f} µm: {results_dict['zero_derivative_points']*1e3:.2f} mm")
# %%
unconcentricities = np.linspace(0.1e-3, 10e-3, 30)
paraxial_spot_sizes = np.zeros_like(unconcentricities)
spot_size_boundaries = np.zeros_like(unconcentricities)
paraxial_NAs = np.zeros_like(unconcentricities)
for i, u in enumerate(unconcentricities):
    results_dict = analyze_potential(
        back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,
        defocus=defocus, T_c=T_c,
        n_design=n_design, diameter=diameter,
        n_actual=n_actual, n_rays=n_rays,
        unconcentricity=u, extract_R_analytically=True, phi_max=phi_max)
    paraxial_spot_sizes[i] = results_dict['spot_size_paraxial']
    paraxial_NAs[i] = results_dict['NA_paraxial']
    try:
        spot_size_boundaries[i] = np.abs(results_dict['zero_derivative_points'])
    except TypeError:
        spot_size_boundaries[i] = np.nan  # If zero_derivative_points is None

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(unconcentricities * 1e3, paraxial_spot_sizes * 1e6, label='Paraxial spot size', color='blue')
ax.plot(unconcentricities * 1e3, spot_size_boundaries * 1e6, label='Boundary of 2nd vs 4th order dominance', color='red')
ax.set_xlabel('Unconcentricity (mm)')
ax.set_ylabel('Spot size / boundary (µm)')
ax.set_title('Effect of Unconcentricity on Paraxial Spot Size and Residual Dominance Boundary')
ax.grid()

# twin y-axis for paraxial NAs
ax2 = ax.twinx()
ax2.plot(unconcentricities * 1e3, paraxial_NAs, label='Paraxial NA', color='orange', linestyle='--')
ax2.set_ylabel('Paraxial NA (unitless)')

# combined legend
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc='best')

# plt.savefig(f"outputs/figures/unconcentricity_vs_spot_size_and_boundary_aspheric={aspheric}_n_design_{n_design:.3f}_n_actual_{n_actual:.3f}_focal_length_{back_focal_length * 1e3:.1f}mm_defocus_{defocus * 1e3:.1f}mm_Tc_{T_c * 1e3:.1f}mm_diameter_{diameter * 1e3:.2f}mm.svg", dpi=300)
plt.show()
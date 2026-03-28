from simple_analysis_scripts.potential_analysis.analyze_potential import *

copy_image = False
copy_input_parameters = True
copy_cavity_parameters = False
max_NA_for_polynomial = 1.0000000000e-02
fourth_order_target_value_log10 = 0.0000000000e+00
n_rays = 30
marginal_ray_NA = 3.0000000000e-01
first_arm_NA = 1.5200000000e-01
lens_type = 'avantier'
mirror_setting_mode = 'Set distance and unconcentricity'
negative_lens_refractive_index = 1.4500000000e+00
negative_lens_center_thickness = 4.3500000000e-03
large_elements_CA = 5.0000000000e-02
right_mirror_ROC = 5.0000000000e-01
right_mirror_distance_to_negative_lens_front = 1.0000000000e-02
right_mirror_ROC_fine = 5.0000000000e-01
unconcentricity = 3.5818000000e+00
negative_lens_defocus_power = -1.2710100000e+01
negative_lens_R_2_inverse = 6.6600000000e+01
desired_focus = 7.0800000000e-02
negative_lens_back_relative_position = 2.5000000000e-01

right_mirror_ROC_fine = widget_convenient_exponent(right_mirror_ROC_fine, scale=-10)
fourth_order_target_value = widget_convenient_exponent(fourth_order_target_value_log10, scale=0)
phi_max_marginal = np.arcsin(marginal_ray_NA)
phi_max_polynomial = np.arcsin(max_NA_for_polynomial)
if mirror_setting_mode == "Set ROC":
    right_mirror_distance_to_negative_lens_front = None
    unconcentricity = None
    right_mirror_ROC += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance to spherical":
    right_mirror_ROC = None
    unconcentricity = None
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance and unconcentricity":
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine
    unconcentricity = widget_convenient_exponent(unconcentricity, scale=-10)
    right_mirror_ROC = None
else:
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine
    unconcentricity = None

n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(
    lens_type=lens_type,
    dn=0)


def f_root_focal_length(negative_lens_focal_length_inverse):
    if negative_lens_focal_length_inverse > 0:
        return 100 + 100 * negative_lens_focal_length_inverse, np.nan, np.nan
    if np.isclose(negative_lens_focal_length_inverse, 0):
        negative_lens_focal_length = np.inf
    else:
        negative_lens_focal_length = 1 / negative_lens_focal_length_inverse
    negative_lens_R_2_inverse = -1

    def f_root_lens_position(negative_lens_back_relative_position):
        if negative_lens_back_relative_position < 0:
            return -100 - 100 * negative_lens_back_relative_position
        cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design,
                                               T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length,
                                               R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                               diameter_first_lens=diameter,
                                               approximate_focus_distance_long_arm=desired_focus,
                                               negative_lens_focal_length=negative_lens_focal_length,
                                               negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                               negative_lens_back_relative_position=negative_lens_back_relative_position,
                                               negative_lens_refractive_index=negative_lens_refractive_index,
                                               negative_lens_center_thickness=negative_lens_center_thickness,
                                               first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC,
                                               right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
                                               large_elements_CA=large_elements_CA, unconcentricity=unconcentricity)
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
        marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False,
                                            n_arms=5)
        if np.isnan(marginal_ray.origin[-1, 1]):
            two_y_marginal_over_CA_right = 10 * negative_lens_back_relative_position  # np.inf
        else:
            two_y_marginal_over_CA_right = 2 * np.abs(marginal_ray.origin[-1, 1]) / large_elements_CA
        value_for_root = two_y_marginal_over_CA_right - 1
        return value_for_root

    try:
        negative_lens_back_relative_position = newton(func=f_root_lens_position, x0=desired_focus, tol=1e-6,
                                                      maxiter=100)
    except RuntimeError:
        print(
            f"lens position solver did not converge for desired focus {desired_focus:.3f} m and focal length of ={negative_lens_focal_length:.2f}")

    def f_root_lens_right(negative_lens_R_2_inverse):
        cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design,
                                               T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length,
                                               R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                               diameter_first_lens=diameter,
                                               approximate_focus_distance_long_arm=desired_focus,
                                               negative_lens_focal_length=negative_lens_focal_length,
                                               negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                               negative_lens_back_relative_position=negative_lens_back_relative_position,
                                               negative_lens_refractive_index=negative_lens_refractive_index,
                                               negative_lens_center_thickness=negative_lens_center_thickness,
                                               first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC,
                                               right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
                                               large_elements_CA=large_elements_CA, unconcentricity=unconcentricity)
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
        marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False,
                                            n_arms=5)
        alpha_left = np.arccos(np.abs(
            cavity.arms[2].surface_1.normal_at_a_point(marginal_ray[3].origin) @ marginal_ray[2].k_vector)) * 360 / (
                             2 * np.pi)
        alpha_right = np.arccos(np.abs(
            cavity.arms[4].surface_0.normal_at_a_point(marginal_ray[4].origin) @ marginal_ray[4].k_vector)) * 360 / (
                              2 * np.pi)
        angles_difference = alpha_left - alpha_right
        if np.isnan(angles_difference):
            angles_difference = 10 + 10 * negative_lens_R_2_inverse
        return angles_difference

    try:
        negative_lens_R_2_inverse = newton(func=f_root_lens_right, x0=negative_lens_R_2_inverse, tol=1e-6, maxiter=100)
    except RuntimeError:
        print(
            f"Did not converge for desired focus {desired_focus:.3} m and negative lens back relative position {negative_lens_back_relative_position}")

    cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design,
                                           T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length,
                                           R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                           diameter_first_lens=diameter,
                                           approximate_focus_distance_long_arm=desired_focus,
                                           negative_lens_focal_length=negative_lens_focal_length,
                                           negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                           negative_lens_back_relative_position=negative_lens_back_relative_position,
                                           negative_lens_refractive_index=negative_lens_refractive_index,
                                           negative_lens_center_thickness=negative_lens_center_thickness,
                                           first_arm_NA=first_arm_NA,
                                           right_mirror_ROC=right_mirror_ROC,
                                           right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
                                           large_elements_CA=large_elements_CA, unconcentricity=unconcentricity)
    results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=phi_max_polynomial,
                                                  print_tests=False)
    return results_dict['polynomial_residuals_opposite'].coef[
        2], negative_lens_back_relative_position, negative_lens_R_2_inverse,


f_root_focal_length_for_solver = lambda negative_lens_back_relative_position: \
    f_root_focal_length(negative_lens_back_relative_position)[
        0] - fourth_order_target_value  # Arbitrary small positive value
negative_lens_focal_length_inverse, results_report = newton(func=f_root_focal_length_for_solver, x0=-10,
                                                            tol=1e-6, maxiter=100, disp=False, full_output=True)
negative_lens_focal_length = 1 / negative_lens_focal_length_inverse
_, negative_lens_back_relative_position, negative_lens_R_2_inverse, = f_root_focal_length(
    negative_lens_focal_length_inverse)

cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design, T_c_first_lens=T_c,
                                       back_focal_length_first_lens=back_focal_length, R_1_first_lens=R_1,
                                       R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                       diameter_first_lens=diameter, approximate_focus_distance_long_arm=desired_focus,
                                       negative_lens_focal_length=negative_lens_focal_length,
                                       negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                       negative_lens_back_relative_position=negative_lens_back_relative_position,
                                       negative_lens_refractive_index=negative_lens_refractive_index,
                                       negative_lens_center_thickness=negative_lens_center_thickness,
                                       large_elements_CA=large_elements_CA,
                                       first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC,
                                       right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
                                       unconcentricity=unconcentricity)
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=phi_max_polynomial,
                                              print_tests=False)
marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5)
marginal_ray_final_step_inverted = Ray(origin=marginal_ray.origin[-2, :], k_vector=-marginal_ray.k_vector[-2, :])
metadata_dict = {
    'marginal ray NA': marginal_ray_NA,
    'y_marginal lens left (mm)': np.abs(marginal_ray[3].origin[1]) * 1e3,
    'y_marginal lens right (mm)': np.abs(marginal_ray[4].origin[1]) * 1e3,
    'y_marginal right mirror (mm)': np.abs(marginal_ray.origin[-1, 1]) * 1e3,
    '2y_marginal/R lens left': 2 * np.abs(marginal_ray[3].origin[1]) / cavity.arms[2].surface_1.radius,
    '2y_marginal/R lens right': 2 * np.abs(marginal_ray[4].origin[1]) / cavity.arms[3].surface_1.radius,
    '2y_marginal/CA lens left': 2 * np.abs(marginal_ray[3].origin[1]) / large_elements_CA,
    '2y_marginal/R mirror right': 2 * np.abs(marginal_ray.origin[-1, 1]) / cavity.arms[4].surface_1.radius,
    '2y_marginal/CA right mirror': 2 * np.abs(marginal_ray.origin[-1, 1]) / large_elements_CA,
    'negative lens left radius (mm)': cavity.surfaces[3].radius * 1e3,
    'negative lens right radius (mm)': cavity.surfaces[4].radius * 1e3,
    'negative lens focal length': negative_lens_focal_length * 1e3,
    'right mirror radius (mm)': cavity.surfaces[-1].radius * 1e3,
    'right mirror distance to negative lens (mm)': np.linalg.norm(
        cavity.surfaces[-1].center - cavity.surfaces[-2].center) * 1e3,
    'fourth order polynomial coefficient (m^-3)': results_dict['polynomial_residuals_opposite'].coef[2],
    'sixth order polynomial coefficient (m^-5)': results_dict['polynomial_residuals_opposite'].coef[3],
    'NA left arm': cavity.arms[0].mode_parameters.NA[0],
    'NA middle arm': cavity.arms[2].mode_parameters.NA[0],
    'NA right arm': cavity.arms[4].mode_parameters.NA[0],
    'Incidence angle left (deg)': np.arccos(
        np.abs(cavity.arms[2].surface_1.normal_at_a_point(marginal_ray[3].origin) @ marginal_ray[2].k_vector)) * 360 / (
                                          2 * np.pi),
    'Incidence angle right (deg)': np.arccos(np.abs(
        -cavity.arms[4].surface_0.normal_at_a_point(marginal_ray[4].origin) @ marginal_ray[4].k_vector)) * 360 / (
                                           2 * np.pi),
}
# if metadata_dict['fourth order polynomial coefficient (m^-3)'] > 0 and not np.isnan(metadata_dict['2y_marginal/CA right mirror']):
print(f"desired_focus =                        {desired_focus:.6f}\n"
      f"negative_lens_back_relative_position = {negative_lens_back_relative_position:.6f}\n"
      f"negative_lens_defocus_power =          {1 / negative_lens_focal_length:.6f}\n"
      f"negative_lens_R_2_inverse =            {negative_lens_R_2_inverse:.6f}\n"
      f"large_elements_CA =                    {large_elements_CA:.3f}\n"
      f"marginal_ray_NA =                      {marginal_ray_NA:.3f}\n"
      f"fourth order polynomial coefficient (m^-3): {metadata_dict['fourth order polynomial coefficient (m^-3)']}\n")

fig, ax = plt.subplots(2, 1, figsize=(20, 20))
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True, fig_and_ax=(fig, ax))
marginal_ray.plot(ax=ax[1])
marginal_ray_final_step_inverted.plot(ax=ax[1], linestyle='dashed', color='red')
fig.suptitle(
    f"marginal ray NA: {metadata_dict['marginal ray NA']:.3f}\n2y_marginal/CA lens left = {metadata_dict['2y_marginal/CA lens left']:.2f}, 2y_marginal/R lens left: {metadata_dict['2y_marginal/R lens left']:.2f}, 2y_marginal/R lens right: {metadata_dict['2y_marginal/R lens right']:.2f}\n2y_marginal/R mirror right: {metadata_dict['2y_marginal/R mirror right']:.2f}, 2y_marginal/CA right mirror: {metadata_dict['2y_marginal/CA right mirror']:.2f}\n"
    f"NA short arm = {metadata_dict['NA left arm']:.3f}, NA middle arm = {metadata_dict['NA middle arm']:.3f}, NA right arm = {metadata_dict['NA right arm']:.3f}\nincidence angle left = {metadata_dict['Incidence angle left (deg)']:.2f} deg, incidence angle right = {metadata_dict['Incidence angle right (deg)']:.2f} deg")
ax[1].set_title(
    f"lens left radius: {metadata_dict['negative lens left radius (mm)']:.2f}mm, lens right radius = {metadata_dict['negative lens right radius (mm)']:.2f}mm, lens focal length = {metadata_dict['negative lens focal length']:.2f}mm, lens n = {negative_lens_refractive_index:.2f}, lens T_c = {negative_lens_center_thickness * 1e3:.2f}mm\nright mirror radius = {metadata_dict['right mirror radius (mm)']:.3f} mm, right mirror distance to negative lens = {metadata_dict['right mirror distance to negative lens (mm)']:.3f} mm, unconcentricity={unconcentricity if mirror_setting_mode == 'Set distance and unconcentricity' else None}")
ax[1].set_ylim(-large_elements_CA / 1.8,
               large_elements_CA / 1.8)  # (-np.abs(final_intersection[1])*1.1, np.abs(final_intersection[1]))
# ax[1].hlines([-12.5, ])
ax[1].grid()
ax[1].set_aspect('equal')
fig.tight_layout()

if copy_cavity_parameters:
    pyperclip.copy(results_dict['cavity'].formatted_textual_params)

plt.show()
print(pd.Series(metadata_dict))
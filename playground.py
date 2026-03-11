# from matplotlib import use
from utils import angles_of_unit_vector
# use('TkAgg')
from simple_analysis_scripts.potential_analysis.analyze_potential import *
from scipy.optimize import newton

copy_image = False
copy_input_parameters = True
copy_cavity_parameters = False
max_NA_for_polynomial = 0.05
n_rays = 30
marginal_ray_NA = 1.50000000e-01
first_arm_NA = 1.5200000000e-01
lens_type = 'avantier'
mirror_setting_mode = "Set both"
right_mirror_ROC = 1e-01
right_mirror_distance_to_negative_lens_front = 2e-2
right_mirror_ROC_fine = 1.0000000000e-01
n_actual_spherical = 1.4500000000e+00
negative_lens_defocus_power = -1.2710100000e+01
negative_lens_R_2_inverse = 6.6600000000e+01
desired_focus = 1.0070000000e-01
negative_lens_back_relative_position = 2.5000000000e-01

right_mirror_ROC_fine = widget_convenient_exponent(right_mirror_ROC_fine, scale=-10)
phi_max_marginal = np.arcsin(marginal_ray_NA)
phi_max_polynomial = np.arcsin(max_NA_for_polynomial)
if mirror_setting_mode == "Set ROC":
    right_mirror_distance_to_negative_lens_front = None
    right_mirror_ROC += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance to spherical":
    right_mirror_ROC = None
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine

n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(
    lens_type=lens_type,
    dn=0)

for desired_focus in np.linspace(0.05, 1, 20):
    for negative_lens_back_relative_position in np.linspace(-desired_focus/2, 1, 20):
        if np.abs(negative_lens_back_relative_position) < desired_focus / 5:
            continue
        negative_lens_focal_length = -1
        negative_lens_R_2_inverse = 1
        print(f"{desired_focus=:.3f} m, {negative_lens_back_relative_position=:.3f}", end="\r")

        def f_root_lens_right(negative_lens_R_2_inverse):
            cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design,
                                                   T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length,
                                                   R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                                   diameter_first_lens=diameter,
                                                   approximate_focus_distance_long_arm=desired_focus,
                                                   negative_lens_focal_length=negative_lens_focal_length,
                                                   negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                                   negative_lens_back_relative_position=negative_lens_back_relative_position,
                                                   negative_lens_refractive_index=1.45, negative_lens_center_thickness=3.45e-3,
                                                   first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC,
                                                   right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front, )
            marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
            marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5)
            #
            if np.isnan(marginal_ray[4].origin[1]):
                return np.inf
            else:
                two_y_marginal_over_R_lens_right = 2 * np.abs(marginal_ray[4].origin[1]) / cavity.arms[3].surface_1.radius
            return two_y_marginal_over_R_lens_right - 1

        try:
            negative_lens_R_2_inverse = newton(func=f_root_lens_right, x0=negative_lens_R_2_inverse, tol=1e-6, maxiter=100)
        except RuntimeError:
            print(f"Did not converge for desired focus {desired_focus} m and negative lens back relative position {negative_lens_back_relative_position}")
            continue

        def f_root_mirror(negative_focal_length_inverse: float):
            if negative_focal_length_inverse == 0:
                negative_lens_focal_length = np.inf
            else:
                negative_lens_focal_length = 1 / negative_focal_length_inverse
            cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design,
                                                   T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length,
                                                   R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed,
                                                   diameter_first_lens=diameter,
                                                   approximate_focus_distance_long_arm=desired_focus,
                                                   negative_lens_focal_length=negative_lens_focal_length,
                                                   negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                                   negative_lens_back_relative_position=negative_lens_back_relative_position,
                                                   negative_lens_refractive_index=1.45, negative_lens_center_thickness=3.45e-3,
                                                   first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC,
                                                   right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front, )
            marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
            marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5)
            if np.isnan(marginal_ray.origin[-1, 1]):
                two_y_marginal_over_CA_right = -10 * negative_focal_length_inverse  # np.inf
            else:
                two_y_marginal_over_CA_right = 2 * np.abs(marginal_ray.origin[-1, 1]) / 25e-3
            value_for_root = two_y_marginal_over_CA_right - 1
            return value_for_root
        try:
            negative_lens_focal_length_inverse = newton(func=f_root_mirror, x0=1/negative_lens_defocus_power, tol=1e-6, maxiter=100, disp=False, full_output=True)
        except RuntimeError:
            print(f"Did not converge for desired focus {desired_focus} m and negative lens back relative position {negative_lens_back_relative_position}")
            continue
        negative_lens_focal_length = 1 / negative_lens_focal_length_inverse[0]

        cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design, T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length, R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed, diameter_first_lens=diameter, approximate_focus_distance_long_arm=desired_focus,
                                               negative_lens_focal_length=negative_lens_focal_length, negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                               negative_lens_back_relative_position=negative_lens_back_relative_position, negative_lens_refractive_index=1.45, negative_lens_center_thickness=3.45e-3, first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC, right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front, )
        results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=phi_max_polynomial, print_tests=False)
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector = unit_vector_of_angles(theta=0, phi=phi_max_marginal))
        marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5)
        metadata_dict = {
        'marginal ray NA': marginal_ray_NA,
        'y_marginal lens left (mm)': np.abs(marginal_ray[3].origin[1])*1e3,
        'y_marginal lens right (mm)': np.abs(marginal_ray[4].origin[1])*1e3,
        'y_marginal right mirror (mm)': np.abs(marginal_ray.origin[-1, 1])*1e3,
        '2y_marginal/R lens left': 2 * np.abs(marginal_ray[3].origin[1]) / cavity.arms[2].surface_1.radius,
        '2y_marginal/R lens right': 2 * np.abs(marginal_ray[4].origin[1]) / cavity.arms[3].surface_1.radius,
        '2y_marginal/CA lens left': 2 * np.abs(marginal_ray[3].origin[1]) / 25e-3,
        '2y_marginal/R mirror right': 2 * np.abs(marginal_ray.origin[-1, 1]) / cavity.arms[4].surface_1.radius,
        '2y_marginal/CA right mirror': 2 * np.abs(marginal_ray.origin[-1, 1]) / 25e-3,
        'negative lens left radius (mm)': cavity.surfaces[3].radius*1e3,
        'negative lens right radius (mm)': cavity.surfaces[4].radius*1e3,
        'negative lens focal length': negative_lens_focal_length*1e3,
        'right mirror radius (mm)': cavity.surfaces[-1].radius*1e3,
        'right mirror distance to negative lens (mm)': np.linalg.norm(cavity.surfaces[-1].center - cavity.surfaces[-2].center) * 1e3,
        'fourth order polynomial coefficient (m^-3)': results_dict['polynomial_residuals_mirror'].coef[2],
        'sixth order polynomial coefficient (m^-5)': results_dict['polynomial_residuals_mirror'].coef[3],
        'NA left arm': cavity.arms[0].mode_parameters.NA[0],
        'NA middle arm': cavity.arms[2].mode_parameters.NA[0],
        'NA right arm': cavity.arms[4].mode_parameters.NA[0],
        'Incidence angle left (deg)': np.arccos(-cavity.arms[2].surface_1.normal_at_a_point(marginal_ray[3].origin) @ marginal_ray[2].k_vector) * 360 / (2*np.pi),
        'Incidence angle right (deg)': np.arccos(-cavity.arms[4].surface_0.normal_at_a_point(marginal_ray[4].origin) @ marginal_ray[4].k_vector) * 360 / (2*np.pi),
        }
        if metadata_dict['fourth order polynomial coefficient (m^-3)'] > 0 and not np.isnan(metadata_dict['2y_marginal/CA right mirror']):
            print(f"desired_focus =                             {desired_focus:.6f}\n"
                  f"negative_lens_back_relative_position =      {negative_lens_back_relative_position:.6f}\n"
                  f"negative_lens_defocus_power =         {negative_lens_focal_length_inverse[0]:.6f}\n"
                  f"negative_lens_R_2_inverse =           {negative_lens_R_2_inverse:.6f}\n"
                  f"fourth order polynomial coefficient (m^-3): {metadata_dict['fourth order polynomial coefficient (m^-3)']}")
            fig, ax = plot_results(results_dict=results_dict, far_away_plane=True)
            marginal_ray.plot(ax=ax[1])
            fig.suptitle(f"marginal ray NA: {metadata_dict['marginal ray NA']:.3f}\n2y_marginal/CA lens left = {metadata_dict['2y_marginal/CA lens left']:.2f}, 2y_marginal/R lens left: {metadata_dict['2y_marginal/R lens left']:.2f}, 2y_marginal/R lens right: {metadata_dict['2y_marginal/R lens right']:.2f}\n2y_marginal/R mirror right: {metadata_dict['2y_marginal/R mirror right']:.2f}, 2y_marginal/CA right mirror: {metadata_dict['2y_marginal/CA right mirror']:.2f}\n"
                         f"NA short arm = {metadata_dict['NA left arm']:.3f}, NA middle arm = {metadata_dict['NA middle arm']:.3f}, NA right arm = {metadata_dict['NA right arm']:.3f}\nincidence angle left = {metadata_dict['Incidence angle left (deg)']:.2f} deg, incidence angle right = {metadata_dict['Incidence angle right (deg)']:.2f} deg")
            ax[1].set_title(f"negative lens left radius: {metadata_dict['negative lens left radius (mm)']:.2f}mm, negative lens right radius = {metadata_dict['negative lens right radius (mm)']:.2f}mm, negative lens focal length = {metadata_dict['negative lens focal length']:.2f} mm\nright mirror radius = {metadata_dict['right mirror radius (mm)']:.3f} mm, right mirror distance to negative lens = {metadata_dict['right mirror distance to negative lens (mm)']:.3f} mm")
            ax[1].set_ylim(-16e-3, 16e-3)
            # ax[1].hlines([-12.5, ])
            ax[1].grid()
            # fig.tight_layout()
            plt.show()
            break

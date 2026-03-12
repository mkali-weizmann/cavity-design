# from matplotlib import use
import numpy as np

from utils import angles_of_unit_vector
# use('TkAgg')
from simple_analysis_scripts.potential_analysis.analyze_potential import *
from scipy.optimize import newton

copy_image = False
copy_input_parameters = True
copy_cavity_parameters = False
max_NA_for_polynomial = 5.0000000000e-02
n_rays = 30
marginal_ray_NA = 2.700000000e-01
first_arm_NA = 1.5200000000e-01
lens_type = 'avantier'
mirror_setting_mode = 'Set both'
negative_lens_refractive_index = 1.4500000000e+00
large_elements_CA = 5.0000000000e-02
right_mirror_ROC = 2.0000000000e-01
right_mirror_distance_to_negative_lens_front = 2.0000000000e-02
right_mirror_ROC_fine = 0
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

desired_foci = np.array([0.05, 0.06, 0.07, 0.09, 0.1, 0.13, 0.15, 0.2])
large_elements_CAs = np.linspace(0.025, 0.05, 4)
large_mirror_position = np.zeros((len(desired_foci), len(large_elements_CAs)))
angle_of_incidence_right_lens = np.zeros((len(desired_foci), len(large_elements_CAs)))

for i, desired_focus in enumerate(desired_foci):
    for j, large_elements_CA in enumerate(large_elements_CAs):
        f_root_negative_lens_position = lambda negative_lens_back_relative_position: gnerate_negative_lens_cavity_smart(phi_max_marginal, phi_max_polynomial, n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter, n_rays, first_arm_NA, negative_lens_refractive_index, large_elements_CA, right_mirror_ROC, right_mirror_distance_to_negative_lens_front, negative_lens_defocus_power, negative_lens_R_2_inverse, desired_focus, negative_lens_back_relative_position)[0] - 0.02
        try:
            negative_lens_back_relative_position, results_report = newton(func=f_root_negative_lens_position, x0=2 * desired_focus, tol=1e-6, maxiter=100, disp=False, full_output=True)
        except ValueError:
            print(f"Failed to find root for desired_focus = {desired_focus:.6f}, large_elements_CA = {large_elements_CA:.3f}")
            continue
        _, negative_lens_focal_length, negative_lens_R_2_inverse, cavity = gnerate_negative_lens_cavity_smart(phi_max_marginal, phi_max_polynomial, n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter, n_rays, first_arm_NA, negative_lens_refractive_index, large_elements_CA, right_mirror_ROC, right_mirror_distance_to_negative_lens_front, negative_lens_defocus_power, negative_lens_R_2_inverse, desired_focus, negative_lens_back_relative_position)
        results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=np.arcsin(max_NA_for_polynomial), print_tests=False)
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=np.arcsin(marginal_ray_NA)))
        marginal_ray = cavity.propagate_ray(ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5)
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
            'fourth order polynomial coefficient (m^-3)': results_dict['polynomial_residuals_mirror'].coef[2],
            'sixth order polynomial coefficient (m^-5)': results_dict['polynomial_residuals_mirror'].coef[3],
            'NA left arm': cavity.arms[0].mode_parameters.NA[0],
            'NA middle arm': cavity.arms[2].mode_parameters.NA[0],
            'NA right arm': cavity.arms[4].mode_parameters.NA[0],
            'Incidence angle left (deg)': np.arccos(
                -cavity.arms[2].surface_1.normal_at_a_point(marginal_ray[3].origin) @ marginal_ray[2].k_vector) * 360 / (
                                                      2 * np.pi),
            'Incidence angle right (deg)': np.arccos(
                -cavity.arms[4].surface_0.normal_at_a_point(marginal_ray[4].origin) @ marginal_ray[4].k_vector) * 360 / (
                                                       2 * np.pi),
        }
        # if metadata_dict['fourth order polynomial coefficient (m^-3)'] > 0 and not np.isnan(metadata_dict['2y_marginal/CA right mirror']):
        print(f"desired_focus =                        {desired_focus:.6f}\n"
              f"negative_lens_back_relative_position = {negative_lens_back_relative_position:.6f}\n"
              f"negative_lens_defocus_power =          {1 / negative_lens_focal_length:.6f}\n"
              f"negative_lens_R_2_inverse =            {negative_lens_R_2_inverse:.6f}\n"
              f"large_elements_CA =                    {large_elements_CA:.3f}\n"
              f"marginal_ray_NA =                      {marginal_ray_NA:.3f}\n"
              f"Angle of incidence on lens concave side (deg) = {metadata_dict['Incidence angle left (deg)']:.0f}\n"
              f"large_mirror_position =                         {metadata_dict['right mirror distance to negative lens (mm)']:.3f} mm\n"
              f"fourth order polynomial coefficient (m^-3):     {metadata_dict['fourth order polynomial coefficient (m^-3)']}\n")
        large_mirror_position[i, j] = cavity.surfaces[-1].center[0]
        angle_of_incidence_right_lens[i, j] = metadata_dict['Incidence angle right (deg)']
        # fig, ax = plot_results(results_dict=results_dict, far_away_plane=True)
        # marginal_ray.plot(ax=ax[1])
        # fig.suptitle(f"marginal ray NA: {metadata_dict['marginal ray NA']:.3f}\n2y_marginal/CA lens left = {metadata_dict['2y_marginal/CA lens left']:.2f}, 2y_marginal/R lens left: {metadata_dict['2y_marginal/R lens left']:.2f}, 2y_marginal/R lens right: {metadata_dict['2y_marginal/R lens right']:.2f}\n2y_marginal/R mirror right: {metadata_dict['2y_marginal/R mirror right']:.2f}, 2y_marginal/CA right mirror: {metadata_dict['2y_marginal/CA right mirror']:.2f}\n"
        #              f"NA short arm = {metadata_dict['NA left arm']:.3f}, NA middle arm = {metadata_dict['NA middle arm']:.3f}, NA right arm = {metadata_dict['NA right arm']:.3f}\nincidence angle left = {metadata_dict['Incidence angle left (deg)']:.2f} deg, incidence angle right = {metadata_dict['Incidence angle right (deg)']:.2f} deg")
        # ax[1].set_title(f"negative lens left radius: {metadata_dict['negative lens left radius (mm)']:.2f}mm, negative lens right radius = {metadata_dict['negative lens right radius (mm)']:.2f}mm, negative lens focal length = {metadata_dict['negative lens focal length']:.2f} mm\nright mirror radius = {metadata_dict['right mirror radius (mm)']:.3f} mm, right mirror distance to negative lens = {metadata_dict['right mirror distance to negative lens (mm)']:.3f} mm")
        # ax[1].set_ylim(-large_elements_CA/1.8, large_elements_CA/1.8)#(-np.abs(final_intersection[1])*1.1, np.abs(final_intersection[1]))
        # ax[1].grid()
        # plt.show()
# %%
valid_results = large_mirror_position > 0
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
for j, large_elements_CA in enumerate(large_elements_CAs):
        ax[0].scatter(desired_foci[valid_results[:, j]]*1e2, large_mirror_position[valid_results[:, j], j]*1e2, label=f'CA = {large_elements_CA*1e3:.0f} mm')
        ax[1].scatter(desired_foci[valid_results[:, j]]*1e2, angle_of_incidence_right_lens[valid_results[:, j], j], label=f'CA = {large_elements_CA*1e3:.0f} mm')

ax[0].set_xlabel('Positive lens to middle focal point distance (cm)')
ax[0].set_ylabel('Right mirror position (cm)')
ax[1].set_xlabel('Positive lens to middle focal point distance (cm)')
ax[1].set_ylabel('Angle of incidence on lens concave side (deg)')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[1].set_ylim(0, 90)
ax[0].set_ylim(0, 150)
plt.suptitle(f"systems with vanishing fourth order, marginal ray NA = {marginal_ray_NA:.3f}, large mirror ROC = {right_mirror_ROC:.3f} m")
plt.tight_layout()
plt.show()


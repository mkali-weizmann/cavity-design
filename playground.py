from simple_analysis_scripts.potential_analysis.analyze_potential import *

copy_image = False
copy_input_parameters = True
copy_cavity_parameters = False
n_rays = 50
phi_max = 1.5200000000e-01
first_arm_NA = 1.5200000000e-01
lens_type = 'avantier'
mirror_setting_mode = 'Set ROC'
right_mirror_ROC = 1.0000000000e-01
right_mirror_distance_to_negative_lens_front = 3.3700000000e-02
right_mirror_ROC_fine = 4.6459717707e-11
n_actual_spherical = 1.4500000000e+00
negative_lens_defocus_power = -1.2710100000e+01
negative_lens_R_2_inverse = 6.6600000000e+01
desired_focus = 1.0070000000e-01
negative_lens_back_relative_position = 2.5000000000e-01

right_mirror_ROC_fine = widget_convenient_exponent(right_mirror_ROC_fine, scale=-10)
if mirror_setting_mode == "Set ROC":
    right_mirror_distance_to_negative_lens_front = None
    right_mirror_ROC += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance to spherical":
    right_mirror_ROC = None
    right_mirror_distance_to_negative_lens_front += right_mirror_ROC_fine
negative_lens_focal_length = 1 / negative_lens_defocus_power
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(
    lens_type=lens_type,
    dn=0)
cavity = generate_negative_lens_cavity(n_actual_first_lens=n_actual, n_design_first_lens=n_design, T_c_first_lens=T_c, back_focal_length_first_lens=back_focal_length, R_1_first_lens=R_1, R_2_first_lens=R_2, R_2_signed_first_lens=R_2_signed, diameter_first_lens=diameter, approximate_focus_distance_long_arm=desired_focus,
                                       negative_lens_focal_length=negative_lens_focal_length, negative_lens_R_2_inverse=negative_lens_R_2_inverse,
                                       negative_lens_back_relative_position=negative_lens_back_relative_position, negative_lens_refractive_index=1.45, negative_lens_center_thickness=3.45e-3, first_arm_NA=first_arm_NA, right_mirror_ROC=right_mirror_ROC, right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front, )
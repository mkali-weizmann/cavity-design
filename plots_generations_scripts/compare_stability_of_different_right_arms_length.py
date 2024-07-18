import time

import matplotlib.pyplot as plt

from cavity import *

NA_left = 1.5800000000e-01
mirror_on_waist = False
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
set_R_left_to_collimate = True
R_left = 2.5000000000e-02
R_left_fine = -1.3552527156e-20
set_R_right_to_collimate = False
set_R_right_to_equalize_angles = True
set_R_right_to_R_left = False
R_right = 5.0000000000e-03
R_right_fine = -1.3552527156e-20
auto_set_right_arm_length = False
right_arm_length = 3.0000000000e-01
lens_fixed_properties = 'sapphire'
mirrors_fixed_properties = 'ULE'
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
add_unheated_cavity = False
print_input_parameters = True
print_cavity_parameters = True

waist_to_lens += waist_to_lens_fine
R_left += R_left_fine
R_right += R_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span

for right_arm_length in np.logspace(-1, 0, 10):

    if not set_R_right_to_collimate:
        def cavity_generator(R_left_):
            cavity = mirror_lens_mirror_cavity_general_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
                                                                 R_left=R_left_, R_right=R_right, T_c=0, T_edge=T_edge,
                                                                 right_arm_length=right_arm_length,
                                                                 lens_fixed_properties=lens_fixed_properties,
                                                                 mirrors_fixed_properties=mirrors_fixed_properties,
                                                                 symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                                                 lambda_0_laser=1064e-9, power=2e4, set_h_instead_of_w=True,
                                                                 auto_set_right_arm_length=auto_set_right_arm_length,
                                                                 set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                                 set_R_right_to_R_left=set_R_right_to_R_left)
            return cavity
    else:
        def cavity_generator(R_right_):
            cavity = mirror_lens_mirror_cavity_general_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
                                                                 R_left=R_left, R_right=R_right_, T_c=0, T_edge=T_edge,
                                                                 right_arm_length=right_arm_length,
                                                                 lens_fixed_properties=lens_fixed_properties,
                                                                 mirrors_fixed_properties=mirrors_fixed_properties,
                                                                 symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                                                 lambda_0_laser=1064e-9, power=2e4, set_h_instead_of_w=True,
                                                                 auto_set_right_arm_length=auto_set_right_arm_length,
                                                                 set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                                 set_R_right_to_R_left=set_R_right_to_R_left)
            return cavity

    if set_R_left_to_collimate or set_R_right_to_collimate:
        if set_R_left_to_collimate:
            x0 = R_left
        else:
            x0 = R_right
        cavity = find_required_value_for_desired_change(cavity_generator=cavity_generator,
                                                        # Takes a float as input and returns a cavity
                                                        desired_parameter=lambda cavity: 1 / cavity.arms[
                                                            2].mode_parameters_on_surfaces[0].z_minus_z_0[0],
                                                        desired_value=-2 / right_arm_length,
                                                        x0=x0)
    else:
        cavity = cavity_generator(R_left)

    plot_mirror_lens_mirror_cavity_analysis(cavity,
                                            auto_set_x=auto_set_x,
                                            x_span=x_span,
                                            auto_set_y=auto_set_y,
                                            y_span=y_span,
                                            T_edge=T_edge,
                                            camera_center=camera_center,
                                            add_unheated_cavity=add_unheated_cavity)
    plt.show()

    # %%
    tolerance_matrix = cavity.generate_tolerance_matrix()

    # # %%
    overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                     shift_size=30,
                                                     )
    # # %%
    cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                    arm_index_for_NA=2)
    plt.suptitle(f'right_arm_length = {right_arm_length:.2e}')
    plt.show()


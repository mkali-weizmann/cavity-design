from cavity import *

lambda_laser = 1.064e-6
power = 2e4
symmetric_left_arm = False
waist_to_left_mirror = -5.0000000000e-03
waist_to_left_mirror_fine = 0.0000000000e+00
NA_left = 1.2200000000e-01
right_arm_length = 3.0000000000e-01
right_mirror_radius_shift = 0.0000000000e+00
right_mirror_position_shift = 0.0000000000e+00
auto_set_right_arm_length = True
mirror_on_waist = False
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.5449990000e-03
set_h_instead_of_w = True
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
set_R_right_to_equalize_angles = False
set_R_right_to_R_left = True
R_left = -1.0000000000e+00
R_left_small_fine = -9.9000000000e-06
R_right = -2.184796220e+00
R_right__fine = -9.9000000000e-06
T_c = 0.0000000000e+00
T_c_fine = 0.0000000000e+00
lens_fixed_properties = 'fused_silica'
mirrors_fixed_properties = 'ULE'
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
add_unheated_cavity = False
print_input_parameters = True
print_parameters = False

R_right = 10 ** R_right + R_right__fine
R_left = 10 ** R_left + R_left_small_fine
T_c += T_c_fine

h_divided_by_spot_size = 2.8

x_span = 10 ** x_span
y_span = 10 ** y_span

def cavity_generator(R_left):
    cavity = mirror_lens_mirror_cavity_general_generator(NA_left=NA_left,
                                                         waist_to_lens=waist_to_lens,
                                                         h=h,
                                                         R_left=R_left,
                                                         R_right=R_right,
                                                         T_c=T_c,
                                                         T_edge=T_edge,
                                                         right_arm_length=right_arm_length,
                                                         lens_fixed_properties=lens_fixed_properties,
                                                         mirrors_fixed_properties=mirrors_fixed_properties,
                                                         symmetric_left_arm=symmetric_left_arm,
                                                         waist_to_left_mirror=waist_to_left_mirror,
                                                         lambda_laser=lambda_laser,
                                                         power=power,
                                                         set_h_instead_of_w=set_h_instead_of_w,
                                                         auto_set_right_arm_length=auto_set_right_arm_length,
                                                         set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                         set_R_right_to_R_left=set_R_right_to_R_left)
    return cavity

cavity = find_required_value_for_desired_change(cavity_generator=cavity_generator,
                                                # Takes a float as input and returns a cavity
                                                desired_parameter=lambda cavity: 1 / cavity.arms[
                                                    2].mode_parameters_on_surface_0.z_minus_z_0[0],
                                                desired_value=-2 / right_arm_length,
                                                x0=0.005, print_progress=True)

plot_mirror_lens_mirror_cavity_analysis(cavity,
                                        auto_set_x=auto_set_x,
                                        x_span=x_span,
                                        auto_set_y=auto_set_y,
                                        y_span=y_span,
                                        camera_center=camera_center,
                                        add_unheated_cavity=add_unheated_cavity)
plt.show()
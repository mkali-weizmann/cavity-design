from cavity import *
from matplotlib import ticker

lambda_0_laser = 1064e-9
NA_left = 1.7200000000e-01
mirror_on_waist = False
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
set_R_left_to_collimate = True
R_left = 8.7532900000e-03
R_left_fine = -1.3552527156e-20
set_R_right_to_collimate = False
set_R_right_to_equalize_angles = False
set_R_right_to_R_left = True
R_right = 8.7532900000e-03
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
print_cavity_parameters = False


# index 1 represents the right arm, index 2 the inner part of the lens, and index 3 the left arm.
assert not (
            set_R_left_to_collimate and set_R_right_to_collimate), "Too many solutions: can't set automatically both R_left to collimate and R_right to collimate"
assert not (
                       set_R_right_to_collimate + set_R_right_to_equalize_angles + set_R_right_to_R_left) > 1, "Too many constraints on R_right"


waist_to_lens += waist_to_lens_fine
R_left += R_left_fine
R_right += R_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span

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
                                        camera_center=camera_center,
                                        add_unheated_cavity=add_unheated_cavity)
plt.ylim(-0.005, 0.005)
plt.show()
if print_cavity_parameters:
    pretty_print_array(cavity.to_array)


# params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000632553e-03+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
#           [ 6.3393015143e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  2.3479617901e-02+0.j,  6.2124542986e-03+0.j,  1.0000000000e+00+0.j,  2.6786030286e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
#           [ 3.0767860303e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5038941544e-01+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])
#
# cavity = Cavity.from_params(params=params,
#                             standing_wave=True,
#                             lambda_0_laser=lambda_0_laser,
#                             names=['left mirror', 'lens-left', 'lens_right',  'right mirror'],
#                             set_central_line=True,
#                             set_mode_parameters=True,
#                             set_initial_surface=False,
#                             t_is_trivial=True,
#                             p_is_trivial=True,
#                             power=2e4)
# plot_mirror_lens_mirror_cavity_analysis(cavity)
#
# NA = 0.138
# beginning_ray = Ray(origin=np.array([0, 0, 0]), k_vector=np.array([np.cos(np.arcsin(NA)), NA, 0]))
# second_ray = cavity.physical_surfaces[1].reflect_ray(beginning_ray)
# third_ray = cavity.physical_surfaces[2].reflect_ray(second_ray)
#
# ax = plt.gca()
# beginning_ray.plot(ax=ax, color='r')
# second_ray.plot(ax=ax, color='r')
# third_ray.plot(ax=ax, color='r')
#
#
# plt.xlim(0.004, 0.008)
# plt.ylim(0, 0.002)
#
#
# ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
#
# ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
# plt.title("Mines")
# plt.tight_layout()
# plt.show()

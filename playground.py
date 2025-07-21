from cavity import *

NA_left = 1.5600000000e-01
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e+00
set_R_left_to_collimate = False
R_small_mirror = 5.0000000000e-03
R_left = 2.4220000000e-02
R_left_fine = -1.3552527156e-20
set_R_right_to_collimate = False
set_R_right_to_equalize_angles = False
set_R_right_to_R_left = False
R_right = 5.4880000000e-03
R_right_fine = -1.3552527156e-20
collimation_mode = 'symmetric arm'
auto_set_big_mirror_radius = False
big_mirror_radius = 2.0000000000e-01
auto_set_right_arm_length = False
right_arm_length = 3.3460890300e-01
lens_fixed_properties = 'sapphire'
mirrors_fixed_properties = 'ULE'
auto_set_x = True
x_span = -1.5700000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
camera_center = 2
add_unheated_cavity = False
waist_to_left_mirror = None

big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
right_arm_length = None if auto_set_right_arm_length else right_arm_length
waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
R_left += widget_convenient_exponent(R_left_fine)
R_right += widget_convenient_exponent(R_right_fine)
x_span = 10 ** x_span
y_span = 10 ** y_span

cavity = mirror_lens_mirror_cavity_generator(
    NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
    R_left=R_left, R_right=R_right, T_c=0,
    T_edge=T_edge, lens_fixed_properties=lens_fixed_properties,
    mirrors_fixed_properties=mirrors_fixed_properties,
    R_small_mirror=R_small_mirror,
    waist_to_left_mirror=waist_to_left_mirror,
    lambda_0_laser=1064e-9, power=2e4,
    set_h_instead_of_w=True,
    collimation_mode=collimation_mode,
    big_mirror_radius=big_mirror_radius,
    right_arm_length=right_arm_length,
    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
    set_R_right_to_R_left=set_R_right_to_R_left,
    set_R_left_to_collimate=set_R_left_to_collimate,
    set_R_right_to_collimate=set_R_right_to_collimate
)

plot_mirror_lens_mirror_cavity_analysis(
    cavity,
    auto_set_x=auto_set_x,
    x_span=x_span,
    auto_set_y=auto_set_y,
    y_span=y_span,
    T_edge=T_edge,
    camera_center=camera_center,
    add_unheated_cavity=add_unheated_cavity
)
plt.show()
# %%
n_arms = 1
ABCD_matrices = cavity.ABCD_matrices[n_arms-1::-1]
if n_arms == 1:
    ABCD_first_n = ABCD_matrices[0]  # Only one arm, no need to reverse
else:
    ABCD_first_n = np.linalg.multi_dot(ABCD_matrices)

refractive_index_n = cavity.arms[n_arms].n
first_mode = cavity.arms[0].mode_parameters_on_surface_0
mode_n = propagate_local_mode_parameter_through_ABCD(first_mode, ABCD_first_n, n_1=1, n_2=refractive_index_n)
actual_mode_n = cavity.arms[n_arms].mode_parameters_on_surface_0

print(mode_n.q[0])
print(actual_mode_n.q[0])
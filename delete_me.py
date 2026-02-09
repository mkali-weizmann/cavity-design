from cavity import *

waist_to_left_mirror = None
NA_left = 1.5000000000e-01
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = -5.8407300310e+00
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
auto_set_right_arm_length = True
right_arm_length = 4.0000000000e-01
lens_fixed_properties = 'sapphire'
mirrors_fixed_properties = 'ULE'
T_edge = 1.0000000000e-03
h = 3.8750000000e-03


big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
right_arm_length = None if auto_set_right_arm_length else right_arm_length
waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
R_left += widget_convenient_exponent(R_left_fine)
R_right += widget_convenient_exponent(R_right_fine)


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


assert np.all(np.isclose(cavity.mode_parameters[0].center, np.array([[0, 0.00000000e+00, 0.00000000e+00], [0, 0.00000000e+00, 0.00000000e+00]]))), f'cavity_smart_generation_test failed: center should be approximately [[8.67361738e-19, 0.00000000e+00, 0.00000000e+00], [8.67361738e-19, 0.00000000e+00, 0.00000000e+00]], instead got {cavity.mode_parameters[0].center}'
assert np.all(np.isclose(cavity.mode_parameters[0].z_R, np.array([1.50525208e-05, 1.50525208e-05]))), f'cavity_smart_generation_test failed: z_R should be approximately 1.50525208e-05, instead got {cavity.mode_parameters[0].z_R}'
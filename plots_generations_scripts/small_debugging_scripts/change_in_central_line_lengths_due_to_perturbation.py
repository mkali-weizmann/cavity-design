from cavity import *

NA_left = 15.2000000000e-02
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
auto_set_x = True
x_span = -1.5700000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
camera_center = 2
add_unheated_cavity = False
copy_input_parameters = True
copy_cavity_parameters = False
eval_box = ''
waist_to_left_mirror = None

big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
right_arm_length = None if auto_set_right_arm_length else right_arm_length
waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
R_left += widget_convenient_exponent(R_left_fine)
R_right += widget_convenient_exponent(R_right_fine)
x_span = 10 ** x_span
y_span = 10 ** y_span

cavity_0 = mirror_lens_mirror_cavity_generator(
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
    set_R_right_to_collimate=set_R_right_to_collimate,
    use_paraxial_ray_tracing=True
)

params = cavity_0.to_params
cavity_0.plot()
# %%
PERTURBATION_ELEMENT_INDEX = 1  # 0 for the left mirror, 1 for lens
PERTURBATION_VALUE = 2.8e-8
PERTURBATION_PARAMETER = ParamsNames.y
CORRECTION_ELEMENT_INDEX = 2  # this is always 2 because we correct with the large mirror
CORRECTION_PARAMETER = ParamsNames.phi

perturbation_pointer = PerturbationPointer(element_index=PERTURBATION_ELEMENT_INDEX,
                                           parameter_name=PERTURBATION_PARAMETER,
                                           perturbation_value=PERTURBATION_VALUE)

perturbed_cavity = perturb_cavity(cavity=cavity_0, perturbation_pointer=perturbation_pointer)

fig, ax = plt.subplots()
cavity_0.plot(laser_color='r', ax=ax)
perturbed_cavity.plot(laser_color='g', ax=ax)
plt.grid()
plt.show()

# print_differences:
for name, cavity in zip(['cavity_0', 'perturbed_cavity'], [cavity_0, perturbed_cavity]):
    print(name)
    print('central line lengths:\n', [l.length for l in cavity.central_line])
    print('round_trip_ABCD:\n', cavity.ABCD_round_trip)
    round_trip_eigen_mode = local_mode_parameters_of_round_trip_ABCD(
        round_trip_ABCD=cavity.ABCD_round_trip, lambda_0_laser=cavity.lambda_0_laser, n=cavity.arms[0].n
    )
    print('round_trip_eigen_mode:\n', round_trip_eigen_mode.q)



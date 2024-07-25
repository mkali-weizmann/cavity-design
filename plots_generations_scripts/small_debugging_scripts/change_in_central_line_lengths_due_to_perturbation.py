from cavity import *

lambda_0_laser = 1064e-9
NA_left = 1.500000000e-01
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
use_paraxial_ray_tracing = True

waist_to_lens += waist_to_lens_fine
R_left += R_left_fine
R_right += R_right_fine
x_span = 10 ** x_span
y_span = 10 ** y_span

lengths = np.logspace(-1, 0, 5)
min_tolerances = np.zeros_like(lengths)


cavity_0 = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
                                             R_left=R_left, R_right=R_right, T_c=0, T_edge=T_edge,
                                             right_arm_length=right_arm_length,
                                             lens_fixed_properties=lens_fixed_properties,
                                             mirrors_fixed_properties=mirrors_fixed_properties,
                                             symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                             lambda_0_laser=1064e-9, set_h_instead_of_w=True,
                                             auto_set_right_arm_length=auto_set_right_arm_length,
                                             set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                             set_R_right_to_R_left=set_R_right_to_R_left,
                                             debug_printing_level=1, power=2e4,
                                             use_paraxial_ray_tracing=use_paraxial_ray_tracing,
                                             set_R_left_to_collimate=set_R_left_to_collimate,
                                             set_R_right_to_collimate=set_R_right_to_collimate)
params = cavity_0.to_params
cavity_0.plot()
# %%
PERTURBATION_ELEMENT_INDEX = 0  # 0 for the left mirror, 1 for lens
PERTURBATION_VALUE = 4e-6
PERTURBATION_PARAMETER = ParamsNames.phi
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



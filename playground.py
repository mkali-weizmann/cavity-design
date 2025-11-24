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
    set_R_right_to_collimate=set_R_right_to_collimate,
    use_paraxial_ray_tracing=True
)

# plot_mirror_lens_mirror_cavity_analysis(
#     cavity,
#     auto_set_x=auto_set_x,
#     x_span=x_span,
#     auto_set_y=auto_set_y,
#     y_span=y_span,
#     T_edge=T_edge,
#     camera_center=camera_center,
#     add_unheated_cavity=add_unheated_cavity,
#     diameters=[7.75e-3, 7.75e-3, 7.75e-3, 0.0254]
# )
#
# plt.show()

# %%

perturbation_pointer = PerturbationPointer(element_index=1, parameter_name='y', perturbation_value=32.8e-9),
cavity_perturbed = perturb_cavity(cavity=cavity,
                                  perturbation_pointer=perturbation_pointer,
                                  set_central_line=True,
                                  set_mode_parameters=True)
plot_mirror_lens_mirror_cavity_analysis(
    cavity_perturbed,
    auto_set_x=auto_set_x,
    x_span=x_span,
    auto_set_y=auto_set_y,
    y_span=y_span,
    T_edge=T_edge,
    camera_center=camera_center,
    add_unheated_cavity=add_unheated_cavity,
    diameters=[7.75e-3, 7.75e-3, 7.75e-3, 0.0254]
)
x_1, y_1, x_2, y_2 = cavity_perturbed.physical_surfaces[0].origin[0], cavity_perturbed.physical_surfaces[0].origin[1], cavity_perturbed.physical_surfaces[3].origin[0], cavity_perturbed.physical_surfaces[3].origin[1]
plt.scatter([x_1, x_2], [y_1, y_2])
# plt.xlim(x_2 - 1e-5, x_2 + 1e-5)
# plt.ylim(y_2 - 1e-5, y_2 + 1e-5)
plt.show()
# %%
D_ABCD_original_propagation = [cavity.arms[arm_index].ABCD_matrix_free_space[2:,2:] for arm_index in range(len(cavity.arms))]
D_ABCD_perturbed_propagation = [cavity_perturbed.arms[i].ABCD_matrix_free_space[2:,2:] for i in range(len(cavity_perturbed.arms))]
D_ABCDs_delta_normalized_propagation = [(ABCD_perturbed - ABCD) / np.where(ABCD == 0, 1, ABCD) for ABCD, ABCD_perturbed in zip(D_ABCD_original_propagation, D_ABCD_perturbed_propagation)]
D_ABCD_original_reflection = [cavity.arms[i].ABCD_matrix_reflection[2:,2:] for i in range(len(cavity.arms))]
D_ABCD_perturbed_reflection = [cavity_perturbed.arms[i].ABCD_matrix_reflection[2:,2:] for i in range(len(cavity_perturbed.arms))]
D_ABCDs_delta_normalized_reflection = [(ABCD_perturbed - ABCD) / np.where(ABCD == 0, 1, ABCD) for ABCD, ABCD_perturbed in zip(D_ABCD_original_reflection, D_ABCD_perturbed_reflection)]
D_ABCD_roundtrip_original = cavity.ABCD_round_trip[2:,2:]
D_ABCD_roundtrip_perturbed = cavity_perturbed.ABCD_round_trip[2:,2:]
D_starting_local_mode_parameters_original = local_mode_parameters_of_round_trip_ABCD(D_ABCD_roundtrip_original, n=1)
D_starting_local_mode_parameters_perturbed = local_mode_parameters_of_round_trip_ABCD(D_ABCD_roundtrip_perturbed, n=1)
D_lengths_original = np.array([cavity.arms[i].central_line.length for i in range(len(cavity.arms))])
D_lengths_perturbed = np.array([cavity_perturbed.arms[i].central_line.length for i in range(len(cavity_perturbed.arms))])
D_lengths_difference = D_lengths_perturbed - D_lengths_original
D_cos_theta_perturbed = np.array([np.abs(cavity_perturbed.arms[i].central_line.k_vector @ cavity_perturbed.arms[i].surface_1.outwards_normal) for i in range(len(cavity.arms))])
D_inclination_angles = np.arccos(D_cos_theta_perturbed)
D_lengths_difference_supposed = D_lengths_original * (1 / np.cos(D_inclination_angles) - 1)

# %%
lengths_original = [cavity.arms[i].central_line.length for i in range(len(cavity.arms))]
lengths_perturbed = [cavity_perturbed.arms[i].central_line.length for i in range(len(cavity_perturbed.arms))]

# %%
np.arccos(cavity.arms[0].central_line.k_vector @ cavity_perturbed.arms[0].central_line.k_vector)
# %%

ABCD_stacked = np.hstack(cavity.ABCD_matrices)
ABCD_stacked_perturbed = np.hstack(cavity_perturbed.ABCD_matrices)
difference_ABCD = (ABCD_stacked_perturbed - ABCD_stacked) / np.where(ABCD_stacked == 0, 1, ABCD_stacked)
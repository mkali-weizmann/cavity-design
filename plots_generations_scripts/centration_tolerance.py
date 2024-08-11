from cavity import *

from matplotlib.lines import Line2D

NA_left = 1.5600000000e-01
mirror_on_waist = False
x_2_perturbation = 0.0000000000e00
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e00
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
lens_fixed_properties = "sapphire"
mirrors_fixed_properties = "ULE"
auto_set_x = True
x_span = -1.0000000000e00
auto_set_y = True
y_span = -2.9000000000e00
camera_center = 2
add_unheated_cavity = False
print_input_parameters = True
print_cavity_parameters = True
use_paraxial_ray_tracing = True
right_arm_length = 0.3

waist_to_lens += waist_to_lens_fine
R_left += R_left_fine
R_right += R_right_fine
x_span = 10**x_span
y_span = 10**y_span


cavity = mirror_lens_mirror_cavity_generator(
    NA_left=NA_left,
    waist_to_lens=waist_to_lens,
    h=h,
    R_left=R_left,
    R_right=R_right,
    T_c=0,
    T_edge=T_edge,
    right_arm_length=right_arm_length,
    lens_fixed_properties=lens_fixed_properties,
    mirrors_fixed_properties=mirrors_fixed_properties,
    symmetric_left_arm=True,
    waist_to_left_mirror=5e-3,
    lambda_0_laser=LAMBDA_0_LASER,
    set_h_instead_of_w=True,
    auto_set_right_arm_length=auto_set_right_arm_length,
    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
    set_R_right_to_R_left=set_R_right_to_R_left,
    debug_printing_level=1,
    power=2e4,
    use_paraxial_ray_tracing=use_paraxial_ray_tracing,
    set_R_left_to_collimate=set_R_left_to_collimate,
    set_R_right_to_collimate=set_R_right_to_collimate,
)

plot_mirror_lens_mirror_cavity_analysis(
    cavity,
    auto_set_x=auto_set_x,
    x_span=x_span,
    auto_set_y=auto_set_y,
    y_span=y_span,
    T_edge=T_edge,
    camera_center=camera_center,
    add_unheated_cavity=add_unheated_cavity,
)
plt.show()

cavity_params_decomposed = [surface.to_params for surface in cavity.physical_surfaces]

cavity_decomposed = Cavity.from_params(
    params=cavity_params_decomposed,
    standing_wave=True,
    lambda_0_laser=LAMBDA_0_LASER,
    names=["Left Mirror", "Lens - Left", "Lens - Right", "Mirror - Right"],
    set_central_line=True,
    set_mode_parameters=True,
    set_initial_surface=False,
    t_is_trivial=True,
    p_is_trivial=True,
    power=2e4,
    use_brute_force_for_central_line=True,
    use_paraxial_ray_tracing=True,
)
# %%
perturbable_params_names = [ParamsNames.phi, ParamsNames.x]
tolerance_matrix = cavity_decomposed.generate_tolerance_matrix(perturbable_params_names=perturbable_params_names)


overlaps_series = cavity_decomposed.generate_overlap_series(
    shifts=2 * np.abs(tolerance_matrix[:, :]), shift_size=100, perturbable_params_names=perturbable_params_names
)
cavity_decomposed.generate_overlaps_graphs(
    overlaps_series=overlaps_series,
    tolerance_matrix=tolerance_matrix[:, :],
    arm_index_for_NA=2,
    perturbable_params_names=perturbable_params_names,
)
plt.show()
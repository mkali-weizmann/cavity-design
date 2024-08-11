import matplotlib.pyplot as plt

from cavity import *

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

for i, right_arm_length in enumerate(lengths):
    cavity = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
                                                 R_left=R_left, R_right=R_right, T_c=0, T_edge=T_edge,
                                                 right_arm_length=right_arm_length,
                                                 lens_fixed_properties=lens_fixed_properties,
                                                 mirrors_fixed_properties=mirrors_fixed_properties,
                                                 symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                                 lambda_0_laser=LAMBDA_0_LASER, set_h_instead_of_w=True,
                                                 auto_set_right_arm_length=auto_set_right_arm_length,
                                                 set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                 set_R_right_to_R_left=set_R_right_to_R_left,
                                                 debug_printing_level=1, power=2e4,
                                                 use_paraxial_ray_tracing=use_paraxial_ray_tracing,
                                                 set_R_left_to_collimate=set_R_left_to_collimate,
                                                 set_R_right_to_collimate=set_R_right_to_collimate)

    plot_mirror_lens_mirror_cavity_analysis(cavity,
                                            auto_set_x=auto_set_x,
                                            x_span=x_span,
                                            auto_set_y=auto_set_y,
                                            y_span=y_span,
                                            T_edge=T_edge,
                                            camera_center=camera_center,
                                            add_unheated_cavity=add_unheated_cavity)
    plt.xlim(-0.06, 1.01)
    plt.show()

    # %%
    tolerance_matrix = cavity.generate_tolerance_matrix()

    # # %%
    overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                     shift_size=100,
                                                     )
    # # %%
    cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                    arm_index_for_NA=2)
    plt.suptitle(generate_mirror_lens_mirror_cavity_textual_summary(cavity, h=h, T_edge=T_edge))
    plt.tight_layout()
    plt.show()

    min_tolerances[i] = np.nanmin(np.abs(tolerance_matrix))
# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
scales = ['linear', 'log']
for i, ax in enumerate(ax):
    ax.plot(lengths, min_tolerances)
    ax.set_yscale(scales[i])
    ax.grid()
    ax.set_xlabel('Long arm length')
    ax.set_ylabel('Minimum tolerance (lens transversal displacement)')
plt.suptitle('Stability of the cavity with respect to the long arm length')
plt.tight_layout()
plt.savefig('figures/right_arm_length_stability.png')
plt.show()

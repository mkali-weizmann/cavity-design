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

PERTURBATION_ELEMENT_INDEX = 0  # 2 for the left mirror, 1 for lens
PERTURBATION_VALUE = 1e-8
PERTURBATION_PARAMETER = ParamsNames.phi
CORRECTION_ELEMENT_INDEX = 2  # this is always 2 because we correct with the large mirror
CORRECTION_PARAMETER = ParamsNames.phi

perturbation_pointer = PerturbationPointer(element_index=PERTURBATION_ELEMENT_INDEX,
                                           parameter_name=PERTURBATION_PARAMETER,
                                           perturbation_value=PERTURBATION_VALUE)

perturbed_cavity = perturb_cavity(cavity=cavity_0, perturbation_pointer=perturbation_pointer)

cavity_0.plot()
plt.show()


def overlap_extractor(cavity):
    overlap = np.abs(calculate_cavities_overlap(cavity_0, cavity))
    return overlap

correction_pointer = PerturbationPointer(element_index=CORRECTION_ELEMENT_INDEX,
                                         parameter_name=CORRECTION_PARAMETER)

corrected_cavity = find_required_perturbation_for_desired_change(cavity=perturbed_cavity,
                                                                 perturbation_pointer=correction_pointer,
                                                                 desired_parameter=overlap_extractor,
                                                                 desired_value=1,
                                                                 x0=0,
                                                                 xtol=1e-10)

print(f"perturbation_value: {getattr(corrected_cavity.to_params[2], PERTURBATION_PARAMETER):.2e}, "
      f"correction_value: {getattr(corrected_cavity.to_params[0], CORRECTION_PARAMETER)}")

print(f"Perturbed overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity)):.2e}, "
      f"Corrected_overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity)):.2e}")


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
plt.suptitle('Overlap between cavities')
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=perturbed_cavity, ax=ax[0], axis_span=1e-5)
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=corrected_cavity, ax=ax[1], axis_span=1e-5)
ax[0].set_title(f'Perturbed cavity, 1-overlap = {1-np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity)):.2e}')
ax[1].set_title(f'Corrected cavity, 1-overlap = {1-np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity)):.2e}')
plt.tight_layout()
plt.show()

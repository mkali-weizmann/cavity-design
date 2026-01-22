from cavity import *
from matplotlib.lines import Line2D

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


cavity_0 = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h, R_left=R_left,
                                               R_right=R_right, T_c=0, T_edge=T_edge, right_arm_length=right_arm_length,
                                               lens_fixed_properties=lens_fixed_properties,
                                               mirrors_fixed_properties=mirrors_fixed_properties,
                                               symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                               lambda_0_laser=LAMBDA_0_LASER, set_h_instead_of_w=True,
                                               set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                               set_R_right_to_R_left=set_R_right_to_R_left,
                                               set_R_right_to_collimate=set_R_right_to_collimate,
                                               set_R_left_to_collimate=set_R_left_to_collimate,
                                               auto_set_right_arm_length=auto_set_right_arm_length,
                                               debug_printing_level=1, power=2e4,
                                               use_paraxial_ray_tracing=use_paraxial_ray_tracing)
params = cavity_0.to_params
# %%
PERTURBATION_ELEMENT_INDEX = 1  # 0 for the left mirror, 1 for lens
PERTURBATION_VALUE = 4.5e-6
PERTURBATION_PARAMETER = ParamsNames.x
CORRECTION_ELEMENT_INDEX = 2  # this is always 2 because we correct with the large mirror
CORRECTION_PARAMETER = ParamsNames.x

perturbation_pointer = PerturbationPointer(element_index=PERTURBATION_ELEMENT_INDEX,
                                           parameter_name=PERTURBATION_PARAMETER,
                                           perturbation_value=PERTURBATION_VALUE)

perturbed_cavity = perturb_cavity(cavity=cavity_0, perturbation_pointer=perturbation_pointer)

fig, ax = plt.subplots()
cavity_0.plot(laser_color='r', ax=ax)
perturbed_cavity.plot(laser_color='g', ax=ax)
plt.grid()
plt.show()

# %%

def overlap_extractor(cavity):
    overlap = np.abs(calculate_cavities_overlap(cavity_0, cavity))
    return overlap

correction_pointer = PerturbationPointer(element_index=CORRECTION_ELEMENT_INDEX,
                                         parameter_name=CORRECTION_PARAMETER)
if PERTURBATION_PARAMETER in [ParamsNames.y, ParamsNames.phi]:
    delta_y = cavity_0.params[1].y - cavity_0.physical_surfaces[0].origin[1]
    delta_x = cavity_0.params[1].x - cavity_0.physical_surfaces[0].origin[0]
    delta_x_long_arm = cavity_0.params[2].x - cavity_0.physical_surfaces[2].origin[0]

corrected_cavity, correction_value = find_required_perturbation_for_desired_change(cavity=perturbed_cavity,
                                                                 perturbation_pointer=correction_pointer,
                                                                 desired_parameter=overlap_extractor,
                                                                 desired_value=1,
                                                                 x0=-3e-3,
                                                                 xtol=1e-10,
                                                                 print_progress=True)
# %%
print(f"perturbation_value: {getattr(corrected_cavity.to_params[2], PERTURBATION_PARAMETER):.2e}, "
      f"correction_value: {getattr(corrected_cavity.to_params[0], CORRECTION_PARAMETER)}")

print(f"Perturbed 1-overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity)):.2e}, "
      f"Corrected 1-overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity)):.2e}")
sup_title = (f"perturbation value in {cavity_0.names[PERTURBATION_ELEMENT_INDEX]}: $\Delta {PERTURBATION_PARAMETER}=${perturbation_pointer.perturbation_value:.2e},"
             f" correction value in {cavity_0.names[CORRECTION_ELEMENT_INDEX]}: $\Delta {CORRECTION_PARAMETER}=${correction_value:.2e}")

file_path = (f'figures/overlap-perturbation-correction/pert_elem={cavity_0.names[PERTURBATION_ELEMENT_INDEX]} pert_parm={PERTURBATION_PARAMETER}'
             f' corr_param={CORRECTION_PARAMETER} plane=')

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
plt.suptitle(sup_title)
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=perturbed_cavity, ax=ax[0], axis_span=1e-5)
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=corrected_cavity, ax=ax[1], axis_span=1e-5)
one_minus_overlap_perturbed = 1 - np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity))
one_minus_overlap_corrected = 1 - np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity))
if one_minus_overlap_perturbed < 1e-3:
    perturbed_title = f'Perturbed cavity, 1-overlap = {one_minus_overlap_perturbed:.2e}'
else:
    perturbed_title = f'Perturbed cavity, overlap = {1-one_minus_overlap_perturbed:.3f}'

if one_minus_overlap_corrected < 1e-3:
    corrected_title = f'Corrected cavity, 1-overlap = {one_minus_overlap_corrected:.2e}'
else:
    corrected_title = f'Corrected cavity, overlap = {1-one_minus_overlap_corrected:.3f}'

ax[0].set_title(perturbed_title)
ax[1].set_title(corrected_title)
plt.tight_layout()
plt.savefig(f'{file_path}transverse.svg', dpi=300, bbox_inches='tight')
plt.show()

# create a legend:
custom_lines = [Line2D([0], [0], color='r', lw=1, linestyle='--', label='Original cavity'),
                Line2D([0], [0], color='g', lw=1, linestyle='--', label='Perturbed cavity'),
                Line2D([0], [0], color='b', lw=1, linestyle='--', label='Corrected cavity')]

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
for i, plane, plane_description in zip((0, 1), ('xy', 'xz'), ('perturbation plane', 'transverse plane')):
    cavity_0.plot(laser_color='r', ax=ax[i], plane=plane)
    perturbed_cavity.plot(laser_color='g', ax=ax[i], plane=plane)
    corrected_cavity.plot(laser_color='b', ax=ax[i], plane=plane)

    ax[i].set_title(plane_description)
    ax[i].set_ylim(-4e-3, 4e-3)
    print(ax[i].get_ylim())
    ax[i].annotate('blue line on top of red line appear purple', xy=(ax[i].get_xlim()[0] + (ax[i].get_xlim()[1] - ax[i].get_xlim()[0]) / 3, ax[i].get_ylim()[0] + (ax[i].get_ylim()[1] - ax[i].get_ylim()[0]) / 4))

    ax[i].legend(handles=custom_lines, loc='lower center')

plt.suptitle(sup_title)
plt.tight_layout()
plt.savefig(f'{file_path}longitudinal.svg', dpi=300, bbox_inches='tight')
plt.show()
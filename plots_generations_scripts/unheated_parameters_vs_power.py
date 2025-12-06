from matplotlib import use
use('TkAgg')
from cavity import *

radii = (5e-3, 5e-3)  # 5 mm
NA_left = 1.5300000000e-01
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
waist_to_left_mirror = None

big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
right_arm_length = None if auto_set_right_arm_length else right_arm_length
waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
R_left += widget_convenient_exponent(R_left_fine)
R_right += widget_convenient_exponent(R_right_fine)

# %%
NA=0.1
powers = np.concatenate(([0], 4408 - np.logspace(1, 3.5, 20)[::-1], np.linspace(6000, 10000, 5)))
NA_initials = np.zeros((len(powers), 3))
tolerances = np.zeros((len(powers), 3))
tolerances_initial = np.zeros((len(powers), 3))
cavity_fabry_perot = fabry_perot_generator(radii=radii,
                                               NA=NA,
                                               power=0,
                                               debug_printing_level=1)
cavity_fabry_perot_params = cavity_fabry_perot.to_params
cavity_fabry_perot_params_copy = copy.deepcopy(cavity_fabry_perot_params)
cavity_fabry_perot_params_copy[0].material_properties.alpha_expansion *= -1  # Negative TEC
cavity_fabry_perot_negative_TEC = Cavity.from_params(cavity_fabry_perot_params_copy, lambda_0_laser=1064e-9,
                                                      t_is_trivial=True, p_is_trivial=True, power=0)
cavity_mirror_lens_mirror = mirror_lens_mirror_cavity_generator(NA_left=NA, waist_to_lens=waist_to_lens, h=h,
                                                                R_left=R_left, R_right=R_right, T_c=0,
                                                                T_edge=T_edge,
                                                                lens_fixed_properties=lens_fixed_properties,
                                                                mirrors_fixed_properties=mirrors_fixed_properties,
                                                                R_small_mirror=R_small_mirror,
                                                                waist_to_left_mirror=waist_to_left_mirror,
                                                                lambda_0_laser=1064e-9, set_h_instead_of_w=True,
                                                                collimation_mode=collimation_mode,
                                                                big_mirror_radius=big_mirror_radius,
                                                                right_arm_length=right_arm_length,
                                                                set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                                set_R_right_to_R_left=set_R_right_to_R_left,
                                                                set_R_right_to_collimate=set_R_right_to_collimate,
                                                                set_R_left_to_collimate=set_R_left_to_collimate,
                                                                power=0,
                                                                debug_printing_level=2)

for j, cavity in tqdm(enumerate([cavity_fabry_perot, cavity_fabry_perot_negative_TEC, cavity_mirror_lens_mirror])):
    # mirror_shift_tolerance = cavity.calculate_parameter_tolerance(
    #     perturbation_pointer=PerturbationPointer(element_index=0, parameter_name=ParamsNames.y))
    for i, power in tqdm(enumerate(powers), total=len(powers), leave=False):
        cavity.power = power
        unheated_version = cavity.thermal_transformation()
        NA_unheated = unheated_version.mode_parameters[0].NA[0]
        tolerance_initial = unheated_version.calculate_parameter_tolerance(perturbation_pointer=PerturbationPointer(element_index=0, parameter_name=ParamsNames.y))
        NA_initials[i, j] = NA_unheated
        # tolerances[i, j] = mirror_shift_tolerance
        tolerances_initial[i, j] = tolerance_initial


# %%
plt.close('all')
fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.plot(powers, NA_initials[:, 0], label='Fabry-Perot Cavity')
plt.plot(powers, NA_initials[:, 1], label='Fabry-Perot Cavity with Negative TEC')
plt.plot(powers, NA_initials[:, 2], label='Mirror-Lens-Mirror Cavity')
plt.title(f"Initial NA vs Power for Fabry-Perot Cavity (High power NA={NA})")
plt.xlabel("Power (W)")
plt.ylabel("Initial NA")
plt.grid(True)
plt.legend()
plt.savefig(f'outputs\\figures\initial_NA_vs_power_high_power_NA={NA}.png')
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 8))
plt.plot(powers, np.abs(tolerances_initial[:, 0]), label='Fabry-Perot Cavity')
plt.plot(powers, np.abs(tolerances_initial[:, 1]), label='Fabry-Perot Cavity with Negative TEC')
plt.plot(powers, np.abs(tolerances_initial[:, 2]), label='Mirror-Lens-Mirror Cavity')
plt.title(f"Slow Perturbations Tolerance vs Power (High power NA={NA})")
plt.xlabel("Power (W)")
plt.ylabel("Initial Mirror Shift Tolerance (µm)")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig(f'outputs\\figures\initial_tolerance_vs_power_high_power_NA={NA}.png')
plt.show()

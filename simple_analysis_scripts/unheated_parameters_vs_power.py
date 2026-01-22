from matplotlib import use
from scipy.interpolate import interp1d

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


NA = np.linspace(0.02, 0.5, 10000)
L = 0.01
unconcentricity = 4 * LAMBDA_0_LASER ** 2 / (L * np.pi**2 * NA**4)
unconcentricity_mirror_lens_mirror = unconcentricity * 41  # arms' lengths ratio + 1
R = 5e-3
y_tolerance_fabry_perot = unconcentricity * np.tan(NA * 0.46)
alpha_tolerance_fabry_perot = unconcentricity * np.tan(NA *0.46) / R
y_tolerance_mirror_lens_mirror = unconcentricity_mirror_lens_mirror * np.tan(NA * 0.46)
alpha_tolerance_mirror_lens_mirror = unconcentricity_mirror_lens_mirror * np.tan(NA * 0.46) / R

interpolation_tolerance_fabry_perot = interp1d(NA, alpha_tolerance_fabry_perot, kind='cubic')
interpolation_tolerance_mirror_lens_mirror = interp1d(NA, alpha_tolerance_mirror_lens_mirror, kind='cubic')


# %%
NA=0.1
powers = np.concatenate(([0], 4408 - np.logspace(1, 3.5, 20)[::-1], np.linspace(6000, 10000, 5)))
NA_initials = np.zeros((len(powers), 3))
tolerances = np.zeros((len(powers), 3))
tolerances_initial = np.zeros((len(powers), 3))

tolerances_angles = np.zeros((len(powers), 3))
tolerances_initial_angles = np.zeros((len(powers), 3))

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
        tolerance_initial = unheated_version.calculate_parameter_tolerance(perturbation_pointer=PerturbationPointer(element_index=0, parameter_name=ParamsNames.phi))
        if len(unheated_version.physical_surfaces) == 2:
            tolerance_initial_angles_interpolated = interpolation_tolerance_fabry_perot(NA_unheated)
        else:
            tolerance_initial_angles_interpolated = interpolation_tolerance_mirror_lens_mirror(NA_unheated)
        NA_initials[i, j] = NA_unheated
        # tolerances[i, j] = mirror_shift_tolerance
        tolerances_initial[i, j] = tolerance_initial
        tolerances_initial_angles[i, j] = tolerance_initial_angles_interpolated


# %%
plt.close('all')

# --- Common styling ---
title_fs = 30
label_fs = 28
legend_fs = 16
tick_fs = 26
lw = 3

# --- FIGURE 1 ---
fig1, ax1 = plt.subplots(figsize=(14, 10))
plt.plot(powers, NA_initials[:, 0], label='Fabry-Perot Cavity', linewidth=lw)
plt.plot(powers, NA_initials[:, 1], label='Fabry-Perot Cavity with Negative TEC', linewidth=lw)
plt.plot(powers, NA_initials[:, 2], label='Mirror-Lens-Mirror Cavity', linewidth=lw)

plt.title(f"Initial NA vs Power for Fabry-Perot Cavity (High power NA={NA})", fontsize=title_fs)
plt.xlabel("Power (W)", fontsize=label_fs)
plt.ylabel("Initial NA", fontsize=label_fs)
plt.tick_params(labelsize=tick_fs)
plt.grid(True)
plt.legend(fontsize=legend_fs)

plt.savefig(f'outputs\\figures\\initial_NA_vs_power_high_power_NA={NA}.png')
plt.show()


# --- FIGURE 2 ---
fig2, ax2 = plt.subplots(figsize=(14, 10))
plt.plot(powers, np.abs(tolerances_initial[:, 0]), label='Fabry-Perot Cavity', linewidth=lw)
plt.plot(powers, np.abs(tolerances_initial[:, 1]), label='Fabry-Perot Cavity with Negative TEC', linewidth=lw)
plt.plot(powers, np.abs(tolerances_initial[:, 2]), label='Mirror-Lens-Mirror Cavity', linewidth=lw)

plt.title(f"Slow Perturbations Tolerance vs Power (High power NA={NA})", fontsize=title_fs)
plt.xlabel("Power (W)", fontsize=label_fs)
plt.ylabel("Mirror Tilt Tolerance [rad]", fontsize=label_fs)
plt.yscale('log')
plt.tick_params(labelsize=tick_fs)
plt.grid(True)
plt.legend(fontsize=legend_fs)

plt.savefig(f'outputs\\figures\\initial_tolerance_vs_power_high_power_NA={NA} - tilt.png')
plt.show()


# --- FIGURE 3 ---
fig3, ax3 = plt.subplots(figsize=(14, 10))
plt.plot(powers, np.abs(tolerances_initial_angles[:, 0]), label='Fabry-Perot Cavity', linewidth=lw)
plt.plot(powers, np.abs(tolerances_initial_angles[:, 1]), label='Fabry-Perot Cavity with Negative TEC', linewidth=lw)
plt.plot(powers, np.abs(tolerances_initial_angles[:, 2]), label='Mirror-Lens-Mirror Cavity', linewidth=lw)

plt.title(f"Slow Perturbations Tolerance vs Power (High power NA={NA})", fontsize=title_fs)
plt.xlabel("Power (W)", fontsize=label_fs)
plt.ylabel("Mirror Tilt Tolerance [rad]", fontsize=label_fs)
plt.yscale('log')
plt.tick_params(labelsize=tick_fs)
plt.grid(True)
plt.legend(fontsize=legend_fs)

plt.savefig(f'outputs\\figures\\initial_tolerance_vs_power_high_power_NA={NA} - no aberrations - tilt.png')
plt.show()

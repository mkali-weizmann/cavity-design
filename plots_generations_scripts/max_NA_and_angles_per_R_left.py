import matplotlib.pyplot as plt
import numpy as np

from cavity import *

NA_left = 1.7200000000e-01
mirror_on_waist = False
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.0000000000e-03
waist_to_lens_fine = 0.0000000000e+00
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
set_R_left_to_collimate = False
set_R_right_to_collimate = True
set_R_right_to_equalize_angles = False
set_R_right_to_R_left = False
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
print_cavity_parameters = False

waist_to_lens += waist_to_lens_fine
x_span = 10 ** x_span
y_span = 10 ** y_span

import time

N = 50

Rs = np.linspace(5e-3, 25e-3, N)
NAs = np.zeros_like(Rs)
angels_left = np.zeros_like(Rs)
angels_right = np.zeros_like(Rs)
R_rights = np.zeros_like(Rs)

for i, R_left in enumerate(Rs):
    def cavity_generator_outer(NA_left_):
        def cavity_generator_inner(R_right_):
            cavity = mirror_lens_mirror_cavity_generator(NA_left=NA_left_, waist_to_lens=waist_to_lens, h=h,
                                                         R_left=R_left, R_right=R_right_, T_edge=T_edge,
                                                         right_arm_length=right_arm_length,
                                                         lens_fixed_properties=lens_fixed_properties,
                                                         mirrors_fixed_properties=mirrors_fixed_properties,
                                                         symmetric_left_arm=True, waist_to_left_mirror=5e-3,
                                                         lambda_0_laser=1064e-9, set_h_instead_of_w=True,
                                                         auto_set_right_arm_length=auto_set_right_arm_length,
                                                         set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                                                         set_R_right_to_R_left=set_R_right_to_R_left, power=2e4)
            return cavity


        cavity = find_required_value_for_desired_change(cavity_generator=cavity_generator_inner,
                                                        # Takes a float as input and returns a cavity
                                                        desired_parameter=lambda cavity: 1 / cavity.arms[
                                                            2].mode_parameters_on_surface_0.z_minus_z_0[0],
                                                        desired_value=-2 / right_arm_length,
                                                        x0=7e-3)
        return cavity

    cavity = find_required_value_for_desired_change(cavity_generator=cavity_generator_outer,
                                                    desired_parameter=lambda cavity: cavity.arms[1].mode_parameters_on_surfaces[1].spot_size[0],
                                                    desired_value=1e-3,
                                                    x0=0.1)
    NAs[i] = cavity.arms[0].mode_parameters.NA[0]
    angels_left[i] = calculate_incidence_angle(surface=cavity.arms[0].surface_1,
                                               mode_parameters=cavity.arms[0].mode_parameters)
    angels_right[i] = calculate_incidence_angle(surface=cavity.arms[2].surface_0,
                                                mode_parameters=cavity.arms[2].mode_parameters)
    R_rights[i] = cavity.arms[1].surface_1.radius

    # plot_mirror_lens_mirror_cavity_analysis(cavity,
    #                                         auto_set_x=auto_set_x,
    #                                         x_span=x_span,
    #                                         auto_set_y=auto_set_y,
    #                                         y_span=y_span,
    #                                         camera_center=camera_center,
    #                                         add_unheated_cavity=add_unheated_cavity)
    # plt.ylim(-0.005, 0.005)
    # plt.show()

# Plot NAs, angles right and left on the same plt.axes:

import matplotlib.ticker as ticker

fig, ax1 = plt.subplots(2, 1, figsize=(10, 11.5))
color = 'tab:red'
ax1[0].set_xlabel('R_left [mm]')
ax1[0].set_ylabel('NA', color=color)
line1, = ax1[0].plot(Rs*1e3, NAs, color=color, label='NA')
ax1[0].tick_params(axis='y', labelcolor=color)
ax1[0].set_ylim(0, 0.2)
ax1[0].yaxis.set_major_locator(ticker.LinearLocator(15))


ax2 = ax1[0].twinx()
color = 'tab:blue'
ax2.set_ylabel('Angle [rad]', color=color)
line2, = ax2.plot(Rs*1e3, angels_left, color=color, label='Angle Left')
line3, = ax2.plot(Rs*1e3, angels_right, color='green', label='Angle Right')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 22)
ax2.yaxis.set_major_locator(ticker.LinearLocator(15))
ax1[0].grid()
# Add legend
plt.legend(handles=[line1, line2, line3], loc='lower right')

ax2.annotate('Equal angles config.', xy=(Rs[-1]*1e3, angels_left[-1]), xytext=(-120, -50), textcoords='offset points',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
             arrowprops=dict(facecolor='black', shrink=0.05))

ax1[0].annotate('Max NA config..', xy=(Rs[0]*1e3, NAs[0]), xytext=(0, -50), textcoords='offset points',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
             arrowprops=dict(facecolor='black', shrink=0.05))


ax1[1].set_xlabel('R_left [mm]')
ax1[1].set_ylabel('R_right [mm]')
ax1[1].plot(Rs*1e3, R_rights*1e3)
ax1[1].grid()

plt.suptitle('Maximally allowed NA, angles of incidence and R_right\n'
             'as a function of R_left of the lens')
fig.tight_layout()

# save and show:
plt.savefig('figures/figures for sharing/max_NA_and_angles_per_R_left.svg', bbox_inches='tight')
plt.savefig('figures/figures for sharing/max_NA_and_angles_per_R_left.png', bbox_inches='tight')
plt.show()

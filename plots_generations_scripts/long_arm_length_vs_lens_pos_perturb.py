from cavity import *
from matplotlib import use
use('TkAgg')


NA_left = 1.0000000e-01
set_mirror_radius = True
R_small_mirror = 5.0000000000e-03
waist_to_lens = 5.0000000000e-03
waist_to_left_mirror = None
waist_to_lens_fine = -1.2995777570e+00
set_R_left_to_collimate = False
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
camera_center = 2

big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
right_arm_length = None if auto_set_right_arm_length else right_arm_length
waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
R_left += widget_convenient_exponent(R_left_fine)
R_right += widget_convenient_exponent(R_right_fine)
x_span = 10 ** x_span
y_span = 10 ** y_span
T_edge = 1e-3
h = 7.75e-3 / 2

m = 10
max_perturbation = 1e-4
lens_perturbations = np.linspace(-max_perturbation, max_perturbation, 2*m + 1)
long_arm_lengths = np.zeros(len(lens_perturbations))
fig, ax = plt.subplots(6, 1, figsize=(14, 16))

for idx in [0, 2, 4]:
    ax[idx].set_visible(False)

for i, lens_perturbation in enumerate(lens_perturbations):
    distance_to_lens_temp = waist_to_lens + lens_perturbation
    cavity = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=distance_to_lens_temp, h=h,
                                             R_left=R_left,
                                             R_right=R_right, T_c=0,
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
                                             set_R_right_to_collimate=set_R_right_to_collimate
    )
    long_arm_lengths[i] = cavity.arms[2].central_line.length
    print(lens_perturbation)
    if i in [0, m, 2*m]:
        plot_mirror_lens_mirror_cavity_analysis(cavity,
                                            auto_set_x=auto_set_x,
                                            x_span=x_span,
                                            auto_set_y=auto_set_y,
                                            y_span=y_span,
                                            camera_center=camera_center,
                                            ax=ax[2*i // m + 1],)
    fig.tight_layout()
plt.savefig(r'figures\long_arm_length_vs_lens_pos_perturb - systems.svg')
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
plt.show()

# %%
# derivative of long arm length with respect to lens perturbation:
dl_dperturbation = (long_arm_lengths[1:] - long_arm_lengths[:-1]) / (lens_perturbations[1:] - lens_perturbations[:-1])

# Save the results to a file
fig, ax = plt.subplots(2, 1, figsize=(12, 12))
x_mm = lens_perturbations * 1e3
long_arm_lengths_mm = long_arm_lengths * 1e3
ax[0].plot(x_mm, long_arm_lengths_mm, marker='o', linestyle='-')
ax[0].set_xlabel('Lens Position Perturbation (mm)')
ax[0].set_ylabel('Long Arm Length (mm)')
ax[0].set_title(f'Long Arm Length vs Lens Position Perturbation, short arm NA={NA_left:.3f}')
ax[0].grid()
# derivative in mm/mm: multiply lengths by 1e3 and x by 1e3 -> derivative unchanged
ax[1].plot(x_mm[:-1], dl_dperturbation, marker='o', linestyle='-')
ax[1].set_xlabel('Lens Position Perturbation (mm)')
ax[1].set_ylabel('dL_long / dLens_x (mm/mm)')
ax[1].set_title(f'Derivative of Long Arm Length vs Lens Position Perturbation, short arm NA={NA_left:.3f}')
ax[1].grid()
plt.subplots_adjust(hspace=0.35)  # <-- Increased vertical space
plt.savefig(f'figures\long_arm_length_vs_lens_pos_perturb NA={NA_left*1000:.0f}.png')
plt.show()



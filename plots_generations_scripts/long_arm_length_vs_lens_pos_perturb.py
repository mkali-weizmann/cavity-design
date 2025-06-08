from cavity import *
import pickle as pkl
from matplotlib import rc
from matplotlib import use
use('TkAgg')


NA_left = 1.5600000000e-01
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

max_perturbation = 1e-4
lens_perturbations = np.linspace(-max_perturbation, max_perturbation, 51)
long_arm_lengths = np.zeros(len(lens_perturbations))
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
    if np.isclose(lens_perturbation,0):
        plot_mirror_lens_mirror_cavity_analysis(cavity,
                                            auto_set_x=auto_set_x,
                                            x_span=x_span,
                                            auto_set_y=auto_set_y,
                                            y_span=y_span,
                                            camera_center=camera_center)
        plt.show()


plt.figure(figsize=(8, 6))
plt.plot(lens_perturbations, long_arm_lengths, marker='o', linestyle='-')
plt.xlabel('Lens Position Perturbation (m)')
plt.ylabel('Long Arm Length (m)')
plt.title('Long Arm Length vs Lens Position Perturbation')
plt.grid()
plt.tight_layout()
plt.savefig(r'figures\long_arm_length_vs_lens_pos_perturb.png', dpi=300)
plt.show()


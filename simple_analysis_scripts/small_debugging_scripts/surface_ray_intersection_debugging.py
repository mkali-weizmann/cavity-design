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
    use_paraxial_ray_tracing=False
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

perturbation_pointer = PerturbationPointer(element_index=1, parameter_name='y', perturbation_value=35e-9),
perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer, set_central_line=False, set_mode_parameters=False)
# Set a breakpoint at this line in Cavity.find_central_line_standing_wave:
#                 def f_reduced(phi):
#         ---->      z, y = self.f_roots_standing_wave(np.array([theta_default, phi]))
#                     if np.isnan(y):
#                         y = np.inf * np.sign(phi)
#                     return y
# %% Then run this line:
perturbed_cavity.set_central_line()

# %% then from the debugger run this code:
# %%
# And then run this code from the evaluate window:
# phi_array = np.linspace(-3e-2, 3e-2, 100)
# y_array = np.zeros_like(phi_array)
# for i, phi_0 in enumerate(phi_array):
#     _, y_0 = self.f_roots_standing_wave(np.array([theta_default, phi_0]))
#     y_array[i] = y_0
#
# plt.plot(phi_array, y_array)
# plt.axhline(0, linestyle='--', label='other mirror center of curvature', color='gray')
# plt.title('Transverse displacement as a function of starting angle - paraxial')
# plt.xlabel('starting angle [rad]')
# plt.ylabel('final displacement')
# plt.legend()
# plt.savefig(f'outputs\\figures\\paraxial approximation debugging {self.use_paraxial_ray_tracing}.svg')
# plt.show()

from cavity import *

NA_left = 6.7400000000e-02
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

NAs = np.linspace(0.07, 0.09, 5)  # np.concatenate((np.linspace(0.038, 0.05, 70), np.linspace(0.051, 0.16, 20)))
tolerances_mirror_lens_mirror = np.zeros((len(NAs), 3, 5))
tolerances_fabry_perot = np.zeros((len(NAs), 2, 4))
for i, NA in (pbar_NAs := tqdm(enumerate(NAs), total=len(NAs))):
    pbar_NAs.set_description(f'NA={NA:.3f}')
    cavity = mirror_lens_mirror_cavity_generator(NA_left=NA, waist_to_lens=waist_to_lens, h=h,
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
                                                                    power=2e4)
    perturbation_pointer = PerturbationPointer(element_index=0, parameter_name=ParamsNames.y, perturbation_value=np.linspace(-4e-6, 4e-6, 1000))
    cavity.calculate_parameter_tolerance(perturbation_pointer)
    overlap_series = cavity.calculated_shifted_cavity_overlap_integral(
                        perturbation_pointer=perturbation_pointer)


    plt.plot(perturbation_pointer.perturbation_value, overlap_series)
    plt.xlabel('bit mirror shift (m)')
    plt.ylabel('Overlap integral')
    plt.title(f'Cavity Overlap Integral vs Lens Shift\nNA={NA:.3f}')
    plt.grid()
    plt.show()
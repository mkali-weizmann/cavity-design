from cavity import *

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
x_span = 10 ** x_span
y_span = 10 ** y_span


NAs = np.linspace(0.03, 0.16, 20)

def fabry_perot_generator(radii: Tuple[float, float], NA: float, lambda_0_laser=LAMBDA_0_LASER):
    w_0 = w_0_of_NA(NA=NA, lambda_laser=lambda_0_laser)
    mode_0 = ModeParameters(center=np.array([0, 0, 0]),
                            k_vector=np.array([1, 0, 0]),
                            lambda_0_laser=LAMBDA_0_LASER,
                            w_0=np.array([w_0, w_0]),
                            n=1,
                            principle_axes=np.array([[1, 0, 0], [0, 1, 0]]))
    mirror_1 = match_a_mirror_to_mode(mode=mode_0,
                                      material_properties=PHYSICAL_SIZES_DICT['material_properties_fused_silica'],
                                      R=radii[0])
    mirror_2 = match_a_mirror_to_mode(mode=mode_0,
                                      material_properties=PHYSICAL_SIZES_DICT['material_properties_fused_silica'],
                                      R=-radii[1])
    return Cavity(physical_surfaces=[mirror_1, mirror_2],
                  lambda_0_laser=lambda_0_laser,
                  t_is_trivial=True,
                  p_is_trivial=True)


tolerances_df_mirror_lens_mirror = np.zeros((len(NAs), 3, 5))
tolerances_df_fabry_perot = np.zeros((len(NAs), 2, 4))

for i, NA in (pbar_NAs := tqdm(enumerate(NAs), total=len(NAs))):
    pbar_NAs.set_description(f'NA={NA:.3f}')
    cavity_mirror_lens_mirror = mirror_lens_mirror_cavity_generator(NA_left=NA_left, waist_to_lens=waist_to_lens, h=h,
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

    cavity_fabry_perot = fabry_perot_generator(radii=(R_small_mirror, R_small_mirror), NA=NA)

    cavity_mirror_lens_mirror.debug_printing_level = 2
    cavity_fabry_perot.debug_printing_level = 2

    tolerance_df_mirror_lens_mirror = cavity_mirror_lens_mirror.generate_tolerance_dataframe()
    tolerance_df_fabry_perot = cavity_fabry_perot.generate_tolerance_dataframe()

    tolerances_df_mirror_lens_mirror[i, :, :] = np.abs(tolerance_df_mirror_lens_mirror)
    tolerances_df_fabry_perot[i, :, :] = np.abs(tolerance_df_fabry_perot)

    # cavity_mirror_lens_mirror.generate_overlaps_graphs(tolerance_dataframe=tolerance_df_mirror_lens_mirror)
    # cavity_fabry_perot.generate_overlaps_graphs(tolerance_dataframe=tolerance_df_fabry_perot)


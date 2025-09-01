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
                  p_is_trivial=True,
                  standing_wave=True)

# %% DELETE ME
NA = 3.8e-02
# cavity_mirror_lens_mirror = mirror_lens_mirror_cavity_generator(NA_left=NA, waist_to_lens=waist_to_lens, h=h,
#                                                                 R_left=R_left, R_right=R_right, T_c=0,
#                                                                 T_edge=T_edge,
#                                                                 lens_fixed_properties=lens_fixed_properties,
#                                                                 mirrors_fixed_properties=mirrors_fixed_properties,
#                                                                 R_small_mirror=R_small_mirror,
#                                                                 waist_to_left_mirror=waist_to_left_mirror,
#                                                                 lambda_0_laser=1064e-9, set_h_instead_of_w=True,
#                                                                 collimation_mode=collimation_mode,
#                                                                 big_mirror_radius=big_mirror_radius,
#                                                                 right_arm_length=right_arm_length,
#                                                                 set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
#                                                                 set_R_right_to_R_left=set_R_right_to_R_left,
#                                                                 set_R_right_to_collimate=set_R_right_to_collimate,
#                                                                 set_R_left_to_collimate=set_R_left_to_collimate,
#                                                                 power=2e4)

params_original = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.988973493761732e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=2.199758914379698e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
         ]
cavity_mirror_lens_mirror = Cavity.from_params(params=params_original, standing_wave=True,
                                lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=True, set_central_line=True, set_mode_parameters=True)
cavity_mirror_lens_mirror.generate_tolerance_dataframe()

# %%

NAs = np.linspace(0.02, 0.16, 100)

tolerances_mirror_lens_mirror = np.zeros((len(NAs), 3, 5))
tolerances_fabry_perot = np.zeros((len(NAs), 2, 4))

for i, NA in (pbar_NAs := tqdm(enumerate(NAs), total=len(NAs))):
    pbar_NAs.set_description(f'NA={NA:.3f}')
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
                                                                    power=2e4)

    cavity_fabry_perot = fabry_perot_generator(radii=(R_small_mirror, R_small_mirror), NA=NA)

    cavity_mirror_lens_mirror.debug_printing_level = 0
    cavity_fabry_perot.debug_printing_level = 0

    tolerance_df_mirror_lens_mirror = cavity_mirror_lens_mirror.generate_tolerance_dataframe()
    tolerance_df_fabry_perot = cavity_fabry_perot.generate_tolerance_dataframe()

    tolerances_mirror_lens_mirror[i, :, :] = np.abs(tolerance_df_mirror_lens_mirror)
    tolerances_fabry_perot[i, :, :] = np.abs(tolerance_df_fabry_perot)

    # cavity_mirror_lens_mirror.generate_overlaps_graphs(tolerance_dataframe=tolerance_df_mirror_lens_mirror)
    # plt.show()
    # cavity_fabry_perot.generate_overlaps_graphs(tolerance_dataframe=tolerance_df_fabry_perot)
    # plt.show()

# %%
# from matplotlib import use
# use('Qt5Agg')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
x_index, y_index, phi_index, r_1_index = 0, 1, 2, 3
ax[0].set_title('Tilt tolerances')
ax[0].plot(NAs, tolerances_mirror_lens_mirror[:, 0, phi_index], label='Small mirror')
ax[0].plot(NAs, tolerances_mirror_lens_mirror[:, 1, phi_index], label='Lens')
ax[0].plot(NAs, tolerances_mirror_lens_mirror[:, 2, phi_index], label='Large mirror')
ax[0].plot(NAs, tolerances_fabry_perot[:, 0, phi_index], label='Fabry-Perot cavity')
ax[0].set_xlabel('NA')
ax[0].set_ylabel('Tilt tolerance')
ax[0].legend()
ax[0].grid()
ax[0].set_yscale('log')

ax[1].set_title('Lateral shift tolerances')
ax[1].plot(NAs, tolerances_mirror_lens_mirror[:, 0, y_index], label='Small mirror')
ax[1].plot(NAs, tolerances_mirror_lens_mirror[:, 1, y_index], label='Lens')
ax[1].plot(NAs, tolerances_mirror_lens_mirror[:, 2, y_index], label='Large mirror')
ax[1].plot(NAs, tolerances_fabry_perot[:, 0, y_index], label='Fabry-Perot cavity')
ax[1].set_xlabel('NA')
ax[1].set_ylabel('Tilt tolerance')
ax[1].legend()
ax[1].grid()
ax[1].set_yscale('log')

plt.show()
# %%




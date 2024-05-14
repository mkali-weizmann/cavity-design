from cavity import *

lambda_laser = 1064e-9
symmetric_left_arm = False
waist_to_left_mirror = -5.0000000000e-03
waist_to_left_mirror_fine = 0.0000000000e+00
NA_3 = 1.0000000000e-01
right_arm_length = -5.2343530200e-01
right_mirror_radius_shift = 0.0000000000e+00
right_mirror_position_shift = 0.0000000000e+00
auto_set_right_arm_length = True
mirror_on_waist = False
x_2_perturbation = 0.0000000000e+00
waist_to_lens = 5.0000000000e-03
set_h_instead_of_w = True
T_edge = 1.0000000000e-03
h = 3.8750000000e-03
R_left = 5e-3
R_left_small_perturbation = 0
R_right = 5e-3
R_right_small_perturbation = 0
w = 4e-3
lens_fixed_properties = 'fused_silica'
mirrors_fixed_properties = 'ULE'
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
print_input_parameters = True
print_parameters = True

power = 2e-4
h_divided_by_spot_size = 2.8

mirrors_material_properties = convert_material_to_mirror_or_lens(
    PHYSICAL_SIZES_DICT[f"thermal_properties_{mirrors_fixed_properties}"], 'mirror')
lens_material_properties = convert_material_to_mirror_or_lens(
    PHYSICAL_SIZES_DICT[f"thermal_properties_{lens_fixed_properties}"], 'lens')

if set_h_instead_of_w:
    assert R_left, f"transverse radius of lens ({h:.2e}), can not be bigger than radius of curvature ({R_left:.2e})"
    assert R_right, f"transverse radius of lens ({h:.2e}), can not be bigger than radius of curvature ({R_right:.2e})"
    dT_c_left = R_left * (1-np.sqrt(1-h**2 / R_left**2))
    dT_c_right = R_right * (1-np.sqrt(1-h**2 / R_right**2))
    w = T_edge + dT_c_left + dT_c_right

right_arm_length = 10 ** right_arm_length

x_span = 10 ** x_span
y_span = 10 ** y_span

# Generate left arm's mirror:
z_R_3 = lambda_laser / (np.pi * NA_3 ** 2)
# left_lens_x = waist_to_lens  # - w / 2
left_waist_x = 0  # left_lens_x - waist_to_lens
if symmetric_left_arm:
    x_3 = left_waist_x - waist_to_lens + waist_to_left_mirror_fine
else:
    x_3 = waist_to_left_mirror + waist_to_left_mirror_fine
mode_3_center = np.array([left_waist_x, 0, 0])

mode_3_k_vector = np.array([1, 0, 0])
mode_3 = ModeParameters(center=np.stack([mode_3_center, mode_3_center], axis=0), k_vector=mode_3_k_vector,
                        z_R=np.array([z_R_3, z_R_3]), principle_axes=np.array([[0, 0, 1], [0, 1, 0]]), lambda_laser=lambda_laser)
mirror_3 = match_a_mirror_to_mode(mode_3, x_3 - mode_3.center[0, 0], mirrors_material_properties)
# Generate lens:
# if lens_material_properties_override:
alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens, intensity_reflectivity, intensity_transmittance, temperature = lens_material_properties.to_array
n = PHYSICAL_SIZES_DICT['refractive_indices'][lens_fixed_properties]
x_2_left = left_waist_x + waist_to_lens

surface_3 = CurvedRefractiveSurface(center=np.array([x_2_left, 0, 0]), radius=R_left, outwards_normal=np.array([-1, 0, 0]), n_1=1, n_2=n, curvature_sign=-1, name='lens_left', thermal_properties=lens_material_properties)
local_mode_3 = mode_3.local_mode_parameters(np.linalg.norm(surface_3.center - mode_3.center[0]))
surface_1 = find_equal_angles_surface(mode_before_lens=mode_3, surface_0=surface_3,
                                      T_edge=T_edge, h=h, lambda_laser=lambda_laser)
lens_params_right = surface_1.to_params
lens_params_left = surface_3.to_params

local_mode_1 = local_mode_2_of_lens_parameters(np.array([R_left, w, n]), local_mode_3)
mode_1 = local_mode_1.to_mode_parameters(location_of_local_mode_parameter=surface_1.center,
                                         k_vector=mode_3_k_vector, lambda_laser=lambda_laser)
lens_params_right = surface_1.to_params
lens_params_left = surface_3.to_params

if auto_set_right_arm_length:
    z_minus_z_0 = - local_mode_1.z_minus_z_0[0]
elif mirror_on_waist:
    z_minus_z_0 = 0
else:
    z_minus_z_0 = local_mode_1.z_minus_z_0[0] + right_arm_length
mirror_1 = match_a_mirror_to_mode(mode_1, z_minus_z_0, mirrors_material_properties)
mirror_1.radius += right_mirror_radius_shift
mirror_1.origin += np.array([right_mirror_position_shift - right_mirror_radius_shift, 0, 0])
mirror_3_params = mirror_3.to_params
mirror_1_params = mirror_1.to_params
params = np.stack([mirror_1_params, lens_params_right, lens_params_left, mirror_3_params], axis=0)
params[1, 0] += x_2_perturbation
params[2, 0] += x_2_perturbation
cavity = Cavity.from_params(params,
                            lambda_laser=lambda_laser,
                            standing_wave=True,
                            p_is_trivial=True,
                            t_is_trivial=True,
                            set_mode_parameters=True,
                            names=['Right mirror', 'lens_right', 'lens_left', 'Left mirror'],
                            power=power)

unheated_cavity = cavity.thermal_transformation()
fig, ax = plt.subplots(2, 1, figsize=(16, 12))
cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[0])

minimal_width_lens = find_minimal_width_for_spot_size_and_radius(radius=R_left,
                                                                 spot_size_radius=
                                                                 cavity.arms[0].mode_parameters_on_surfaces[
                                                                     1].spot_size(cavity.lambda_laser)[0],
                                                                 T_edge=T_edge,
                                                                 h_divided_by_spot_size=h_divided_by_spot_size)
geometric_feasibility = True
spot_size_lens_right = cavity.arms[0].mode_parameters_on_surfaces[1].spot_size(cavity.lambda_laser)[0]
minimal_h_lens = h_divided_by_spot_size * spot_size_lens_right

if set_h_instead_of_w:
    CA_5mm_divided_by_2spot_size = 2.5e-3 / spot_size_lens_right
    if CA_5mm_divided_by_2spot_size < 2:
        geometric_feasibility = False
    angle_right = cavity.arms[0].calculate_incidence_angle(1)
    angle_left = cavity.arms[2].calculate_incidence_angle(0)

    lens_specs_string = f"R = {R_left:.3e},  D = {2 * h:.3e}, T_edge = {T_edge:.2e}, T_c = {w:.3e}\n spot_size (2w) = {2 * spot_size_lens_right:.3e},   5mm / 2w_spot_size = {CA_5mm_divided_by_2spot_size:.3e},   lens is wide enough = {geometric_feasibility},   {angle_left=:.1f},   {angle_right=:.1f}"
else:
    if w < minimal_width_lens:
        geometric_feasibility = False
    lens_specs_string = f"R_lens = {R_left:.3e},  w_lens = {w:.3e}, minimal_w_lens = {minimal_width_lens:.2e}, minimal_h_lens={minimal_h_lens:.3e}, lens is thick enough = {geometric_feasibility}"
ax[0].set_title(
    f"short arm NA = {cavity.arms[2].mode_parameters.NA[0]:.3e},  short arm length = {np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[3].center):.3e} [m]\n"  # 
    f"long arm NA = {cavity.arms[0].mode_parameters.NA[0]:.3e},  long arm length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.3e} [m]\n"  # 
    f"{lens_specs_string}, \n"
    f"R_left = {mirror_3.radius:.3e}, spot diameters left mirror = {2 * cavity.arms[2].mode_parameters_on_surfaces[1].spot_size(cavity.lambda_laser)[0]:.2e}, R_right = {mirror_1.radius:.3e}, spot diameters right mirror = {2 * cavity.arms[0].mode_parameters_on_surfaces[0].spot_size(cavity.lambda_laser)[0]:.2e}")

plt.grid()
if auto_set_x:
    # cavity_length = mirror_1.center[0] - mirror_3.center[0]
    # ax[0].set_xlim(mirror_3.center[0] - 0.01 * cavity_length, mirror_1.center[0] + 0.01 * cavity_length)
    ax[0].set_xlim(mirror_3.center[0] - 0.01, x_2 + 0.4)
if auto_set_y:
    y_lim = maximal_lens_height(R_left, w) * 1.1
else:
    y_lim = y_span
ax[0].set_ylim(-y_lim, y_lim)
ax[0].grid()
if print_parameters:
    print(f"{params=}")
unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1])
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_ylim(ax[0].get_ylim())
ax[1].set_title(
    f"unheated_cavity, short arm NA={unheated_cavity.arms[2].mode_parameters.NA[0]:.2e}, Left spot size = {2 * unheated_cavity.arms[2].mode_parameters_on_surface_1.spot_size(lambda_laser=1064e-9)[0]:.2e}")
plt.subplots_adjust(hspace=0.35)
fig.tight_layout()
plt.show()
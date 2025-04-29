import matplotlib.pyplot as plt

from cavity import *
import pickle as pkl
from matplotlib import rc
from matplotlib.lines import Line2D

np.set_printoptions(precision=10, linewidth=151)
LEGEND_FONT_SIZE = 15
TITLE_FONT_SIZE = 13
fig, ax = plt.subplots(2, 2, figsize=(13, 7))


def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"


def print_parameters_func(local_parameters):
    for key, value in local_parameters.items():
        if key.startswith("t_") or key.startswith("p_") or key in ["theta", "phi"]:
            print(f"{key} = 1j*{value / np.pi:.10e}")
        elif isinstance(value, float):
            print(f"{key} = {value:.10e}")
        else:
            print(f"{key} = {value}")


with open('data/params_dict.pkl', 'rb') as f:
    params_dict = pkl.load(f)



font = {'size'   : 10}
rc('font', **font)
lambda_0_laser = 1.064e-6


power = 5.0000000000e+04
waist_to_left_mirror = 1.0000000000e-02
symmetric_left_arm = True
mode_3_center = -8.1540000000e-03
mode_3_center_fine = -1.0000000000e-05
x_2 = 4.0000000000e-03
x_2_perturbation = 0.0000000000e+00
NA_3 = 1.0006595100e-01
right_arm_length = -1.0000000000e+00
right_mirror_radius_shift = 0.0000000000e+00
right_mirror_position_shift = 0.0000000000e+00
auto_set_right_arm_length = True
mirror_on_waist = False
R = -1.8064380000e+00
R_small_perturbation = 0.0000000000e+00
w = -2.3635000000e+00
w_small_perturbation = 0.0000000000e+00
n = 1.7600000000e+00
lens_thermal_properties_override = True
lens_fixed_properties = 'sapphire'
alpha_lens = 5.5000000000e-06
beta_lens = 1.0000000000e-06
kappa_lens = 4.6060000000e+01
dn_dT_lens = 1.1700000000e-05
nu_lens = 3.0000000000e-01
alpha_absorption_lens = 3.0000000000e-01
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = False
y_span = 1.46e-3
camera_center = 2
print_input_parameters = True
print_parameters = False


R = 10 ** R
R += R_small_perturbation
w = 10 ** w
w += w_small_perturbation
right_arm_length = 10 ** right_arm_length
x_span = 10 ** x_span
# y_span = 10 ** y_span
mode_3_center += mode_3_center_fine
if print_input_parameters:
    print_parameters_func(locals())
# Generate left arm's mirror:
w_0_3 = lambda_0_laser / (np.pi * NA_3)
if symmetric_left_arm:
    half_length = (x_2 - w/2) - mode_3_center
    x_3 = mode_3_center - half_length
else:
    x_3 = mode_3_center - waist_to_left_mirror
mode_3_center = np.array([mode_3_center, 0, 0])
mode_3_k_vector = np.array([1, 0, 0])
mode_3 = ModeParameters(center=np.stack([mode_3_center, mode_3_center], axis=0), k_vector=mode_3_k_vector,
                        w_0=np.array([w_0_3, w_0_3]),
                        principle_axes=np.array([[0, 0, 1], [0, 1, 0]]), lambda_0_laser=lambda_0_laser)
mirror_3 = match_a_mirror_to_mode(mode_3, x_3 - mode_3.center[0, 0], PHYSICAL_SIZES_DICT['material_properties_ULE'])

# Generate lens:
if lens_thermal_properties_override:
    alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens = PHYSICAL_SIZES_DICT[
        f"thermal_properties_{lens_fixed_properties}"].to_array
    n = PHYSICAL_SIZES_DICT['refractive_indices'][lens_fixed_properties]
lens_params = np.array(
    [x_2, 0, 0, 0, R, n, w, 1, 0, 0, alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens,
     1])
surface_3, surface_1 = generate_lens_from_params(lens_params, names=['lens_left', 'lens_right'])
local_mode_3 = mode_3.local_mode_parameters(np.linalg.norm(surface_3.center - mode_3.center[0]))
local_mode_1 = local_mode_2_of_lens_parameters(np.array([R, w, n]), local_mode_3)
mode_1 = local_mode_1.to_mode_parameters(location_of_local_mode_parameter=surface_1.center,
                                         k_vector=mode_3_k_vector, lambda_0_laser=lambda_0_laser)

if auto_set_right_arm_length:
    z_minus_z_0 = - local_mode_1.z_minus_z_0[0]
elif mirror_on_waist:
    z_minus_z_0 = 0
else:
    z_minus_z_0 = local_mode_1.z_minus_z_0[0] + right_arm_length

mirror_1 = match_a_mirror_to_mode(mode_1, z_minus_z_0, PHYSICAL_SIZES_DICT['material_properties_ULE'])
mirror_1.radius += right_mirror_radius_shift
mirror_1.origin += np.array([right_mirror_position_shift-right_mirror_radius_shift, 0, 0])
mirror_3_params = mirror_3.to_params
mirror_1_params = mirror_1.to_params
lens_params = lens_params.astype(np.complex128)
lens_params[3] = 1j
params = np.stack([mirror_1_params, lens_params, mirror_3_params], axis=0)
params[1, 0] += x_2_perturbation

cavity = Cavity.from_params(params,
                            lambda_0_laser=lambda_0_laser,
                            standing_wave=True,
                            p_is_trivial=True,
                            t_is_trivial=True,
                            set_mode_parameters=True,
                            names=['Left mirror', 'Lens_left', 'lens_right', 'Right mirror'],
                            power=power)
unheated_cavity = cavity.thermal_transformation()
cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[0, 0])
ax[0, 0].set_title("High Power", fontsize=TITLE_FONT_SIZE)
plt.grid()
if auto_set_x:
    cavity_length = mirror_1.center[0] - mirror_3.center[0]
    ax[0, 0].set_xlim(mirror_3.center[0] - 0.01 * cavity_length, mirror_1.center[0] + 0.01 * cavity_length)
if auto_set_y:
    y_lim = maximal_lens_height(R, w) * 1.1
else:
    y_lim = y_span
ax[0, 0].set_ylim(-y_lim, y_lim)
legend_elements = [Line2D([0], [0], color='red', ls='--', lw=1, label=f'short arm NA={cavity.arms[2].mode_parameters.NA[0]:.3f}')]
ax[0, 0].legend(handles=legend_elements, fontsize=LEGEND_FONT_SIZE, loc='lower center')
if print_parameters:
    print(f"{params=}")
unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[0, 1])
ax[0, 1].set_xlim(ax[0, 0].get_xlim())
ax[0, 1].set_ylim(ax[0, 0].get_ylim())
ax[0, 1].set_title(
    f"Low Power - Sapphire Lens", fontsize=TITLE_FONT_SIZE)
legend_elements = [Line2D([0], [0], color='red', ls='--', lw=1, label=f'short arm NA={unheated_cavity.arms[2].mode_parameters.NA[0]:.3f}')]
ax[0, 1].legend(handles=legend_elements, fontsize=LEGEND_FONT_SIZE, loc='lower center')

plt.grid()
plt.show()
# %%

power = 5.0000000000e+04
waist_to_left_mirror = 1.0000000000e-02
symmetric_left_arm = True
mode_3_center = -8.1540000000e-03
mode_3_center_fine = -1.0000000000e-05
x_2 = 4.0000000000e-03
x_2_perturbation = 0.0000000000e+00
NA_3 = 1.0006595100e-01
right_arm_length = -1.0000000000e+00
right_mirror_radius_shift = 0.0000000000e+00
right_mirror_position_shift = 0.0000000000e+00
auto_set_right_arm_length = True
mirror_on_waist = False
R = -2.0279240100e+00
R_small_perturbation = -2.8508000000e-05
w = -2.3635000000e+00
w_small_perturbation = 0.0000000000e+00
n = 1.4550000000e+00
lens_thermal_properties_override = False
lens_fixed_properties = 'sapphire'
alpha_lens = 4.8000000000e-07
beta_lens = 1.0000000000e-06
kappa_lens = 1.3800000000e+00
dn_dT_lens = 1.2000000000e-05
nu_lens = 1.5000000000e-01
alpha_absorption_lens = 1.0000000000e-03
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
print_input_parameters = True
print_parameters = False


R = 10 ** R
R += R_small_perturbation
w = 10 ** w
w += w_small_perturbation
right_arm_length = 10 ** right_arm_length
x_span = 10 ** x_span
y_span = 10 ** y_span
mode_3_center += mode_3_center_fine
if print_input_parameters:
    print_parameters_func(locals())
# Generate left arm's mirror:
w_0_3 = lambda_0_laser / (np.pi * NA_3 ** 2)
if symmetric_left_arm:
    half_length = (x_2 - w/2) - mode_3_center
    x_3 = mode_3_center - half_length
else:
    x_3 = mode_3_center - waist_to_left_mirror
mode_3_center = np.array([mode_3_center, 0, 0])
mode_3_k_vector = np.array([1, 0, 0])
mode_3 = ModeParameters(center=np.stack([mode_3_center, mode_3_center], axis=0), k_vector=mode_3_k_vector,
                        w_0=np.array([w_0_3, w_0_3]),
                        principle_axes=np.array([[0, 0, 1], [0, 1, 0]]), lambda_0_laser=lambda_0_laser)
mirror_3 = match_a_mirror_to_mode(mode_3, x_3 - mode_3.center[0, 0], PHYSICAL_SIZES_DICT['material_properties_ULE'])

# Generate lens:
if lens_thermal_properties_override:
    alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens = PHYSICAL_SIZES_DICT[
        f"thermal_properties_{lens_fixed_properties}"].to_array
    n = PHYSICAL_SIZES_DICT['refractive_indices'][lens_fixed_properties]
lens_params = np.array(
    [x_2, 0, 0, 0, R, n, w, 1, 0, 0, alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens,
     1])
surface_3, surface_1 = generate_lens_from_params(lens_params, names=['lens_left', 'lens_right'])
local_mode_3 = mode_3.local_mode_parameters(np.linalg.norm(surface_3.center - mode_3.center[0]))
local_mode_1 = local_mode_2_of_lens_parameters(np.array([R, w, n]), local_mode_3)
mode_1 = local_mode_1.to_mode_parameters(location_of_local_mode_parameter=surface_1.center,
                                         k_vector=mode_3_k_vector, lambda_0_laser=lambda_0_laser)

if auto_set_right_arm_length:
    z_minus_z_0 = - local_mode_1.z_minus_z_0[0]
elif mirror_on_waist:
    z_minus_z_0 = 0
else:
    z_minus_z_0 = local_mode_1.z_minus_z_0[0] + right_arm_length

mirror_1 = match_a_mirror_to_mode(mode_1, z_minus_z_0, PHYSICAL_SIZES_DICT['material_properties_ULE'])
mirror_1.radius += right_mirror_radius_shift
mirror_1.origin += np.array([right_mirror_position_shift-right_mirror_radius_shift, 0, 0])
mirror_3_params = mirror_3.to_params
mirror_1_params = mirror_1.to_params
lens_params = lens_params.astype(np.complex128)
lens_params[3] = 1j
params = np.stack([mirror_1_params, lens_params, mirror_3_params], axis=0)
params[1, 0] += x_2_perturbation

cavity = Cavity.from_params(params,
                            lambda_0_laser=lambda_0_laser,
                            standing_wave=True,
                            p_is_trivial=True,
                            t_is_trivial=True,
                            set_mode_parameters=True,
                            names=['Left mirror', 'Lens_left', 'lens_right', 'Right mirror'],
                            power=power)
unheated_cavity = cavity.thermal_transformation()
# cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1, 1])
# ax[1, 1].set_title("Heated Cavity")
# plt.grid()
# if auto_set_x:
#     cavity_length = mirror_1.center[0] - mirror_3.center[0]
#     ax[1, 1].set_xlim(mirror_3.center[0] - 0.01 * cavity_length, mirror_1.center[0] + 0.01 * cavity_length)
# if auto_set_y:
#     y_lim = maximal_lens_height(R, w) * 1.1
# else:
#     y_lim = y_span
# ax[1, 1].set_ylim(-y_lim, y_lim)
# if print_parameters:
#     print(f"{params=}")


unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1, 1])

ax[1, 1].set_xlim(ax[0, 0].get_xlim())
ax[1, 1].set_ylim(ax[0, 0].get_ylim())
ax[1, 1].set_title(
    f"Low Power - Fused Silica Lens", fontsize=TITLE_FONT_SIZE)

legend_elements = [Line2D([0], [0], color='red', ls='--', lw=1, label=f'short arm NA={unheated_cavity.arms[2].mode_parameters.NA[0]:.3f}')]
ax[1, 1].legend(handles=legend_elements, loc='lower center', fontsize=LEGEND_FONT_SIZE)


power = 5.0000000000e+04
waist_to_left_mirror = 1.0000000000e-02
symmetric_left_arm = True
mode_3_center = -8.1540000000e-03
mode_3_center_fine = -1.0000000000e-05
x_2 = 4.0000000000e-03
x_2_perturbation = 0.0000000000e+00
NA_3 = 1.0006595100e-01
right_arm_length = -1.0000000000e+00
right_mirror_radius_shift = 0.0000000000e+00
right_mirror_position_shift = 0.0000000000e+00
auto_set_right_arm_length = True
mirror_on_waist = False
R = -1.8064380000e+00
R_small_perturbation = 0.0000000000e+00
w = -2.3635000000e+00
w_small_perturbation = 0.0000000000e+00
n = 1.7600000000e+00
lens_thermal_properties_override = False
lens_fixed_properties = 'sapphire'
alpha_lens = 5.5000000000e-06
beta_lens = 0
kappa_lens = 4.6060000000e+01
dn_dT_lens = 0
nu_lens = 3.0000000000e-01
alpha_absorption_lens = 3.0000000000e-01
auto_set_x = True
x_span = -1.0000000000e+00
auto_set_y = True
y_span = -2.9000000000e+00
camera_center = 2
print_input_parameters = True
print_parameters = False


R = 10 ** R
R += R_small_perturbation
w = 10 ** w
w += w_small_perturbation
right_arm_length = 10 ** right_arm_length
x_span = 10 ** x_span
y_span = 10 ** y_span
mode_3_center += mode_3_center_fine
if print_input_parameters:
    print_parameters_func(locals())
# Generate left arm's mirror:
w_0_3 = lambda_0_laser / (np.pi * NA_3 ** 2)
if symmetric_left_arm:
    half_length = (x_2 - w/2) - mode_3_center
    x_3 = mode_3_center - half_length
else:
    x_3 = mode_3_center - waist_to_left_mirror
mode_3_center = np.array([mode_3_center, 0, 0])
mode_3_k_vector = np.array([1, 0, 0])
mode_3 = ModeParameters(center=np.stack([mode_3_center, mode_3_center], axis=0), k_vector=mode_3_k_vector,
                        z_R=np.array([w_0_3, w_0_3]),
                        principle_axes=np.array([[0, 0, 1], [0, 1, 0]]), lambda_0_laser=lambda_0_laser)
mirror_3 = match_a_mirror_to_mode(mode_3, x_3 - mode_3.center[0, 0], PHYSICAL_SIZES_DICT['material_properties_ULE'])

# Generate lens:
if lens_thermal_properties_override:
    alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens = PHYSICAL_SIZES_DICT[
        f"thermal_properties_{lens_fixed_properties}"].to_array
    n = PHYSICAL_SIZES_DICT['refractive_indices'][lens_fixed_properties]
lens_params = np.array(
    [x_2, 0, 0, 0, R, n, w, 1, 0, 0, alpha_lens, beta_lens, kappa_lens, dn_dT_lens, nu_lens, alpha_absorption_lens,
     1])
surface_3, surface_1 = generate_lens_from_params(lens_params, names=['lens_left', 'lens_right'])
local_mode_3 = mode_3.local_mode_parameters(np.linalg.norm(surface_3.center - mode_3.center[0]))
local_mode_1 = local_mode_2_of_lens_parameters(np.array([R, w, n]), local_mode_3)
mode_1 = local_mode_1.to_mode_parameters(location_of_local_mode_parameter=surface_1.center,
                                         k_vector=mode_3_k_vector, lambda_0_laser=lambda_0_laser)

if auto_set_right_arm_length:
    z_minus_z_0 = - local_mode_1.z_minus_z_0[0]
elif mirror_on_waist:
    z_minus_z_0 = 0
else:
    z_minus_z_0 = local_mode_1.z_minus_z_0[0] + right_arm_length

mirror_1 = match_a_mirror_to_mode(mode_1, z_minus_z_0, PHYSICAL_SIZES_DICT['material_properties_ULE'])
mirror_1.radius += right_mirror_radius_shift
mirror_1.origin += np.array([right_mirror_position_shift-right_mirror_radius_shift, 0, 0])
mirror_3_params = mirror_3.to_params
mirror_1_params = mirror_1.to_params
lens_params = lens_params.astype(np.complex128)
lens_params[3] = 1j
params = np.stack([mirror_1_params, lens_params, mirror_3_params], axis=0)
params[1, 0] += x_2_perturbation

cavity = Cavity.from_params(params,
                            lambda_0_laser=lambda_0_laser,
                            standing_wave=True,
                            p_is_trivial=True,
                            t_is_trivial=True,
                            set_mode_parameters=True,
                            names=['Left mirror', 'Lens_left', 'lens_right', 'Right mirror'],
                            power=power)
unheated_cavity = cavity.thermal_transformation()
# cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1, 0])
# ax[1, 0].set_title("Heated Cavity")
# plt.grid()
# if auto_set_x:
#     cavity_length = mirror_1.center[0] - mirror_3.center[0]
#     ax[1, 0].set_xlim(mirror_3.center[0] - 0.01 * cavity_length, mirror_1.center[0] + 0.01 * cavity_length)
# if auto_set_y:
#     y_lim = maximal_lens_height(R, w) * 1.1
# else:
#     y_lim = y_span
# ax[1, 0].set_ylim(-y_lim, y_lim)
# if print_parameters:
#     print(f"{params=}")
# legend_elements = [Line2D([0], [0], color='red', ls='--', lw=1, label=f'short arm NA={cavity.arms[2].mode_parameters.NA[0]:.3f}')]
# ax[1, 0].legend(handles=legend_elements, loc='lower center', fontsize=LEGEND_FONT_SIZE)

unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1, 0])

ax[1, 0].set_xlim(ax[0, 0].get_xlim())
ax[1, 0].set_ylim([-2e-3, 2e-3])
ax[1, 0].set_title(
    f"Low Power - No lens thermal transformation, only mirrors", fontsize=TITLE_FONT_SIZE)

legend_elements = [Line2D([0], [0], color='red', ls='--', lw=1, label=f'short arm NA={unheated_cavity.arms[2].mode_parameters.NA[0]:.3f}')]
ax[1, 0].legend(handles=legend_elements, loc='lower center', fontsize=LEGEND_FONT_SIZE)



for i in [0, 1]:
    for j in [0, 1]:
        ax[i, j].yaxis.set_major_formatter(plt.FuncFormatter(mm_format))
        ax[i, j].xaxis.set_major_formatter(plt.FuncFormatter(cm_format))
        ax[i, j].set_ylabel("y [mm]")
        ax[i, j].set_xlabel("x [cm]")
        ax[i, j].grid()

plt.tight_layout()

plt.savefig(f"figures/thermal_lens_comparison.png", dpi=600)

plt.show()

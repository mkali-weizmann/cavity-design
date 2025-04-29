from cavity import *
import pickle as pkl
from matplotlib.lines import Line2D
from matplotlib import rc

FONT_SIZE_AXIS_LABELS = 14
FONT_SIZE_TITLES = 14
FONT_SIZE_TICKS = 12

def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"


def kW_format(value, tick_number):
    return f"{value * 1e-3:.0f}"


font = {'size': 11}
rc('font', **font)
with open('data/params_dict.pkl', 'rb') as f:
    params_dict = pkl.load(f)

axis_span = 0.005
camera_center = -1
names = ['Right Mirror', 'lens', 'Left Mirror']

params = params_dict['Sapphire, NA=0.2-0.0365, L1=0.3 - High NA axis']
x_2_values = np.concatenate([np.linspace(-6e-5, 0, 20)])  # np.array([0]),
params_temp = params.copy()
params_temp[1, 0] = x_2_values[-1]  # Change the position of the lens

cavity = Cavity.from_params(params=params_temp, standing_wave=True,
                            lambda_0_laser=LAMBDA_0_LASER, names=names, t_is_trivial=True, p_is_trivial=True)
fig, ax = plt.subplots(figsize=(6, 4))
cavity.plot(ax=ax, plane='xz')  #

ax.yaxis.set_major_formatter(plt.FuncFormatter(cm_format))
ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format))
ax.set_ylabel("y [cm]")
ax.set_xlabel("x [cm]")

# ax.set_ymargin(0.3)
# ax.set_xlim(x_3 - 0.01, x_1 + 0.01)
# ax.set_ylim(-0.002, 0.002)
ax.set_title(
    f"short arm NA={cavity.arms[2].mode_parameters.NA[0]:.3f}, short arm length = {np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[3].center)*100:.0f}cm\n"
    f"long arm NA={cavity.arms[0].mode_parameters.NA[0]:.3f},  long arm length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center)*100:.0f}cm")
plt.savefig('figures/systems/mirror-lens-mirror_high_NA_ratio.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
NAs_2, tolerance_matrix_2 = generate_tolerance_of_NA(params, parameter_index_for_NA_control=(1, 0), arm_index_for_NA=2,
                                                     parameter_values=x_2_values, lambda_0_laser=lambda_0_laser,
                                                     t_is_trivial=True, p_is_trivial=True, return_cavities=False,
                                                     print_progress=True)
tolerance_matrix_2 = np.abs(tolerance_matrix_2)
# %%
x_1 = 2.0000000000e-02
y_1 = 0.0000000000e+00
t_1 = 0.0000000000e+00
p_1 = 0.0000000000e+00
r_1 = 1.00000071e-02
x_3 = 0
y_3 = 0.0000000000e+00
t_3 = 0.0000000000e+00
p_3 = 0.0000000000e+00
lambda_0_laser = 1064e-09
r_3 = r_1
p_3 += np.pi
names = ['Right Mirror', 'Left Mirror']
# INDICES_DICT = {'x': 0, 'y': 1, 'theta': 2, 'phi': 3, 'r': 4, 'n_1': 5, 'w': 6, 'n_2': 7, 'z': 8, 'curvature_sign': 9,
#                 'alpha_thermal_expansion': 10, 'beta_power_absorption': 11, 'kappa_thermal_conductivity': 12,
#                 'dn_dT': 13,
#                 'nu_poisson_ratio': 14, 'alpha_volume_absorption': 15, 'surface_type': 16}
params = np.array([[x_1, y_1, t_1, p_1, r_1, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['material_properties_ULE'].to_array, 0],
                   # [x_2, y_2, t_lens, p_2, r_2, n_in, w_2, n_out, 0, 0, 1],
                   [x_3, y_3, t_3, p_3, r_3, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['material_properties_ULE'].to_array, 0]])
ratios = 1 - np.concatenate((np.array([0]), np.logspace(-7, -3.5, 10, endpoint=True)))
x_1_values = x_1 * ratios
params_temp = params.copy()
params_temp[0, 0] = x_1_values[0]

cavity = Cavity.from_params(params=params_temp, standing_wave=True,
                            lambda_0_laser=lambda_0_laser, names=names, t_is_trivial=True, p_is_trivial=True)
font = {'size': 11}
rc('font', **font)
fig, ax = plt.subplots(figsize=(6, 4))
cavity.plot(axis_span=1, ax=ax)
ax.set_xlim(-0.005, 0.3205)
length = x_1 - x_3 + 0.002
ax.set_ylim(-0.006, 0.006)

def cm_format_shifted(value, tick_number):
    return f"{value * 1e2 - 2:.1f}"
ax.yaxis.set_major_formatter(plt.FuncFormatter(cm_format))
ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format_shifted))
ax.set_ylabel("y [cm]")
ax.set_xlabel("x [cm]")
ax.set_title(
    f"Fabry Perot, NA={cavity.arms[0].mode_parameters.NA[0]:.3f}, length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center)*100:.0f}cm")

plt.savefig('figures/systems/Fabry-Perot_scaled_x.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
NAs_1, tolerance_matrix_1 = generate_tolerance_of_NA(params, parameter_index_for_NA_control=(0, 0), arm_index_for_NA=0,
                                                     parameter_values=x_1_values, t_is_trivial=True, p_is_trivial=True,
                                                     return_cavities=False, print_progress=True)
tolerance_matrix_1 = np.abs(tolerance_matrix_1)
# %%
font = {'size': 20}
rc('font', **font)
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Tilt Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 2, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 2, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 2, :], label='2-arms, lens', linestyle='--', color='b')
ax.plot(NAs_2, tolerance_matrix_2[2, 2, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [rad]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_tilt.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Axial Displacement Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 0, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 0, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 0, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 0, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_lateral.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Transversal Displacement Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 1, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 1, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 1, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 1, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_transversal.svg', dpi=300, bbox_inches='tight')
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10))
plt.title("Radius Change Tolerance")
ax.plot(NAs_1, tolerance_matrix_1[0, 4, :], label='Fabry-Perot', linestyle=':', color='g')
ax.plot(NAs_2, tolerance_matrix_2[0, 4, :], label='2-arms, right mirror', linestyle='-', color='g')
ax.plot(NAs_2, tolerance_matrix_2[1, 4, :], label='2-arms, lens', linestyle='--', color='b', linewidth=2.2)
ax.plot(NAs_2, tolerance_matrix_2[2, 4, :], label='2-arms, left mirror', linestyle='-', color='orange')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Numerical Aperture')
ax.set_ylabel('Tolerance [m]')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig('figures/NA-tolerance/comparison_radius.svg', dpi=300, bbox_inches='tight')
plt.show()

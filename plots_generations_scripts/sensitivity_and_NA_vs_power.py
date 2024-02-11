import matplotlib.pyplot as plt

from cavity import *
import pickle as pkl
from matplotlib.lines import Line2D

FONT_SIZE_AXIS_LABELS = 14
FONT_SIZE_TITLES = 14
FONT_SIZE_TICKS = 12

def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"


def kW_format(value, tick_number):
    return f"{value * 1e-3:.0f}"


# def scientific_format(value, tick_number):
#     return f"{value:.2e}"


def plot_high_power_and_low_power_cavity(cavity, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    unheated_cavity = cavity.thermal_transformation()
    unheated_cavity.plot(ax=ax, laser_color='b')
    cavity.plot(ax=ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(mm_format))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format))
    custom_lines = [Line2D([0], [0], color='r', ),
                    Line2D([0], [0], color='b')]
    ax.legend(custom_lines, ['High power', 'Low power'])
    ax.set_ylabel("y [mm]", fontsize=FONT_SIZE_AXIS_LABELS)
    ax.set_xlabel("x [cm]", fontsize=FONT_SIZE_AXIS_LABELS)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)


def generate_low_power_NA_plots(cavity, powers: np.ndarray, arm_for_NA_measurement: int, title: str="low power NA vs. power\nhigh power NA is set to 0.1"
                                , create_new_axes: bool=True):
    N = len(powers)
    resulted_low_power_NAs = np.zeros(N)
    for i, power in enumerate(powers):
        cavity.power = power
        unheated_cavity = cavity.thermal_transformation()
        resulted_low_power_NAs[i] = unheated_cavity.arms[arm_for_NA_measurement].mode_parameters.NA[0]
    if create_new_axes:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.plot(powers, resulted_low_power_NAs)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(kW_format))
    plt.xlabel('Laser power, kW', fontsize=FONT_SIZE_AXIS_LABELS)
    plt.ylabel('Low power NA', fontsize=FONT_SIZE_AXIS_LABELS)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)


def generate_tolerance_series(cavity, surface_to_tilt_index: int, powers: np.ndarray):
    N = len(powers)
    tolerances_low_power = np.zeros(N)
    for i, power in enumerate(powers):
        print(i, end='\r')
        cavity.power = power
        unheated_cavity = cavity.thermal_transformation()
        tolerances_low_power[i] = np.abs(
            unheated_cavity.calculate_parameter_critical_tolerance(parameter_index=(surface_to_tilt_index, 2), accuracy=1e-4))
    return tolerances_low_power


def generate_tolerance_plots(powers: np.ndarray,
                             tolerances: np.ndarray,
                             title: str = "Tolerance as a function of power",
                             create_new_axes: bool=True):
    if create_new_axes:
        fig, ax = plt.subplots(figsize=(8, 4.6))
    plt.plot(powers, tolerances)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(kW_format))
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(scientific_format))
    plt.xlabel('Laser power [kW]', fontsize=FONT_SIZE_AXIS_LABELS)
    plt.ylabel('Tolerance [$\mathrm{rad}$]', fontsize=FONT_SIZE_AXIS_LABELS)
    plt.yscale('log')
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)

# %% Mirror-lens-mirror cavity:
with open('data/params_dict.pkl', 'rb') as f:
    params_dict = pkl.load(f)
params = params_dict['Sapphire, NA=0.1-0.039, L1=0.3 - High NA axis']

lambda_laser = 1064e-9
power_laser = 2e4
powers = np.logspace(2, 5, 40)
cavity_mirror_lens_mirror = Cavity.from_params(params=params, set_initial_surface=False, standing_wave=True,
                                lambda_laser=lambda_laser, power=power_laser, p_is_trivial=True, t_is_trivial=True)
plot_high_power_and_low_power_cavity(cavity_mirror_lens_mirror, "The modes in the cavity for ~0kW and for 20kW circulating power")
plt.savefig('figures/high_power_low_power_cavity_mirror_lens_mirror.svg')
plt.show()
generate_low_power_NA_plots(cavity=cavity_mirror_lens_mirror,
                       powers=powers,
                       arm_for_NA_measurement=2,
                       title="Low power NA vs. laser power\nHigh power NA is fixed to 0.1")
plt.savefig('figures/low_power_NA_vs_power_mirror_lens_mirror.svg')
plt.show()
# tolerance_serise_mirror_lens_mirror = generate_tolerance_series(cavity_mirror_lens_mirror, surface_to_tilt_index=3, powers=powers)
generate_tolerance_plots(powers=powers, tolerances=tolerance_serise_mirror_lens_mirror, title="Tolerance to tilt of the small arm's mirrors vs. laser power\nMirror-lens-mirror cavity")
plt.savefig('figures/tilt_tolerance_vs_power_mirror_lens_mirror.svg')
plt.show()

# %% Do the same for a Fabry-Perot cavity:
x_1 = 2.0000000000e-02
y_1 = 0.0000000000e+00
t_1 = 0.0000000000e+00
p_1 = 0.0000000000e+00
r_1 = 1.00001145e-02
x_3 = 0
y_3 = 0.0000000000e+00
t_3 = 0.0000000000e+00
p_3 = 0.0000000000e+00
lambda_laser = 1064e-09
r_3 = r_1
p_3 += np.pi
names = ['Right Mirror', 'Left Mirror']
params = np.array([[x_1, y_1, t_1, p_1, r_1, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['thermal_properties_ULE'].to_array, 0],
                   [x_3, y_3, t_3, p_3, r_3, 0, 0, 0, 0, 1, *PHYSICAL_SIZES_DICT['thermal_properties_ULE'].to_array, 0]])
powers = np.linspace(0, 4.9e3, 40)
cavity_fabry_perot = Cavity.from_params(params=params, standing_wave=True,
                            lambda_laser=lambda_laser, power=4.9e3,
                            names=names, t_is_trivial=True, p_is_trivial=True)
plot_high_power_and_low_power_cavity(cavity_fabry_perot, "The modes in the cavity for ~0kW and for 4.9kW circulating power")
plt.savefig('figures/high_power_low_power_cavity_fabry_perot.svg')
plt.show()
generate_low_power_NA_plots(cavity=cavity_fabry_perot,
                       powers=powers,
                       arm_for_NA_measurement=0,
                       title="Low power NA vs. laser power\nHigh power NA is fixed to 0.1")
plt.savefig('figures/low_power_NA_vs_power_fabry_perot.svg')
plt.show()
# tolerance_series_fabry_perot = generate_tolerance_series(cavity_fabry_perot, surface_to_tilt_index=0, powers=powers)
generate_tolerance_plots(powers=powers, tolerances=tolerance_series_fabry_perot, title="Tolerance to mirror's tilt vs. laser power\nFabry-Perot cavity")

plt.savefig('figures/tilt_tolerance_vs_power_fabry_perot.svg')
plt.show()


# %% Do the same for a Fabry-Perot cavity with a negative thermal expansion coefficient:
x_1 = 2.0000000000e-02
y_1 = 0.0000000000e+00
t_1 = 0.0000000000e+00
p_1 = 0.0000000000e+00
r_1 = 1.00001145e-02
x_3 = 0
y_3 = 0.0000000000e+00
t_3 = 0.0000000000e+00
p_3 = 0.0000000000e+00
lambda_laser = 1064e-09
r_3 = r_1
p_3 += np.pi
names = ['Right Mirror', 'Left Mirror']
thermal_properties_inverse_expansion = PHYSICAL_SIZES_DICT['thermal_properties_ULE']
thermal_properties_inverse_expansion.alpha_expansion *= -1
params = np.array([[x_1, y_1, t_1, p_1, r_1, 0, 0, 0, 0, 1, *thermal_properties_inverse_expansion.to_array, 0],
                   [x_3, y_3, t_3, p_3, r_3, 0, 0, 0, 0, 1, *thermal_properties_inverse_expansion.to_array, 0]])

cavity_fabry_perot_thermally_inverted = Cavity.from_params(params=params, standing_wave=True,
                            lambda_laser=lambda_laser, power=4.9e3,
                            names=names, t_is_trivial=True, p_is_trivial=True)
powers = np.linspace(0, 4.9e3, 40)
plot_high_power_and_low_power_cavity(cavity_fabry_perot_thermally_inverted, "The modes in the cavity for ~0kW and for 4.9kW circulating power\n"
                                 "Mirrors have a negative thermal expansion coefficient")
plt.savefig('figures/high_power_low_power_cavity_fabry_perot_thermally_inverted.svg')
plt.show()
generate_low_power_NA_plots(cavity=cavity_fabry_perot_thermally_inverted,
                       powers=powers,
                       arm_for_NA_measurement=0,
                       title="Low power NA as a function of the laser power high power NA is fixed to 0.1\n"
                             "Mirrors have a negative thermal expansion coefficient")
plt.savefig('figures/low_power_NA_vs_power_fabry_perot_thermally_inverted.svg')
plt.show()
# tolerance_series_fabry_perot_thermally_inverted = generate_tolerance_series(cavity_fabry_perot_thermally_inverted, surface_to_tilt_index=0, powers=powers)
generate_tolerance_plots(powers=powers, tolerances=tolerance_series_fabry_perot_thermally_inverted, title="Tolerance to mirror's tilt vs. laser power\nFabry-Perot cavity with a negative thermal expansion coefficient")
plt.savefig('figures/tilt_tolerance_vs_power_fabry_perot_thermally_inverted.svg')
plt.show()
# %% low_power NA Comparison:
fig, ax = plt.subplots(figsize=(8, 4.5))
generate_low_power_NA_plots(cavity=cavity_mirror_lens_mirror,
                            powers=powers,
                            arm_for_NA_measurement=2,
                            create_new_axes=False)
generate_low_power_NA_plots(cavity=cavity_fabry_perot,
                            powers=powers,
                            arm_for_NA_measurement=0,
                            create_new_axes=False)
generate_low_power_NA_plots(cavity=cavity_fabry_perot_thermally_inverted,
                            powers=powers,
                            arm_for_NA_measurement=0,
                            create_new_axes=False)
plt.legend(['Mirror-Lens-Mirror', 'Fabry-Perot', 'Fabry-Perot with negative thermal expansion coefficient'])
plt.savefig('figures/low_power_NA_comparison.svg')
plt.show()
# %% tolerance comparison:
# tolerance_serise_mirror_lens_mirror_narrowed = generate_tolerance_series(cavity_mirror_lens_mirror, surface_to_tilt_index=3, powers=powers)
fig, ax = plt.subplots(figsize=(8, 4.5))
generate_tolerance_plots(powers=powers, tolerances=tolerance_serise_mirror_lens_mirror_narrowed, create_new_axes=False)
generate_tolerance_plots(powers=powers, tolerances=tolerance_series_fabry_perot, create_new_axes=False)
generate_tolerance_plots(powers=powers, tolerances=tolerance_series_fabry_perot_thermally_inverted, create_new_axes=False)
plt.legend(['Mirror-Lens-Mirror', 'Fabry-Perot', 'Fabry-Perot with negative thermal expansion coefficient'])
plt.savefig('figures/tilt_tolerance_vs_power_comparison.svg')
plt.show()
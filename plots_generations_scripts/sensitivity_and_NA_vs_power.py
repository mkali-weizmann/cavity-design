import matplotlib.pyplot as plt

from cavity import *
import pickle as pkl
from matplotlib.lines import Line2D

FONT_SIZE_AXIS_LABELS = 14
FONT_SIZE_TITLES = 14
FONT_SIZE_TICKS = 12
save_figs = False

powers_100kW = np.logspace(0, 5, 40)
powers_5kW = np.linspace(0, 4.9e3, 40)
powers_20kW = np.linspace(0, 2e4, 40)


def mm_format(value, tick_number):
    return f"{value * 1e3:.2f}"


def cm_format(value, tick_number):
    return f"{value * 1e2:.2f}"


def kW_format(value, tick_number):
    return f"{value * 1e-3:.0f}"


def plot_high_power_and_low_power_cavity(cavity, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    unheated_cavity = cavity.thermal_transformation()
    unheated_cavity.plot(ax=ax, laser_color="b")
    cavity.plot(ax=ax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(mm_format))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(cm_format))
    custom_lines = [
        Line2D(
            [0],
            [0],
            color="r",
        ),
        Line2D([0], [0], color="b"),
    ]
    ax.legend(custom_lines, ["High power", "Low power"])
    ax.set_ylabel("y [mm]", fontsize=FONT_SIZE_AXIS_LABELS)
    ax.set_xlabel("x [cm]", fontsize=FONT_SIZE_AXIS_LABELS)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)
    plt.grid()


def generate_low_power_NA_plots(
    cavity,
    powers: np.ndarray,
    arm_for_NA_measurement: int,
    title: str = "low power NA vs. power\nhigh power NA is set to 0.1",
    create_new_axes: bool = True,
):
    N = len(powers)
    resulted_low_power_NAs = np.zeros(N)
    for i, power in enumerate(powers):
        cavity.power = power
        unheated_cavity = cavity.thermal_transformation()
        resulted_low_power_NAs[i] = unheated_cavity.arms[
            arm_for_NA_measurement
        ].mode_parameters.NA[0]
    if create_new_axes:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.grid()
    plt.plot(powers, resulted_low_power_NAs)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(kW_format))
    plt.xlabel("Laser power, kW", fontsize=FONT_SIZE_AXIS_LABELS)
    plt.ylabel("Low power NA", fontsize=FONT_SIZE_AXIS_LABELS)
    ylim = plt.gca().get_ylim()
    plt.ylim(0, ylim[1] * 1.1)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)


def generate_tolerance_series(cavity, surface_to_tilt_index: int, powers: np.ndarray):
    N = len(powers)
    tolerances_low_power = np.zeros(N)
    for i, power in enumerate(powers):
        print(i, end="\r")
        cavity.power = power
        unheated_cavity = cavity.thermal_transformation()
        tolerances_low_power[i] = np.abs(
            unheated_cavity.calculate_parameter_tolerance(perturbation_pointer=(surface_to_tilt_index, 2),
                                                          accuracy=1e-4)
        )
    return tolerances_low_power


def generate_tolerance_plots(
    powers: np.ndarray,
    tolerances: np.ndarray,
    title: str = "Tolerance as a function of power",
    create_new_axes: bool = True,
):
    if create_new_axes:
        fig, ax = plt.subplots(figsize=(8, 4.6))
    plt.plot(powers, tolerances)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(kW_format))
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(scientific_format))
    plt.xlabel("Laser power [kW]", fontsize=FONT_SIZE_AXIS_LABELS)
    plt.ylabel("Tolerance [$\mathrm{rad}$]", fontsize=FONT_SIZE_AXIS_LABELS)
    plt.yscale("log")
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.title(title, fontsize=FONT_SIZE_TITLES)
    plt.grid()


# %% Mirror-lens-mirror cavity:
with open("data/params_dict.pkl", "rb") as f:
    params_dict = pkl.load(f)
params = params_dict[
    "Sapphire, NA=0.2, L1=0.3m, w=5.9mm, R_lens=15.16mm - High NA Axis"
]
params[0, INDICES_DICT['intensity_reflectivity']] = PHYSICAL_SIZES_DICT['material_properties_ULE'].intensity_reflectivity
params[2, INDICES_DICT['intensity_reflectivity']] = PHYSICAL_SIZES_DICT['material_properties_ULE'].intensity_reflectivity
params = np.array([[ 3.2550206224e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.6128989168e-01+0.j,  0.0000000000e+00+0.j,
         0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
         1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,
                      np.nan+0.j,  0.0000000000e+00+0.j],
       [ 1.7347234760e-18+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+1.j,  8.9127992977e-03+0.j,  1.4550000000e+00+0.j,
         5.9000000000e-03+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  4.8000000000e-07+0.j,  1.0000000000e-06+0.j,
         1.3800000000e+00+0.j,  1.2000000000e-05+0.j,  1.5000000000e-01+0.j,  1.0000000000e-03+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,
                      np.nan+0.j,  1.0000000000e+00+0.j],
       [-2.0800928000e-02+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  8.9254720322e-03+0.j,  0.0000000000e+00+0.j,
         0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
         1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,
                      np.nan+0.j,  0.0000000000e+00+0.j]])

# params = np.array([[ 3.2519036040e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.6113400154e-01+0.j,  0.0000000000e+00+0.j,
#          0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
#          1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,
#                       np.nan+0.j,  0.0000000000e+00+0.j],
#        [ 1.7347234760e-18+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+1.j,  1.4886177934e-02+0.j,  1.7600000000e+00+0.j,
#          5.9000000000e-03+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,
#          4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,
#                       np.nan+0.j,  1.0000000000e+00+0.j],
#        [-2.0800928000e-02+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  8.9254720322e-03+0.j,  0.0000000000e+00+0.j,
#          0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
#          1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,
#                       np.nan+0.j,  0.0000000000e+00+0.j]])


power_laser = 2e4
cavity_mirror_lens_mirror = Cavity.from_params(
    params=params,
    set_initial_surface=False,
    standing_wave=True,
    lambda_0_laser=LAMBDA_0_LASER,
    power=power_laser,
    p_is_trivial=True,
    t_is_trivial=True,
    names=["Right Mirror", "Lens", "Left Mirror"],
)
# %%
cavity_mirror_lens_mirror.plot()
plt.xlim(-0.02, 0.02)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.title('Sapphire')
plt.show()

cavity_mirror_lens_mirror.specs(save_specs_name='fused_silica comparison')

# %%
unheated_cavity = cavity_mirror_lens_mirror.thermal_transformation()
unheated_cavity_lens_params = unheated_cavity.to_params()[1:3, :]
old_cavity_lens_params = cavity_mirror_lens_mirror.to_params()[1, :]
new_radius = (
    unheated_cavity_lens_params[0, INDICES_DICT["r"]]
    + unheated_cavity_lens_params[1, INDICES_DICT["r"]]
) / 2
new_index_of_refraction = (
    unheated_cavity_lens_params[0, INDICES_DICT["n_2"]]
    + unheated_cavity_lens_params[1, INDICES_DICT["n_1"]]
) / 2
new_params = copy.copy(cavity_mirror_lens_mirror.to_params())
new_params[1, INDICES_DICT["r"]] = new_radius
new_params[1, INDICES_DICT["n_1"]] = new_index_of_refraction
new_unheated_cavity = Cavity.from_params(
    params=new_params,
    set_initial_surface=False,
    standing_wave=True,
    lambda_0_laser=LAMBDA_0_LASER,
    power=0,
    p_is_trivial=True,
    t_is_trivial=True,
    names=["Right Mirror", "Lens", "Left Mirror"],
)
print("Hot cavity:")
cavity_mirror_lens_mirror_specs_ule = cavity_mirror_lens_mirror.specs(save_specs_name="hot_fuesed_silica",
                                                                      tolerance_matrix=True, print_specs=True)
print("Cold cavity:")
new_unheated_cavity_specs_ule = new_unheated_cavity.specs(save_specs_name="cold_fuesed_silica", tolerance_matrix=True,
                                                          print_specs=True)


# %%

plot_high_power_and_low_power_cavity(
    cavity_mirror_lens_mirror,
    "The modes in the cavity for low power and for 20kW circulating power.\nMirror-lens-mirror cavity",
)
# if save_figs:
#     plt.savefig("figures/high_power_low_power_cavity_mirror_lens_mirror.svg")
# plt.show()
# generate_low_power_NA_plots(
#     cavity=cavity_mirror_lens_mirror,
#     powers=powers_100kW,
#     arm_for_NA_measurement=2,
#     title="Low power NA vs. laser power\nHigh power NA is fixed to 0.1",
# )
#
# plt.savefig("figures/low_power_NA_vs_power_mirror_lens_mirror.svg")
# plt.show()
# tolerance_serise_mirror_lens_mirror = generate_tolerance_series(
#     cavity_mirror_lens_mirror, surface_to_tilt_index=3, powers=powers_100kW
# )
# generate_tolerance_plots(
#     powers=powers_100kW,
#     tolerances=tolerance_serise_mirror_lens_mirror,
#     title="Tolerance to tilt of the small arm's mirrors vs. laser power\nMirror-lens-mirror cavity",
# )
# # if save_figs:
# # plt.savefig('figures/tilt_tolerance_vs_power_mirror_lens_mirror.svg')
# plt.show()


# %% Do the same for a Fabry-Perot cavity:
# x_1 = 2.0000000000e-02
# y_1 = 0.0000000000e00
# t_1 = 0.0000000000e00
# p_1 = 0.0000000000e00
# r_1 = 1.00001145e-02
# x_3 = 0
# y_3 = 0.0000000000e00
# t_3 = 0.0000000000e00
# p_3 = 0.0000000000e00
# lambda_0_laser = 1064e-09
# r_3 = r_1
# p_3 += np.pi
# names = ["Right Mirror", "Left Mirror"]
# params = np.array(
#     [
#         [
#             x_1,
#             y_1,
#             t_1,
#             p_1,
#             r_1,
#             0,
#             0,
#             0,
#             0,
#             1,
#             *PHYSICAL_SIZES_DICT["material_properties_ULE"].to_array,
#             0,
#         ],
#         [
#             x_3,
#             y_3,
#             t_3,
#             p_3,
#             r_3,
#             0,
#             0,
#             0,
#             0,
#             1,
#             *PHYSICAL_SIZES_DICT["material_properties_ULE"].to_array,
#             0,
#         ],
#     ]
# )
#
# cavity_fabry_perot = Cavity.from_params(
#     params=params,
#     standing_wave=True,
#     lambda_0_laser=lambda_0_laser,
#     power=3.5e3,
#     names=names,
#     t_is_trivial=True,
#     p_is_trivial=True,
# )
# plot_high_power_and_low_power_cavity(
#     cavity_fabry_perot,
#     "The modes in the cavity for low power and for 4.9kW circulating power\nFabry-Perot cavity",
# )
# if save_figs:
#     plt.savefig("figures/high_power_low_power_cavity_fabry_perot.svg")
# plt.show()
# generate_low_power_NA_plots(
#     cavity=cavity_fabry_perot,
#     powers=powers_5kW,
#     arm_for_NA_measurement=0,
#     title="Low power NA vs. laser power\nHigh power NA is fixed to 0.1",
# )
# plt.savefig("figures/low_power_NA_vs_power_fabry_perot.svg")
# plt.show()
# tolerance_series_fabry_perot = generate_tolerance_series(
#     cavity_fabry_perot, surface_to_tilt_index=0, powers=powers_5kW
# )
# generate_tolerance_plots(
#     powers=powers_5kW,
#     tolerances=tolerance_series_fabry_perot,
#     title="Tolerance to mirror's tilt vs. laser power\nFabry-Perot cavity",
# )
# if save_figs:
#     plt.savefig("figures/tilt_tolerance_vs_power_fabry_perot.svg")
# plt.show()
#
#
# # %% Do the same for a Fabry-Perot cavity with a negative thermal expansion coefficient:
# x_1 = 2.0000000000e-02
# y_1 = 0.0000000000e00
# t_1 = 0.0000000000e00
# p_1 = 0.0000000000e00
# r_1 = 1.00001145e-02
# x_3 = 0
# y_3 = 0.0000000000e00
# t_3 = 0.0000000000e00
# p_3 = 0.0000000000e00
# lambda_0_laser = 1064e-09
# r_3 = r_1
# p_3 += np.pi
# names = ["Right Mirror", "Left Mirror"]
# thermal_properties_inverse_expansion = PHYSICAL_SIZES_DICT["material_properties_ULE"]
# thermal_properties_inverse_expansion.alpha_expansion *= -1
# params = np.array(
#     [
#         [
#             x_1,
#             y_1,
#             t_1,
#             p_1,
#             r_1,
#             0,
#             0,
#             0,
#             0,
#             1,
#             *thermal_properties_inverse_expansion.to_array,
#             0,
#         ],
#         [
#             x_3,
#             y_3,
#             t_3,
#             p_3,
#             r_3,
#             0,
#             0,
#             0,
#             0,
#             1,
#             *thermal_properties_inverse_expansion.to_array,
#             0,
#         ],
#     ]
# )
#
# cavity_fabry_perot_thermally_inverted = Cavity.from_params(
#     params=params,
#     standing_wave=True,
#     lambda_0_laser=lambda_0_laser,
#     power=4.9e3,
#     names=names,
#     t_is_trivial=True,
#     p_is_trivial=True,
# )
# plot_high_power_and_low_power_cavity(
#     cavity_fabry_perot_thermally_inverted,
#     "The modes in the cavity for low power and for 4.9kW circulating power\n"
#     "Fabry-Perot cavity with a negative thermal expansion coefficient",
# )
# if save_figs:
#     plt.savefig(
#         "figures/high_power_low_power_cavity_fabry_perot_thermally_inverted.svg"
#     )
# plt.show()
# generate_low_power_NA_plots(
#     cavity=cavity_fabry_perot_thermally_inverted,
#     powers=powers_100kW,
#     arm_for_NA_measurement=0,
#     title="Low power NA as a function of the laser power high power NA is fixed to 0.1\n"
#     "Fabry-Perot cavity with a negative thermal expansion coefficient",
# )
# # plt.savefig('figures/low_power_NA_vs_power_fabry_perot_thermally_inverted.svg')
# plt.show()
# tolerance_series_fabry_perot_thermally_inverted = generate_tolerance_series(
#     cavity_fabry_perot_thermally_inverted, surface_to_tilt_index=0, powers=powers_100kW
# )
# generate_tolerance_plots(
#     powers=powers_100kW,
#     tolerances=tolerance_series_fabry_perot_thermally_inverted,
#     title="Tolerance to mirror's tilt vs. laser power\nFabry-Perot cavity with a negative thermal expansion coefficient",
# )
# if save_figs:
#     plt.savefig("figures/tilt_tolerance_vs_power_fabry_perot_thermally_inverted.svg")
# plt.show()
#
# # %% low_power NA Comparison:
# fig, ax = plt.subplots(figsize=(8, 4.5))
#
# generate_low_power_NA_plots(
#     cavity=cavity_mirror_lens_mirror,
#     powers=powers_20kW,
#     arm_for_NA_measurement=2,
#     create_new_axes=False,
# )
# generate_low_power_NA_plots(
#     cavity=cavity_fabry_perot,
#     powers=powers_5kW,
#     arm_for_NA_measurement=0,
#     create_new_axes=False,
# )
# generate_low_power_NA_plots(
#     cavity=cavity_fabry_perot_thermally_inverted,
#     powers=powers_20kW,
#     arm_for_NA_measurement=0,
#     create_new_axes=False,
# )
# plt.legend(
#     [
#         "Mirror-Lens-Mirror",
#         "Fabry-Perot",
#         "Fabry-Perot with negative\nthermal expansion coefficient",
#     ]
# )
# ax.set_ylim(0, 0.3)
# if save_figs:
#     plt.savefig("figures/low_power_NA_comparison.svg")
# plt.show()
#
# # %% tolerance comparison:
# tolerance_serise_mirror_lens_mirror_narrowed = generate_tolerance_series(
#     cavity_mirror_lens_mirror, surface_to_tilt_index=3, powers=powers_20kW
# )
# tolerance_series_fabry_perot_thermally_inverted_narrowed = generate_tolerance_series(
#     cavity_fabry_perot_thermally_inverted, surface_to_tilt_index=0, powers=powers_20kW
# )
# fig, ax = plt.subplots(figsize=(8, 4.5))
# generate_tolerance_plots(
#     powers=powers_20kW,
#     tolerances=tolerance_serise_mirror_lens_mirror_narrowed,
#     create_new_axes=False,
# )
# generate_tolerance_plots(
#     powers=powers_5kW, tolerances=tolerance_series_fabry_perot, create_new_axes=False
# )
# generate_tolerance_plots(
#     powers=powers_20kW,
#     tolerances=tolerance_series_fabry_perot_thermally_inverted_narrowed,
#     create_new_axes=False,
# )
# plt.legend(
#     [
#         "Mirror-Lens-Mirror",
#         "Fabry-Perot",
#         "Fabry-Perot with negative thermal expansion coefficient",
#     ]
# )
# if save_figs:
#     plt.savefig("figures/tilt_tolerance_vs_power_comparison.svg")
# plt.show()

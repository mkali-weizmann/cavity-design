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

params = [
          OpticalElementParams(name='Small Mirror'           ,surface_type='curved_mirror'                  , x=-4.988973493761732e-03  , y=0                       , z=0                       , theta=0                       , phi=-1e+00 * np.pi          , r_1=5e-03                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  )),
          OpticalElementParams(name='Lens'                   ,surface_type='thick_lens'                     , x=6.387599281689135e-03   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2.422e-02               , r_2=5.488e-03               , curvature_sign=CurvatureSigns.concave, T_c=2.913797540986543e-03   , n_inside_or_after=1.76e+00                , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=1.76e+00                , alpha_expansion=5.5e-06                 , beta_surface_absorption=1e-06                   , kappa_conductivity=4.606e+01               , dn_dT=1.17e-05                , nu_poisson_ratio=3e-01                   , alpha_volume_absorption=1e-02                   , intensity_reflectivity=1e-04                   , intensity_transmittance=9.99899e-01             , temperature=np.nan                  )),
          OpticalElementParams(name='Big Mirror'             ,surface_type='curved_mirror'                  , x=2.199758914379698e-01   , y=0                       , z=0                       , theta=0                       , phi=0                       , r_1=2e-01                   , r_2=np.nan                  , curvature_sign=CurvatureSigns.concave, T_c=np.nan                  , n_inside_or_after=1e+00                   , n_outside_or_before=1e+00                   , material_properties=MaterialProperties(refractive_index=None                    , alpha_expansion=7.5e-08                 , beta_surface_absorption=1e-06                   , kappa_conductivity=1.31e+00                , dn_dT=None                    , nu_poisson_ratio=1.7e-01                 , alpha_volume_absorption=None                    , intensity_reflectivity=9.99889e-01             , intensity_transmittance=1e-04                   , temperature=np.nan                  ))
         ]

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
plot_mirror_lens_mirror_cavity_analysis(cavity_mirror_lens_mirror)
# %%
cavity_mirror_lens_mirror.plot()

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
                                                                      tolerance_dataframe=True, print_specs=True)
print("Cold cavity:")
new_unheated_cavity_specs_ule = new_unheated_cavity.specs(save_specs_name="cold_fuesed_silica",
                                                          tolerance_dataframe=True, print_specs=True)


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
# material_properties_inverse_expansion = PHYSICAL_SIZES_DICT["material_properties_ULE"]
# material_properties_inverse_expansion.alpha_expansion *= -1
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
#             *material_properties_inverse_expansion.to_array,
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
#             *material_properties_inverse_expansion.to_array,
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

from matplotlib import use
use("TkAgg")

from cavity_design import *
from tqdm import tqdm


elements=[LASER_OPTIK_MIRROR, EKSMA_LENS_20MM_ASPHERIC, THOLABS_100MM_PLANO_CONVEX_LENS, COASTLINE_20CM_MIRROR]

# %% 2d map:
def equality_equation(x, coef):
    quad_deriv = 2 * coef[1] * x
    higher_deriv = sum(2 * n * coef[n] * x ** (2 * n - 1) for n in range(2, len(coef)))
    return abs(quad_deriv) - abs(higher_deriv)

cavity = Cavity(elements=elements, standing_wave=True, lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=False, set_central_line=True, set_mode_parameters=True)
collimation_point = cavity[0].radius + back_focal_length_of_lens_object(lens_object=cavity[1])
long_arm_lengths = np.array([28e-2, 29e-2, 30e-2, 31e-2, 32e-2, 33e-2, 34e-2, 37e-2])
mid_arm_length = 1.6e-2
short_arm_lengths = np.linspace(collimation_point-1e-3, collimation_point+4e-4, 100)

NAs = np.zeros(shape=(len(long_arm_lengths), len(short_arm_lengths)))
mode_spacings = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
# zero_derivative_points = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
polynomial_derivatives_equality_stable = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
polynomial_derivatives_equality_metastable = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
focii_to_lens = np.zeros(shape=(len(long_arm_lengths)))

cavity.place_element(element=cavity[2], position = (collimation_point + cavity[1].T_c + mid_arm_length) * RIGHT, reference_center=cavity[0], recalculate_optic=False)

for i, long_arm_length in tqdm(enumerate(long_arm_lengths)):  #
    cavity.place_element(element=cavity[-1], position=long_arm_length * RIGHT, reference_center=cavity.surfaces[-2],
                         recalculate_optic=False)
    flag=False
    for j, short_arm_length in enumerate(short_arm_lengths):
        cavity.place_element(element=cavity[1], position=short_arm_length * RIGHT, reference_center=cavity[0],
                             recalculate_optic=True)
        NAs[i, j] = cavity.arms[0].mode_parameters.NA[0]
        mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
        if not np.isnan(cavity.arms[0].mode_parameters.NA[0]):
            results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=50,
                                                          phi_max=np.arcsin(0.3), print_tests=False)
            potential_polynomial = results_dict["polynomial_residuals_mirror"].coef
            a = 3 * potential_polynomial[3]
            b = 2 * potential_polynomial[2]
            c = 1 * potential_polynomial[1]
            quadratic_solution_metastable = np.sqrt((- b - np.sqrt(b**2 - 4 * a * c)) / (2 * a))
            quadratic_solution_stable = np.sqrt((- b + np.sqrt(b ** 2 + 4 * a * c)) / (2 * a))
            # zero_derivative_points[i, j] = results_dict["zero_derivative_point"]
            polynomial_derivatives_equality_stable[i, j] = quadratic_solution_stable
            polynomial_derivatives_equality_metastable[i, j] = quadratic_solution_metastable

        if j == 0:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
        if cavity.arms[0].mode_parameters.NA[0] > 0.07 and flag is False:
            focii_to_lens[i] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0] - cavity.surfaces[-1].radius
            flag=True

# %%
plot_different_axes = True
plt.close('all')
if plot_different_axes:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
else:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()
    # ax.set_zorder(ax2.get_zorder() + 1)  # Make ax "on top" for coordinate display

for i in range(len(long_arm_lengths)):
    na_line, = ax.plot(short_arm_lengths * 1e3, NAs[i, :], label=f"Mirror-to-spherical = {long_arm_lengths[i]*100:.2f}cm")
    if i == 0:
        ax.plot(short_arm_lengths * 1e3, polynomial_derivatives_equality_stable[i, :], color=na_line.get_color(), linestyle='-.', alpha=0.3, label="Stable quadratic-quartic equality")
        ax.plot(short_arm_lengths * 1e3, polynomial_derivatives_equality_metastable[i, :], color=na_line.get_color(), linestyle=':', alpha=0.3, label="Metastable quadratic-quartic equality")
        # ax.plot(short_arm_lengths * 1e3, zero_derivative_points[i, :], color=na_line.get_color(), linestyle='-.', label=f"maximally allowed NA")
    else:
        ax.plot(short_arm_lengths * 1e3, polynomial_derivatives_equality_stable[i, :], color=na_line.get_color(), linestyle='-.', alpha=0.3)
        ax.plot(short_arm_lengths * 1e3, polynomial_derivatives_equality_metastable[i, :], color=na_line.get_color(), linestyle=':', alpha=0.3)
        # ax.plot(short_arm_lengths * 1e3, zero_derivative_points[i, :], color=na_line.get_color(), linestyle='-.')
    ax2.plot(short_arm_lengths * 1e3, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Mirror-to-spherical = {long_arm_lengths[i]*100:.2f}cm")
ax2.set_ylim(0, 300)
ax2.set_ylabel("Mode Spacing [MHz]")
ax.set_ylabel('Short Arm Numerical Aperture')
ax.axvline(collimation_point * 1e3, color='k', linestyle='--', linewidth=1, label='Collimation point')
ax.set_ylim(0, 0.2)
ax.grid()
if plot_different_axes:
    ax2.set_xlabel('Short Arm Length (mm)')
    ax2.axvline(collimation_point * 1e3, color='k', linestyle='--', linewidth=1)
    ax2.grid()
    fig.subplots_adjust(right=0.68)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    ax.set_title(f"aspheric = {cavity[1].name}, spherical lens = {cavity[2].name}")
else:
    ax.set_xlabel('Short Arm Length (mm)')
    fig.subplots_adjust(right=0.68)
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), borderaxespad=0.0)
    plt.title(f"spherical focal length = {focal_length_of_lens_object(cavity[2]) * 1000:.0f} mm")
obsidian_path=get_obsidian_save_path(filename='NA as a function of mirrors - eksma 20mm lens.svg')
plt.savefig(obsidian_path)
plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(len(long_arm_lengths)):
#     ax.plot(NAs[i, :], mode_spacings[i, :] / 1e6, label=f"Long arm={long_arm_lengths[i]*1e2:.2f}cm")
# ax.set_xlabel("NA")
# ax.set_ylabel("Mode spacing [MHz]")
# ax.legend()
# ax.set_yscale('log')
# ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1, 2, 3, 4, 5, 6, 7, 8, 9], numticks=15))
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.grid()
# plt.tight_layout()
# plt.show()


# %% mode spacing and NA:
# short_arm_lengths = np.array([7.35e-3, 7.45e-3, 7.55e-3, 7.65e-3])
# NAs = np.linspace(0.03, 0.15, 100)
# long_arm_lengths = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
# mode_spacings = np.full(shape=(len(short_arm_lengths), len(NAs)), fill_value=np.nan)
# for i, short_arm_length in enumerate(short_arm_lengths):
#     optical_system = OpticalSystem.from_params(params=params[0:-1], use_paraxial_ray_tracing=True, lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True,)
#     optical_system.place_elements(elements=optical_system[1], position=short_arm_length * RIGHT, reference_center=optical_system[0])
#     for j, NA in enumerate(NAs):
#         try:
#             cavity=optical_system.complete_to_cavity(NA=NA, end_mirror_ROC=2e-1)
#             long_arm_lengths[i, j] = cavity.surfaces[-1].center[0] - cavity.surfaces[2].center[0]
#             mode_spacings[i, j] = cavity.mode_spacing_transversal_apparent
#         except ValueError:
#             continue
#
#
# plt.close('all')
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()
# for i in range(len(short_arm_lengths)):
#     ax1.plot(long_arm_lengths[i, :]*100, NAs, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - NA")
#     ax2.plot(long_arm_lengths[i, :]*100, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm - df")
# ax1.legend()
# ax1.grid()
# ax1.set_xlabel("Large mirror to aspheric distance [cm]")
# ax1.set_ylabel("NA")
# ax2.set_ylabel("Mode spacing [MHz]")
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(len(short_arm_lengths)):
#     ax.plot(NAs, mode_spacings[i, :] / 1e6, label=f"Short arm={short_arm_lengths[i]*1e3:.2f}mm")
# ax.set_xlabel("NA")
# ax.set_ylabel("Mode spacing [MHz]")
# ax.legend()
# ax.grid()
# plt.tight_layout()
# plt.show()


# %% Long arm perturbation
# perturbations_large_mirror = np.linspace(-4e-2, 4e-2, 100)
#
# NAs = np.zeros_like(perturbations_large_mirror)
# long_arm_lengths = np.zeros_like(perturbations_large_mirror)
# for i, perturbation_value in enumerate(perturbations_large_mirror):
#     perturbation_pointer = PerturbationPointer(element_index=2, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
#     perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
#     NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
#     long_arm_lengths[i] = perturbed_cavity.arms[2].central_line.length
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(long_arm_lengths * 1e2, NAs, marker='o', markersize=2)
# ax.set_xlabel('Long arm length (cm)')
# ax.set_ylabel('Short Arm Numerical Aperture')
# ax.set_title('Short Arm Numerical Aperture as a Function of Long Arm Length')
# ax.grid()
# plt.tight_layout()
# # plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
# plt.show()

# %% Short arm perturbation
# perturbations_aspheric_lens = np.linspace(-5e-5, 2e-4, 100)
#
# NAs = np.zeros_like(perturbations_aspheric_lens)
# short_arm_lengths = np.zeros_like(perturbations_aspheric_lens)
# for i, perturbation_value in enumerate(perturbations_aspheric_lens):
#     perturbation_pointer = PerturbationPointer(element_index=0, parameter_name=ParamsNames.x, perturbation_value=perturbation_value)
#     perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointer)
#     NAs[i] = perturbed_cavity.arms[0].mode_parameters.NA[0]
#     short_arm_lengths[i] = perturbed_cavity.arms[0].central_line.length
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(short_arm_lengths * 1e2, NAs, marker='o', markersize=2)
# ax.set_xlabel('Short arm length (cm)')
# ax.set_ylabel('Short Arm Numerical Aperture')
# ax.set_title('Short Arm Numerical Aperture as a Function of Small arm length')
# ax.grid()
# plt.tight_layout()
# # plt.savefig(r'figures\NA_as_a_function_of_small_mirror')
# plt.show()
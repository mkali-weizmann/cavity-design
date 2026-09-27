from matplotlib import use
use("TkAgg")

from cavity_design import *
from scipy.optimize import brentq
from tqdm import tqdm


elements=[LASER_OPTIK_MIRROR, EDMUND_4MM_ASPHERIC_16701, COASTLINE_20CM_MIRROR]

# %% 2d map:
def equality_equation(x, coef):
    quad_deriv = 2 * coef[1] * x
    higher_deriv = sum(2 * n * coef[n] * x ** (2 * n - 1) for n in range(2, len(coef)))
    return abs(quad_deriv) - abs(higher_deriv)

cavity = Cavity(elements=elements, standing_wave=True, lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True, use_paraxial_ray_tracing=False, set_central_line=True, set_mode_parameters=True)
short_arm_length_collimation = cavity[0].radius + back_focal_length_of_lens_object(lens_object=cavity[1])
long_arm_lengths = np.arange(34e-2, 66e-2, 4e-2)#  np.array([38e-2])
mid_arm_length = 1.6e-2
short_arm_lengths = np.linspace(short_arm_length_collimation, short_arm_length_collimation + 2e-4, 200)  # np.linspace(7.32e-3, 7.35e-3, 500)#

spherical_lens_nominal_position = short_arm_length_collimation + cavity[1].T_c + mid_arm_length # Measured from small mirror.
Large_to_small_mirror_minus_long_arm_length_nominal = short_arm_length_collimation + cavity[1].T_c if len(elements) == 3\
    else spherical_lens_nominal_position + cavity[2].T_c

PHI_MAX_POTENTIAL = np.arcsin(0.3)


def stable_branch_state(short_arm_length):
    """NA, discriminant, the plotted root, and the fold value, at one short arm length.

    The stable branch solves 3*p3*u**2 + 2*p2*u - p1 = 0 with u = NA**2. Its two roots merge and vanish where the
    discriminant b**2 + 4*a*c changes sign, so that sign is what brackets the fold."""
    cavity.place_element(
        element=cavity[1], position=short_arm_length * RIGHT, reference_center=cavity[0], recalculate_optic=True
    )
    NA = cavity.arms[0].mode_parameters.NA[0]
    if np.isnan(NA):
        return np.nan, np.nan, np.nan, np.nan
    coef = analyze_potential_given_cavity(
        cavity=cavity, n_rays=50, phi_max=PHI_MAX_POTENTIAL, print_tests=False
    )["polynomial_residuals_mirror"].coef
    a, b, c = 3 * coef[3], 2 * coef[2], coef[1]
    discriminant = b**2 + 4 * a * c
    u_fold = -b / (2 * a)
    x_fold = np.sqrt(u_fold) if u_fold > 0 else np.nan
    if discriminant < 0:
        return NA, discriminant, np.nan, x_fold
    u = (-b + np.sqrt(discriminant)) / (2 * a)
    return NA, discriminant, (np.sqrt(u) if u > 0 else np.nan), x_fold


NAs = np.zeros(shape=(len(long_arm_lengths), len(short_arm_lengths)))
mode_spacings = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
# zero_derivative_points = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
polynomial_derivatives_equality_stable = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
polynomial_derivatives_equality_metastable = np.full(shape=(len(long_arm_lengths), len(short_arm_lengths)), fill_value=np.nan)
focii_to_lens = np.zeros(shape=(len(long_arm_lengths)))
edge_refinements = []

if len(elements) > 3:
    cavity.place_element(element=cavity[2], position = spherical_lens_nominal_position * RIGHT, reference_center=cavity[0], recalculate_optic=False)

for i, long_arm_length in tqdm(enumerate(long_arm_lengths)):
    cavity.place_element(element=cavity[-1], position=(long_arm_length + Large_to_small_mirror_minus_long_arm_length_nominal) * RIGHT, reference_center=cavity.elements[0],
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

    # The stable branch is born in a fold near the right edge of the stability window and then drops steeply, so
    # its crossing with the mode NA is narrower than one step of `short_arm_lengths` (0.14um vs 1um at 38cm) and
    # the scan above steps straight over it. Root-find the fold and the crossing instead of refining the grid.
    edge_refinements.append(None)
    stable_short_arm_lengths = short_arm_lengths[~np.isnan(NAs[i, :])]
    if len(stable_short_arm_lengths) == 0:
        continue
    short_arm_length_edge = stable_short_arm_lengths[-1]
    step = short_arm_lengths[1] - short_arm_lengths[0]
    short_arm_length_before_fold = None
    for k in range(1, 12):
        candidate = short_arm_length_edge - k * step
        NA_candidate, discriminant_candidate, _, _ = stable_branch_state(candidate)
        if np.isnan(NA_candidate):
            break
        if discriminant_candidate < 0:
            short_arm_length_before_fold = candidate
            break
    if short_arm_length_before_fold is None:
        continue
    short_arm_length_fold = brentq(
        lambda short_arm_length: stable_branch_state(short_arm_length)[1],
        short_arm_length_before_fold, short_arm_length_edge, xtol=1e-12,
    )
    NA_at_fold, _, _, NA_fold = stable_branch_state(short_arm_length_fold)

    # brentq can land a hair on the negative-discriminant side of the fold, where the root is still nan.
    short_arm_lengths_fine = np.linspace(
        short_arm_length_fold + (short_arm_length_edge - short_arm_length_fold) * 1e-4, short_arm_length_edge, 80
    )
    NAs_fine = np.full_like(short_arm_lengths_fine, np.nan)
    stable_roots_fine = np.full_like(short_arm_lengths_fine, np.nan)
    for k, short_arm_length in enumerate(short_arm_lengths_fine):
        NAs_fine[k], _, stable_roots_fine[k], _ = stable_branch_state(short_arm_length)

    def stable_branch_minus_NA(short_arm_length):
        NA, _, stable_root, _ = stable_branch_state(short_arm_length)
        return stable_root - NA

    if (stable_roots_fine[0] - NAs_fine[0]) > 0 > (stable_roots_fine[-1] - NAs_fine[-1]):
        short_arm_length_crossing = brentq(
            stable_branch_minus_NA, short_arm_lengths_fine[0], short_arm_lengths_fine[-1], xtol=1e-12,
        )
        NA_crossing = stable_branch_state(short_arm_length_crossing)[0]
    else:
        short_arm_length_crossing, NA_crossing = None, None

    edge_refinements[i] = {
        "short_arm_length_fold": short_arm_length_fold, "NA_fold": NA_fold, "NA_at_fold": NA_at_fold,
        "short_arm_lengths_fine": short_arm_lengths_fine, "NAs_fine": NAs_fine,
        "stable_roots_fine": stable_roots_fine,
        "short_arm_length_crossing": short_arm_length_crossing, "NA_crossing": NA_crossing,
    }
    print(f"\nLong arm {long_arm_length * 1e2:.2f}cm: stable branch born at "
          f"{short_arm_length_fold * 1e3:.7f}mm with NA={NA_fold:.4f} (mode NA there {NA_at_fold:.4f})")
    if short_arm_length_crossing is None:
        print("  born below the mode NA -> no crossing")
    else:
        print(f"  crosses the mode NA at {short_arm_length_crossing * 1e3:.7f}mm, NA={NA_crossing:.4f} "
              f"({(short_arm_length_crossing - short_arm_length_fold) * 1e6:.3f}um past the fold)")

# %% First plot
LABEL_FONTSIZE = 20  # 2x the matplotlib default of 10
TICK_FONTSIZE = 20  # 2x the matplotlib default of 10
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 13  # The NA legend carries one entry per long arm plus four extras, so it does not fit at 20.
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
    # The fold and the crossing live inside a single step of the scan grid, so draw that stretch from the
    # root-found refinement instead - otherwise both curves are sampled straight past each other.
    refinement = edge_refinements[i]
    if refinement is not None:
        ax.plot(refinement["short_arm_lengths_fine"] * 1e3, refinement["NAs_fine"], color=na_line.get_color(), linewidth=1.2)
        ax.plot(refinement["short_arm_lengths_fine"] * 1e3, refinement["stable_roots_fine"], color=na_line.get_color(), linestyle='-.', alpha=0.6)
        if refinement["short_arm_length_crossing"] is not None:
            ax.plot(refinement["short_arm_length_crossing"] * 1e3, refinement["NA_crossing"], marker='*',
                    markersize=18, color=na_line.get_color(), markeredgecolor='k', markeredgewidth=0.8, linestyle='',
                    label="Stable branch meets mode NA" if i == 0 else None)
    ax2.plot(short_arm_lengths * 1e3, mode_spacings[i, :] / 1e6, linestyle='--', label=f"Mirror-to-spherical = {long_arm_lengths[i]*100:.2f}cm")
ax2.set_ylim(0, 300)
ax2.set_ylabel("Mode Spacing [MHz]", fontsize=LABEL_FONTSIZE)
ax.set_ylabel('Short Arm Numerical Aperture', fontsize=LABEL_FONTSIZE)
ax.tick_params(labelsize=TICK_FONTSIZE)
ax2.tick_params(labelsize=TICK_FONTSIZE)
ax.axvline(short_arm_length_collimation * 1e3, color='k', linestyle='--', linewidth=1, label='Collimation point')
ax.set_ylim(0, 0.23)
ax.grid()
if plot_different_axes:
    ax2.set_xlabel('Short Arm Length (mm)', fontsize=LABEL_FONTSIZE)
    ax2.axvline(short_arm_length_collimation * 1e3, color='k', linestyle='--', linewidth=1)
    ax2.grid()
    fig.subplots_adjust(right=0.70)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=LEGEND_FONTSIZE)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=LEGEND_FONTSIZE)
    ax.set_title(f"aspheric = {cavity[1].name}, spherical lens = {cavity[2].name if len(elements) == 4 else None}", fontsize=TITLE_FONTSIZE)
else:
    ax.set_xlabel('Short Arm Length (mm)', fontsize=LABEL_FONTSIZE)
    fig.subplots_adjust(right=0.68)
    ax.legend(loc='center left', bbox_to_anchor=(1.12, 0.5), borderaxespad=0.0, fontsize=LABEL_FONTSIZE)
    plt.title(f"spherical focal length = {focal_length_of_lens_object(cavity[2]) * 1000:.0f} mm", fontsize=TITLE_FONTSIZE)
# obsidian_path=get_obsidian_save_path(filename='NA as a function of mirrors - 16701 - no spherical.svg')
# plt.savefig(obsidian_path)
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
# %%
from matplotlib import use
use("QtAgg")  # for interactive plotting in Spyder; version-agnostic so it picks up PySide6
from cavity_design import *
from scipy.optimize import brentq
from tqdm import tqdm

# Automates the "Single aspheric lens cavity - Existing elements" widget of analyze_potential.ipynb: for every short
# arm length, find the long arm length at which the x^4 coefficient of the mirror deviation fit vanishes, and read the
# x^2 coefficient and the ground state energy there.
aspheric_lens_model = EDMUND_4MM_ASPHERIC_16701
n_rays = 50
max_NA_for_polynomial = 0.3
long_arm_lengths_coarse = np.linspace(0.02, 0.6, 59)  # The widget's slider range. The roots are bracketed on this grid.

cavity_temp = Cavity(elements=[LASER_OPTIK_MIRROR, aspheric_lens_model, COASTLINE_20CM_MIRROR],
                     standing_wave=True, use_paraxial_ray_tracing=False, p_is_trivial=True, t_is_trivial=True,
                     lambda_0_laser=LAMBDA_0_LASER)
collimation_point = cavity_temp[0].radius + back_focal_length_of_lens_object(lens_object=aspheric_lens_model)
short_arm_lengths = collimation_point + np.linspace(-1e-4, 3e-4, 41)  # = collimation point + the widget's defocus


def cavity_and_mirror_polynomial(long_arm_length):
    cavity = cavity_temp.set_NA_by_moving_last_mirror(end_mirror_distance_to_last_element=long_arm_length)
    results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=np.arcsin(max_NA_for_polynomial),
                                                  print_tests=False, potential_horizontal_axis_in_NAs=True)
    return cavity, results_dict["polynomial_residuals_mirror"].coef  # A polynomial in NA**2: coef[1] is x^2, coef[2] is x^4.


def fourth_order_coefficient(long_arm_length):
    return cavity_and_mirror_polynomial(long_arm_length)[1][2]


# a_4 crosses zero twice along the long arm: going - -> + at tens of cm (only past some defocus), and + -> - at a few
# cm. Only the first is kept, so a short arm length has at most one root (none when that crossing is not in the window).
roots = []
for short_arm_length in tqdm(short_arm_lengths):
    cavity_temp.place_element(element=cavity_temp[1], position=short_arm_length * RIGHT,
                              recalculate_optic=False, reference_center=cavity_temp[0])
    a_4_coarse = np.array([fourth_order_coefficient(long_arm_length) for long_arm_length in long_arm_lengths_coarse])
    rising_crossings = np.nonzero((a_4_coarse[:-1] < 0) & (a_4_coarse[1:] > 0))[0]
    for k in rising_crossings:
        long_arm_length_root = brentq(fourth_order_coefficient, long_arm_lengths_coarse[k],
                                      long_arm_lengths_coarse[k + 1], xtol=1e-9)
        cavity, coefficients = cavity_and_mirror_polynomial(long_arm_length_root)
        NA_short_arm = cavity.arms[0].mode_parameters.NA[0]
        # nan NA means the cavity is paraxially unstable there - the root exists, but there is no mode.
        ground_state_energy = np.nan if np.isnan(NA_short_arm) else np.abs(energy_level(cavity=cavity))
        roots.append({"short_arm_length": short_arm_length, "long_arm_length": long_arm_length_root,
                      "a_2": coefficients[1], "a_4": coefficients[2], "NA_short_arm": NA_short_arm,
                      "ground_state_energy": ground_state_energy})

for root in roots:
    print(f"short arm = {root['short_arm_length'] * 1e3:.5f} mm (defocus {(root['short_arm_length'] - collimation_point) * 1e6:+.1f} um): "
          f"long arm = {root['long_arm_length'] * 1e2:.3f} cm, a_2 = {root['a_2']:.3e}, NA short arm = {root['NA_short_arm']:.3f}, "
          f"ground state energy = {root['ground_state_energy'] * 1e9:.2f} nm")

# %% Plot
short_arm_lengths_roots = np.array([root["short_arm_length"] for root in roots])
stable = ~np.isnan([root["NA_short_arm"] for root in roots])
quantities = [("long_arm_length", 1e2, "Long arm length at $a_4=0$ [cm]"),
              ("a_2", 1, "$a_2$ (x$^2$ coefficient) at $a_4=0$"),
              ("NA_short_arm", 1, "Short arm NA at $a_4=0$"),
              ("ground_state_energy", 1e9, "Ground state energy at $a_4=0$ [nm]")]

fig, axes = plt.subplots(len(quantities), 1, figsize=(10, 12), sharex=True)
for ax, (key, scale, label) in zip(axes, quantities):
    values = np.array([root[key] for root in roots]) * scale
    ax.plot(short_arm_lengths_roots[stable] * 1e3, values[stable], "o", label="Stable cavity")
    ax.set_ylabel(label)
    ax.axvline(collimation_point * 1e3, color="k", linestyle="--", linewidth=1)
    ax.grid()
# Roots where the cavity has no mode are shown only here: their a_2 is orders of magnitude off and would squash
# the a_2 panel, and NA / energy are nan there anyway.
axes[0].plot(short_arm_lengths_roots[~stable] * 1e3, np.array([root["long_arm_length"] for root in roots])[~stable] * 1e2,
             "x", color="gray", label="Unstable cavity (no mode)")
axes[1].axhline(0, color="k", linewidth=0.8)
axes[0].legend()
axes[0].set_title(f"{aspheric_lens_model.name}: where the x$^4$ term of the mirror deviation vanishes\n"
                  f"(fit up to NA={max_NA_for_polynomial}, dashed = collimation point)")
axes[-1].set_xlabel("Short arm length [mm]")
fig.tight_layout()
plt.show()

from cavity import *
from matplotlib import ticker

lambda_0_laser = 1064e-9

# Original lens
params = np.array([[ 3.0791359873e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5039269433e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
          [ 6.4567993648e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+1.j,  5.4883362661e-03+0.j,  2.4224615887e-02+0.j,  1.0000000000e+00+0.j,  2.9135987295e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
          [-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000387360e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

cavity_0 = Cavity.from_params(params=params,
                              standing_wave=True,
                              lambda_0_laser=lambda_0_laser,
                              names=['left mirror', 'lens-left', 'lens_right', 'right mirror'],
                              set_central_line=True,
                              set_mode_parameters=True,
                              set_initial_surface=False,
                              t_is_trivial=True,
                              p_is_trivial=True,
                              power=2e4)

PERTURBATION_ELEMENT_INDEX = 2  # 2 for the left mirror, 1 for lens
PERTURBATION_VALUE = 1e-6
PERTURBATION_PARAMETER = 'theta'
CORRECTION_PARAMETER = 'theta'

perturbed_params = params.copy()
perturbed_params[PERTURBATION_ELEMENT_INDEX, INDICES_DICT[PERTURBATION_PARAMETER]] += PERTURBATION_VALUE

perturbed_cavity = perturb_cavity(cavity_0,
                                  (PERTURBATION_ELEMENT_INDEX, INDICES_DICT[PERTURBATION_PARAMETER]),
                                  PERTURBATION_VALUE)

cavity_0.plot()
plt.show()


def overlap(cavity):
    overlap = np.abs(calculate_cavities_overlap(cavity_0, cavity))
    return overlap


corrected_cavity = find_required_perturbation_for_desired_change(cavity=perturbed_cavity,
                                                                 parameter_index_to_change=(
                                                                 0, INDICES_DICT[CORRECTION_PARAMETER]),
                                                                 desired_parameter=overlap,
                                                                 desired_value=1,
                                                                 x0=0,
                                                                 xtol=1e-10)

print(f"perturbation_value: {getattr(corrected_cavity.to_params[2], PERTURBATION_PARAMETER):.2e}, "
      f"correction_value: {getattr(corrected_cavity.to_params[0], CORRECTION_PARAMETER)}")

print(f"Perturbed overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity)):.2e}, "
      f"Corrected_overlap: {1-np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity)):.2e}")


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
plt.suptitle('Overlap between cavities')
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=perturbed_cavity, ax=ax[0])
plot_2_cavity_perturbation_overlap(cavity=cavity_0, second_cavity=corrected_cavity, ax=ax[1])
ax[0].set_title(f'Perturbed cavity, 1-overlap = {1-np.abs(calculate_cavities_overlap(cavity_0, perturbed_cavity)):.2e}')
ax[1].set_title(f'Corrected cavity, 1-overlap = {1-np.abs(calculate_cavities_overlap(cavity_0, corrected_cavity)):.2e}')
plt.tight_layout()
plt.show()

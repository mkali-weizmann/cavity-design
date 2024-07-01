import time

import numpy as np

from cavity import *

lambda_0_laser = 1064e-9
params = np.array([[-5.0000000000e-03 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, -0.0000000000e+00 - 1.j,
                    5.0000387360e-03 + 0.j, np.nan + 0.j, 1.0000000000e+00 + 0.j, np.nan + 0.j, 1.0000000000e+00 + 0.j,
                    0.0000000000e+00 + 0.j, 1.0000000000e+00 + 0.j, np.nan + 0.j, 7.5000000000e-08 + 0.j,
                    1.0000000000e-06 + 0.j, 1.3100000000e+00 + 0.j, np.nan + 0.j, 1.7000000000e-01 + 0.j, np.nan + 0.j,
                    9.9988900000e-01 + 0.j, 1.0000000000e-04 + 0.j, np.nan + 0.j, 0.0000000000e+00 + 0.j],
                   [6.4567993648e-03 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j,
                    2.4224615887e-02 + 0.j, 5.4883362661e-03 + 0.j, 1.0000000000e+00 + 0.j, 2.9135987295e-03 + 0.j,
                    1.7600000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 1.0000000000e+00 + 0.j, np.nan + 0.j, np.nan + 0.j,
                    np.nan + 0.j, np.nan + 0.j, np.nan + 0.j, np.nan + 0.j, np.nan + 0.j, np.nan + 0.j, np.nan + 0.j,
                    np.nan + 0.j, 1.0000000000e+00 + 0.j],
                   [3.0791359873e-01 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j,
                    1.5039269433e-01 + 0.j, np.nan + 0.j, 1.0000000000e+00 + 0.j, np.nan + 0.j, 1.0000000000e+00 + 0.j,
                    0.0000000000e+00 + 0.j, 1.0000000000e+00 + 0.j, np.nan + 0.j, 7.5000000000e-08 + 0.j,
                    1.0000000000e-06 + 0.j, 1.3100000000e+00 + 0.j, np.nan + 0.j, 1.7000000000e-01 + 0.j, np.nan + 0.j,
                    9.9988900000e-01 + 0.j, 1.0000000000e-04 + 0.j, np.nan + 0.j, 0.0000000000e+00 + 0.j]])
cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['left mirror', 'lens-left', 'lens_right', 'right mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,

                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4)
perturbed_cavity = perturb_cavity(cavity, (0, INDICES_DICT['z']), shift_value=1e-5)
# %%
N_resolution = 3
range_limit = 1e-4
zoom_factor = 1.9
center = np.array([0.0005382701814471093, -0.1086493299834081, 0, 0])
N_iterations = 50
fig, ax = plt.subplots(N_iterations, 3, figsize=(24, N_iterations*3))

# def find_central_line(
for i in range(N_iterations):
    print("center: ", center)
    print("range: ", range_limit)
    initial_parameters = generate_initial_parameters_grid(center,
                                                          range_limit,
                                                          N_resolution,
                                                          p_is_trivial=True,
                                                          t_is_trivial=False)
    diff = perturbed_cavity.f_roots(initial_parameters)
    # diff[..., 0] *= 100
    diff_norm = np.linalg.norm(diff, axis=-1)

    smallest_elements_index = np.unravel_index(np.argmin(diff_norm), diff.shape[:-1])
    center = initial_parameters[smallest_elements_index]
    range_limit /= zoom_factor
    print(f"{i}: min diff: ", np.min(diff_norm), end='\n\n')
    ax[i, 0].imshow(diff_norm)
    # plt.colorbar()
    # Add a dot at the minimum:
    ax[i, 0].scatter(smallest_elements_index[1], smallest_elements_index[0], color='r')

    diff_1d = diff_norm[smallest_elements_index[0], :]
    ax[i, 1].plot(initial_parameters[0, :, 1], diff_1d)
    ax[i, 1].axvline(initial_parameters[0, smallest_elements_index[1], 1], color='r')
    ax[i, 1].set_title(f'{i}: t')

    diff_1d = diff_norm[:, smallest_elements_index[1]]
    ax[i, 2].plot(initial_parameters[:, 0, 0], diff_1d)
    ax[i, 2].axvline(initial_parameters[smallest_elements_index[0], 0, 0], color='r')
    ax[i, 2].set_title(f'{i}: x')

# fig_1.show()
# fig_2.show()
# fig_3.show()




# initial_parameters_x_shifted = np.array([1e-10, 0, 0, 0])
# initial_parameters_y_shifted = np.array([0, 0, 1e-10, 0])
# initial_parameters_t_shifted = np.array([0, 1e-10, 0, 0])
# initial_parameters_p_shifted = np.array([0, 0, 0, 1e-10])
# initial_parameters_diverging = np.array([0, 1, 0, 0])
# initial_parameters_trivial = np.array([0, 0, 0, 0])
# initial_parameters = np.stack([initial_parameters_x_shifted,
#                                 initial_parameters_t_shifted,
#                                 initial_parameters_y_shifted,
#                                 initial_parameters_p_shifted,
# initial_parameters_diverging,
# initial_parameters_trivial
# ], axis=0)
# %%
# ray_history = cavity.trace_ray(ray=initial_ray)
# final_position_and_angles, ray_history_parametric = cavity.trace_ray_parametric(initial_parameters)
# diff = cavity.f_roots(initial_parameters)
# diff_norm = np.linalg.norm(diff, axis=-1)
#
# smallest_elements_xytp = np.unravel_index(np.argmin(np.linalg.norm(diff, axis=-1)), diff.shape[:-1])
# %%
# diff_reduced_norm = diff_norm[:, 4, :, 4]

# smallest_elements_xytp = np.unravel_index(np.nanargmin(diff_reduced_norm), diff_reduced_norm.shape)

plt.imshow(diff_norm)
plt.colorbar()
plt.show()
# %%
#
diff_1d = diff_norm[3, :]

plt.plot(diff_1d)
plt.title('t')
plt.show()

# %%
d = 5e-10
ts = np.linspace(-d, d, 101)
initial_parameters = np.repeat(ts, 4).reshape(101, 4)
initial_parameters[:, [0, 2, 3]] = 0

cavity.f_roots(initial_parameters)

# %%
d = 2e-11
norms = np.zeros(100)
ss = np.linspace(-d, d, 100)
for i, s in enumerate(ss):
    initial_parameters = np.array([0, s, 0, 0])
    diff = perturbed_cavity.f_roots(initial_parameters)
    diff_norm = np.linalg.norm(diff, axis=-1)
    norms[i] = diff_norm

plt.plot(ss, norms)
plt.title('P')
plt.show()

from cavity import *

lambda_0_laser = 1064e-9
params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000387360e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
          [ 6.4567993648e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  2.4224615887e-02+0.j,  5.4883362661e-03+0.j,  1.0000000000e+00+0.j,  2.9135987295e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
          [ 3.0791359873e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5039269433e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

# swap first and third rows of params:
params[[0, 2]] = params[[2, 0]]
params[1, [4, 5]] = params[1, [5, 4]]
params[1, 3] += 1j


cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['left mirror', 'lens-left', 'lens_right',  'right mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4)
# plot_mirror_lens_mirror_cavity_analysis(cavity)
#
# ax = plt.gca()
# ax.set_xlim(-0.05, 0.35)
#
# for NA in np.logspace(-3, np.log10(0.01), 5):
#     beginning_ray = Ray(origin=np.array([0, 0, 0]), k_vector=np.array([np.cos(np.arcsin(NA)), NA, 0]))
#     second_ray = cavity.physical_surfaces[1].reflect_ray(beginning_ray)
#     third_ray = cavity.physical_surfaces[2].reflect_ray(second_ray)
#     fourth_ray = cavity.physical_surfaces[3].reflect_ray(third_ray)
#     fifth_ray = cavity.physical_surfaces_ordered[4].reflect_ray(fourth_ray)
#     sixth_ray = cavity.physical_surfaces_ordered[5].reflect_ray(fifth_ray)
#
#     beginning_ray.plot(ax=ax, color='r')
#     second_ray.plot(ax=ax, color='g')
#     third_ray.plot(ax=ax, color='b')
#     fourth_ray.plot(ax=ax, color='k')
#     fifth_ray.plot(ax=ax, color='b')
#     sixth_ray.plot(ax=ax, color='b')
#
# # plt.xlim(0.05, 0.25)
# plt.ylim(-1e-4, 1e-4)
#
# plt.show()

# plt.show()

# %%
tolerance_matrix = cavity.generate_tolerance_matrix(print_progress=True)

# # %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                 shift_size=30,
                                                 print_progress=True)
# # %%
cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                arm_index_for_NA=2)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
cavity.specs(print_specs=True, contracted=False)


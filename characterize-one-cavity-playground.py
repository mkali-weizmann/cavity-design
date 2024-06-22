from cavity import *

lambda_0_laser = 1064e-9
# params = np.array([[-5.00000000e-03+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, -0.00000000e+00-1.j,  5.00006326e-03+0.j, np.nan+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j, np.nan+0.j,  7.50000000e-08+0.j,  1.00000000e-06+0.j,  1.31000000e+00+0.j,        np.nan+0.j,  1.70000000e-01+0.j,  0.00000000e+00+0.j,  9.99889000e-01+0.j,  1.00000000e-04+0.j, np.nan+0.j,  0.00000000e+00+0.j],
#           [ 5.00000000e-03+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+1.j,  2.28050832e-02+0.j, np.nan+0.j,  1.00000000e+00+0.j,  0.00000000e+00+0.j,  1.76000000e+00+0.j,  0.00000000e+00+0.j, -1.00000000e+00+0.j,  1.76000000e+00+0.j,  5.50000000e-06+0.j,  1.00000000e-06+0.j,  4.60600000e+01+0.j,  1.17000000e-05+0.j,  3.00000000e-01+0.j,  1.00000000e-02+0.j,  1.00000000e-04+0.j,  9.99899000e-01+0.j, np.nan+0.j,  2.00000000e+00+0.j],
#           [ 7.68128933e-03+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  6.23756826e-03+0.j, np.nan+0.j,  1.76000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j,  1.76000000e+00+0.j,        np.nan+0.j, np.nan+0.j, np.nan+0.j,        np.nan+0.j, np.nan+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, np.nan+0.j,  2.00000000e+00+0.j],
#           [ 3.07681289e-01+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.50391711e-01+0.j, np.nan+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j, np.nan+0.j,  7.50000000e-08+0.j,  1.00000000e-06+0.j,  1.31000000e+00+0.j,        np.nan+0.j,  1.70000000e-01+0.j,  0.00000000e+00+0.j,  9.99889000e-01+0.j,  1.00000000e-04+0.j, np.nan+0.j,  0.00000000e+00+0.j]])
params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000632553e-03+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
          [ 6.3393015143e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  2.3479617901e-02+0.j,  6.2124542986e-03+0.j,  1.0000000000e+00+0.j,  2.6786030286e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,   np.nan+0.j,          np.nan+0.j,   np.nan+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
          [ 3.0767860303e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5038941544e-01+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,   np.nan+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])
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
plot_mirror_lens_mirror_cavity_analysis(cavity)

plt.show()



# %%
tolerance_matrix = cavity.generate_tolerance_matrix(print_progress=False)

# # %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                 shift_size=30,
                                                 print_progress=False)
# # %%
cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                arm_index_for_NA=2)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
# # plt.show()
# %%
cavity.specs(print_specs=True, contracted=False)


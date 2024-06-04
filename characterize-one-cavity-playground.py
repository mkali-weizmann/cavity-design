from cavity import *
b = np.array([[-5.00000000e-03+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, -0.00000000e+00-1.j,  5.00058899e-03+0.j, np.nan+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j, np.nan+0.j,  7.50000000e-08+0.j,  1.00000000e-06+0.j,  1.31000000e+00+0.j,        np.nan+0.j,  1.70000000e-01+0.j,  0.00000000e+00+0.j,  9.99889000e-01+0.j,  1.00000000e-04+0.j, np.nan+0.j,  0.00000000e+00+0.j],
          [ 1.12776050e-02+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+1.j,  1.00046500e-02+0.j, np.nan+0.j,  1.00000000e+00+0.j,  0.00000000e+00+0.j,  1.45500000e+00+0.j,  0.00000000e+00+0.j, -1.00000000e+00+0.j,  1.45500000e+00+0.j,  4.80000000e-07+0.j,  1.00000000e-06+0.j,  1.38000000e+00+0.j,  1.20000000e-05+0.j,  1.50000000e-01+0.j,  1.00000000e-03+0.j,  1.00000000e-04+0.j,  9.99899000e-01+0.j, np.nan+0.j,  2.00000000e+00+0.j],
          [ 1.38394828e-02+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00039460e-02+0.j, np.nan+0.j,  1.45500000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j,  1.45500000e+00+0.j,  4.80000000e-07+0.j,  1.00000000e-06+0.j,  1.38000000e+00+0.j,  1.20000000e-05+0.j,  1.50000000e-01+0.j,  1.00000000e-03+0.j,  1.00000000e-04+0.j,  9.99899000e-01+0.j, np.nan+0.j,  2.00000000e+00+0.j],
          [ 3.16305967e-01+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.51649353e-01+0.j, np.nan+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j, np.nan+0.j,  7.50000000e-08+0.j,  1.00000000e-06+0.j,  1.31000000e+00+0.j,        np.nan+0.j,  1.70000000e-01+0.j,  0.00000000e+00+0.j,  9.99889000e-01+0.j,  1.00000000e-04+0.j, np.nan+0.j,  0.00000000e+00+0.j]])

cavity = Cavity.from_params(params=b,
                            standing_wave=True,
                            lambda_laser=1064e-9,
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

# %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                 shift_size=30,
                                                 print_progress=False)

# %%
cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                arm_index_for_NA=2)
# plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
#     dpi=300, bbox_inches='tight')
plt.show()
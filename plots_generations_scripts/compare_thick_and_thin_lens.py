from cavity import *
# %%

for w in ['3mm', '4mm', '8mm']:
    params = params_dict[f'Sapphire, NA=0.2, L1=0.3, w={w} - High NA axis']

    mirror_on_waist = True
    auto_set_axes = 1.0000000000e+00
    axis_span = None
    camera_center = -1
    lambda_laser = 1064e-9
    names = ['Right Mirror', 'lens', 'Left Mirror']

    cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_laser=lambda_laser, names=names, t_is_trivial=True, p_is_trivial=True)

    title = f"short arm NA={cavity.arms[2].local_mode_parameters.NA[0]:.2e}, short arm length = {np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[3].center):.2e} [m]\n" + \
            f"long arm NA={cavity.arms[0].local_mode_parameters.NA[0]:.2e},   long arm length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} [m] w={w}"

    # fig, ax = plt.subplots(figsize=(13, 5))
    # cavity.plot(axis_span=axis_span, camera_center=camera_center, ax=ax, plane='xz')  #
    # ax.set_title(title)
    # plt.savefig(
    #     f'figures/systems/mirror-lens-mirror_high_NA_ratio_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} w={w} {systems_names[i]}.svg',
    #     dpi=300, bbox_inches='tight')
    # plt.show()
    # %%
    tolerance_matrix = cavity.generate_tolerance_matrix(print_progress=False)

    # %%
    overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                     shift_size=30,
                                                     print_progress=False)

    # %%
    cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                    arm_index_for_NA=2)
    plt.suptitle(title)
    # plt.savefig(f'figures/NA tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
    #     dpi=300, bbox_inches='tight')
    plt.show()
# %% Calculate the minimal thickness of a lens:
# params = params_dict[f'Sapphire, NA=0.2, L1=0.3, w=4.8mm - High NA axis']
# params = np.insert(params, -1, 0, axis=1)
params = np.array([[ 3.0695707324e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5001380651e-01+0.j,  0.0000000000e+00+0.j,
         0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
         1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9989900000e-01+0.j,  1.0000000000e-04+0.j,
                      np.nan+0.j,  0.0000000000e+00+0.j],
       [ 4.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+1.j,  1.5166312489e-02+0.j,  1.7600000000e+00+0.j,
         5.9000000000e-03+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,
         4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,
                      np.nan+0.j,  1.0000000000e+00+0.j],
       [-1.7378000000e-02+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  9.2140077806e-03+0.j,  0.0000000000e+00+0.j,
         0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,
         1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  9.9989900000e-01+0.j,  1.0000000000e-04+0.j,
                      np.nan+0.j,  0.0000000000e+00+0.j]])
lambda_laser = 1064e-9
names = ['Right Mirror', 'lens', 'Left Mirror']

cavity = Cavity.from_params(params=params, standing_wave=True,
                            lambda_laser=lambda_laser, names=names, t_is_trivial=True, p_is_trivial=True)
cavity.plot()
plt.xlim(-0.025, 0.01)
R = np.real(params[1, INDICES_DICT['r']])
spot_size = cavity.arms[0].mode_parameters_on_surface_1.spot_size(cavity.lambda_laser)[0]
h = 3 * spot_size
d = R * (1 - np.sqrt(1- h**2 / R**2))
alpha = np.arcsin(h / R)
d_alternative = R * (1 - np.cos(alpha))
print(f"minimal thickness when allowing sharp edges: {2 * d}\nminimal thickness when adding one milimeter of edge thickness: {2 * d + 1e-3}")



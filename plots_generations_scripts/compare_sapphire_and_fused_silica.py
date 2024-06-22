from cavity import *


# %% ### Comparing sapphire and fused silica lenses, lengths measured from len's surfaces:
with open('data/params_dict.pkl', 'rb') as f:
    params_dict = pkl.load(f)

lenses_types = ['Sapphire', 'Fused Silica']
for i, params_name in enumerate(['Sapphire, NA=0.2, L1=0.3, w=3mm - High NA axis',  'Fused Silica, NA=0.2, L1=0.3, w=3mm - High NA axis']):
    w='3mm'
    params = params_dict[params_name]


    mirror_on_waist = True
    auto_set_axes = 1.0000000000e+00
    axis_span = None
    camera_center = -1
    lambda_0_laser = 1064e-9
    names = ['Right Mirror', 'lens', 'Left Mirror']

    cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_0_laser=lambda_0_laser, names=names, t_is_trivial=True, p_is_trivial=True)

    # fig, ax = plt.subplots(figsize=(13, 5))
    # cavity.plot(axis_span=axis_span, camera_center=camera_center, ax=ax, plane='xz')  #
    # ax.set_xlim(x_3 - 0.01, x_1 + 0.01)
    # ax.set_ylim(-0.002, 0.002)
    title = f"short arm NA={cavity.arms[2].mode_parameters.NA[0]:.2e}, short arm length = {np.linalg.norm(cavity.surfaces[2].center - cavity.surfaces[3].center):.2e} [m]\n" + \
            f"long arm NA={cavity.arms[0].mode_parameters.NA[0]:.2e},   long arm length = {np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} [m] {lenses_types[i]}"

    # ax.set_title(title)
    # plt.savefig(
        # f'figures/systems/mirror-lens-mirror_high_NA_ratio_NA_1_{cavity.arms[0].mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} w={w}.svg',
        # dpi=300, bbox_inches='tight')
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
    # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} w={w}.svg',
        # dpi=300, bbox_inches='tight')
    plt.show()

# %% Comparing sapphire and fused silica lenses, lengths measured from len's center:
params_list = [np.array([[ 3.0391834335e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.4923071669e-01+0.j,  0.0000000000e+00+0.j,
     0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.0000000000e-08+0.j,  1.0000000000e-06+0.j,
     1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j],
   [ 4.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+1.j,  1.3646319612e-02+0.j,  1.7600000000e+00+0.j,
     2.9964809220e-03+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,
     4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  3.0000000000e-02+0.j,  1.0000000000e+00+0.j],
   [-1.5179523539e-02+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  8.8406496482e-03+0.j,  0.0000000000e+00+0.j,
     0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.0000000000e-08+0.j,  1.0000000000e-06+0.j,
     1.3100000000e+00+0.j,  0.0000000000e+00+0.j,  1.7000000000e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j]]),
               np.array(
                   [[3.0395758172e-01 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j,
                     1.4925034392e-01 + 0.j, 0.0000000000e+00 + 0.j,
                     0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 1.0000000000e+00 + 0.j,
                     1.0000000000e-08 + 0.j, 1.0000000000e-06 + 0.j,
                     1.3100000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 1.7000000000e-01 + 0.j, 0.0000000000e+00 + 0.j,
                     0.0000000000e+00 + 0.j],
                    [4.0000000000e-03 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 1.j,
                     8.1699108961e-03 + 0.j, 1.4550000000e+00 + 0.j,
                     2.9964809220e-03 + 0.j, 1.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j,
                     4.8000000000e-07 + 0.j, 1.0000000000e-06 + 0.j,
                     1.3800000000e+00 + 0.j, 1.2000000000e-05 + 0.j, 1.5000000000e-01 + 0.j, 1.0000000000e-03 + 0.j,
                     1.0000000000e+00 + 0.j],
                    [-1.5179523539e-02 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j,
                     -0.0000000000e+00 - 1.j,
                     8.8406496482e-03 + 0.j, 0.0000000000e+00 + 0.j,
                     0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 1.0000000000e+00 + 0.j,
                     1.0000000000e-08 + 0.j, 1.0000000000e-06 + 0.j,
                     1.3100000000e+00 + 0.j, 0.0000000000e+00 + 0.j, 1.7000000000e-01 + 0.j, 0.0000000000e+00 + 0.j,
                     0.0000000000e+00 + 0.j]])
               ]
systems_names = ["Sapphire", "Fused Silica"]
for i, params in enumerate(params_list):
    w='3mm'
    # params = params_dict[params_name]


    mirror_on_waist = True
    auto_set_axes = 1.0000000000e+00
    axis_span = None
    camera_center = -1
    lambda_0_laser = 1064e-9
    names = ['Right Mirror', 'lens', 'Left Mirror']

    cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_0_laser=lambda_0_laser, names=names, t_is_trivial=True, p_is_trivial=True)

    # fig, ax = plt.subplots(figsize=(13, 5))
    # cavity.plot(axis_span=axis_span, camera_center=camera_center, ax=ax, plane='xz')  #
    # ax.set_xlim(x_3 - 0.01, x_1 + 0.01)
    # ax.set_ylim(-0.002, 0.002)
    title = f"short arm NA={cavity.arms[2].mode_parameters.NA[0]:.2e}, short arm length = {np.linalg.norm(params[1, 0] - cavity.surfaces[3].center):.2e} [m] {systems_names[i]}\n" + \
            f"long arm NA={cavity.arms[0].mode_parameters.NA[0]:.2e},   long arm length = {np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} [m] w={w}"

    # ax.set_title(title)
    # plt.savefig(
        # f'figures/systems/mirror-lens-mirror_high_NA_ratio_NA_1_{cavity.arms[0].mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(cavity.surfaces[1].center - cavity.surfaces[0].center):.2e} w={w} {systems_names[i]}.svg',
        # dpi=300, bbox_inches='tight')
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
    # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
        # dpi=300, bbox_inches='tight')
    plt.show()
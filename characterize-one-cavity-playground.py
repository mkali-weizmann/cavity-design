from cavity import *

lambda_0_laser = 1064e-9


# Old cavity params in array new format:
                    #  x                    y                    t                   p                    r_1                  r_2                 n_outside_or_before  T_c                  n_inside_or_after    z                    curvature_sign       material_refractive_index alpha_expansion      beta_surface_absorption kappa_conductivity   dn_dT                nu_poisson_ratio    alpha_volume_absorption  intensity_reflectivity   intensity_transmittance   temperature surface_type
# params = np.array([[ 3.06163358e-01+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, 0.00000000e+00+0.j,  1.50009986e-01+0.j,  0,                  1,                   0.00000000e+00+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j,  1.00000000e+00+0.j,  0,                        7.50000000e-08+0.j,  1.00000000e-06+0.j,     1.31000000e+00+0.j,  0.00000000e+00+0.j,  1.70000000e-01+0.j, 0.00000000e+00+0.j,      0,                        0,                        0,          0],
#                    [ 4.00000000e-03+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, 0.00000000e+00+1.j,  1.56186512e-02+0.j,  1.56186512e-02+0.j, 1,                   4.33012068e-03+0.j,  1.76,                0.00000000e+00+0.j,  -1,                  1.76,                     5.50000000e-06+0.j,  1.00000000e-06+0.j,     4.60600000e+01+0.j,  1.17000000e-05+0.j,  3.00000000e-01+0.j, 3.00000000e-02+0.j,      0,                        0,                        0,          1],
#                    [-1.81677697e-02+0.j,  0.00000000e+00+0.j,  0.00000000e+00+0.j, -0.00000000e+00-1.j, 1.00013618e-02+0.j,  0.00000000e+00+0.j, 1,                   0.00000000e+00+0.j,  0.00000000e+00+0.j,  0,                   1.00000000e+00+0.j,  0,                        7.50000000e-08+0.j,  1.00000000e-06+0.j,     1.31000000e+00+0.j,  0.00000000e+00+0.j,  1.70000000e-01+0.j, 0.00000000e+00+0.j,      0,                        0,                        0,          0]])


# Old cavity params in new format:
# params = [OpticalElementParams(surface_type=0.0, x=3.0616335818e-01,  y=0.0, z=0.0, t=0.0, p=0.0,   r_1=0.15000998609,  r_2=np.nan,          curvature_sign=1.0, T_c=np.nan,         n_inside_or_after=1.0,  n_outside_or_before=1.0, material_properties=MaterialProperties(refractive_index=np.nan, alpha_expansion=7.5e-08, beta_surface_absorption=1e-06, kappa_conductivity=1.31, dn_dT=np.nan, nu_poisson_ratio=0.17, alpha_volume_absorption=np.nan, intensity_reflectivity=0.999889, intensity_transmittance=0.0001, temperature=np.nan)),
#           OpticalElementParams(surface_type=1.0, x=4.0000000000e-03,  y=0.0, z=0.0, t=0.0, p=np.pi, r_1=0.015618651198, r_2=0.015618651198, curvature_sign=1.0, T_c=4.33012068e-03, n_inside_or_after=1.76, n_outside_or_before=1.0, material_properties=MaterialProperties(refractive_index=np.nan, alpha_expansion=np.nan, beta_surface_absorption=np.nan, kappa_conductivity=np.nan, dn_dT=np.nan, nu_poisson_ratio=np.nan, alpha_volume_absorption=np.nan, intensity_reflectivity=np.nan, intensity_transmittance=np.nan, temperature=np.nan)),
#           OpticalElementParams(surface_type=0.0, x=-1.8167769661e-02, y=0.0, z=0.0, t=0.0, p=np.pi, r_1=0.010001361829, r_2=np.nan,          curvature_sign=1.0, T_c=np.nan,         n_inside_or_after=1.0,  n_outside_or_before=1.0, material_properties=MaterialProperties(refractive_index=np.nan, alpha_expansion=7.5e-08, beta_surface_absorption=1e-06, kappa_conductivity=1.31, dn_dT=np.nan, nu_poisson_ratio=0.17, alpha_volume_absorption=np.nan, intensity_reflectivity=0.999889, intensity_transmittance=0.0001, temperature=np.nan))]


# New cavity params:
params = np.array([[-5.0000000000e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j, -0.0000000000e+00-1.j,  5.0000281233e-03+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j],
                   [ 6.5057257992e-03+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  7.9679319133e-03+0.j,  7.9679319133e-03+0.j,  1.0000000000e+00+0.j,  3.0114515984e-03+0.j,  1.7600000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,  1.7600000000e+00+0.j,  5.5000000000e-06+0.j,  1.0000000000e-06+0.j,  4.6060000000e+01+0.j,  1.1700000000e-05+0.j,  3.0000000000e-01+0.j,  1.0000000000e-02+0.j,  1.0000000000e-04+0.j,  9.9989900000e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j],
                   [ 3.0801145160e-01+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.5039504639e-01+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  1.0000000000e+00+0.j,  0.0000000000e+00+0.j,  1.0000000000e+00+0.j,   np.nan+0.j,  7.5000000000e-08+0.j,  1.0000000000e-06+0.j,  1.3100000000e+00+0.j,   np.nan+0.j,  1.7000000000e-01+0.j,   np.nan+0.j,  9.9988900000e-01+0.j,  1.0000000000e-04+0.j,   np.nan+0.j,  0.0000000000e+00+0.j]])

# # swap first and third rows of params:
params[[0, 2]] = params[[2, 0]]
params[1, [4, 5]] = params[1, [5, 4]]
params[1, 3] += 1j


cavity = Cavity.from_params(params=params,
                            standing_wave=True,
                            lambda_0_laser=lambda_0_laser,
                            names=['Right mirror', 'Lens', 'Left Mirror'],
                            set_central_line=True,
                            set_mode_parameters=True,
                            set_initial_surface=False,
                            t_is_trivial=True,
                            p_is_trivial=True,
                            power=2e4, use_brute_force_for_central_line=True)
cavity.plot()
# plot_mirror_lens_mirror_cavity_analysis(cavity)
plt.show()
# %%
for shift_value in np.logspace(-6, -6, 1):
    print(f"\nshift_value: {shift_value:.2e}")
    perturbed_cavity = perturb_cavity(cavity, (0, 3), shift_value)


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
plt.show()

# plt.show()

# %%
tolerance_matrix = cavity.generate_tolerance_matrix()

# %%
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :]),
                                                 shift_size=50,
                                                 )
# # %%
cavity.generate_overlaps_graphs(overlaps_series=overlaps_series, tolerance_matrix=tolerance_matrix[:, :],
                                arm_index_for_NA=2)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
cavity.specs(print_specs=True, contracted=False)


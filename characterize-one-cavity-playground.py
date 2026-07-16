from cavity_design import *
# from simple_analysis_scripts.potential_analysis.analyze_potential import energy_level, analyze_potential_given_cavity, plot_results
# np.seterr(all="raise")
# %%
cavity_paraxial = Cavity(
    elements=[
        LASER_OPTIK_MIRROR.to_position(np.array([-0.005, 0.0, 0.0])),
        EKSMA_LENS_20MM_ASPHERIC.to_position(np.array([0.017623230771841976, 0.0, 0.0])),
        DUMMY_LENS.to_position(np.array([0.03305723077184197, 0.0, 0.0])),
        CurvedMirror(name='End mirror', radius=0.2, outwards_normal=np.array([1.0, -0.0, -0.0]), center=np.array([0.4511688875799871, 0.0, 0.0]), curvature_sign=-1, diameter=0.0254, material_properties=MaterialProperties(refractive_index=1.45, alpha_expansion=5.2e-07, beta_surface_absorption=1e-06, kappa_conductivity=1.38, dn_dT=1.2e-05, nu_poisson_ratio=0.16, alpha_volume_absorption=0.001, intensity_reflectivity=0.0001, intensity_transmittance=0.999899, temperature=np.nan)),
    ],
    standing_wave=True,     lambda_0_laser=1.064e-06,     t_is_trivial=True,     p_is_trivial=True,     use_paraxial_ray_tracing=False,
)
# %%
plot_mirror_lens_mirror_cavity_analysis(cavity_paraxial)
plt.xlim([-6e-3, 305e-3])
plt.show()
# %%
perturbable_params_names=['x', 'y', 'phi']
tolerance_df = cavity_paraxial.generate_tolerance_dataframe(perturbable_params_names=perturbable_params_names)
tolerance_matrix = tolerance_df.to_numpy()
## %%
overlaps_series = cavity_paraxial.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix),
                                                 shift_numel=50,
                                                 perturbable_params_names=perturbable_params_names,)
## %%
cavity_paraxial.generate_overlaps_graphs(arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
                                overlaps_series=overlaps_series, perturbable_params_names=perturbable_params_names)
# # plt.savefig(f'figures/NA-tolerance/mirror-lens-mirror_high_NA_ratio_smart_choice_tolerance_NA_1_{cavity.arms[0].local_mode_parameters.NA[0]:.2e}_L_1_{np.linalg.norm(params[1, 0] - cavity.surfaces[0].center):.2e} w={w}.svg',
# #     dpi=300, bbox_inches='tight')
plt.show()
# %%
cavity_non_paraxial = copy.deepcopy(cavity_paraxial)
cavity_non_paraxial.use_paraxial_ray_tracing = False

print(2*energy_level(cavity_non_paraxial))
results_dict = analyze_potential_given_cavity(cavity=cavity_non_paraxial, n_rays=30, phi_max=0.2)
plot_results(results_dict=results_dict, far_away_plane=True)
plt.show()




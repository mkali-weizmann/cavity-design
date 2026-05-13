from simple_analysis_scripts.potential_analysis.analyze_potential import *
back_focal_length_aspheric = 5.9000000000e-03
back_focal_length_aspheric_fine = 0.0000000000e+00
defocus = -1.7347234760e-18
defocus_fine = -1.7763568394e-15
spherical_aspherical_distance = 5.0000000000e-03
desired_focus = 2.0000000000e-01
mirror_setting_mode = 'Set NA'
unconcentricity = 5.0000000000e-04
NA_small_arm = 1.5000000000e-01
max_NA_for_polynomial = 3.0000000000e-01
n_rays = 50
T_c_aspheric = 3.6000000000e-03
T_c_spherical = 4.3500000000e-03
n_aspheric_design = 1.5800000000e+00
n_aspheric_actual = 1.5700000000e+00
n_spherical = 1.5100000000e+00
diameter = 1.2700000000e-02

back_focal_length_aspheric += widget_convenient_exponent(back_focal_length_aspheric_fine)
defocus = 0  # += widget_convenient_exponent(defocus_fine)
unconcentricity = widget_convenient_exponent(unconcentricity)

cavity = generate_two_positive_lenses_cavity(defocus=defocus, back_focal_length_aspheric=back_focal_length_aspheric,
                                             T_c_aspheric=T_c_aspheric, n_aspheric_design=n_aspheric_design,
                                             n_aspheric_actual=n_aspheric_actual, n_spherical=n_spherical,
                                             T_c_spherical=T_c_spherical, unconcentricity=unconcentricity,
                                             NA_small_arm=NA_small_arm, mirror_setting_mode=mirror_setting_mode,
                                             diameter=diameter,
                                             spherical_aspherical_distance=spherical_aspherical_distance,
                                             desired_focus=desired_focus)
results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=np.arcsin(max_NA_for_polynomial),
                                              print_tests=False, )
unconcentricity = np.nan if mirror_setting_mode == 'set NA' else unconcentricity
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True, unconcentricity=unconcentricity)
ax[1].set_title(
    f"f_spherical={focal_length_of_lens(R_1=results_dict['cavity'].surfaces[3].radius, R_2=-results_dict['cavity'].surfaces[3].radius, T_c=results_dict['cavity'].surfaces[4].center[0] - results_dict['cavity'].surfaces[3].center[0], n=results_dict['cavity'].surfaces[3].n_2) * 1e3:.3f} mm, "
    f"EFL aspheric = {focal_length_of_lens(R_1=np.inf, R_2=-results_dict['cavity'].surfaces[2].radius, T_c=results_dict['cavity'].surfaces[2].center[0] - results_dict['cavity'].surfaces[1].center[0], n=results_dict['cavity'].surfaces[1].n_2) * 1e3:.3f}, "
    f"BFL aspheric = {back_focal_length_aspheric * 1e3:.3f} unconcentricity={unconcentricity * 1e3:.3f} mm\n"
    f"waist to lens left={results_dict['cavity'].surfaces[1].center[0]:.2e}, waist to lens right={results_dict['cavity'].mode_parameters[4].center[0, 0] - results_dict['cavity'].surfaces[4].center[0]:.2e}")
ax[1].set_ylim(-0.01, 0.01)

fig.tight_layout()
plt.show()
print(results_dict['ray_sequence'][2].k_vector)
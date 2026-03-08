from simple_analysis_scripts.potential_analysis.analyze_potential import *
dn = 0
lens_types = ["aspheric - lab", "spherical - like labs aspheric", "avantier", "aspheric - like avantier"]
lens_type = lens_types[2]
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(
    lens_type=lens_type, dn=dn)
n_rays = 400
unconcentricity = 2.24255506e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.04
desired_focus = 200e-3
print_tests = True

defocus = choose_source_position_for_desired_focus_analytic(
    desired_focus=desired_focus,
    T_c=T_c,
    n_design=n_design,
    diameter=diameter,
    back_focal_length=back_focal_length,
    R_1=R_1,
    R_2=R_2_signed,
)

optical_system, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed,
                                                                back_focal_length=back_focal_length,
                                                                defocus=defocus, T_c=T_c, n_design=n_design,
                                                                diameter=diameter, n_actual=n_actual, )
rays_0 = initialize_rays(n_rays=n_rays, phi_max=phi_max)
results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity,
                                 print_tests=print_tests)
assert np.isclose(
    np.abs(results_dict["zero_derivative_points"] * 1e3), 0.15342637331775477
), f"Potential single lens test failed: expected zero derivative point at approximately 0.15342637331775477 mm but got {results_dict['zero_derivative_points']*1e3} mm"

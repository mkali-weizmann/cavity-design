from simple_analysis_scripts.potential_analysis.analyze_potential import *

copy_input_parameters = False
copy_cavity_parameters = False
copy_image = False
eval_box = ''
first_arm_NA = 1.5000000000e-01
n_rays = 30
max_NA_for_polynomial = 2.5000000000e-01
R_left = 7.2900000000e-03
R_right = 4.3600000000e-03
T_c = 4.0000000000e-03
waist_to_lens = 5.0000000000e-03
a_2_left = 0.0000000000e+00
a_4_left = -6.2470000000e+00
a_6_left = 0.0000000000e+00
spherical_right = False
a_2_right = 0.0000000000e+00
a_4_right = -6.2400000000e+00
a_6_right = 0.0000000000e+00

OPTICAL_AXIS = RIGHT
diameter = 7.75e-3
a_2_left = widget_convenient_exponent(a_2_left, scale=0)
a_4_left = widget_convenient_exponent(a_4_left, scale=0)
a_6_left = widget_convenient_exponent(a_6_left, scale=0)
laseroptik_mirror = CurvedMirror(radius=5e-3, outwards_normal=LEFT, center=np.array([-0.00499995, 0, 0]),
                                 curvature_sign=CurvatureSigns.concave, diameter=7.75e-3, name="LaserOptik mirror",
                                 material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"])
polynomial_coefficients_back = np.array(
    [0, a_2_left, a_4_left, a_6_left])  # widget_convenient_exponent(-2.470e-1,scale=0)
lens_left = AsphericRefractiveSurface.pseudo_spherical(radius=R_left,
                                                       polynomial_coefficients=polynomial_coefficients_back,
                                                       center=np.array([waist_to_lens, 0, 0]),
                                                       n_1=1,
                                                       n_2=1.45,
                                                       outwards_normal=-OPTICAL_AXIS,
                                                       name="Lens aspherical back",
                                                       material_properties=PHYSICAL_SIZES_DICT[
                                                           "material_properties_fused_silica"],
                                                       thickness=T_c / 2,
                                                       diameter=diameter,
                                                       curvature_sign=CurvatureSigns.convex
                                                       )
a_2_right = widget_convenient_exponent(a_2_right, scale=0)
a_4_right = widget_convenient_exponent(a_4_right, scale=0)
a_6_right = widget_convenient_exponent(a_6_right, scale=0)
polynomial_coefficients_front = np.array(
    [0, a_2_right, a_4_right, a_6_right])  # widget_convenient_exponent(-2.470e-1,scale=0)
lens_right = AsphericRefractiveSurface.pseudo_spherical(radius=R_right,
                                                        polynomial_coefficients=polynomial_coefficients_front,
                                                        center=lens_left.center + np.array([T_c, 0, 0]),
                                                        n_1=1.45,
                                                        n_2=1,
                                                        outwards_normal=OPTICAL_AXIS,
                                                        name="Lens aspherical front",
                                                        material_properties=PHYSICAL_SIZES_DICT[
                                                            "material_properties_fused_silica"],
                                                        thickness=T_c / 2,
                                                        diameter=diameter,
                                                        curvature_sign=CurvatureSigns.concave
                                                        )

optical_system = OpticalSystem(surfaces=[laseroptik_mirror, lens_left, lens_right],
                               lambda_0_laser=LAMBDA_0_LASER, t_is_trivial=True, p_is_trivial=True,
                               use_paraxial_ray_tracing=False)
cavity = optical_system_to_cavity_completion(optical_system=optical_system, end_mirror_ROC=0.286876428009918,
                                             end_mirror_distance_to_last_element=0.56984127)  # , NA=first_arm_NA
params_dummy = cavity.to_params
cavity_dummy = Cavity.from_params(params_dummy, lambda_0_laser=LAMBDA_0_LASER, t_is_trivial=True, p_is_trivial=True,
                                  use_paraxial_ray_tracing=False)

results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays, phi_max=np.arcsin(max_NA_for_polynomial),
                                              print_tests=False)
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True, fig_and_ax=(fig, ax))
ax[1].grid()
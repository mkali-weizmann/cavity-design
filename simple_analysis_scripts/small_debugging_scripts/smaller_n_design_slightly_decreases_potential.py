# from matplotlib import use
# use('TkAgg')
from cavity_design.analyze_potential import *

back_focal_length_aspheric = 5.9000000000e-03
defocuses = [5.805e-4, 2.95e-4]
n_aspheric_actuals = [1.56, 1.58]
desired_focus = 2.0000000000e-01
NA_small_arm = 1.5000000000e-01
max_NA_for_polynomial = 3.0000000000e-01
n_rays = 10
T_c_aspheric = 3.6000000000e-03
n_aspheric_design = 1.5800000000e+00
diameter = 1.2700000000e-02
OPTICAL_AXIS = RIGHT
# back_center = back_focal_length_aspheric * OPTICAL_AXIS # WHY DOESN'T IT WORK?!?!
# aspheric_flat, aspheric_curved = Surface.from_params(
#         generate_aspheric_lens_params(
#             back_focal_length=back_focal_length_aspheric,
#             T_c=T_c_aspheric,
#             n=n_aspheric_design,
#             forward_normal=OPTICAL_AXIS,
#             flat_faces_center=back_center,
#             diameter=diameter,
#             polynomial_degree=16,
#             name="aspheric_lens_automatic",
#         )
#     )
#
# aspheric_flat.n_2 = n_aspheric_actual
# aspheric_curved.n_1 = n_aspheric_actual
#
# defocus = choose_source_position_for_desired_focus_analytic(T_c=T_c_aspheric, R_1=np.inf, R_2=-aspheric_curved.radius, n=n_aspheric_actual, back_focal_length = None, desired_focus=200e-3, diameter = None)
# print(defocus)
for i in range(len(defocuses)):
    defocus = defocuses[i]
    n_aspheric_actual = n_aspheric_actuals[i]
    back_center = (back_focal_length_aspheric + defocus) * OPTICAL_AXIS
    aspheric_flat, aspheric_curved = Surface.from_params(
            generate_aspheric_lens_params(
                back_focal_length=back_focal_length_aspheric,
                T_c=T_c_aspheric,
                n=n_aspheric_design,
                forward_normal=OPTICAL_AXIS,
                flat_faces_center=back_center,
                diameter=diameter,
                polynomial_degree=16,
                name="aspheric lens",
            )
        )
    aspheric_flat.n_2 = n_aspheric_actual
    aspheric_curved.n_1 = n_aspheric_actual


    optical_system = OpticalSystem(surfaces=[aspheric_flat, aspheric_curved], t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=False)
    optical_system_with_small_mirror = OpticalSystem(surfaces=[LASER_OPTIK_MIRROR, *optical_system.surfaces],
                                                         t_is_trivial=True, p_is_trivial=True,
                                                         use_paraxial_ray_tracing=False, lambda_0_laser=LAMBDA_0_LASER)
    cavity = optical_system_to_cavity_completion(optical_system=optical_system_with_small_mirror, NA=NA_small_arm, unconcentricity=None,
                                                 end_mirror_ROC=2e-1)

    results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=n_rays,
                                                  phi_max=np.arcsin(max_NA_for_polynomial), print_tests=False)
    print(cavity.surfaces[-1].center[0], defocus, results_dict['residual_distances_opposite'][-1], n_aspheric_actual)
    plt.close('all')
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    plot_results(results_dict=results_dict, fig_and_ax=(fig, ax))

    plt.show()

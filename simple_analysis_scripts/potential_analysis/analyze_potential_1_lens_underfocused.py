from matplotlib import use
# use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from simple_analysis_scripts.potential_analysis.analyze_potential import *

lens_type = 'aspheric - lab'
desired_focus = -0.2
n_rays = 50
phi_max = 1.4800000000e-01
n_actual_spherical = 1.4500000000e+00
mirror_setting_mode = 'Set ROC'
first_arm_NA = 1.5200000000e-01
right_mirror_ROC = 0.5
right_mirror_distance_to_lens_front = 5.1300000000e-02
right_mirror_ROC_fine = 4.6081573018e-11
right_mirror_ROC_fine = widget_convenient_exponent(right_mirror_ROC_fine, scale=-10)

if mirror_setting_mode == "Set ROC":
    right_mirror_distance_to_lens_front = None
    right_mirror_ROC += right_mirror_ROC_fine
elif mirror_setting_mode == "Set distance to spherical":
    right_mirror_ROC = None
    right_mirror_distance_to_lens_front += right_mirror_ROC_fine

n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type=lens_type, dn=0)
defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed)
optical_system_lens, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual)
mirror_left = CurvedMirror(radius=5e-3, outwards_normal=LEFT, origin=ORIGIN, curvature_sign=CurvatureSigns.concave, name="LaserOptik mirror", diameter=7.75e-3, material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"])
optical_system_without_large_mirror = OpticalSystem.from_params(params=[mirror_left.to_params, *optical_system_lens.params],
                                                    lambda_0_laser=LAMBDA_0_LASER, p_is_trivial=True, t_is_trivial=True,
                                                    use_paraxial_ray_tracing=True)
cavity = fixed_NA_cavity_generator(optical_system=optical_system_without_large_mirror,
                                   NA=first_arm_NA,
                                   end_mirror_ROC=right_mirror_ROC,
                                   end_mirror_distance_to_last_element=right_mirror_distance_to_lens_front,)

plot_mirror_lens_mirror_cavity_analysis(cavity=cavity, CA=12.7e-3)
plt.show()

tolerance_df = cavity.generate_tolerance_dataframe()
tolerance_matrix = tolerance_df.to_numpy()
overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix), shift_numel=50)
cavity.generate_overlaps_graphs(arm_index_for_NA=0, tolerance_dataframe=tolerance_df,
                                overlaps_series=overlaps_series)
plt.show()

# %% An old version that sets unconcentricity instead of NA
# n_actual = 1.8000000000e+00
# dn = 0.0000000000e+00
# n_rays = 300
# unconcentricity = 3.0000000000e-03
# phi_max = 1.5000000000e-01
# desired_focus = -0.1297
# end_mirror_ROC = 0.5
# T_c = 4.3500000000e-03
# diameter = 1.2700000000e-02
# plot = True
# lens_type = 'aspheric - lab'
# copy_input_parameters = True
# copy_cavity_parameters = False
# eval_box = ''
# copy_image = False
#
# n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type, dn)
# defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,)
# optical_system_lens, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual, )
# rays_0 = initialize_rays(n_rays=n_rays, phi_max=phi_max)
# results_dict = analyze_potential(optical_system=optical_system_lens, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=False, end_mirror_ROC=end_mirror_ROC)
#
# fig, ax = plot_results(results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After aspheric surface"])
# center = results_dict["center_of_curvature"]
# plt.suptitle(
#     f"lens_type={lens_type}, desired_focus = {desired_focus:.3e}m, n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {back_focal_length * 1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus * 1e3:.1f} mm, T_c: {T_c * 1e3:.1f} mm, Diameter: {diameter * 1e3:.2f} mm"
# )
# plt.show()
# print(results_dict['cavity'])
from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system
from simple_analysis_scripts.potential_analysis.analyze_potential import *



n_actual = 1.8000000000e+00
dn = 0.0000000000e+00
n_rays = 30
unconcentricity = 3.0000000000e-03
phi_max = 2.5000000000e-01
desired_focus = -40e-02
T_c = 4.3500000000e-03
diameter = 1.2700000000e-02
plot = True
lens_type = 'aspheric - lab'
copy_input_parameters = True
copy_cavity_parameters = False
eval_box = ''
copy_image = False

n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type, dn)
defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,)
optical_system, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual, )
rays_0 = initialize_rays(n_rays=n_rays, phi_max=phi_max)
results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=False, end_mirror_ROC=0.5)



fig, ax = plot_results(results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After aspheric surface"])
center = results_dict["center_of_curvature"]
plt.suptitle(
    f"lens_type={lens_type}, desired_focus = {desired_focus:.3e}m, n_design: {n_design:.3f}, n_actual: {n_actual:.3f}, Lens focal length: {back_focal_length * 1e3:.1f} mm, Defocus: z_lens -> z_lens + {defocus * 1e3:.1f} mm, T_c: {T_c * 1e3:.1f} mm, Diameter: {diameter * 1e3:.2f} mm"
)
plt.show()
# %%
results_dict['cavity'].params
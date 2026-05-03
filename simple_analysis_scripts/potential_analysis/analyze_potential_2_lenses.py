from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system

from simple_analysis_scripts.potential_analysis.analyze_potential import *

# %%

copy_input_parameters = True
copy_cavity_parameters = False
copy_image = False
eval_box = ''
n_actual_aspheric = 1.4500000000e+00
n_rays = 50
unconcentricity = 5.0451000000e+00
max_NA_for_polynomial = 2.5000000000e-01
defocus = 4.7300000000e-04
back_focal_length_aspheric = 7.9000000000e-03
T_c_aspheric = 4.4500000000e-03
n_design_aspheric = 1.4500000000e+00
n_design_spherical = 1.45000000000e+00
n_actual_spherical = 1.45000000000e+00
T_c_spherical = 4.4500000000e-03
f_spherical = 1.0000000000e-01
diameter = 1.2700000000e-02

unconcentricity = widget_convenient_exponent(unconcentricity)
optical_system = generate_two_lenses_optical_system(defocus=defocus, back_focal_length_aspheric=back_focal_length_aspheric, T_c_aspheric=T_c_aspheric, n_design_aspheric=n_design_aspheric, n_actual_aspheric=n_actual_aspheric, n_design_spherical=n_design_spherical, n_actual_spherical=n_actual_spherical, T_c_spherical=T_c_spherical, f_spherical=f_spherical, diameter=diameter,)
rays_0 = initialize_rays(n_rays=n_rays, phi_max=np.arcsin(max_NA_for_polynomial))
results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=False, end_mirror_ROC=2e-1)
fig, ax = plot_results(results_dict=results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False)
ax[1].set_title(f"f_spherical={f_spherical*1e3:.3f} mm, unconcentricity={unconcentricity*1e3:.3f} mm, waist to lens left={results_dict['cavity'].surfaces[1].center[0]:.2e}, waist to lens right={results_dict['cavity'].mode_parameters[4].center[0, 0] - results_dict['cavity'].surfaces[4].center[0]:.2e}")
ax[1].set_ylim(-0.01, 0.01)
fig.tight_layout()
plt.show()
# %% GAMES

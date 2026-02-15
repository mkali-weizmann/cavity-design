from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system

from simple_analysis_scripts.potential_analysis.analyze_potential import *

# %%

plot = True
print_tests = True
n_rays = 50
unconcentricity = -1e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.2
defocus = 0.0029887489470528557
back_focal_length_aspheric = 20e-3
T_c_aspheric = 4.35e-3
n_design_aspheric = 1.45
n_actual_aspheric = 1.45
n_design_spherical = 1.45
n_actual_spherical = 1.45
T_c_spherical = 4.35e-3
f_spherical = 100e-3
diameter = 12.7e-3

optical_system = generate_two_lenses_optical_system(defocus=defocus, back_focal_length_aspheric=back_focal_length_aspheric, T_c_aspheric=T_c_aspheric, n_design_aspheric=n_design_aspheric, n_actual_aspheric=n_actual_aspheric, n_design_spherical=n_design_spherical, n_actual_spherical=n_actual_spherical, T_c_spherical=T_c_spherical, f_spherical=f_spherical, diameter=diameter,)
rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)
results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=True)
plot_results(results_dict=results_dict, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False)
plt.show()

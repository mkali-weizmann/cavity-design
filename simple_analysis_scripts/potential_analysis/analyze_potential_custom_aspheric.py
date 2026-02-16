from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc
from simple_analysis_scripts.potential_analysis.analyze_potential import *

# %%
plot = True
print_tests = True
n_rays = 15
unconcentricity = 1e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.05
OPTICAL_AXIS = RIGHT
back_focal_length_aspheric = 20e-3
T_c_aspheric = 4.35e-3
n_design_aspheric = 1.45
n_actual_aspheric = 1.45
diameter = 12.7e-3
defocus = -0.002
back_center = (back_focal_length_aspheric - defocus) * OPTICAL_AXIS

aspheric_flat, aspheric_front = Surface.from_params(
        generate_aspheric_lens_params(
            back_focal_length=back_focal_length_aspheric,
            T_c=T_c_aspheric,
            n=n_design_aspheric,
            forward_normal=OPTICAL_AXIS,
            flat_faces_center=back_center,
            diameter=diameter,
            polynomial_degree=8,
            name="aspheric_lens_automatic",
        )
    )

r_spherical = 200e-3
spherical_back_alternative = CurvedRefractiveSurface(radius=r_spherical,
                                                     curvature_sign=CurvatureSigns.convex,
                                                     outwards_normal=-OPTICAL_AXIS,
                                                     center=aspheric_flat.center, n_1=1, n_2=1.45,
                                                     name="spherical_lens_alternative",
                                                     material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
                                                     thickness=T_c_aspheric / 2,
                                                     diameter=diameter,)
polynomial_coefficients_back = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # widget_convenient_exponent(-2.470e-1,scale=0)
outwards_normal = -OPTICAL_AXIS * np.sign(polynomial_coefficients_back[1]) if polynomial_coefficients_back[1] != 0 else -OPTICAL_AXIS
if polynomial_coefficients_back[1] < 0:
    polynomial_coefficients_back[1] *= -1
aspherical_back_alternative = AsphericRefractiveSurface(polynomial_coefficients=polynomial_coefficients_back,
                                                        center=aspheric_flat.center,
                                                        n_1=1,
                                                        n_2=1.45,
                                                        outwards_normal=outwards_normal,
                                                        name="aspherical_lens_alternative",
                                                        material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
                                                        thickness=T_c_aspheric / 2,
                                                        diameter=diameter,
                                                        curvature_sign=outwards_normal @ OPTICAL_AXIS
                                                        )

optical_system_original = OpticalSystem(surfaces=[aspheric_flat, aspheric_front],
                                       t_is_trivial=True,
                                       p_is_trivial=True,
                                       given_initial_central_line=True,
                                       use_paraxial_ray_tracing=False,)

optical_system_spherical_back = OpticalSystem(surfaces=[spherical_back_alternative, aspheric_front],
                                              t_is_trivial=True,
                                              p_is_trivial=True,
                                              given_initial_central_line=True,
                                              use_paraxial_ray_tracing=False,)

optical_system_aspherical_back = OpticalSystem(surfaces=[aspherical_back_alternative, aspheric_front],
                                               t_is_trivial=True,
                                               p_is_trivial=True,
                                               given_initial_central_line=True,
                                               use_paraxial_ray_tracing=False,)

rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)
# %%
# results_dict_original = analyze_potential(optical_system=optical_system_original, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=True)
# fig, ax = plot_results(results_dict=results_dict_original, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After aspherical surface"])
# plt.show()
#
# # %%
# results_dict_spherical_back = analyze_potential(optical_system=optical_system_spherical_back, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=True)
# fig, ax = plot_results(results_dict=results_dict_spherical_back, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After spherical surface"])
# plt.show()
# %%
results_dict_aspherical_back = analyze_potential(optical_system=optical_system_aspherical_back, rays_0=rays_0, unconcentricity=unconcentricity, print_tests=True)
fig, ax = plot_results(results_dict=results_dict_aspherical_back, far_away_plane=True, unconcentricity=unconcentricity, potential_x_axis_angles=False, rays_labels=["Before lens", "After flat surface", "After aspherical surface"])
plt.show()
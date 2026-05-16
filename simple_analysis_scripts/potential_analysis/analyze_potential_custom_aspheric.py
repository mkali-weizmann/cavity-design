from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc
from cavity_design import *

# %%
R_1 = 6.574e-3
R_2 = 5.009e-3
T_c = 4.098e-3
waist_to_lens = 5.5e-3
OPTICAL_AXIS = RIGHT
diameter = 7.75e-3
a_2 = 1 / (2*R_1)
a_4 = 1000
a_6 = 0

laseroptik_mirror = CurvedMirror(radius=5e-3, outwards_normal=LEFT, center=np.array([-0.00499995, 0, 0]), curvature_sign=CurvatureSigns.concave, diameter=7.75e-3, name="LaserOptik mirror", material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"])
polynomial_coefficients_back = np.array([0, a_2, a_4, a_6])  # widget_convenient_exponent(-2.470e-1,scale=0)
lens_left = AsphericRefractiveSurface(polynomial_coefficients=polynomial_coefficients_back,
                                                  center=np.array([waist_to_lens, 0, 0]),
                                                  n_1=1,
                                                  n_2=1.45,
                                                  outwards_normal=-OPTICAL_AXIS,
                                                  name="Lens aspherical back",
                                                  material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
                                                  thickness=T_c / 2,
                                                  diameter=diameter,
                                                  curvature_sign=CurvatureSigns.convex
                                                        )
lens_right = CurvedRefractiveSurface(radius=R_2,
                                     curvature_sign=CurvatureSigns.concave,
                                     outwards_normal=OPTICAL_AXIS,
                                     center = lens_left.center + np.array([T_c, 0, 0]),
                                     name="Lens_right",
                                     n_1=1.45,
                                     n_2=1,
                                     material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
                                     thickness=T_c / 2,
                                     diameter=diameter,
                                    )

optical_system = OpticalSystem(elements=[laseroptik_mirror, lens_left, lens_right], lambda_0_laser=LAMBDA_0_LASER,
                               t_is_trivial=True, p_is_trivial=True, use_paraxial_ray_tracing=False)
cavity = optical_system_to_cavity_completion(optical_system=optical_system, NA=0.15)
# # %%
# cavity.plot()
# plt.show()

results_dict = analyze_potential_given_cavity(cavity=cavity, n_rays=30, phi_max=0.3)
plot_results(results_dict=results_dict, far_away_plane=True)
plt.show()
# %%
plot = True
print_tests = True
n_rays = 15
unconcentricity = 1e-3
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

optical_system_original = OpticalSystem(elements=[aspheric_flat, aspheric_front], t_is_trivial=True, p_is_trivial=True,
                                        given_initial_central_line=True, use_paraxial_ray_tracing=False)

optical_system_spherical_back = OpticalSystem(elements=[spherical_back_alternative, aspheric_front], t_is_trivial=True,
                                              p_is_trivial=True, given_initial_central_line=True,
                                              use_paraxial_ray_tracing=False)

optical_system_aspherical_back = OpticalSystem(elements=[aspherical_back_alternative, aspheric_front],
                                               t_is_trivial=True, p_is_trivial=True, given_initial_central_line=True,
                                               use_paraxial_ray_tracing=False)

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
fig, ax = plot_results(results_dict=results_dict_aspherical_back, far_away_plane=True, unconcentricity=unconcentricity,
                       potential_horizontal_axis_in_NAs=False,
                       rays_labels=["Before lens", "After flat surface", "After aspherical surface"])
plt.show()
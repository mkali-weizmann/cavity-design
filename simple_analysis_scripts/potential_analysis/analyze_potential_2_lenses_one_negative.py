from matplotlib import use

use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc
from simple_analysis_scripts.potential_analysis.analyze_potential import *

# %%
plot = True
print_tests = True
n_rays = 15
unconcentricity = 10e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.05
OPTICAL_AXIS = RIGHT
back_focal_length_aspheric = 20e-3
T_c_aspheric = 4.35e-3
n_design_aspheric = 1.45
n_actual_aspheric = 1.45
n_design_spherical = 1.45
n_actual_spherical = 1.45
T_c_spherical = 4.35e-3
spherical_radii = 200e-3
relative_position_spherical = 0.3
desired_focus = 200e-3
lens_type = "aspheric - lab"
dn = 0

n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = known_lenses_generator(lens_type, dn)
defocus = choose_source_position_for_desired_focus_analytic(desired_focus=desired_focus, T_c=T_c, n_design=n_design, diameter=diameter, back_focal_length=back_focal_length, R_1=R_1, R_2=R_2_signed,)
optical_system_initial, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length, defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter, n_actual=n_actual, )


output_ROC = optical_system_initial.output_radius_of_curvature(initial_distance=optical_system_initial.surfaces[0].center[0])


spherical_back_center = optical_system_initial.surfaces[-1].center + output_ROC * (1+relative_position_spherical) * OPTICAL_AXIS
spherical_back = CurvedRefractiveSurface(
    radius=spherical_radii,
    center=spherical_back_center,
    outwards_normal=OPTICAL_AXIS,
    n_1=1,
    n_2=n_actual_spherical,
    name="spherical_lens_concave_back",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
    thickness=T_c_spherical / 2,
    diameter=diameter,
    curvature_sign=CurvatureSigns.concave,
)

spherical_front = CurvedRefractiveSurface(
    radius=spherical_radii,
    center=spherical_back_center + T_c_spherical * OPTICAL_AXIS,
    outwards_normal=-OPTICAL_AXIS,
    n_1=n_actual_spherical,
    n_2=1,
    name="spherical_lens_concave_front",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
    thickness=T_c_spherical / 2,
    diameter=diameter,
    curvature_sign=CurvatureSigns.convex,
)

optical_system = OpticalSystem(
    surfaces=[*optical_system_initial.surfaces, spherical_back, spherical_front],
    t_is_trivial=True,
    p_is_trivial=True,
    use_paraxial_ray_tracing=False,
    given_initial_central_line=True,
)

rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)


results_dict = analyze_potential(
    optical_system=optical_system,
    rays_0=rays_0,
    unconcentricity=unconcentricity,
    end_mirror_ROC=20e-2,
    print_tests=print_tests,
)
plot_results(
    results_dict=results_dict,
    far_away_plane=True,
    unconcentricity=unconcentricity,
    potential_x_axis_angles=False,
)

plt.show()

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
diameter = 12.7e-3
defocus = -0.002
n_design_spherical = 1.45
n_actual_spherical = 1.45
T_c_spherical = 4.35e-3


back_center = (back_focal_length_aspheric - defocus) * OPTICAL_AXIS
aspheric_flat, aspheric_curved = Surface.from_params(
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
aspheric_flat.n_2 = n_actual_aspheric
aspheric_curved.n_1 = n_actual_aspheric

optical_system_aspheric = OpticalSystem(
    surfaces=[aspheric_flat, aspheric_curved],
    t_is_trivial=True,
    p_is_trivial=True,
    use_paraxial_ray_tracing=False,
    given_initial_central_line=True,
)

output_ROC = optical_system_aspheric.output_radius_of_curvature(initial_distance=back_center[0])

spherical_radii = 200e-3
relative_position_spherical = 0.3
spherical_back_center = aspheric_curved.center + output_ROC * (1+relative_position_spherical) * OPTICAL_AXIS
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
    surfaces=[aspheric_flat, aspheric_curved, spherical_back, spherical_front],
    t_is_trivial=True,
    p_is_trivial=True,
    use_paraxial_ray_tracing=False,
    given_initial_central_line=True,
)

rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)
# %%

ray_sequence = optical_system.propagate_ray(rays_0, propagate_with_first_surface_first=True)
R_analytical = optical_system.output_radius_of_curvature(
    initial_distance=np.linalg.norm(rays_0.origin[0, :] - optical_system.arms[0].surface_0.center)
)
# %%
# ax = optical_system.plot()
# ray_sequence.plot(ax=ax)
# output_rays_inverse = Ray(origin=ray_sequence[-1].origin, k_vector=-ray_sequence[-1].k_vector)
# output_rays_inverse.plot(ax=ax, linestyle="dashed", linewidth=0.5, color='red')
# plt.show()
# %%
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

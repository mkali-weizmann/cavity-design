from matplotlib import use
use("TkAgg")  # or 'Qt5Agg', 'GTK3Agg', etc. depending on your system

from simple_analysis_scripts.analyze_potential import *

# %%
OPTICAL_AXIS = RIGHT
dn = 0
lens_types = ["aspheric - lab", "spherical - like labs aspheric", "avantier", "aspheric - like avantier"]
lens_type = lens_types[0]
n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = generate_input_parameters_for_lenses(
    lens_type=lens_type, dn=dn
)
n_rays = 50
unconcentricity = 2.24255506e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
phi_max = 0.04
desired_focus = 200e-3
plot = True
print_tests = False

defocus = 0.0029887489470528557

back_center = (back_focal_length - defocus) * OPTICAL_AXIS
aspheric_flat, aspheric_curved = Surface.from_params(
    generate_aspheric_lens_params(
        back_focal_length=back_focal_length,
        T_c=T_c,
        n=n_design,
        forward_normal=OPTICAL_AXIS,
        flat_faces_center=back_center,
        diameter=diameter,
        polynomial_degree=8,
        name="aspheric_lens_automatic",
    )
)
aspheric_flat.n_2 = n_actual
aspheric_curved.n_1 = n_actual

optical_system = OpticalSystem(
    surfaces=[aspheric_flat, aspheric_curved],
    t_is_trivial=True,
    p_is_trivial=True,
    given_initial_central_line=True,
    use_paraxial_ray_tracing=False,
)

n_spherical = 1.45
n_design_spherical = n_actual + dn
T_c_spherical = 4.35e-3
f_spherical = 100e-3
R = (
    f_spherical * (n_design_spherical - 1) * (1 + np.sqrt(1 - T_c / (f_spherical * n_design_spherical)))
)  # This is the R value that results in f=f_lens
R_1_spherical = R
R_2_spherical = R
back_focal_length_spherical = back_focal_length_of_lens(R_1=R_1_spherical, R_2=-R_2_spherical, n=n_design_spherical, T_c=T_c_spherical)
diameter = 12.7e-3
lens_distance_to_aspheric_output_COC = image_of_a_point_with_thick_lens(
    distance_to_face_1=desired_focus, R_1=R_2_spherical, R_2=-R_1_spherical, n=n_actual, T_c=T_c_spherical
)
aspheric_output_ROC = optical_system.output_radius_of_curvature(initial_distance=back_focal_length - defocus)
lens_distance_to_aspheric_curved_face = lens_distance_to_aspheric_output_COC + aspheric_output_ROC  # aspheric_output_ROC Should be negative, so this is effectively a subtraction
spherical_0 = CurvedRefractiveSurface(
    radius=np.abs(R_1_spherical),
    outwards_normal=-OPTICAL_AXIS,
    center=aspheric_curved.center + lens_distance_to_aspheric_curved_face * OPTICAL_AXIS,
    n_1=1,
    n_2=n_actual,
    curvature_sign=CurvatureSigns.convex,
    name="spherical_0",
    thickness=T_c / 2,
    diameter=diameter,
)

spherical_1 = CurvedRefractiveSurface(
    radius=np.abs(R_2_spherical),
    outwards_normal=OPTICAL_AXIS,
    center=spherical_0.center + T_c_spherical * OPTICAL_AXIS,
    n_1=n_actual,
    n_2=1,
    curvature_sign=CurvatureSigns.concave,
    name="spherical_1",
    thickness=T_c / 2,
    diameter=diameter,
)

optical_system_combined = OpticalSystem(
    surfaces=[aspheric_flat, aspheric_curved, spherical_0, spherical_1],
    t_is_trivial=True,
    p_is_trivial=True,
    given_initial_central_line=True,
    use_paraxial_ray_tracing=False,
)

fig, ax = plt.subplots()
optical_system_combined.plot(ax=ax)
ray_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max)
ray_sequence = optical_system_combined.propagate_ray(ray_0, propagate_with_first_surface_first=True)
ray_sequence.plot(ax=ax, color="red", linewidth=0.1, label="aspheric only", alpha=1)
plt.show()


# %%
d_0 = ray_sequence.cumulative_optical_path_length[1, 0]  # Assumes the first ray is the optical axis ray.
wavefront_points_initial = ray_sequence.parameterization(d_0, optical_path_length=True)
R_numerical, center_of_curvature_numerical = extract_matching_sphere(
    wavefront_points_initial[..., 0, :], wavefront_points_initial[..., 1, :], ray_sequence.k_vector[-1, 0, :]
)
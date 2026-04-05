import numpy as np
from scipy.optimize import newton

from cavity import *
from matplotlib.lines import Line2D


# %%
def initialize_rays(
    n_rays=100,
    phi_max: Optional[float] = None,
    starting_mirror: Optional[CurvedMirror] = None,
):
    if starting_mirror is not None:
        tilt_angles = np.linspace(0, phi_max, n_rays)
        initial_arc_lengths = tilt_angles * starting_mirror.radius
        initial_rays_origin = starting_mirror.parameterization(np.zeros_like(initial_arc_lengths), -initial_arc_lengths)
        orthonormal_direction = unit_vector_of_angles(
            theta=np.zeros_like(tilt_angles), phi=tilt_angles + np.pi * (1 - starting_mirror.inwards_normal[0]) / 2
        )  # Assume system is alligned with x-axis
        rays_0 = Ray(origin=initial_rays_origin, k_vector=orthonormal_direction, n=1)
    else:
        phi = np.linspace(0, phi_max, n_rays)
        ray_origin = ORIGIN  # optical_axis * defocus
        rays_0 = Ray(origin=ray_origin, k_vector=unit_vector_of_angles(theta=0, phi=phi), n=1)
    return rays_0


def known_lenses_generator(lens_type, dn):
    if lens_type == "aspheric - lab":
        # back_focal_length = back_focal_length_of_lens(R_1=24.22e-3, R_2=-5.49e-3, n=1.8, T_c=2.91e-3)
        # diameter = 7.75e-3
        back_focal_length = 20e-3
        R_1 = None
        R_2 = None
        R_2_signed = None
        n_actual = 1.45
        n_design = n_actual + dn
        T_c = 4.35e-3
        diameter = 12.7e-3
        # This results in this value of R_2: -0.017933320598319306 for n=1.8 and -0.010350017052321312 for n=1.45
    elif lens_type == "spherical - like labs aspheric":
        n_actual = 1.45
        n_design = n_actual + dn
        T_c = 4.35e-3
        f_lens = focal_length_of_lens(
            R_1=np.inf, R_2=-0.010350017052321312, n=1.45, T_c=4.35e-3
        )  # Same as the aspheric ones.
        R = (
            f_lens * (n_design - 1) * (1 + np.sqrt(1 - T_c / (f_lens * n_design)))
        )  # This is the R value that results in f=f_lens
        R_1 = R
        R_2 = R
        R_2_signed = -R_2
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n_design, T_c=T_c)
        diameter = 12.7e-3
    elif lens_type == "avantier":
        # Avantier lenses:
        n_actual = 1.76
        n_design = n_actual + dn
        R_1 = 24.22e-3
        R_2 = 5.488e-3
        R_2_signed = -R_2
        T_c = 0.002913797540986543
        diameter = 7.75e-3
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=-R_2, n=n_design, T_c=T_c)
    elif lens_type == "aspheric - like avantier":
        n_actual = 1.76
        n_design = n_actual + dn
        back_focal_length = 0.0042325  # This results in the save focal length as the avantier lens.
        R_1 = None
        R_2 = None
        R_2_signed = None
        T_c = 2.91e-3
        diameter = 7.75e-3
    else:
        raise ValueError(
            "lens_type must be either 'aspheric - lab', 'spherical - like labs aspheric', 'avantier',  'aspheric - like avantier'"
        )
    return n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter


def choose_source_position_for_desired_focus_analytic(
    back_focal_length, desired_focus, T_c, n_design, diameter, R_1=None, R_2=None
):
    if R_1 is None and R_2 is None:
        p = LensParams(n=n_design, f=back_focal_length, T_c=T_c)
        coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=8)
        R_2 = -1 / (2 * coeffs[1])
        R_1 = np.inf
    elif R_1 is not None and R_2 is not None:
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n_design, T_c=T_c)
    else:
        raise ValueError("Either both R_1 and R_2 must be provided, or neither.")
    distance_to_flat_face = image_of_a_point_with_thick_lens(
        distance_to_face_1=desired_focus, R_1=-R_2, R_2=-R_1, n=n_design, T_c=T_c
    )
    defocus = back_focal_length - distance_to_flat_face
    return defocus


def generate_one_lens_optical_system(
    R_1: Optional[float] = None,  # For a spherical lens
    R_2: Optional[float] = None,  # For a spherical lens
    back_focal_length: Optional[float] = None,  # For an aspheric lens
    defocus=0,
    T_c=3e-3,
    n_design=1.8,
    diameter=12.7e-3,
    n_actual=None,
):
    optical_axis = RIGHT
    # Enrich input arguments:
    if n_actual is None:
        n_actual = n_design
    if R_1 is not None and R_2 is not None:
        back_focal_length = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n_design, T_c=T_c)
        params = OpticalElementParams(
            name="spherical_lens",
            surface_type=SurfacesTypes.thick_lens,
            x=back_focal_length - defocus + T_c / 2,
            y=0,
            z=0,
            r_1=np.abs(R_1),
            r_2=-np.abs(R_2),
            theta=0,
            phi=0,
            T_c=T_c,
            n_inside_or_after=n_actual,
            n_outside_or_before=1,
            diameter=diameter,
            curvature_sign=CurvatureSigns.convex,
            polynomial_coefficients=None,
            material_properties=PHYSICAL_SIZES_DICT["material_properties_sapphire"],
        )
    elif back_focal_length is not None:
        back_center = (back_focal_length - defocus) * optical_axis
        params = generate_aspheric_lens_params(
            back_focal_length=back_focal_length,
            T_c=T_c,
            n=n_design,
            forward_normal=optical_axis,
            flat_faces_center=back_center,
            diameter=diameter,
            polynomial_degree=8,
            name="Aspheric lens",
        )
        params.n_inside_or_after = n_actual
    else:
        raise ValueError("Either R_1 and R_2, or back_focal_length must be provided.")

    optical_system = OpticalSystem.from_params(
        params=[params],
        lambda_0_laser=LAMBDA_0_LASER,
        given_initial_central_line=True,
        use_paraxial_ray_tracing=False,
    )
    return optical_system, optical_axis


def generate_two_lenses_optical_system(
    defocus: float,
    back_focal_length_aspheric: float,
    T_c_aspheric: float,
    n_design_aspheric: float,
    n_actual_aspheric: float,
    n_design_spherical: float,
    n_actual_spherical: float,
    T_c_spherical: float,
    f_spherical: float,
    diameter: float = 12.7e-3,
):
    OPTICAL_AXIS = RIGHT
    desired_focus = 200e-3
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

    optical_system = OpticalSystem(
        surfaces=[aspheric_flat, aspheric_curved],
        t_is_trivial=True,
        p_is_trivial=True,
        given_initial_central_line=True,
        use_paraxial_ray_tracing=False,
    )

    R = (
        f_spherical * (n_design_spherical - 1) * (1 + np.sqrt(1 - T_c_aspheric / (f_spherical * n_design_spherical)))
    )  # This is the R value that results in f=f_lens
    R_1_spherical = R
    R_2_spherical = R
    lens_distance_to_aspheric_output_COC = image_of_a_point_with_thick_lens(
        distance_to_face_1=desired_focus, R_1=R_2_spherical, R_2=-R_1_spherical, n=n_actual_aspheric, T_c=T_c_spherical
    )
    aspheric_output_ROC = optical_system.output_radius_of_curvature(
        initial_distance=back_focal_length_aspheric - defocus)
    lens_distance_to_aspheric_curved_face = (
        lens_distance_to_aspheric_output_COC + aspheric_output_ROC
    )  # aspheric_output_ROC Should be negative, so this is effectively a subtraction
    spherical_0 = CurvedRefractiveSurface(
        radius=np.abs(R_1_spherical),
        outwards_normal=-OPTICAL_AXIS,
        center=aspheric_curved.center + lens_distance_to_aspheric_curved_face * OPTICAL_AXIS,
        n_1=1,
        n_2=n_actual_spherical,
        curvature_sign=CurvatureSigns.convex,
        name="spherical_0",
        thickness=T_c_aspheric / 2,
        diameter=diameter,
    )

    spherical_1 = CurvedRefractiveSurface(
        radius=np.abs(R_2_spherical),
        outwards_normal=OPTICAL_AXIS,
        center=spherical_0.center + T_c_spherical * OPTICAL_AXIS,
        n_1=n_actual_spherical,
        n_2=1,
        curvature_sign=CurvatureSigns.concave,
        name="spherical_1",
        thickness=T_c_aspheric / 2,
        diameter=diameter,
    )

    optical_system_combined = OpticalSystem(
        surfaces=[aspheric_flat, aspheric_curved, spherical_0, spherical_1],
        t_is_trivial=True,
        p_is_trivial=True,
        given_initial_central_line=True,
        use_paraxial_ray_tracing=False,
    )

    return optical_system_combined


def generate_negative_lens_cavity(
    n_actual_first_lens,
    n_design_first_lens,
    T_c_first_lens,
    back_focal_length_first_lens,
    R_1_first_lens,
    R_2_first_lens,
    R_2_signed_first_lens,
    diameter_first_lens,
    approximate_focus_distance_long_arm: float,
    negative_lens_focal_length: float,
    negative_lens_R_2_inverse: float,
    negative_lens_back_relative_position,
    negative_lens_refractive_index,
    negative_lens_center_thickness,
    first_arm_NA: float,
    right_mirror_ROC: Optional[float] = None,
    right_mirror_distance_to_negative_lens_front: Optional[float] = None,
    large_elements_CA: float = 25e-3,
    unconcentricity: Optional[float] = None,
):
    defocus = choose_source_position_for_desired_focus_analytic(
        desired_focus=approximate_focus_distance_long_arm,
        T_c=T_c_first_lens,
        n_design=n_design_first_lens,
        diameter=diameter_first_lens,
        back_focal_length=back_focal_length_first_lens,
        R_1=R_1_first_lens,
        R_2=R_2_signed_first_lens,
    )
    optical_system_lens, optical_axis = generate_one_lens_optical_system(
        R_1=R_1_first_lens,
        R_2=R_2_signed_first_lens,
        back_focal_length=back_focal_length_first_lens,
        defocus=defocus,
        T_c=T_c_first_lens,
        n_design=n_design_first_lens,
        diameter=diameter_first_lens,
        n_actual=n_actual_first_lens,
    )
    mirror_left = CurvedMirror(
        radius=5e-3,
        outwards_normal=LEFT,
        origin=ORIGIN,
        curvature_sign=CurvatureSigns.concave,
        name="LaserOptik mirror",
        diameter=7.75e-3,
        material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
    )
    f, n, r_2_inverse, T_c = (
        negative_lens_focal_length,
        negative_lens_refractive_index,
        negative_lens_R_2_inverse,
        negative_lens_center_thickness,
    )
    negative_lens_R_1_inverse = (1 / (f * (n - 1)) + r_2_inverse) / ((n - 1) * T_c * r_2_inverse / n + 1)
    if negative_lens_R_1_inverse == 0:
        negative_lens_R_1 = np.inf
    else:
        negative_lens_R_1 = 1 / negative_lens_R_1_inverse
    if negative_lens_R_2_inverse == 0:
        negative_lens_R_2 = np.inf
    else:
        negative_lens_R_2 = 1 / negative_lens_R_2_inverse
    negative_lens_back_center = (
        approximate_focus_distance_long_arm + negative_lens_back_relative_position
    ) * optical_axis + optical_system_lens.surfaces[-1].center
    negative_lens_params = OpticalElementParams(
        x=negative_lens_back_center[0] + negative_lens_center_thickness / 2,
        y=0,
        z=0,
        r_1=negative_lens_R_1,
        r_2=negative_lens_R_2,
        theta=0,
        phi=0,
        T_c=negative_lens_center_thickness,
        n_inside_or_after=negative_lens_refractive_index,
        n_outside_or_before=1,
        diameter=large_elements_CA,
        curvature_sign=np.nan,
        name="Negative Lens",
        material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
        polynomial_coefficients=None,
        surface_type=SurfacesTypes.thick_lens,
    )
    optical_system_without_last_mirror = OpticalSystem.from_params(
        params=[mirror_left.to_params, *optical_system_lens.params, negative_lens_params],
        lambda_0_laser=LAMBDA_0_LASER,
        p_is_trivial=True,
        t_is_trivial=True,
        use_paraxial_ray_tracing=False,
    )
    if r_2_inverse > 0 and right_mirror_distance_to_negative_lens_front is not None:
        r_2 = 1 / r_2_inverse
        right_mirror_distance_to_negative_lens_front += (
            1 - np.cos(np.arcsin(large_elements_CA / (2 * r_2)))
        ) * r_2  # Adjust for the fact that the lens surface is curved, so the distance to the front of the lens is not the same as the distance to the center of the lens.

    if right_mirror_distance_to_negative_lens_front is not None and unconcentricity is not None:
        optical_system_lenses_only = OpticalSystem.from_params(
            params=[*optical_system_lens.params, negative_lens_params],
            lambda_0_laser=LAMBDA_0_LASER,
            p_is_trivial=True,
            t_is_trivial=True,
            use_paraxial_ray_tracing=False,
        )
        R_analytical = optical_system_lenses_only.output_radius_of_curvature(source_position=ORIGIN)
        center_of_curvature = (
            optical_system_lenses_only.surfaces[-1].center + (R_analytical - unconcentricity) * optical_axis
        )
        right_mirror_center = (
            optical_system_lenses_only.surfaces[-1].center + right_mirror_distance_to_negative_lens_front * optical_axis
        )
        right_mirror_ROC = float(np.linalg.norm(center_of_curvature - right_mirror_center))
        mirror_right = CurvedMirror(
            radius=right_mirror_ROC,
            center=right_mirror_center,
            diameter=large_elements_CA,
            outwards_normal=RIGHT,
            curvature_sign=CurvatureSigns.concave,
            name="big mirror",
            material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
        )
        cavity = Cavity.from_params(
            params=[mirror_left.to_params, *optical_system_lens.params, negative_lens_params, mirror_right.to_params],
            lambda_0_laser=LAMBDA_0_LASER,
            p_is_trivial=True,
            t_is_trivial=True,
            use_paraxial_ray_tracing=False,
        )
    else:
        cavity = optical_system_to_cavity_completion(optical_system=optical_system_without_last_mirror, NA=first_arm_NA,
                                                     end_mirror_distance_to_last_element=right_mirror_distance_to_negative_lens_front,
                                                     end_mirror_ROC=right_mirror_ROC,
                                                     material_properties=PHYSICAL_SIZES_DICT[
                                                         "material_properties_fused_silica"],
                                                     diameter=large_elements_CA)
    return cavity


def generate_negative_lens_cavity_smart(
    phi_max_marginal,
    phi_max_polynomial,
    n_actual,
    n_design,
    T_c,
    back_focal_length,
    R_1,
    R_2,
    R_2_signed,
    diameter,
    n_rays,
    first_arm_NA,
    negative_lens_refractive_index,
    large_elements_CA,
    right_mirror_ROC,
    right_mirror_distance_to_negative_lens_front,
    negative_lens_defocus_power,
    negative_lens_R_2_inverse,
    desired_focus,
    negative_lens_back_relative_position,
):
    negative_lens_focal_length = -1
    negative_lens_R_2_inverse = 1

    def f_root_lens_right(negative_lens_R_2_inverse):
        cavity = generate_negative_lens_cavity(
            n_actual_first_lens=n_actual,
            n_design_first_lens=n_design,
            T_c_first_lens=T_c,
            back_focal_length_first_lens=back_focal_length,
            R_1_first_lens=R_1,
            R_2_first_lens=R_2,
            R_2_signed_first_lens=R_2_signed,
            diameter_first_lens=diameter,
            approximate_focus_distance_long_arm=desired_focus,
            negative_lens_focal_length=negative_lens_focal_length,
            negative_lens_R_2_inverse=negative_lens_R_2_inverse,
            negative_lens_back_relative_position=negative_lens_back_relative_position,
            negative_lens_refractive_index=negative_lens_refractive_index,
            negative_lens_center_thickness=3.45e-3,
            first_arm_NA=first_arm_NA,
            right_mirror_ROC=right_mirror_ROC,
            right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
        )
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
        marginal_ray = cavity.propagate_ray(
            ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5
        )
        #
        if np.isnan(marginal_ray[4].origin[1]):
            return np.inf
        else:
            two_y_marginal_over_R_lens_right = 2 * np.abs(marginal_ray[4].origin[1]) / cavity.arms[3].surface_1.radius
        return two_y_marginal_over_R_lens_right - 1

    try:
        negative_lens_R_2_inverse = newton(func=f_root_lens_right, x0=negative_lens_R_2_inverse, tol=1e-6, maxiter=100)
    except RuntimeError:
        print(
            f"Did not converge for desired focus {desired_focus} m and negative lens back relative position {negative_lens_back_relative_position}"
        )

    def f_root_mirror(negative_focal_length_inverse: float):
        if negative_focal_length_inverse == 0:
            negative_lens_focal_length = np.inf
        else:
            negative_lens_focal_length = 1 / negative_focal_length_inverse
        cavity = generate_negative_lens_cavity(
            n_actual_first_lens=n_actual,
            n_design_first_lens=n_design,
            T_c_first_lens=T_c,
            back_focal_length_first_lens=back_focal_length,
            R_1_first_lens=R_1,
            R_2_first_lens=R_2,
            R_2_signed_first_lens=R_2_signed,
            diameter_first_lens=diameter,
            approximate_focus_distance_long_arm=desired_focus,
            negative_lens_focal_length=negative_lens_focal_length,
            negative_lens_R_2_inverse=negative_lens_R_2_inverse,
            negative_lens_back_relative_position=negative_lens_back_relative_position,
            negative_lens_refractive_index=negative_lens_refractive_index,
            negative_lens_center_thickness=3.45e-3,
            first_arm_NA=first_arm_NA,
            right_mirror_ROC=right_mirror_ROC,
            right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
        )
        marginal_ray_initial = Ray(origin=ORIGIN, k_vector=unit_vector_of_angles(theta=0, phi=phi_max_marginal))
        marginal_ray = cavity.propagate_ray(
            ray=marginal_ray_initial, propagate_with_first_surface_first=False, n_arms=5
        )
        if np.isnan(marginal_ray.origin[-1, 1]):
            two_y_marginal_over_CA_right = -10 * negative_focal_length_inverse  # np.inf
        else:
            two_y_marginal_over_CA_right = 2 * np.abs(marginal_ray.origin[-1, 1]) / large_elements_CA
        value_for_root = two_y_marginal_over_CA_right - 1
        return value_for_root

    try:
        negative_lens_focal_length_inverse, results_report = newton(
            func=f_root_mirror, x0=1 / negative_lens_defocus_power, tol=1e-6, maxiter=100, disp=False, full_output=True
        )
    except RuntimeError:
        print(
            f"Did not converge for desired focus {desired_focus} m and negative lens back relative position {negative_lens_back_relative_position}"
        )
    negative_lens_focal_length = 1 / negative_lens_focal_length_inverse

    cavity = generate_negative_lens_cavity(
        n_actual_first_lens=n_actual,
        n_design_first_lens=n_design,
        T_c_first_lens=T_c,
        back_focal_length_first_lens=back_focal_length,
        R_1_first_lens=R_1,
        R_2_first_lens=R_2,
        R_2_signed_first_lens=R_2_signed,
        diameter_first_lens=diameter,
        approximate_focus_distance_long_arm=desired_focus,
        negative_lens_focal_length=negative_lens_focal_length,
        negative_lens_R_2_inverse=negative_lens_R_2_inverse,
        negative_lens_back_relative_position=negative_lens_back_relative_position,
        negative_lens_refractive_index=negative_lens_refractive_index,
        negative_lens_center_thickness=3.45e-3,
        first_arm_NA=first_arm_NA,
        right_mirror_ROC=right_mirror_ROC,
        right_mirror_distance_to_negative_lens_front=right_mirror_distance_to_negative_lens_front,
    )
    results_dict = analyze_potential_given_cavity(
        cavity=cavity, n_rays=n_rays, phi_max=phi_max_polynomial, print_tests=False
    )
    return (
        results_dict["polynomial_residuals_mirror"].coef[2],
        negative_lens_focal_length,
        negative_lens_R_2_inverse,
        cavity,
    )


def analyze_output_wavefront(
    ray_sequence: RaySequence,
    unconcentricity: Optional[float] = None,
    end_mirror_center: Optional[float] = None,
    R_output_analytical: Optional[float] = None,
    end_mirror_ROC: Optional[float] = None,
    print_tests: bool = True,
):
    output_ray = ray_sequence[-1]
    optical_axis = output_ray.k_vector[..., 0, :]
    # Extract all wavefront features at the output surface of the lens.

    relative_optical_path_length = (
        ray_sequence.cumulative_optical_path_length[-2, :] - ray_sequence.cumulative_optical_path_length[-2, 0]
    )
    wavefront_points_initial = output_ray.parameterization(t=-relative_optical_path_length)

    R_output_numerical, center_of_curvature_numerical = extract_matching_sphere(
        wavefront_points_initial[..., 0, :], wavefront_points_initial[..., 1, :], output_ray.k_vector[..., 0, :]
    )
    if R_output_analytical is not None:
        R_output = R_output_analytical
        center_of_curvature = ray_sequence[-1].parameterization(R_output_analytical, optical_path_length=False)[
            0, :
        ]  # Along the optical axis.
        if print_tests:
            print(
                f"Analytical/numerical output ROC:\n{R_output:.6e}\n{R_output_numerical:.6e} (inaccurate for large or extremeley small dphi)"
            )
            print(
                f"Analytical/numerical center of curvature:\n{np.stack((center_of_curvature, center_of_curvature_numerical), axis=0)} (inacurate for large or extremeley small dphi)"
            )
    else:
        R_output, center_of_curvature = R_output_numerical, center_of_curvature_numerical

    residual_distances_initial = np.abs(R_output) - np.linalg.norm(
        wavefront_points_initial - center_of_curvature, axis=-1
    )
    polynomial_residuals_initial = Polynomial.fit(
        wavefront_points_initial[:, 1] ** 2, residual_distances_initial, 4
    ).convert()
    if print_tests:
        print(
            f"Initial wavefront residual from fitted sphere. 2nd order term should be singificantly smaller than 1/(2*R_output) = {1 / (2 * R_output):.3e}, actual: {polynomial_residuals_initial.coef[1]:.3e}"
        )
        print(
            f" Fourth order term: {polynomial_residuals_initial.coef[2]:.3e}, should be significantly larger than y_max ** -2 * second order term = {wavefront_points_initial[-1, 1] ** -2 * polynomial_residuals_initial.coef[1]:.26e}"
        )
    if end_mirror_ROC is None:
        end_mirror_ROC = R_output

    if unconcentricity is None and end_mirror_center is not None:
        end_mirror_origin = end_mirror_center - end_mirror_ROC * optical_axis
        unconcentricity = (center_of_curvature - end_mirror_origin) @ optical_axis
    elif end_mirror_center is None and unconcentricity is not None:
        end_mirror_origin = center_of_curvature - unconcentricity * optical_axis
    else:
        raise ValueError("Either unconcentricity or end_mirror_center must be provided, but not both.")

    # Extract wavefront features at a far away plane (2*ROC - u from the lens):
    wavefront_points_opposite = output_ray.parameterization(
        -relative_optical_path_length + R_output + end_mirror_ROC - unconcentricity, optical_path_length=True
    )

    R_opposite = -(end_mirror_ROC - unconcentricity)  # negative because at this point the beam is diverging.
    R_opposite_numerical, center_of_curvature_opposite_numerical = extract_matching_sphere(
        wavefront_points_opposite[..., 0, :], wavefront_points_opposite[..., 1, :], optical_axis
    )  # Should be the same as the original center of curvature
    if print_tests:
        print(
            f"Far away plane analytical/numerical output ROC:\n{R_opposite:.6e}\n{R_opposite_numerical:.6e} (inaccurate for large or extremeley small dphi)"
        )
        print(
            f"Far away plane analytical/numerical center of curvature:\n{np.stack((center_of_curvature, center_of_curvature_opposite_numerical), axis=0)} (inacurate for large or extremeley small dphi)"
        )

    if R_output_analytical is None:
        R_opposite, center_of_curvature = R_opposite_numerical, center_of_curvature_opposite_numerical

    residual_distances_opposite = np.abs(R_opposite) - np.linalg.norm(
        wavefront_points_opposite - center_of_curvature, axis=-1
    )
    polynomial_residuals_opposite = Polynomial.fit(
        wavefront_points_opposite[:, 1] ** 2, residual_distances_opposite, 6
    ).convert()

    # Analyze unconcentric mirror case:
    residual_distances_mirror = end_mirror_ROC - np.linalg.norm(
        wavefront_points_opposite - end_mirror_origin, axis=-1
    )  # Mirror has a radius of R_output, not R_opposite.
    polynomial_residuals_mirror = Polynomial.fit(
        wavefront_points_opposite[:, 1] ** 2, residual_distances_mirror, 6
    ).convert()

    # Generate dummy points for fitted spheres (used only for plotting, not for calculations):
    points_rel = wavefront_points_initial - center_of_curvature
    phi_dummy = np.linspace(0, np.arctan(points_rel[-1, 1] / points_rel[-1, 0]), wavefront_points_initial.shape[0])
    dummy_points_curvature_initial = center_of_curvature - R_output * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )
    dummy_points_curvature_opposite = center_of_curvature - R_opposite * np.stack(  # R_opposite is negative
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )
    dummy_points_mirror = end_mirror_origin + end_mirror_ROC * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )  # Mirror has the radius of the original wavefront sphere, but centered at the shifted center.

    L_long_arm = R_output + end_mirror_ROC - unconcentricity
    assert (
        L_long_arm > 0
    ), f"Long arm length should be positive, but got {L_long_arm:.3e} m. Try increasing end mirror ROC. The default end mirror ROC works only for output converging wavefront"
    end_mirror_object = CurvedMirror(
        radius=end_mirror_ROC,
        outwards_normal=RIGHT,
        center=end_mirror_origin + end_mirror_ROC * RIGHT,
        curvature_sign=CurvatureSigns.concave,
        name="big mirror",
    )

    # find point of 0 derivative (other than 0) in residual_distances_mirror:
    deriv_mirror = np.gradient(residual_distances_mirror, wavefront_points_opposite[:, 1])
    first_zero_crossings = np.where(np.diff(np.sign(deriv_mirror)))[0]
    if len(first_zero_crossings) > 0:
        zero_derivative_points = wavefront_points_opposite[first_zero_crossings[0], 1]
    else:
        zero_derivative_points = None

    results_dict = {
        "ray_sequence": ray_sequence,
        "wavefront_points_initial": wavefront_points_initial,
        "R_output": R_output,
        "center_of_curvature": center_of_curvature,
        "dummy_points_curvature_initial": dummy_points_curvature_initial,
        "residual_distances_initial": residual_distances_initial,
        "polynomial_residuals_initial": polynomial_residuals_initial,
        "R_opposite": R_opposite,
        "wavefront_points_opposite": wavefront_points_opposite,
        "dummy_points_curvature_opposite": dummy_points_curvature_opposite,
        "dummy_points_mirror": dummy_points_mirror,
        "residual_distances_opposite": residual_distances_opposite,
        "residual_distances_mirror": residual_distances_mirror,
        "polynomial_residuals_opposite": polynomial_residuals_opposite,
        "polynomial_residuals_mirror": polynomial_residuals_mirror,
        "zero_derivative_points": zero_derivative_points,
        "end_mirror_object": end_mirror_object,
    }

    return results_dict


def analyze_potential(
    optical_system: OpticalSystem,
    rays_0: Ray,
    unconcentricity: float,
    end_mirror_ROC: Optional[float] = None,
    small_mirror_object: Optional[CurvedMirror] = None,
    print_tests: bool = True,
):
    ray_sequence = optical_system.propagate_ray(rays_0, propagate_with_first_surface_first=True)
    ray_sequence = (
        ray_sequence.remove_escaped_rays
    )  # Filter out rays that escaped the system (e.g. because of finite size of the lenses or because they were blocked by the mirror)

    R_analytical = optical_system.output_radius_of_curvature(source_position=rays_0.origin[0, :])

    results_dict = analyze_output_wavefront(
        ray_sequence,
        unconcentricity=unconcentricity,
        R_output_analytical=R_analytical,
        end_mirror_ROC=end_mirror_ROC,
        print_tests=print_tests,
    )

    if small_mirror_object is None:
        small_mirror_object = CurvedMirror(
            radius=5e-3,
            outwards_normal=LEFT,
            origin=ORIGIN,
            curvature_sign=CurvatureSigns.concave,
            name="LaserOptik mirror",
            diameter=7.75e-3,
            material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
        )

    cavity = Cavity.from_params(
        params=[small_mirror_object.to_params, *optical_system.to_params, results_dict["end_mirror_object"].to_params],
        lambda_0_laser=LAMBDA_0_LASER,
        t_is_trivial=True,
        p_is_trivial=True,
        use_paraxial_ray_tracing=False,
        standing_wave=True,
    )

    results_dict["optical_system"] = optical_system
    results_dict["cavity"] = cavity

    if cavity.resonating_mode_successfully_traced:
        results_dict["spot_size_paraxial"] = cavity.arms[
            len(cavity.surfaces) - 2
        ].mode_parameters_on_surface_1.spot_size[0]
        results_dict["NA_paraxial"] = cavity.arms[len(cavity.surfaces) - 2].mode_parameters.NA[0]
    else:
        results_dict["spot_size_paraxial"] = np.nan
        results_dict["NA_paraxial"] = np.nan

    return results_dict


def analyze_potential_given_cavity(cavity: Cavity, n_rays: int, phi_max: float, print_tests: bool = True):
    assert np.all(
        np.isclose(cavity.surfaces[0].origin, ORIGIN)
    ), "Currently assumes the center of the small mirror is at the origin for the extraction of the Analytical R. probably it will work otherwise, but needs to be debugged"  #
    optical_system_reduced = OpticalSystem(
        surfaces=cavity.surfaces[1:-1],
        lambda_0_laser=cavity.lambda_0_laser,
        t_is_trivial=cavity.t_is_trivial,
        p_is_trivial=cavity.p_is_trivial,
        use_paraxial_ray_tracing=cavity.use_paraxial_ray_tracing,
        given_initial_central_line=cavity.central_line[1],
    )
    first_mirror = cavity.physical_surfaces[0]
    rays_0 = initialize_rays(n_rays=n_rays, phi_max=phi_max)
    ray_sequence = optical_system_reduced.propagate_ray(rays_0, propagate_with_first_surface_first=True)
    ray_sequence_cleaned = ray_sequence.remove_escaped_rays
    if len(optical_system_reduced.surfaces) == 0:
        R_analytical = None
    else:
        R_analytical = optical_system_reduced.output_radius_of_curvature(source_position=first_mirror.origin)
    if ray_sequence_cleaned.origin.shape[1] == 1:
        raise ValueError(
            "All rays escaped the system, cannot analyze wavefront. Try increasing the number of rays or the maximum angle phi_max."
        )
    results_dict = analyze_output_wavefront(
        ray_sequence=ray_sequence_cleaned,
        R_output_analytical=R_analytical,
        end_mirror_center=cavity.physical_surfaces[-1].center,
        end_mirror_ROC=cavity.physical_surfaces[-1].radius,
        print_tests=print_tests,
    )
    results_dict["optical_system"] = optical_system_reduced
    results_dict["cavity"] = cavity
    if cavity.resonating_mode_successfully_traced:
        results_dict["spot_size_paraxial"] = cavity.arms[
            len(cavity.surfaces) - 2
        ].mode_parameters_on_surface_1.spot_size[0]
        results_dict["NA_paraxial"] = cavity.arms[len(cavity.surfaces) - 2].mode_parameters.NA[0]
    else:
        results_dict["spot_size_paraxial"] = np.nan
        results_dict["NA_paraxial"] = np.nan
    return results_dict


def plot_results(
    results_dict,
    far_away_plane: bool = False,
    unconcentricity: Optional[float] = None,
    potential_x_axis_angles: bool = False,
    rays_labels: Optional[List[str]] = None,
    fig_and_ax=None,
    plot_final_arm_backwards_rays: bool = False,
):
    (ray_sequence, R, center_of_curvature, NA_paraxial, spot_size_paraxial, zero_derivative_points) = (
        results_dict["ray_sequence"],
        results_dict["R_output"],
        results_dict["center_of_curvature"],
        results_dict["NA_paraxial"],
        results_dict["spot_size_paraxial"],
        results_dict["zero_derivative_points"],
    )
    if far_away_plane:
        (
            wavefront_points,
            residual_distances,
            polynomial,
            polynomial_residuals_mirror,
            residual_distances_mirror,
        ) = (
            results_dict["wavefront_points_opposite"],
            results_dict["residual_distances_opposite"],
            results_dict["polynomial_residuals_opposite"],
            results_dict["polynomial_residuals_mirror"],
            results_dict["residual_distances_mirror"],
        )
    else:
        (
            wavefront_points,
            residual_distances,
            polynomial,
            polynomial_residuals_mirror,
            residual_distances_mirror,
        ) = (
            results_dict["wavefront_points_initial"],
            results_dict["residual_distances_initial"],
            results_dict["polynomial_residuals_initial"],
            None,
            None,
        )
    valid_cavity = results_dict["cavity"] is not None
    if fig_and_ax is not None:
        fig, ax = fig_and_ax
    else:
        fig, ax = plt.subplots(2, 1, figsize=(20, 20), constrained_layout=True)
    ray_sequence.plot(ax=ax[1], linewidth=0.5, labels=rays_labels)
    if valid_cavity:
        results_dict["cavity"].plot(ax=ax[1], fine_resolution=True)
    else:
        results_dict["optical_system"].plot(ax=ax[1], fine_resolution=True)
    ax[1].set_xlim(
        ray_sequence.origin[0, 0, 0] - 0.01, results_dict["end_mirror_object"].center[0] + 0.01
    )  # (-1e-3, 100e-3)
    ax[1].set_ylim(-5e-3, 5e-3)  # surface_1.diameter / 2, surface_1.diameter / 2
    ax[1].grid()
    ax[1].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color="purple")
    ax[1].scatter(center_of_curvature[0], center_of_curvature[1], s=20, color="cyan", label="Center of curvature")
    if plot_final_arm_backwards_rays:
        final_rays = ray_sequence[-1]
        backwards_rays = Ray(origin=final_rays.origin, k_vector=-final_rays.k_vector)
        backwards_rays.plot(ax=ax[1], linewidth=0.5, linestyle="--", color="orange", length=0.5)
    ax[1].legend()

    if potential_x_axis_angles:
        angles_theta, angles_phi = angles_of_unit_vector(ray_sequence[0].k_vector)
        potential_x_axis = angles_phi
        potential_x_label = "phi (rad)"
    else:
        potential_x_axis = wavefront_points[:, 1] * 1e3
        potential_x_label = "y (mm)"
    ax[0].plot(
        potential_x_axis,
        residual_distances * 1e6,
        label="wavefront residual from matching sphere",
        marker="o",
        linestyle="",
        color="blue",
        markersize=5,
        alpha=0.6,
    )
    x_fit = np.linspace(np.min(wavefront_points[:, 1]), np.max(wavefront_points[:, 1]), 100)
    ax[0].set_xlim(x_fit[0] * 1e3, x_fit[-1] * 1e3)
    ax[0].set_xlabel(potential_x_label)
    ax[0].set_ylabel("wavefront difference (µm)")
    ax[0].grid()
    # build polynomial term string with ascending powers and .1e formatting, include x^{n} terms
    coeffs_asc = polynomial.coef
    terms_parts = []
    for i, c in enumerate(coeffs_asc):
        s = f"{c:.1e}"
        mant_str, exp_str = s.split("e")
        mant = float(mant_str)
        exp = int(exp_str)
        terms_parts.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
    terms = " + ".join(terms_parts)
    title = "residual distance between wavefront and fitted sphere. fit:\n" + terms
    # Build terms for mirror deviation polynomial (if present)
    terms_parts_mirror = []
    if polynomial_residuals_mirror is not None:
        ax[0].plot(
            potential_x_axis,
            residual_distances_mirror * 1e6,
            marker="x",
            linestyle="",
            color="magenta",
            label="Mirror deviation data",
            markersize=5,
            alpha=0.6,
        )
        coeffs_mirror_asc = polynomial_residuals_mirror.coef
        for i, c in enumerate(coeffs_mirror_asc):
            s = f"{c:.1e}"
            mant_str, exp_str = s.split("e")
            mant = float(mant_str)
            exp = int(exp_str)
            terms_parts_mirror.append(rf"${mant:.1f}\cdot10^{{{exp}}}\,x^{{{2 * i}}}$")
        terms_mirror = " + ".join(terms_parts_mirror)
        if not np.isnan(spot_size_paraxial):
            NA_short_arm = (
                np.nan if results_dict["cavity"] is None else results_dict["cavity"].arms[0].mode_parameters.NA[0]
            )
            mode_terms = f"\n Paraxial spot size: {spot_size_paraxial * 1e3:.2f} mm, NA long arm: {NA_paraxial:.2e}, NA short arm: {NA_short_arm:.2e}"
        else:
            mode_terms = ""
        if unconcentricity is not None:
            unconcentricity_um = unconcentricity * 1e6
        else:
            unconcentricity_um = np.nan
        differential_second_order_mirror = (residual_distances_mirror[1] - residual_distances_mirror[0]) / (wavefront_points[1, 1] - wavefront_points[0, 1]) ** 2
        title += (
            f"\nmirror deviation fit (unconcentricity = {unconcentricity_um:.1f} µm):\n" + terms_mirror + f"a_2_diff = {differential_second_order_mirror:.2e}" + mode_terms
        )
        ax[0].set_title(title)
        if not potential_x_axis_angles:
            ax[0].plot(
                x_fit * 1e3,
                polynomial_residuals_mirror(x_fit**2) * 1e6,
                color="green",
                linestyle="dashed",
                label="Mirror residuals Polynomial fit",
                linewidth=0.5,
            )
            ax[0].plot(
                x_fit * 1e3,
                polynomial(x_fit**2) * 1e6,
                color="red",
                linestyle="dashed",
                label="Matching sphere residuals Polynomial fit",
                linewidth=0.5,
            )
            # Plot intensity profile on a new y axis for ax[0] if NA_paraxial is not None: using the formula: e^{-2y^{2} / (spot_size_paraxial)^{2})}
            if NA_paraxial is not None and spot_size_paraxial is not None:
                ax2 = ax[0].twinx()
                intensity_profile = np.exp(-(2 * x_fit**2) / (spot_size_paraxial) ** 2)
                ax2.plot(
                    x_fit * 1e3,
                    intensity_profile,
                    color="orange",
                    linestyle="dashed",
                    label="Paraxial Gaussian intensity profile",
                    linewidth=1,
                )
                ax2.axvline(
                    spot_size_paraxial * 1e3 * np.sign(x_fit[0] + 1e-19),
                    color="orange",
                    linestyle="dashed",
                    linewidth=1,
                    label="Paraxial spot size ($w_{0}$)",
                )
                legend_line = Line2D(
                    [], [], color="orange", linestyle="dashed", linewidth=1, label="Paraxial Gaussian intensity profile"
                )
                handles, labels = ax[0].get_legend_handles_labels()
                handles.append(legend_line)
                ax2.set_ylabel("Relative intensity (a.u.)")
                ax2.grid(False)

            zero_derivative_point_plot = np.nan if zero_derivative_points is None else zero_derivative_points
            ax[0].axvline(
                zero_derivative_point_plot * 1e3, label="2nd vs 4th order max", color="purple", linestyle="dotted"
            )
            ax[0].legend(handles=handles)
    return fig, ax

from cavity import *
from matplotlib.lines import Line2D


# %%
def initialize_rays(
    defocus: float = 0,
    n_rays=100,
    dphi: Optional[float] = None,
    phi_max: Optional[float] = None,
    diameter: Optional[float] = None,
    back_focal_length: Optional[float] = None,
):
    if phi_max is None and dphi is not None:
        phi = np.arange(0, n_rays) * dphi
    elif phi_max is None and dphi is None:
        phi_max = np.arctan((diameter / 2) / (back_focal_length - defocus) / 2)
        phi = np.linspace(0, phi_max, n_rays)
    else:
        phi = np.linspace(0, phi_max, n_rays)
    ray_origin = ORIGIN  # optical_axis * defocus
    rays_0 = Ray(origin=ray_origin, k_vector=unit_vector_of_angles(theta=0, phi=phi), n=1)
    return rays_0


def analyze_output_wavefront(
    ray_sequence: RaySequence,
    unconcentricity: float,
    R_output_analytical: Optional[float] = None,
    end_mirror_ROC: Optional[float] = None,
    print_tests: bool = True,
):
    output_ray = ray_sequence[-1]
    # Extract all wavefront features at the output surface of the lens.

    relative_optical_path_length = (
        ray_sequence.cumulative_optical_path_length[-2, :] - ray_sequence.cumulative_optical_path_length[-2, 0]
    )
    wavefront_points_initial = output_ray.parameterization(t=-relative_optical_path_length)

    R_output_numerical, center_of_curvature_numerical = extract_matching_sphere(  # TODO: invert R sign convention
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

    # Extract wavefront features at a far away plane (2*ROC - u from the lens):
    wavefront_points_opposite = output_ray.parameterization(
        -relative_optical_path_length + R_output + end_mirror_ROC - unconcentricity, optical_path_length=True
    )

    R_opposite = -(end_mirror_ROC - unconcentricity)  # negative because at this point the beam is diverging.
    R_opposite_numerical, center_of_curvature_opposite_numerical = extract_matching_sphere(
        wavefront_points_opposite[..., 0, :], wavefront_points_opposite[..., 1, :], ray_sequence[-1].k_vector[..., 0, :]
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

    residual_distances_opposite = np.abs(R_opposite) - np.linalg.norm(wavefront_points_opposite - center_of_curvature, axis=-1)
    polynomial_residuals_opposite = Polynomial.fit(
        wavefront_points_opposite[:, 1] ** 2, residual_distances_opposite, 4
    ).convert()

    # Analyze unconcentric mirror case:
    center_of_mirror = center_of_curvature - unconcentricity * output_ray.k_vector[0, :]
    residual_distances_mirror = end_mirror_ROC - np.linalg.norm(
        wavefront_points_opposite - center_of_mirror, axis=-1
    )  # Mirror has a radius of R_output, not R_opposite.
    polynomial_residuals_mirror = Polynomial.fit(
        wavefront_points_opposite[:, 1] ** 2, residual_distances_mirror, 4
    ).convert()
    expected_second_order_term = 1 / 2 * (1 / R_opposite - 1 / R_output)

    # Generate dummy points for fitted spheres (used only for plotting, not for calculations):
    points_rel = wavefront_points_initial - center_of_curvature
    phi_dummy = np.linspace(0, np.arctan(points_rel[-1, 1] / points_rel[-1, 0]), 100)
    dummy_points_curvature_initial = center_of_curvature - R_output * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )
    dummy_points_curvature_opposite = center_of_curvature -R_opposite * np.stack(  # R_opposite is negative
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )
    dummy_points_mirror = center_of_mirror + end_mirror_ROC * np.stack(
        (np.cos(phi_dummy), np.sin(phi_dummy), np.zeros_like(phi_dummy)), axis=-1
    )  # Mirror has the radius of the original wavefront sphere, but centered at the shifted center.

    L_long_arm = R_output + end_mirror_ROC - unconcentricity
    assert (
        L_long_arm > 0
    ), f"Long arm length should be positive, but got {L_long_arm:.3e} m. Try increasing end mirror ROC. The default end mirror ROC works only for output converging wavefront"
    end_mirror_object = CurvedMirror(radius=end_mirror_ROC, outwards_normal=RIGHT,
                                     center=center_of_mirror + end_mirror_ROC * RIGHT,
                                     curvature_sign=CurvatureSigns.concave, name="big mirror")

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


def generate_one_lens_optical_system(
    R_1: Optional[float] = None,
    R_2: Optional[float] = None,
    back_focal_length: Optional[float] = None,
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
        surface_0, surface_1 = CurvedRefractiveSurface(
            radius=np.abs(R_1),
            outwards_normal=-optical_axis,
            center=(back_focal_length - defocus) * optical_axis,
            n_1=1,
            n_2=n_actual,
            curvature_sign=CurvatureSigns.convex,
            name="first surface",
            thickness=T_c / 2,
            diameter=diameter,
        ), CurvedRefractiveSurface(
            radius=np.abs(R_2),
            outwards_normal=optical_axis,
            center=(back_focal_length - defocus + T_c) * optical_axis,
            n_1=n_actual,
            n_2=1,
            curvature_sign=CurvatureSigns.concave,
            name="second surface",
            thickness=T_c / 2,
            diameter=diameter,
        )
    elif back_focal_length is not None:
        back_center = (back_focal_length - defocus) * optical_axis
        surface_0, surface_1 = Surface.from_params(
            generate_aspheric_lens_params(
                back_focal_length=back_focal_length,
                T_c=T_c,
                n=n_design,
                forward_normal=optical_axis,
                flat_faces_center=back_center,
                diameter=diameter,
                polynomial_degree=8,
                name="aspheric_lens_automatic",
            )
        )
        surface_0.n_2 = n_actual
        surface_1.n_1 = n_actual
    else:
        raise ValueError("Either R_1 and R_2, or back_focal_length must be provided.")

    optical_system = OpticalSystem(
        surfaces=[surface_0, surface_1],
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
        initial_distance=back_focal_length_aspheric - defocus
    )
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


def complete_optical_system_to_cavity(results_dict: dict, unconcentricity: float, print_tests: bool = True):
    # Generate cavity for mode analysis:
    optical_system = results_dict["optical_system"]
    mode_parameters_lens_right_outer_side = LocalModeParameters(
        z_minus_z_0=results_dict["R_output"] - unconcentricity / 2,
        lambda_0_laser=LAMBDA_0_LASER,
        n=1,
        z_R=z_R_of_NA(NA=results_dict["NA_paraxial"], lambda_laser=LAMBDA_0_LASER),
    )
    optical_system_inverted = optical_system.invert()
    mode_parameters_right_arm = results_dict["mode_parameters_right_arm"].invert_direction()
    modes_history = optical_system_inverted.propagate_mode_parameters(
        mode_parameters=mode_parameters_right_arm, propagate_with_first_surface_first=True
    )
    output_mode_local = modes_history[-1]
    output_mode = output_mode_local.to_mode_parameters(
        location_of_local_mode_parameter=optical_system_inverted.arms[-1].surface_1.center, k_vector=LEFT
    )

    mirror_left = match_a_mirror_to_mode(
        mode=output_mode,
        R=5e-3,
        name="LaserOptik mirror",
        material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
    )

    cavity = Cavity(
        surfaces=[mirror_left, *optical_system.physical_surfaces, results_dict["end_mirror_object"]],
        lambda_0_laser=LAMBDA_0_LASER,
        t_is_trivial=True,
        p_is_trivial=True,
        use_paraxial_ray_tracing=False,
        standing_wave=True,
    )
    if print_tests:
        print(
            f"NA in the right arm - analytical calculation and numerical cavity solution\n{results_dict['NA_paraxial']:.3e}\n"
            f"{cavity.arms[len(cavity.arms) // 2].mode_parameters.NA[0]:.3e}"
        )

        print(
            f"center of curvature of output wave and mode center in cavity:\n"
            f"{results_dict['center_of_curvature'] - unconcentricity / 2 * RIGHT}\n{cavity.mode_parameters[len(cavity.arms) // 2].center[0, :]}"
        )

        print(
            "Spot size in the right mirror - analytical calculation and numerical cavity solution\n"
            f"{results_dict['spot_size_paraxial'] * 1e3:.3e} mm\n{w_of_q(cavity.arms[len(cavity.arms) // 2].mode_parameters_on_surface_0.q[0], lambda_laser=LAMBDA_0_LASER) * 1e3:.3e} mm"
        )

        print(
            "mode radius of curvature after the optical system - analytical calculation and numerical cavity solution\n"
            f"{results_dict['R_output']*1e3:.3e} mm\n{R_of_q(cavity.arms[len(cavity.arms) // 2 - 1].mode_parameters_on_surface_0.q[0])*1e3:.3e} mm"
        )
    return cavity


def analyze_potential(
    optical_system: OpticalSystem,
    rays_0: Ray,
    unconcentricity: float,
    end_mirror_ROC: Optional[float] = None,
    small_mirror_object: Optional[CurvedMirror] = None,
    print_tests: bool = True,
):
    ray_sequence = optical_system.propagate_ray(rays_0, propagate_with_first_surface_first=True)
    R_analytical = optical_system.output_radius_of_curvature(
        initial_distance=np.linalg.norm(rays_0.origin[0, :] - optical_system.arms[0].surface_0.center)
    )

    results_dict = analyze_output_wavefront(
        ray_sequence,
        unconcentricity=unconcentricity,
        R_output_analytical=R_analytical,
        end_mirror_ROC=end_mirror_ROC,
        print_tests=print_tests,
    )

    if small_mirror_object is None:
        small_mirror_object = CurvedMirror(radius=5e-3, outwards_normal=LEFT, origin=ORIGIN,
                                           curvature_sign=CurvatureSigns.concave, name="LaserOptik mirror",
                                           diameter=7.75e-3, material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"])

    cavity = Cavity(surfaces=[small_mirror_object, *optical_system.surfaces, results_dict["end_mirror_object"]],
                    lambda_0_laser=LAMBDA_0_LASER,
                    t_is_trivial=True,
                    p_is_trivial=True,
                    use_paraxial_ray_tracing=False,
                    standing_wave=True)

    results_dict["optical_system"] = optical_system
    results_dict["cavity"] = cavity

    if cavity.resonating_mode_successfully_traced:
        results_dict["spot_size_paraxial"] = cavity.arms[len(cavity.surfaces) - 1].mode_parameters_on_surface_1.spot_size[0]
        results_dict["NA_paraxial"] = cavity.arms[len(cavity.surfaces) - 1].mode_parameters.NA[0]
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
):
    (optical_system, ray_sequence, R, center_of_curvature, NA_paraxial, spot_size_paraxial, zero_derivative_points) = (
        results_dict["optical_system"],
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
            dummy_points,
            polynomial,
            dummy_points_mirror,
            polynomial_residuals_mirror,
            residual_distances_mirror,
        ) = (
            results_dict["wavefront_points_opposite"],
            results_dict["residual_distances_opposite"],
            results_dict["dummy_points_curvature_opposite"],
            results_dict["polynomial_residuals_opposite"],
            results_dict["dummy_points_mirror"],
            results_dict["polynomial_residuals_mirror"],
            results_dict["residual_distances_mirror"],
        )
    else:
        (
            wavefront_points,
            residual_distances,
            dummy_points,
            polynomial,
            dummy_points_mirror,
            polynomial_residuals_mirror,
            residual_distances_mirror,
        ) = (
            results_dict["wavefront_points_initial"],
            results_dict["residual_distances_initial"],
            results_dict["dummy_points_curvature_initial"],
            results_dict["polynomial_residuals_initial"],
            None,
            None,
            None,
        )
    valid_cavity = results_dict["cavity"] is not None
    fig, ax = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    surface_0, surface_1 = optical_system.physical_surfaces[0], optical_system.physical_surfaces[-1]
    ray_sequence.plot(ax=ax[1, 0], linewidth=0.5, labels=rays_labels)
    ray_sequence.plot(ax=ax[1, 1], linewidth=0.5, labels=rays_labels)
    if valid_cavity:
        results_dict["cavity"].plot(ax=ax[1, 0], fine_resolution=True)
        results_dict["cavity"].plot(ax=ax[1, 1], fine_resolution=True)
    else:
        results_dict["optical_system"].plot(ax=ax[1, 0], fine_resolution=True)
        results_dict["optical_system"].plot(ax=ax[1, 1], fine_resolution=True)
    ax[1, 0].set_xlim(ray_sequence.origin[0, 0, 0] - 0.01, center_of_curvature[0] * 2)  # (-1e-3, 100e-3)
    ax[1, 0].set_ylim(-surface_1.diameter / 2, surface_1.diameter / 2)  # (-4.2e-3, 4.2e-3)
    ax[1, 0].grid()
    ax[1, 0].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color="purple")
    ax[1, 0].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color="cyan", label="Center of curvature")
    ax[1, 0].legend()

    ax[1, 1].set_xlim(
        center_of_curvature[0] -1e-3, center_of_curvature[0] + 1e-3
    )
    ax[1, 1].set_ylim(-ray_sequence.origin[-1, 1, 1] * 0.5, 1 * ray_sequence.origin[-1, 1, 1])  # (-4.2e-3, 4.2e-3)
    ax[1, 1].grid()
    ax[1, 1].scatter(wavefront_points[:, 0], wavefront_points[:, 1], s=8, color="purple")
    ax[1, 1].scatter(center_of_curvature[0], center_of_curvature[1], s=50, color="cyan", label="Center of curvature")

    ax[0, 0].plot(
        wavefront_points[:, 0] * 1e3,
        wavefront_points[:, 1] * 1e3,
        label="wavefront points",
        color="black",
        marker="o",
        linestyle="",
        markersize=5,
        zorder=1,
        alpha=0.4,
    )
    ax[0, 0].plot(
        dummy_points[:, 0] * 1e3,
        dummy_points[:, 1] * 1e3,
        linestyle="dashdot",
        label="Fitted sphere for wavefront",
        color="C0",
        linewidth=2.0,
        alpha=0.95,
        zorder=3,
    )
    ax[0, 0].set_xlabel("x (mm)")
    ax[0, 0].set_ylabel("y (mm)")
    ax[0, 0].grid()
    ax[0, 0].set_title(f"wavefront and fitted sphere: {R*1e3:.2f} mm")
    if dummy_points_mirror is not None:
        ax[0, 0].plot(
            dummy_points_mirror[:, 0] * 1e3,
            dummy_points_mirror[:, 1] * 1e3,
            linestyle="dashed",
            color="magenta",
            label="Fitted sphere for unconcentric mirror",
            linewidth=2.0,
            alpha=0.5,
            zorder=2,
        )
        ax[0, 0].legend()

    if potential_x_axis_angles:
        angles_theta, angles_phi = angles_of_unit_vector(ray_sequence[0].k_vector)
        potential_x_axis = angles_phi
        potential_x_label = "phi (rad)"
    else:
        potential_x_axis = wavefront_points[:, 1] * 1e3
        potential_x_label = "y (mm)"
    ax[0, 1].plot(
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
    ax[0, 1].set_xlabel(potential_x_label)
    ax[0, 1].set_ylabel("wavefront difference (µm)")
    ax[0, 1].grid()
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
        ax[0, 1].plot(
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
        title += (
            f"\nmirror deviation fit (unconcentricity = {unconcentricity * 1e6:.1f} µm):\n" + terms_mirror + mode_terms
        )
        ax[0, 1].set_title(title)
        if not potential_x_axis_angles:
            ax[0, 1].plot(
                x_fit * 1e3,
                polynomial_residuals_mirror(x_fit**2) * 1e6,
                color="green",
                linestyle="dashed",
                label="Mirror residuals Polynomial fit",
                linewidth=0.5,
            )
            ax[0, 1].plot(
                x_fit * 1e3,
                polynomial(x_fit**2) * 1e6,
                color="red",
                linestyle="dashed",
                label="Matching sphere residuals Polynomial fit",
                linewidth=0.5,
            )
            # Plot intensity profile on a new y axis for ax[0, 1] if NA_paraxial is not None: using the formula: e^{-y^{2} / (spot_size_paraxial / 2)^{2})}
            if NA_paraxial is not None and spot_size_paraxial is not None:
                ax2 = ax[0, 1].twinx()
                intensity_profile = np.exp(-(x_fit**2) / (spot_size_paraxial / 2) ** 2)
                ax2.plot(
                    x_fit * 1e3,
                    intensity_profile,
                    color="orange",
                    linestyle="dashed",
                    label="Paraxial Gaussian intensity profile",
                    linewidth=1,
                )
                ax2.axvline(
                    spot_size_paraxial * 1e3 * np.sign(x_fit[0]),
                    color="orange",
                    linestyle="dashed",
                    linewidth=1,
                    label="Paraxial spot size ($w_{0}$)",
                )
                legend_line = Line2D(
                    [], [], color="orange", linestyle="dashed", linewidth=1, label="Paraxial Gaussian intensity profile"
                )
                handles, labels = ax[0, 1].get_legend_handles_labels()
                handles.append(legend_line)
                ax2.set_ylabel("Relative intensity (a.u.)")
                ax2.grid(False)

            zero_derivative_point_plot = np.nan if zero_derivative_points is None else zero_derivative_points
            ax[0, 1].axvline(
                zero_derivative_point_plot * 1e3, label="2nd vs 4th order max", color="purple", linestyle="dotted"
            )
            ax[0, 1].legend(handles=handles)
    return fig, ax


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
        raise ValueError("lens_type must be either 'aspheric - lab', 'spherical - like labs aspheric', 'avantier',  'aspheric - like avantier'")
    return n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter

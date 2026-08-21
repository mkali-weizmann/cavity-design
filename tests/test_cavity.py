import copy

import numpy as np
import pytest
from numpy.polynomial import Polynomial


from cavity_design import (
    SphericalMirror,
    CurvatureSigns,
    Cavity,
    LAMBDA_0_LASER,
    FlatMirror,
    FlatRefractiveSurface,
    AsphericRefractiveSurface,
    Ray,
    perturb_cavity,
    evaluate_cavities_modes_on_surface,
    mirror_lens_mirror_cavity_generator,
    fabry_perot_generator,
    OpticalSystem,
    OpticalSurfaceParams,
    generate_lens_from_params,
    MaterialProperties,
    PerturbationPointer,
    gaussians_overlap_integral,
    LensParams,
    solve_aspheric_profile,
    widget_convenient_exponent,
    choose_source_position_for_desired_focus_analytic,
    known_lenses_generator,
    generate_one_lens_optical_system,
    initialize_rays,
    analyze_potential,
    analyze_potential_given_cavity,
    hessian_ray_tracing,
    hessian_ABCD_matrices,
    mirrors_jacobian,
    C_LIGHT_SPEED,
    ORIGIN,
    RIGHT,
    surfaces_are_equivalent,
    CartesianOval,
    RefractiveCartesianOval,
    signed_vertex_radius_of_a_cartesian_oval,
    SphericalRefractiveSurface,
    LEFT,
    normalize_vector,
    cartesian_oval_longitudinal_expansion,
    PHYSICAL_SIZES_DICT,
    generate_cartesian_oval_lens,
    cartesian_oval_lens_intermediate_image_distance,
    CARTESIAN_OVAL_LENS_SPLITS,
    lensmaker_radius_of_a_surface,
    back_focal_length_of_lens_object,
    focal_length_of_lens_object,
    focal_length_of_lens_formula,
    back_focal_length_of_lens_formula,
)


def test_fabry_perot_mode_finding():
    # Compares the numerical result from the analytical solution for a simple Fabry-Perot cavity
    R_1 = 5e-3
    R_2 = 5e-3
    u = 1e-5
    L = R_1 + R_2 - u
    surface_1 = SphericalMirror(
        radius=R_1,
        outwards_normal=np.array([0, 0, -1]),
        center=np.array([0, 0, -R_1]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    surface_2 = SphericalMirror(
        radius=R_2,
        outwards_normal=np.array([0, 0, 1]),
        center=np.array([0, 0, -R_1 + L]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    cavity = Cavity(
        elements=[surface_1, surface_2],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        power=1e3,
        use_paraxial_ray_tracing=False,
    )
    theoretical_reighly_range = np.sqrt(u * L) / 2
    actual_reighly_range = cavity.arms[0].mode_parameters.z_R[0]

    theoretical_waist = np.sqrt(LAMBDA_0_LASER * theoretical_reighly_range / np.pi)
    actual_waist = cavity.arms[0].mode_parameters.w_0[0]
    # print(f'Theoretical Reighly range: {theoretical_reighly_range}, Actual Reighly range: {actual_reighly_range}')
    # print(f'Theoretical Waist: {theoretical_waist}, Actual Waist: {actual_waist}')
    assert (
        theoretical_reighly_range / actual_reighly_range - 1
    ) < 1e-6, f"Fabry Perot generation failed: Reighly range mismatch - theoretical {theoretical_reighly_range}, actual {actual_reighly_range}"
    assert (
        theoretical_waist / actual_waist - 1
    ) < 1e-6, f"Fabry Perot generation failed: Waist mismatch - theoretical {theoretical_waist}, actual {actual_waist}"


def test_fabry_perot_transversal_mode_spacing():
    # Compares cavity.mode_spacing_transversal_apparent against Siegman's analytical
    # transverse-mode-spacing formula for a two-mirror Fabry-Perot resonator.
    # Theory: Siegman, "Lasers", Ch. 19 (PDF pp. 771-789, esp. 788-789). The resonant
    # frequencies are
    #     nu_qmn = (c / 2L) * [q + (m + n + 1) / pi * arccos(+/- sqrt(g_1 * g_2))],
    # so the spacing between adjacent transverse modes is
    #     df_transverse = (1 / pi) * arccos(+/- sqrt(g_1 * g_2)) * FSR,
    # with the sign of the sqrt taken as the (common) sign of g_1, g_2. Note the argument
    # is sqrt(g_1 * g_2), NOT g_1 * g_2. mode_spacing_transversal_apparent folds the result
    # into [0, FSR/2], so we fold the analytical value the same way before comparing.
    R_1 = 5e-3
    R_2 = 7e-3
    u = 50e-6
    L = R_1 + R_2 - u

    mirror_1 = SphericalMirror(
        radius=R_1,
        origin=np.array([0.0, 0.0, 0.0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.0254 / 2,
        outwards_normal=np.array([-1.0, 0.0, 0.0]),
    )
    mirror_2 = SphericalMirror(
        radius=R_2,
        origin=np.array([-u, 0.0, 0.0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.0254 / 2,
        outwards_normal=np.array([1.0, 0.0, 0.0]),
    )
    cavity = Cavity(
        elements=[mirror_1, mirror_2],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        use_paraxial_ray_tracing=True,
        p_is_trivial=True,
        t_is_trivial=True,
    )

    g_1 = 1 - L / R_1
    g_2 = 1 - L / R_2
    fsr = C_LIGHT_SPEED / (2 * L)

    # --- Beam geometry (Siegman, PDF pp. 771-789) --------------------------------------
    # Waist size, spot sizes on the two mirrors, and mirror-to-waist distances.
    w_0_squared_analytical = (
        L
        * LAMBDA_0_LASER
        / np.pi
        * np.sqrt(g_1 * g_2 * (1 - g_1 * g_2) / (g_1 + g_2 - 2 * g_1 * g_2) ** 2)
    )
    w_1_squared_analytical = L * LAMBDA_0_LASER / np.pi * np.sqrt(g_2 / (g_1 * (1 - g_1 * g_2)))
    w_2_squared_analytical = L * LAMBDA_0_LASER / np.pi * np.sqrt(g_1 / (g_2 * (1 - g_1 * g_2)))
    z_1_analytical = g_2 * (1 - g_1) / (g_1 + g_2 - 2 * g_1 * g_2) * L
    z_2_analytical = g_1 * (1 - g_2) / (g_1 + g_2 - 2 * g_1 * g_2) * L

    w_0_squared_numerical = cavity.mode_parameters[0].w_0 ** 2
    w_1_squared_numerical = cavity.arms[0].mode_parameters_on_surface_0.spot_size ** 2
    w_2_squared_numerical = cavity.arms[0].mode_parameters_on_surface_1.spot_size ** 2
    z_1_numerical = cavity.arms[0].mode_parameters_on_surface_0.z_minus_z_0
    z_2_numerical = cavity.arms[0].mode_parameters_on_surface_1.z_minus_z_0

    # Numerical quantities carry a (tangential, sagittal) pair; both agree for this
    # astigmatism-free cavity. z_minus_z_0 is signed relative to the waist, so compare |.|.
    assert w_0_squared_numerical == pytest.approx(w_0_squared_analytical, rel=1e-6)
    assert w_1_squared_numerical == pytest.approx(w_1_squared_analytical, rel=1e-6)
    assert w_2_squared_numerical == pytest.approx(w_2_squared_analytical, rel=1e-6)
    assert np.abs(z_1_numerical) == pytest.approx(np.abs(z_1_analytical), rel=1e-6)
    assert np.abs(z_2_numerical) == pytest.approx(np.abs(z_2_analytical), rel=1e-6)

    # --- Transverse mode spacing -------------------------------------------------------
    gouy_argument = np.sign(g_1) * np.sqrt(g_1 * g_2)
    df_transverse = np.arccos(gouy_argument) / np.pi * fsr
    # Fold into [0, FSR/2] exactly as mode_spacing_transversal_apparent does.
    df_apparent_analytical = np.abs(np.mod(df_transverse + fsr / 2, fsr) - fsr / 2)

    df_apparent_numerical = cavity.mode_spacing_transversal_apparent

    assert cavity.free_spectral_range == pytest.approx(fsr, rel=1e-9)
    assert df_apparent_numerical == pytest.approx(df_apparent_analytical, rel=1e-6), (
        f"Transverse mode spacing mismatch: analytical {df_apparent_analytical}, "
        f"numerical {df_apparent_numerical}"
    )


def test_mirror_lens_mirror_design():
    _mirror_mat = MaterialProperties(
        refractive_index=None,
        alpha_expansion=7.5e-08,
        beta_surface_absorption=1e-06,
        kappa_conductivity=1.31e00,
        dn_dT=None,
        nu_poisson_ratio=1.7e-01,
        alpha_volume_absorption=None,
        intensity_reflectivity=9.99889e-01,
        intensity_transmittance=1e-04,
        temperature=np.nan,
    )
    _lens_mat = MaterialProperties(
        refractive_index=1.76e00,
        alpha_expansion=5.5e-06,
        beta_surface_absorption=1e-06,
        kappa_conductivity=4.606e01,
        dn_dT=1.17e-05,
        nu_poisson_ratio=3e-01,
        alpha_volume_absorption=1e-02,
        intensity_reflectivity=1e-04,
        intensity_transmittance=9.99899e-01,
        temperature=np.nan,
    )
    params = [
        OpticalSurfaceParams(
            name="Small Mirror",
            surface_type="curved_mirror",
            x=-4.999961263669513e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            material_properties=_mirror_mat,
        ),
        generate_lens_from_params(
            center=np.array([6.387599281689135e-03, 0, 0]),
            forward_direction=np.array([1.0, 0.0, 0.0]),
            r_1=2.422e-02,
            r_2=-5.488e-03,
            T_c=2.913797540986543e-03,
            n_inside=1.76e00,
            n_outside=1e00,
            material_properties=_lens_mat,
            name="Lens",
        ).to_params,
        OpticalSurfaceParams(
            name="Big Mirror",
            surface_type="curved_mirror",
            x=4.078081462362321e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            radius=2e-01,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            material_properties=_mirror_mat,
        ),
    ]

    cavity = Cavity.from_params(
        params=params,
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_central_line=True,
        set_mode_parameters=True,
        set_initial_surface=False,
        t_is_trivial=True,
        p_is_trivial=True,
        power=2e4,
        use_paraxial_ray_tracing=True,
        debug_printing_level=1,
    )

    assert (
        cavity.mode_parameters[0].NA[0] / 0.156 - 1 < 1e-4
    ), f"Cavity generation changed: Numerical NA mismatch: expected 0.156, got {cavity.mode_parameters[0].NA}"
    assert (
        cavity.arms[2].central_line.length / 0.4 - 1 < 1e-4
    ), f"Cavity generation changed: Numerical cavity length mismatch: expected 0.4, got {cavity.arms[2].central_line.length}"


def test_aspheric_lens():
    phi = 0.2
    theta = 0.2

    f = 20.0
    T_c = 3.0
    n_1 = 1
    n_2 = 1.5
    polynomial_coefficients = [
        -5.47939897e-06,
        4.54562088e-02,
        4.02452659e-05,
        5.53445352e-08,
        6.96909906e-11,
    ]  # generated for f=20, Tc=3 in aspheric_lens_generator.py
    polynomial = Polynomial(polynomial_coefficients)

    optical_axis = np.array([np.cos(phi), np.sin(phi), 0])

    diameter = 15
    back_center = f * optical_axis
    front_center = back_center + T_c * optical_axis
    s_1 = FlatRefractiveSurface(
        outwards_normal=optical_axis,
        center=back_center,
        n_1=n_1,
        n_2=n_2,
        diameter=diameter,
    )

    s_2 = AsphericRefractiveSurface(
        center=front_center,
        outwards_normal=optical_axis,
        diameter=diameter,
        polynomial_coefficients=polynomial,
        n_1=n_2,
        n_2=n_1,
    )

    ray_initial = Ray(
        origin=np.array([[0, 0, 0], [0, 0, 0]]),
        k_vector=np.array(
            [
                [np.cos(-theta + phi), np.sin(-theta + phi), 0],
                # [np.cos(phi), np.sin(phi), 0],
                [np.cos(theta / 2 + phi), np.sin(theta / 2 + phi), 0],
            ]
        ),
    )

    ray_inner = s_1.propagate_ray(ray_initial)

    ray_output = s_2.propagate_ray(ray_inner)

    rays_are_collimated = np.allclose(
        ray_output.k_vector @ optical_axis, 1.0, atol=1e-7
    )

    # Plot results for visual inspection:
    # fig, ax = plt.subplots(figsize=(15, 15))
    # intersection_2, normals_2 = s_2.enrich_intersection_geometries(ray_inner)
    # output_direction = s_2.scatter_direction_exact(ray_inner)
    # s_1.plot(ax=ax, label='Back Surface', color='black')
    # ray_initial.plot(ax=ax, label='Initial Ray', color='m')
    # s_2.plot(ax=ax, label='Front Surface', color='orange')
    # ray_inner.plot(ax=ax, label='Inner Ray', color='c')
    # ray_output.plot(ax=ax, label='Output Ray', color='r', length=5)
    # for i in range(intersection_2.shape[0]):
    #     ax.plot([intersection_2[i, 0] - normals_2[i, 0]*2, intersection_2[i, 0] + normals_2[i, 0]*2],
    #             [intersection_2[i, 1] - normals_2[i, 1]*2, intersection_2[i, 1] + normals_2[i, 1]*2],
    #             'g--', label='Normal Vector' if i == 0 else "")
    # for i in range(intersection_2.shape[0]):
    #     ax.plot(intersection_2[i, 0], intersection_2[i, 1], 'ro', label='Intersection' if i == 0 else "")
    # ax.legend()
    # plt.axis('equal')
    # ax.grid()
    # ax.set_title(f"{ray_output.k_vector @ optical_axis}\n{ray_initial.k_vector @ optical_axis}")
    # plt.show()

    assert (
        rays_are_collimated
    ), f"Aspheric lens test failed: output rays are not collimated, dot products: {ray_output.k_vector @ optical_axis}"


def test_aspheric_intersection():
    polynomial_coefficients = [0, 1]
    polynomial = Polynomial(polynomial_coefficients)
    optical_axis = np.array([0, 0, 1])
    diameter = 4
    center = np.array([0, 0, 0])
    s = AsphericRefractiveSurface(
        center=center,
        outwards_normal=optical_axis,
        diameter=diameter,
        polynomial_coefficients=polynomial,
        n_1=1,
        n_2=1.5,
    )
    ray_initial = Ray(
        origin=np.array([[0, 2, -6], [0, 0, -6]]),
        k_vector=np.array([[0, 0, 1], [0, np.sqrt(2) / 2, np.sqrt(2) / 2]]),
    )
    intersections, normals = s.enrich_intersection_geometries(ray_initial)
    expected_intersections = np.array([[0, 2, -4], [0, 2, -4]])
    assert np.allclose(
        intersections, expected_intersections, atol=1e-6
    ), f"Aspheric intersection test failed: expected {expected_intersections}, got {intersections}"


def test_aspheric_comparison_to_edmunds():
    params = LensParams(n=1.5168, f=45.23, T_c=7.24)

    coefficients, y, x = solve_aspheric_profile(
        params, y_max=12.5, n_points=1500, return_raw_values=True
    )
    # Specifically for n=1.511, f=0.04523, Tc=0.00724, edmunds optics element:
    k, C, E, F = (
        -8.00424e-1,
        3.869969e-2,
        1.643994e-6,
        5.887865e-10,
    )  # This lens: https://www.edmundoptics.com/p/25mm-dia-x-50mm-fl-vis-ext-lambda40-aspheric-lens/49344/?srsltid=AfmBOopKmF77SQ5bhSl5JyfRk32CIzQF00e6hZqcFTehRTCqJRp8T1j_
    x_edmund = C * y**2 / (1 + np.sqrt(1 - (1 + k) * C**2 * y**2)) + E * y**4 + F * y**6
    residual = x_edmund - x
    max_residual = np.max(np.abs(residual))
    assert (
        max_residual < 1e-4
    ), f"Aspheric comparison to Edmunds failed: max residual {max_residual}"


def test_perturbation():
    power_laser = 5.0000000000e04
    element_index_0 = 1
    param_name_0 = "y"
    perturbation_value_special_log_0 = 2.4733644800e00
    perturbation_value_special_log_0_fine = 6.7924969500e-01

    _ule_mat = MaterialProperties(
        refractive_index=None,
        alpha_expansion=7.5e-08,
        beta_surface_absorption=1e-06,
        kappa_conductivity=1.31e00,
        dn_dT=None,
        nu_poisson_ratio=1.7e-01,
        alpha_volume_absorption=None,
        intensity_reflectivity=9.99889e-01,
        intensity_transmittance=1e-04,
        temperature=np.nan,
    )
    _sapphire_mat = MaterialProperties(
        refractive_index=1.76e00,
        alpha_expansion=5.5e-06,
        beta_surface_absorption=1e-06,
        kappa_conductivity=4.606e01,
        dn_dT=1.17e-05,
        nu_poisson_ratio=3e-01,
        alpha_volume_absorption=1e-02,
        intensity_reflectivity=1e-04,
        intensity_transmittance=9.99899e-01,
        temperature=np.nan,
    )
    params = [
        OpticalSurfaceParams(
            name="Small Mirror",
            surface_type="curved_mirror",
            x=-4.999954683912563e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=7.75e-03,
            material_properties=_ule_mat,
            polynomial_coefficients=None,
        ),
        generate_lens_from_params(
            center=np.array([6.387599281689135e-03, 0, 0]),
            forward_direction=np.array([1.0, 0.0, 0.0]),
            r_1=2.422e-02,
            r_2=-5.488e-03,
            T_c=2.913797540986543e-03,
            n_inside=1.76e00,
            n_outside=1e00,
            diameter=7.75e-03,
            material_properties=_sapphire_mat,
            name="Lens",
        ).to_params,
        OpticalSurfaceParams(
            name="Big Mirror",
            surface_type="curved_mirror",
            x=4.074677357638641e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            radius=2e-01,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=2.54e-02,
            material_properties=_ule_mat,
            polynomial_coefficients=None,
        ),
    ]

    perturbation_value_0 = widget_convenient_exponent(
        perturbation_value_special_log_0, base=10, scale=-10
    )

    perturbation_value_0_fine = widget_convenient_exponent(
        perturbation_value_special_log_0_fine, base=10, scale=-10
    )

    perturbation_value_0 += perturbation_value_0_fine

    cavity = Cavity.from_params(
        params=params,
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        power=power_laser,
        p_is_trivial=True,
        t_is_trivial=True,
        use_paraxial_ray_tracing=True,
        set_central_line=True,
        set_mode_parameters=True,
    )
    perturbation_pointers = [
        PerturbationPointer(
            element_index=element_index_0,
            parameter_name=param_name_0,
            perturbation_value=perturbation_value_0,
        ),
    ]
    perturbed_cavity = perturb_cavity(
        cavity=cavity, perturbation_pointer=perturbation_pointers
    )

    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_mode = evaluate_cavities_modes_on_surface(
        cavity, perturbed_cavity, arm_index=0
    )
    overlap = gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)
    print("asdasdasd", np.abs(overlap))
    assert (
        np.abs(overlap) - 0.9001508804272882
    ) < 1e-6, f"Perturbation test failed: Expected overlap of 0.9001508804272882 but got {np.abs(overlap)}"
    assert np.all(
        np.isclose(
            perturbed_cavity.central_line[0].k_vector,
            np.array([0.9999149256, -0.0130438341, 0]),
        )
    ), f"Perturbation test failed:  Expected k_vector of [0.99991744, 0.0128496 , 0.        ] but got {perturbed_cavity.central_line[0].k_vector}"


def test_cavity_smart_generation():
    waist_to_left_mirror = None
    NA_left = 1.5000000000e-01
    waist_to_lens = 5.0000000000e-03
    waist_to_lens_fine = -5.8407300310e00
    set_R_left_to_collimate = False
    R_small_mirror = 5.0000000000e-03
    R_left = 2.4220000000e-02
    R_left_fine = -1.3552527156e-20
    set_R_right_to_collimate = False
    set_R_right_to_equalize_angles = False
    set_R_right_to_R_left = False
    R_right = 5.4880000000e-03
    R_right_fine = -1.3552527156e-20
    collimation_mode = "symmetric arm"
    auto_set_big_mirror_radius = False
    big_mirror_radius = 2.0000000000e-01
    auto_set_right_arm_length = True
    right_arm_length = 4.0000000000e-01
    lens_fixed_properties = "sapphire"
    mirrors_fixed_properties = "ULE"
    T_edge = 1.0000000000e-03
    h = 3.8750000000e-03

    big_mirror_radius = None if auto_set_big_mirror_radius else big_mirror_radius
    right_arm_length = None if auto_set_right_arm_length else right_arm_length
    waist_to_lens += widget_convenient_exponent(waist_to_lens_fine)
    R_left += widget_convenient_exponent(R_left_fine)
    R_right += widget_convenient_exponent(R_right_fine)

    cavity = mirror_lens_mirror_cavity_generator(
        NA_left=NA_left,
        waist_to_lens=waist_to_lens,
        h=h,
        R_left=R_left,
        R_right=R_right,
        T_c=0,
        T_edge=T_edge,
        lens_fixed_properties=lens_fixed_properties,
        mirrors_fixed_properties=mirrors_fixed_properties,
        R_small_mirror=R_small_mirror,
        waist_to_left_mirror=waist_to_left_mirror,
        lambda_0_laser=1064e-9,
        power=2e4,
        set_h_instead_of_w=True,
        collimation_mode=collimation_mode,
        big_mirror_radius=big_mirror_radius,
        right_arm_length=right_arm_length,
        set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
        set_R_right_to_R_left=set_R_right_to_R_left,
        set_R_left_to_collimate=set_R_left_to_collimate,
        set_R_right_to_collimate=set_R_right_to_collimate,
    )

    assert np.all(
        np.isclose(
            cavity.mode_parameters[0].center,
            np.array(
                [[0, 0.00000000e00, 0.00000000e00], [0, 0.00000000e00, 0.00000000e00]]
            ),
        )
    ), f"cavity_smart_generation_test failed: center should be approximately [[8.67361738e-19, 0.00000000e+00, 0.00000000e+00], [8.67361738e-19, 0.00000000e+00, 0.00000000e+00]], instead got {cavity.mode_parameters[0].center}"
    assert np.all(
        np.isclose(
            cavity.mode_parameters[0].z_R, np.array([1.50525208e-05, 1.50525208e-05])
        )
    ), f"cavity_smart_generation_test failed: z_R should be approximately 1.50525208e-05, instead got {cavity.mode_parameters[0].z_R}"


def test_fabry_perot_perturbation():
    power_laser = 5.0000000000e04
    element_index_0 = 0
    param_name_0 = "x"
    perturbation_value_special_log_0 = -2.6766707630e00
    perturbation_value_special_log_0_fine = 0.0000000000e00
    element_index_1 = 1
    param_name_1 = "phi"
    perturbation_value_special_log_1 = -2.0200114290e00
    perturbation_value_special_log_1_fine = 1.7763568394e-15
    eval_box = ""
    _fp_mat = MaterialProperties(
        refractive_index=1.45e00,
        alpha_expansion=5.2e-07,
        beta_surface_absorption=1e-06,
        kappa_conductivity=1.38e00,
        dn_dT=1.2e-05,
        nu_poisson_ratio=1.6e-01,
        alpha_volume_absorption=1e-03,
        intensity_reflectivity=1e-04,
        intensity_transmittance=9.99899e-01,
        temperature=np.nan,
    )
    params = [
        OpticalSurfaceParams(
            name="None",
            surface_type="curved_mirror",
            x=-4.999964994473332e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=np.nan,
            material_properties=_fp_mat,
        ),
        OpticalSurfaceParams(
            name="None",
            surface_type="curved_mirror",
            x=4.999964994473332e-03,
            y=0,
            z=0,
            theta=0,
            phi=0,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=np.nan,
            material_properties=_fp_mat,
        ),
    ]
    perturbation_value_0 = widget_convenient_exponent(
        perturbation_value_special_log_0, base=10, scale=-10
    )
    perturbation_value_1 = widget_convenient_exponent(
        perturbation_value_special_log_1, base=10, scale=-10
    )

    perturbation_value_0_fine = widget_convenient_exponent(
        perturbation_value_special_log_0_fine, base=10, scale=-10
    )
    perturbation_value_1_fine = widget_convenient_exponent(
        perturbation_value_special_log_1_fine, base=10, scale=-10
    )

    perturbation_value_0 += perturbation_value_0_fine
    perturbation_value_1 += perturbation_value_1_fine

    cavity = Cavity.from_params(
        params=params,
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        power=power_laser,
        p_is_trivial=True,
        t_is_trivial=True,
        use_paraxial_ray_tracing=False,
        set_central_line=True,
        set_mode_parameters=True,
    )
    perturbation_pointers = [
        PerturbationPointer(
            element_index=element_index_0,
            parameter_name=param_name_0,
            perturbation_value=perturbation_value_0,
        ),
        PerturbationPointer(
            element_index=element_index_1,
            parameter_name=param_name_1,
            perturbation_value=perturbation_value_1,
        ),
    ]
    perturbed_cavity = perturb_cavity(
        cavity=cavity, perturbation_pointer=perturbation_pointers
    )
    if eval_box != "":
        try:
            exec(f"print({eval_box})")
        except (NameError, AttributeError) as e:
            print(f"invalid expression: {e}")
    u = np.linalg.norm(
        perturbed_cavity.physical_surfaces[0].origin
        - perturbed_cavity.physical_surfaces[1].origin
    )
    NA_analytical = np.sqrt(2 * LAMBDA_0_LASER / np.pi) * (
        perturbed_cavity.arms[0].central_line.length * u
    ) ** (-1 / 4)
    NA_numerical = perturbed_cavity.mode_parameters[0].NA[0]
    assert np.isclose(
        NA_numerical, NA_analytical, rtol=0.0001
    ), f"Fabry-Perot perturbation test failed: expected NA of approximately {NA_analytical} but got {NA_numerical}"


def test_complex_cavity_perturbation():
    params = [
        OpticalSurfaceParams(
            name="Small Mirror",
            surface_type="curved_mirror",
            x=-4.999954683912563e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=7.75e-03,
            material_properties=MaterialProperties(
                refractive_index=None,
                alpha_expansion=7.5e-08,
                beta_surface_absorption=1e-06,
                kappa_conductivity=1.31e00,
                dn_dT=None,
                nu_poisson_ratio=1.7e-01,
                alpha_volume_absorption=None,
                intensity_reflectivity=9.99889e-01,
                intensity_transmittance=1e-04,
                temperature=np.nan,
            ),
            polynomial_coefficients=None,
        ),
        [
            OpticalSurfaceParams(
                name="Lens_left",
                surface_type="curved_refractive_surface",
                x=4.930700511195863e-03,
                y=0,
                z=0,
                theta=0,
                phi=1e00 * np.pi,
                radius=2.422e-02,
                curvature_sign=CurvatureSigns.convex,
                T_c=np.nan,
                n_inside_or_after=1.76e00,
                n_outside_or_before=1e00,
                diameter=7.75e-03,
                material_properties=MaterialProperties(
                    refractive_index=1.76e00,
                    alpha_expansion=5.5e-06,
                    beta_surface_absorption=1e-06,
                    kappa_conductivity=4.606e01,
                    dn_dT=1.17e-05,
                    nu_poisson_ratio=3e-01,
                    alpha_volume_absorption=1e-02,
                    intensity_reflectivity=1e-04,
                    intensity_transmittance=9.99899e-01,
                    temperature=np.nan,
                ),
                polynomial_coefficients=None,
            ),
            OpticalSurfaceParams(
                name="Lens_right",
                surface_type="curved_refractive_surface",
                x=7.844498052182406e-03,
                y=0,
                z=0,
                theta=0,
                phi=0,
                radius=5.488e-03,
                curvature_sign=CurvatureSigns.concave,
                T_c=np.nan,
                n_inside_or_after=1e00,
                n_outside_or_before=1.76e00,
                diameter=7.75e-03,
                material_properties=MaterialProperties(
                    refractive_index=1.76e00,
                    alpha_expansion=5.5e-06,
                    beta_surface_absorption=1e-06,
                    kappa_conductivity=4.606e01,
                    dn_dT=1.17e-05,
                    nu_poisson_ratio=3e-01,
                    alpha_volume_absorption=1e-02,
                    intensity_reflectivity=1e-04,
                    intensity_transmittance=9.99899e-01,
                    temperature=np.nan,
                ),
                polynomial_coefficients=None,
            ),
        ],
        OpticalSurfaceParams(
            name="Big Mirror",
            surface_type="curved_mirror",
            x=4.074677357638641e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            radius=2e-01,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=2.54e-02,
            material_properties=MaterialProperties(
                refractive_index=None,
                alpha_expansion=7.5e-08,
                beta_surface_absorption=1e-06,
                kappa_conductivity=1.31e00,
                dn_dT=None,
                nu_poisson_ratio=1.7e-01,
                alpha_volume_absorption=None,
                intensity_reflectivity=9.99889e-01,
                intensity_transmittance=1e-04,
                temperature=np.nan,
            ),
            polynomial_coefficients=None,
        ),
    ]

    cavity_paraxial = Cavity.from_params(
        params=params,
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_central_line=True,
        set_mode_parameters=True,
        t_is_trivial=True,
        p_is_trivial=True,
        power=2e4,
        use_paraxial_ray_tracing=True,
        debug_printing_level=1,
    )
    perturbable_params_names = ["x", "y", "phi"]
    tolerance_df = cavity_paraxial.generate_tolerance_dataframe(
        perturbable_params_names=perturbable_params_names
    )
    tolerance_df_numpy = tolerance_df.to_numpy()
    known_result = np.array(
        [
            [-2.01466680e-06, 3.11409932e-08, -6.23125000e-06],
            [2.02635293e-06, 3.02079512e-08, np.nan],
            [2.02971058e-03, -9.82278238e-07, 4.91029984e-06],
        ]
    )
    # relative tolerance is a bit high because results change slightly from one run to another.
    assert np.allclose(
        np.abs(tolerance_df_numpy), np.abs(known_result), rtol=1e-2, equal_nan=True
    ), f"Complex cavity perturbation test failed: expected tolerance dataframe \n{known_result}\n but got \n{tolerance_df_numpy}"


def test_potential_single_lens():
    dn = 0
    lens_types = [
        "aspheric - lab",
        "spherical - like labs aspheric",
        "avantier",
        "aspheric - like avantier",
    ]
    lens_type = lens_types[2]
    n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = (
        known_lenses_generator(lens_type=lens_type, dn=dn)
    )
    n_rays = 400
    unconcentricity = 2.24255506e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
    phi_max = 0.04
    desired_focus = 200e-3
    print_tests = True

    defocus = choose_source_position_for_desired_focus_analytic(
        back_focal_length=back_focal_length,
        desired_focus=desired_focus,
        T_c=T_c,
        n=n_design,
        diameter=diameter,
        R_1=R_1,
        R_2=R_2_signed,
    )

    optical_system, optical_axis = generate_one_lens_optical_system(
        R_1=R_1,
        R_2=R_2_signed,
        back_focal_length=back_focal_length,
        defocus=defocus,
        T_c=T_c,
        n_design=n_design,
        diameter=diameter,
        n_actual=n_actual,
    )
    rays_0 = initialize_rays(phi_max=phi_max, n_rays=n_rays)
    results_dict = analyze_potential(
        optical_system=optical_system,
        rays_0=rays_0,
        unconcentricity=unconcentricity,
        print_tests=print_tests,
        potential_horizontal_axis_in_NAs=False,
    )
    assert np.isclose(
        np.abs(results_dict["zero_derivative_point"] * 1e3), 0.15342637331775477
    ), f"Potential single lens test failed: expected zero derivative point at approximately 0.15342637331775477 mm but got {results_dict['zero_derivative_point']*1e3} mm"


def test_free_potential_vs_cavity_potential_comparison():
    dn = 0
    lens_types = [
        "aspheric - lab",
        "spherical - like labs aspheric",
        "avantier",
        "aspheric - like avantier",
    ]
    lens_type = lens_types[0]
    n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = (
        known_lenses_generator(lens_type=lens_type, dn=dn)
    )
    n_rays = 30
    unconcentricity = (
        1e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
    )
    phi_max = 0.14
    desired_focus = 200e-3
    print_tests = True

    defocus = choose_source_position_for_desired_focus_analytic(
        back_focal_length=back_focal_length,
        desired_focus=desired_focus,
        T_c=T_c,
        n=n_design,
        diameter=diameter,
        R_1=R_1,
        R_2=R_2_signed,
    )
    optical_system, optical_axis = generate_one_lens_optical_system(
        R_1=R_1,
        R_2=R_2_signed,
        back_focal_length=back_focal_length,
        defocus=defocus,
        T_c=T_c,
        n_design=n_design,
        diameter=diameter,
        n_actual=n_actual,
    )
    rays_0 = initialize_rays(phi_max=phi_max, n_rays=n_rays)
    results_dict = analyze_potential(
        optical_system=optical_system,
        rays_0=rays_0,
        unconcentricity=unconcentricity,
        print_tests=print_tests,
        potential_horizontal_axis_in_NAs=False,
    )
    cavity = results_dict["cavity"]
    print(cavity.ABCD_round_trip)
    results_dict_cavity = analyze_potential_given_cavity(
        cavity=cavity,
        n_rays=30,
        phi_max=0.14,
        print_tests=True,
        potential_horizontal_axis_in_NAs=False,
    )
    assert np.all(
        np.isclose(
            np.array(
                [
                    results_dict["zero_derivative_point"],
                    results_dict["polynomial_residuals_mirror"].coef[2],
                    results_dict["polynomial_residuals_opposite"].coef[1],
                ]
            ),
            np.array(
                [
                    results_dict_cavity["zero_derivative_point"],
                    results_dict_cavity["polynomial_residuals_mirror"].coef[2],
                    results_dict_cavity["polynomial_residuals_opposite"].coef[1],
                ]
            ),
        )
    ), "Results from analyze_potential_given_cavity do not match results from analyze_potential for the same cavity."


def test_spot_size_from_potential_and_ray_tracing():
    _fused_silica_mat = MaterialProperties(
        refractive_index=1.45e00,
        alpha_expansion=5.2e-07,
        beta_surface_absorption=1e-06,
        kappa_conductivity=1.38e00,
        dn_dT=1.2e-05,
        nu_poisson_ratio=1.6e-01,
        alpha_volume_absorption=1e-03,
        intensity_reflectivity=1e-04,
        intensity_transmittance=9.99899e-01,
        temperature=np.nan,
    )
    _sapphire_mat2 = MaterialProperties(
        refractive_index=1.76e00,
        alpha_expansion=5.5e-06,
        beta_surface_absorption=1e-06,
        kappa_conductivity=4.606e01,
        dn_dT=1.17e-05,
        nu_poisson_ratio=3e-01,
        alpha_volume_absorption=1e-02,
        intensity_reflectivity=1e-04,
        intensity_transmittance=9.99899e-01,
        temperature=np.nan,
    )
    params = [
        OpticalSurfaceParams(
            name="LaserOptik mirror",
            surface_type="curved_mirror",
            x=-5e-03,
            y=0,
            z=0,
            theta=0,
            phi=1e00 * np.pi,
            radius=5e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=7.75e-03,
            material_properties=_fused_silica_mat,
            polynomial_coefficients=None,
        ),
        generate_lens_from_params(
            center=np.array([6.776592092031389e-03, 0, 0]),
            forward_direction=np.array([1.0, 0.0, 0.0]),
            r_1=2.422e-02,
            r_2=-5.488e-03,
            T_c=2.913797540986543e-03,
            n_inside=1.76e00,
            n_outside=1e00,
            diameter=7.75e-03,
            material_properties=_sapphire_mat2,
            name="spherical_lens",
        ).to_params,
        generate_lens_from_params(
            center=np.array([4.190164703571147e-01, 0, 0]),
            forward_direction=np.array([1.0, 0.0, 0.0]),
            r_1=-3.561084685817112e-02,
            r_2=1.732922172776388e-01,
            T_c=4.350000000000001e-03,
            n_inside=1.45e00,
            n_outside=1e00,
            diameter=5e-02,
            material_properties=_fused_silica_mat,
            name="Negative Lens",
        ).to_params,
        OpticalSurfaceParams(
            name="big mirror",
            surface_type="curved_mirror",
            x=4.330042644697557e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            radius=6.896719562240133e-02,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=5e-02,
            material_properties=_fused_silica_mat,
            polynomial_coefficients=None,
        ),
    ]

    optical_system_small_elements = OpticalSystem.from_params(
        params[:-1],
        lambda_0_laser=LAMBDA_0_LASER,
        use_paraxial_ray_tracing=False,
        p_is_trivial=True,
        t_is_trivial=True,
    )
    R = params[-1].radius
    u = 5e-6
    # Cavity with a known unconcentricity in the last arm:
    cavity = optical_system_small_elements.complete_to_cavity(
        unconcentricity=u, end_mirror_ROC=R
    )

    results_dict = analyze_potential_given_cavity(
        cavity=cavity,
        n_rays=10,
        phi_max=0.01,
        print_tests=False,
        potential_horizontal_axis_in_NAs=False,
    )
    a_2_numerical = results_dict["polynomial_residuals_mirror"].coef[1]
    a_2_analytical = u / (2 * R**2)
    assert np.isclose(
        a_2_numerical, a_2_analytical, rtol=5e-3
    ), f"Spot size from potential test failed: expected quadratic coefficient of approximately {a_2_analytical} but got {a_2_numerical}"

    hessian = hessian_ABCD_matrices(cavity=cavity, n_rays=1, phi_max=0.01)[
        0, 0
    ]  # First zero because we have one ray, second 0 because hessian is isotropic for non astigmatic systems at the optical axis.

    jacobian = mirrors_jacobian(cavity=cavity)

    hessian_normalized = hessian * jacobian
    a_2_normalized = a_2_analytical * jacobian

    spot_size_squared_from_potential = cavity.lambda_0_laser / (
        np.pi * np.sqrt(-2 * hessian_normalized * a_2_normalized)
    )
    spot_size_squared_from_optics = (
        cavity.arms[len(cavity.arms) // 2 - 1].mode_parameters_on_surface_1.spot_size[0]
        ** 2
    )
    assert np.isclose(
        spot_size_squared_from_potential, spot_size_squared_from_optics, rtol=5e-3
    ), f"Spot size comparison test failed: expected spot size squared from potential of approximately {spot_size_squared_from_potential} but got {spot_size_squared_from_optics}"

    energy_level_hessian_and_potential = (
        np.sqrt(a_2_normalized / (-2 * hessian_normalized))
        * cavity.lambda_0_laser
        / np.pi
    )
    energy_level_hessian_and_spot_size = cavity.lambda_0_laser**2 / (
        2 * np.pi**2 * spot_size_squared_from_potential * hessian_normalized
    )
    assert np.isclose(
        energy_level_hessian_and_potential,
        energy_level_hessian_and_spot_size,
        rtol=1e-3,
    ), f"Energy level comparison test failed: expected energy level from hessian and potential of approximately {energy_level_hessian_and_potential} but got {energy_level_hessian_and_spot_size}"


def test_analytical_hessian_for_fabry_perot():
    u = 1e-6
    R_0 = 5e-3
    R_1 = 15e-3
    cavity = fabry_perot_generator(
        (R_0, R_1),
        unconcentricity=u,
        lambda_0_laser=LAMBDA_0_LASER,
        use_paraxial_ray_tracing=False,
    )
    hessian_ray_tracing_value = hessian_ray_tracing(
        cavity=cavity, n_rays=1, phi_max=0.1
    )[0, 0]
    hessian_ABCD_matrices_value = hessian_ABCD_matrices(
        cavity=cavity, n_rays=1, phi_max=0.1
    )[0, 0]
    hessian_analytical = -R_1 / ((R_0 + R_1) * R_0)
    assert np.isclose(
        hessian_ray_tracing_value, hessian_analytical, rtol=1e-3
    ), f"Hessian ray tracing test failed: expected approximately {hessian_analytical} but got {hessian_ray_tracing_value}"
    assert np.isclose(
        hessian_ABCD_matrices_value, hessian_analytical, rtol=1e-3
    ), f"Hessian ABCD matrices test failed: expected approximately {hessian_analytical} but got {hessian_ABCD_matrices_value}"


# ---------------------------------------------------------------------------
# Nested OpticalSystem (rigid-body) tests
# ---------------------------------------------------------------------------


def _make_lens_group(center_x, forward=np.array([1.0, 0.0, 0.0])):
    """Helper: returns an OpticalSystem wrapping two refractive surfaces (a thin lens group)."""
    from cavity_design import SphericalRefractiveSurface

    T_c = 3e-3
    R = 20e-3
    n_glass = 1.5
    normal_in = -forward
    normal_out = forward
    s1 = SphericalRefractiveSurface(
        radius=R,
        outwards_normal=normal_in,
        center=np.array([center_x - T_c / 2, 0.0, 0.0]),
        n_1=1.0,
        n_2=n_glass,
        curvature_sign=1,
        name="lens_front",
    )
    s2 = SphericalRefractiveSurface(
        radius=R,
        outwards_normal=normal_out,
        center=np.array([center_x + T_c / 2, 0.0, 0.0]),
        n_1=n_glass,
        n_2=1.0,
        curvature_sign=-1,
        name="lens_back",
    )
    lens_center = np.array([center_x, 0.0, 0.0])
    return OpticalSystem(
        [s1, s2],
        use_paraxial_ray_tracing=False,
        given_initial_central_line=None,
        mechanical_center=lens_center,
    )


def test_nested_optical_system_flat_arms():
    """Nesting an OpticalSystem inside another should flatten arms correctly."""
    from cavity_design import SphericalMirror

    R = 50e-3
    L = 100e-3
    m1 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=-1,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([L / 2, 0, 0]),
        curvature_sign=-1,
    )

    sys = OpticalSystem(
        [m1, lens, m2], given_initial_central_line=None, use_paraxial_ray_tracing=False
    )
    # Flat arms: Arm(m1, lens_front), Arm(lens_front, lens_back), Arm(lens_back, m2)
    assert len(sys.arms) == 3, f"Expected 3 arms, got {len(sys.arms)}"
    assert sys.arms[0].surface_0 is m1
    assert sys.arms[0].surface_1 is lens._surfaces[0]
    assert sys.arms[1].surface_0 is lens._surfaces[0]
    assert sys.arms[1].surface_1 is lens._surfaces[1]
    assert sys.arms[2].surface_0 is lens._surfaces[1]
    assert sys.arms[2].surface_1 is m2


def test_nested_to_params_from_params_roundtrip():
    """to_params on a system with a nested group returns a nested list; from_params reconstructs it."""
    from cavity_design import SphericalMirror

    R = 50e-3
    L = 100e-3
    m1 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=-1,
        name="m1",
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([L / 2, 0, 0]),
        curvature_sign=-1,
        name="m2",
    )

    sys = OpticalSystem(
        [m1, lens, m2], given_initial_central_line=None, use_paraxial_ray_tracing=False
    )
    params = sys.to_params

    # params[0] is a single OpticalSurfaceParams, params[1] is a list, params[2] is single
    assert not isinstance(
        params[0], list
    ), "First element should be a flat OpticalSurfaceParams"
    assert isinstance(
        params[1], list
    ), "Second element should be a nested list for the lens group"
    assert len(params[1]) == 2, "Lens group should have 2 surface params"
    assert not isinstance(
        params[2], list
    ), "Third element should be a flat OpticalSurfaceParams"

    # Roundtrip: from_params should reconstruct a system with the same arm count
    reconstructed = OpticalSystem.from_params(params, given_initial_central_line=None)
    assert (
        len(reconstructed.arms) == 3
    ), f"Expected 3 arms after roundtrip, got {len(reconstructed.arms)}"
    # Surface positions should be preserved
    for orig_arm, rec_arm in zip(sys.arms, reconstructed.arms):
        assert np.allclose(
            orig_arm.surface_0.center, rec_arm.surface_0.center, atol=1e-12
        )
        assert np.allclose(
            orig_arm.surface_1.center, rec_arm.surface_1.center, atol=1e-12
        )


def test_rigid_body_translation_perturbation():
    """perturb_cavity with a nested element and translation parameter moves both surfaces."""
    R_mirror = 50e-3
    L = 200e-3
    m1 = SphericalMirror(
        radius=R_mirror,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m1",
        diameter=25e-3,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = SphericalMirror(
        radius=R_mirror,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m2",
        diameter=25e-3,
    )

    cavity = Cavity(
        [m1, lens, m2],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_mode_parameters=False,
    )

    delta_y = 1e-4
    pp = PerturbationPointer(
        element_index=1, parameter_name="y", perturbation_value=delta_y
    )
    new_cavity = perturb_cavity(cavity, [pp])

    # Both lens surfaces should have shifted by delta_y in y
    lens_surfaces_new = [
        arm.surface_0
        for arm in new_cavity.arms
        if hasattr(arm.surface_0, "name")
        and arm.surface_0.name in ("lens_front", "lens_back")
    ]
    for s in lens_surfaces_new:
        assert np.isclose(
            s.center[1], delta_y, atol=1e-14
        ), f"Expected y={delta_y} for {s.name}, got {s.center[1]}"


def test_rigid_body_rotation_perturbation():
    """perturb_cavity with a nested element and rotation parameter rotates both surfaces around mechanical_center."""
    R_mirror = 50e-3
    L = 200e-3
    m1 = SphericalMirror(
        radius=R_mirror,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m1",
        diameter=25e-3,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = SphericalMirror(
        radius=R_mirror,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m2",
        diameter=25e-3,
    )

    cavity = Cavity(
        [m1, lens, m2],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_mode_parameters=False,
    )

    delta_theta = 0.02  # small tilt
    pp = PerturbationPointer(
        element_index=1, parameter_name="theta", perturbation_value=delta_theta
    )
    new_cavity = perturb_cavity(cavity, [pp])

    # Inspect the perturbed params: new_cavity.to_params[1] should be a list of 2 params
    new_params = new_cavity.to_params
    assert isinstance(
        new_params[1], list
    ), "Lens group params should still be a nested list"
    assert len(new_params[1]) == 2

    for sp in new_params[1]:
        # The normal's x-component must be < 1 (lens has been tilted away from pure x-axis)
        normal = np.array(
            [
                np.sin(sp.theta) * np.cos(sp.phi),
                np.sin(sp.theta) * np.sin(sp.phi),
                np.cos(sp.theta),
            ]
        )
        assert (
            abs(normal[0]) < 1.0 - 1e-6
        ), f"Normal should no longer be purely along x after rotation, got normal={normal}"


# ---------------------------------------------------------------------------
# Flexible positioning: undefined poses, setters, relative coordinates,
# object-based rigid-body perturbation
# ---------------------------------------------------------------------------


def _simple_mirror_params(name, x, phi):
    return OpticalSurfaceParams(
        name=name,
        surface_type="curved_mirror",
        x=x,
        y=0,
        z=0,
        theta=0,
        phi=phi,
        radius=5e-3,
        curvature_sign=CurvatureSigns.concave,
        T_c=np.nan,
        n_inside_or_after=1,
        n_outside_or_before=1,
        diameter=0.01,
        material_properties=MaterialProperties(),
    )


def test_surface_undefined_pose_and_setters():
    # A curved surface can be created with undefined (nan) pose, and the setters bring it to a defined state.
    R = 5e-3
    m = SphericalMirror(
        radius=R,
        outwards_normal=None,
        center=None,
        curvature_sign=CurvatureSigns.concave,
    )
    assert not m.positions_defined
    assert np.all(np.isnan(m.center)) and np.all(np.isnan(m.outwards_normal))

    m.outwards_normal = np.array([-1.0, 0.0, 0.0])
    m.center = np.array([-R, 0.0, 0.0])
    assert m.positions_defined
    assert np.allclose(m.center, [-R, 0.0, 0.0], atol=1e-15)
    # origin is the sphere center = vertex + R * inwards_normal = [-R,0,0] + R*[1,0,0] = [0,0,0]
    assert np.allclose(m.origin, [0.0, 0.0, 0.0], atol=1e-15)

    # The curved center setter round-trips with the getter for an arbitrary normal/center.
    m.outwards_normal = np.array([1.0, 2.0, 0.0])
    target_center = np.array([3e-3, -1e-3, 0.5e-3])
    m.center = target_center
    assert np.allclose(m.center, target_center, atol=1e-15)

    # radius setter keeps the vertex fixed and moves the sphere origin.
    m.outwards_normal = np.array([-1.0, 0.0, 0.0])
    m.center = np.array([-R, 0.0, 0.0])
    m.radius = R + 1e-3
    assert np.isclose(m.radius, R + 1e-3)
    assert np.allclose(m.center, [-R, 0.0, 0.0], atol=1e-15)


def test_undefined_optical_system_construction_skips_and_raises():
    # A cavity with an undefined-pose element constructs without error and skips tracing; explicit calls raise.
    R = 5e-3
    m1 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-R, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m2 = SphericalMirror(
        radius=R,
        outwards_normal=None,
        center=None,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )  # undefined
    cavity = Cavity([m1, m2], standing_wave=True, lambda_0_laser=LAMBDA_0_LASER)
    assert not cavity.positions_defined
    assert cavity.central_line is None
    with pytest.raises(ValueError):
        cavity.set_central_line()


def test_relative_coordinate_resolution():
    from cavity_design import OpticalSystem

    p0 = _simple_mirror_params("a", x=-5e-3, phi=np.pi)
    p1_abs = _simple_mirror_params("b", x=5e-3, phi=0)
    # Relative: previous resolved x is -5e-3, a +10e-3 step lands on +5e-3.
    p1_rel = _simple_mirror_params("b", x=10e-3 * 1j, phi=0)

    sys_abs = OpticalSystem.from_params([p0, p1_abs], given_initial_central_line=None)
    sys_rel = OpticalSystem.from_params([p0, p1_rel], given_initial_central_line=None)
    assert np.allclose(
        sys_rel.surfaces[1].center, sys_abs.surfaces[1].center, atol=1e-15
    )
    assert np.isclose(sys_rel.surfaces[1].center[0], 5e-3)

    # A tiny imaginary part (numerical noise) is treated as absolute, not relative.
    p1_tiny = _simple_mirror_params("b", x=5e-3 + 1e-18j, phi=0)
    sys_tiny = OpticalSystem.from_params([p0, p1_tiny], given_initial_central_line=None)
    assert np.isclose(sys_tiny.surfaces[1].center[0], 5e-3)

    # A genuinely mixed real+imaginary coordinate is ambiguous and must raise.
    p1_mixed = _simple_mirror_params("b", x=1e-3 + 3e-3j, phi=0)
    with pytest.raises(ValueError):
        OpticalSystem.from_params([p0, p1_mixed], given_initial_central_line=None)


def test_object_vs_params_rigid_body_parity():
    from cavity_design import (
        apply_rigid_body_perturbation,
        apply_rigid_body_perturbation_to_params,
        unit_vector_of_angles,
    )

    lens = _make_lens_group(
        center_x=0.0
    )  # OpticalSystem with explicit mechanical_center
    base_params = lens.to_params  # list of 2 OpticalSurfaceParams

    for parameter_name, value in [("y", 1e-4), ("theta", 0.02), ("phi", -0.013)]:
        lens_obj = copy.deepcopy(lens)
        params = copy.deepcopy(base_params)
        apply_rigid_body_perturbation(lens_obj, parameter_name, value)
        apply_rigid_body_perturbation_to_params(
            params, parameter_name, value, mechanical_center=lens.mechanical_center
        )
        for surf, p in zip(lens_obj._surfaces, params):
            assert np.allclose(
                surf.center, [p.x, p.y, p.z], atol=1e-12
            ), f"{parameter_name}: center mismatch {surf.center} vs {[p.x, p.y, p.z]}"
            assert np.allclose(
                surf.outwards_normal, unit_vector_of_angles(p.theta, p.phi), atol=1e-12
            ), f"{parameter_name}: normal mismatch"


def test_perturb_cavity_radius_preserves_vertex():
    # The object-based scalar perturbation of a mirror radius keeps the vertex fixed and changes the radius.
    R = 5e-3
    m1 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-R, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m2 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([R - 1e-5, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    cavity = Cavity(
        [m1, m2],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_mode_parameters=False,
    )
    vertex_before = cavity.surfaces[0].center.copy()

    dR = 1e-4
    pp = PerturbationPointer(
        element_index=0, parameter_name="radius", perturbation_value=dR
    )
    new_cavity = perturb_cavity(cavity, [pp], set_mode_parameters=False)

    assert np.isclose(new_cavity.surfaces[0].radius, R + dR)
    assert np.allclose(new_cavity.surfaces[0].center, vertex_before, atol=1e-12)


def test_set_element_position_rigid():
    from cavity_design import set_element_position

    lens = _make_lens_group(
        center_x=0.0
    )  # two surfaces, 3mm apart, centered at the origin
    offset = (
        lens.surfaces[1].center - lens.surfaces[0].center
    )  # rigid separation to be preserved

    target = np.array([1.0, 0.2, -0.1])
    returned = set_element_position(lens, target, reference="first_surface")

    assert returned is lens
    assert np.allclose(lens.surfaces[0].center, target, atol=1e-12)
    # The second surface moved by the same delta, so the rigid separation is unchanged.
    assert np.allclose(
        lens.surfaces[1].center - lens.surfaces[0].center, offset, atol=1e-12
    )

    # mechanical_center reference places the group's mechanical center on the target.
    lens2 = _make_lens_group(center_x=0.0)
    set_element_position(lens2, target, reference="mechanical_center")
    assert np.allclose(lens2.mechanical_center, target, atol=1e-12)

    # Works on a single surface too (reference is its own center).
    m = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
    )
    set_element_position(m, np.array([2.0, 0.0, 0.0]))
    assert np.allclose(m.center, [2.0, 0.0, 0.0], atol=1e-12)


def test_set_element_position_undefined_surface_sets_center():
    from cavity_design import set_element_position

    # An undefined single surface should simply accept the new center (no longer raises).
    s = FlatMirror(name="u", center=None, outwards_normal=np.array([1.0, 0.0, 0.0]))
    assert not s.positions_defined
    returned = set_element_position(s, np.array([0.01, 0.0, 0.0]))
    assert returned is s
    assert np.allclose(s.center, [0.01, 0.0, 0.0], atol=1e-12)
    assert (
        s.positions_defined
    )  # normal was already defined, so setting the center completes the pose


def _make_fabry_perot(u=1e-5):
    R = 5e-3
    L = 2 * R - u
    m0 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([0, 0, -1.0]),
        center=np.array([0, 0, -R]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=R,
        outwards_normal=np.array([0, 0, 1.0]),
        center=np.array([0, 0, -R + L]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    return Cavity(
        elements=[m0, m1],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        power=1e3,
        use_paraxial_ray_tracing=False,
    )


def test_place_element_retraces_and_syncs_standing_wave():
    cavity = _make_fabry_perot()
    assert cavity.central_line_successfully_traced is True
    z_R_before = cavity.arms[0].mode_parameters.z_R[0]

    # Move the second mirror outward along the axis: a different gap => a different (still valid) resonant mode.
    cavity.place_element(
        cavity[1], cavity[1].center + np.array([0.0, 0.0, 2e-5]), recalculate_optic=True
    )

    # The central line and mode parameters were recomputed in place.
    assert cavity.central_line_successfully_traced is True
    assert cavity.resonating_mode_successfully_traced is True
    z_R_after = cavity.arms[0].mode_parameters.z_R[0]
    assert not np.isclose(z_R_before, z_R_after)

    # The standing-wave back-trip surfaces stay consistent with the forward ones (palindrome of centers).
    ordered = [arm.surface_0 for arm in cavity.arms] + [cavity.arms[-1].surface_1]
    centers = np.array([s.center for s in ordered])
    assert np.allclose(centers, centers[::-1], atol=1e-12)


def test_place_element_reference_center_variants():
    # reference_center makes position relative; a Surface reference uses its center.
    cavity = _make_fabry_perot()
    cavity.place_element(
        cavity[1],
        np.array([0.0, 0.0, 5e-3]),
        recalculate_optic=False,
        reference_center=cavity[0],
    )
    assert np.allclose(
        cavity[1].center, cavity[0].center + np.array([0.0, 0.0, 5e-3]), atol=1e-12
    )

    lens_a = _make_lens_group(center_x=0.0)
    lens_b = _make_lens_group(center_x=10e-3)
    system = OpticalSystem(
        [lens_a, lens_b],
        given_initial_central_line=None,
        use_paraxial_ray_tracing=False,
    )
    sep_b = lens_b.surfaces[1].center - lens_b.surfaces[0].center

    # An OpticalSystem reference resolves to its last surface's center; a nested element moves rigidly.
    system.place_element(
        lens_b,
        np.array([5e-3, 0.0, 0.0]),
        recalculate_optic=False,
        reference_center=lens_a,
    )
    assert np.allclose(
        lens_b.surfaces[0].center,
        lens_a.surfaces[-1].center + np.array([5e-3, 0.0, 0.0]),
        atol=1e-12,
    )
    assert np.allclose(
        lens_b.surfaces[1].center - lens_b.surfaces[0].center, sep_b, atol=1e-12
    )

    # Lists of elements are no longer accepted: each element is placed on its own.
    with pytest.raises(TypeError):
        system.place_element(
            [lens_a, lens_b], np.array([1.0, 0.2, 0.0]), recalculate_optic=False
        )


def test_place_element_updates_symmetry_flags():
    # Mirrors perturb_cavity's logic: a z displacement breaks the theta (t) symmetry, a y one the phi (p) symmetry,
    # an x (axial, here) one breaks neither. Start from a cavity that is trivial in both transverse axes.
    def fp_symmetric():
        cav = _make_fabry_perot()
        cav.t_is_trivial = True
        cav.p_is_trivial = True
        return cav

    cav = fp_symmetric()
    cav.place_element(
        cav[1], cav[1].center + np.array([0.0, 0.0, 2e-5]), recalculate_optic=False
    )
    assert cav.t_is_trivial is False and cav.p_is_trivial is True

    cav = fp_symmetric()
    cav.place_element(
        cav[1], cav[1].center + np.array([0.0, 1e-6, 0.0]), recalculate_optic=False
    )
    assert cav.t_is_trivial is True and cav.p_is_trivial is False

    cav = fp_symmetric()
    cav.place_element(
        cav[1], cav[1].center + np.array([1e-6, 0.0, 0.0]), recalculate_optic=False
    )
    assert cav.t_is_trivial is True and cav.p_is_trivial is True


def test_with_element_placed_is_nondestructive():
    cavity = _make_fabry_perot()
    center_before = np.array(cavity[1].center)

    # By index.
    moved = cavity.with_element_placed(1, cavity[1].center + np.array([0.0, 0.0, 2e-5]))
    assert moved is not cavity
    assert np.allclose(
        cavity[1].center, center_before, atol=1e-12
    )  # original untouched
    assert moved[1].center[2] > center_before[2]
    assert moved.central_line_successfully_traced is True

    # By element object, with a Surface reference_center snapshotted from the original.
    moved2 = cavity.with_element_placed(
        cavity[1], np.array([0.0, 0.0, 4e-3]), reference_center=cavity[0]
    )
    assert np.allclose(cavity[1].center, center_before, atol=1e-12)
    assert np.allclose(
        moved2[1].center, cavity[0].center + np.array([0.0, 0.0, 4e-3]), atol=1e-12
    )


def test_to_params_reflects_in_place_edits():
    from cavity_design import OpticalSystem, set_element_position

    # to_params must regenerate from the live elements, not return a copy cached at construction (otherwise an
    # in-place move is invisible to anything that rebuilds from params, e.g. OpticalSystem.complete_to_cavity).
    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    sys = OpticalSystem.from_params(
        [m0.to_params, m1.to_params],
        given_initial_central_line=None,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    assert np.isclose(sys.to_params[1].x, 5e-3)
    set_element_position(sys.elements[1], np.array([8e-3, 0.0, 0.0]))
    assert np.isclose(
        sys.to_params[1].x, 8e-3
    )  # regenerated, not the stale cached value


def test_place_element_refreshes_plain_optical_system_central_line():
    from cavity_design import OpticalSystem

    # A plain OpticalSystem traces its central line via set_given_central_line; with recalculate_optic=True,
    # place_element retraces it from scratch after the move.
    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    sys = OpticalSystem([m0, m1], use_paraxial_ray_tracing=False)
    assert sys.central_line is not None
    assert (
        sys.central_line_successfully_traced is None
    )  # plain OpticalSystem never sets this flag
    assert np.isclose(sys.arms[0].central_line.length, 10e-3)

    sys.place_element(
        sys.elements[1], np.array([8e-3, 0.0, 0.0]), recalculate_optic=True
    )
    # The central line must reflect the moved geometry (mirror at -5mm to the surface now at +8mm).
    assert np.isclose(sys.arms[0].central_line.length, 13e-3)


def test_place_element_resets_optics_and_recalculates_on_demand():
    cavity = _make_fabry_perot()
    assert cavity.central_line_successfully_traced is True

    # Whatever recalculate_optic is, the move discards all previously computed optics: every arm returns to its
    # freshly-initialized state (no central line, empty local mode parameters) and the traced flags are cleared.
    cavity.place_element(
        cavity[1],
        cavity[1].center + np.array([0.0, 0.0, 1e-6]),
        recalculate_optic=False,
    )
    assert cavity.central_line is None
    assert cavity.central_line_successfully_traced is None
    assert cavity.resonating_mode_successfully_traced is None
    for arm in cavity.arms:
        assert arm.central_line is None
        assert np.all(np.isnan(arm.mode_parameters_on_surface_0.q))
        assert np.all(np.isnan(arm.mode_parameters_on_surface_1.q))

    # With recalculate_optic=True the optics are recomputed from scratch, regardless of the pre-move state.
    cavity.place_element(
        cavity[1], cavity[1].center + np.array([0.0, 0.0, 1e-6]), recalculate_optic=True
    )
    assert cavity.central_line_successfully_traced is True
    assert cavity.resonating_mode_successfully_traced is True
    assert np.isfinite(cavity.arms[0].mode_parameters.z_R[0])


def test_surface_level_relative_position_resolved_at_construction():
    from cavity_design import OpticalSystem

    # First surface real (anchor), second encoded as a +10mm relative step in x (pure imaginary).
    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([1j * 10e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )

    sys = OpticalSystem(
        [m0, m1], given_initial_central_line=None, use_paraxial_ray_tracing=False
    )
    assert sys.positions_defined
    assert np.isclose(sys.surfaces[1].center[0], -5e-3 + 10e-3)
    assert not np.iscomplexobj(np.asarray(sys.surfaces[1].center))


def test_surface_level_relative_position_deferred():
    from cavity_design import OpticalSystem, set_element_position

    # First surface undefined, second is a relative step. The system is not defined and the relative stays complex.
    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=None,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([1j * 10e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )

    sys = OpticalSystem(
        [m0, m1], given_initial_central_line=None, use_paraxial_ray_tracing=False
    )
    assert not sys.positions_defined
    # The unresolved relative position is still complex.
    assert np.any(np.abs(np.imag(np.asarray(sys.surfaces[1].center))) > 1e-12)

    # Anchoring the first surface resolves the chain.
    set_element_position(sys, np.array([-5e-3, 0.0, 0.0]))
    assert sys.positions_defined
    assert np.isclose(sys.surfaces[0].center[0], -5e-3)
    assert np.isclose(sys.surfaces[1].center[0], -5e-3 + 10e-3)


def test_cavity_relative_positions_standing_wave():
    # A standing-wave Fabry-Perot defined with the second mirror as a relative step; resolution must also place the
    # mirror-image (inverse) surface correctly.
    u = 1e-5
    sep = 10e-3 - u
    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([1j * sep, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )

    cavity = Cavity(
        [m0, m1],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_mode_parameters=False,
    )
    assert cavity.positions_defined
    assert np.isclose(cavity.surfaces[1].center[0], -5e-3 + sep)
    # The inverse (back-trip) copy of the first surface shares the forward first surface's location.
    ordered = cavity._surfaces
    assert np.allclose(ordered[-1].center, cavity.surfaces[0].center, atol=1e-12)


def test_unresolved_relative_blocks_calculations():
    from cavity_design import OpticalSystem

    m0 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=None,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([1j * 10e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    sys = OpticalSystem(
        [m0, m1], given_initial_central_line=None, use_paraxial_ray_tracing=False
    )
    ray = Ray(origin=np.array([0.0, 0, 0]), k_vector=np.array([1.0, 0, 0]))
    with pytest.raises(ValueError):
        sys.propagate_ray(ray)


def _make_standard_catalog_cavity():
    # The standard mirror-aspheric-lens-mirror cavity, built from (copies of) the catalog elements and placed as in
    # simple_analysis_scripts/mode_spacing_to_NA.py.
    from cavity_design import (
        LASER_OPTIK_MIRROR,
        EDMUND_4p03MM_ASPHERIC,
        THOLABS_100MM_PLANO_CONVEX_LENS,
        COASTLINE_20CM_MIRROR,
        back_focal_length_of_lens_object,
        LEFT,
        RIGHT,
    )

    cavity = Cavity(
        elements=copy.deepcopy(
            [
                # Catalog elements are floating; anchor the first mirror at its historical position (vertex at
                # x=-5mm) and place the rest relative to it below.
                LASER_OPTIK_MIRROR.to_position(5e-3 * LEFT),
                EDMUND_4p03MM_ASPHERIC,
                THOLABS_100MM_PLANO_CONVEX_LENS,
                COASTLINE_20CM_MIRROR,
            ]
        ),
        use_paraxial_ray_tracing=True,
        p_is_trivial=True,
        t_is_trivial=True,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    collimation_point = cavity[0].radius + back_focal_length_of_lens_object(
        lens_object=cavity[1]
    )
    cavity.place_element(
        cavity[1],
        collimation_point * RIGHT,
        recalculate_optic=False,
        reference_center=cavity[0],
    )
    cavity.place_element(
        cavity[2], 0.5 * RIGHT, recalculate_optic=False, reference_center=cavity[1]
    )
    cavity.place_element(
        cavity[3], 1.0 * RIGHT, recalculate_optic=True, reference_center=cavity[2]
    )
    return cavity


def test_init_syntax_round_trip():
    # The textual representation of a cavity is its initialization syntax: exec-ing it must reproduce the cavity.
    cavity = _make_standard_catalog_cavity()
    text = cavity.formatted_init_syntax
    assert text.startswith("cavity = Cavity(")
    namespace = {}
    exec("from cavity_design import *\nimport numpy as np\n" + text, namespace)
    cavity_2 = namespace["cavity"]
    for s1, s2 in zip(cavity.surfaces, cavity_2.surfaces):
        assert type(s1) is type(s2)
        assert np.allclose(s1.center, s2.center, atol=1e-12)
        assert np.allclose(s1.outwards_normal, s2.outwards_normal, atol=1e-12)
    assert np.isclose(
        cavity.free_spectral_range, cavity_2.free_spectral_range, rtol=1e-12
    )
    na_1 = cavity.arms[0].mode_parameters.NA[0]
    na_2 = cavity_2.arms[0].mode_parameters.NA[0]
    assert np.isclose(na_1, na_2, rtol=1e-9, equal_nan=True)


def test_init_syntax_catalog_recognition_and_fallback():
    cavity = _make_standard_catalog_cavity()
    text = cavity.init_syntax
    # Placed catalog elements are rendered compactly (tag + verified against the registry).
    assert "EDMUND_4p03MM_ASPHERIC.to_position(" in text
    assert "COASTLINE_20CM_MIRROR.to_position(" in text
    # A catalog element whose intrinsics were mutated after construction must fall back to its full constructor.
    cavity[1]._surfaces[0].n_2 = 1.9
    text_mutated = cavity.init_syntax
    assert "EDMUND_4p03MM_ASPHERIC" not in text_mutated
    assert "AsphericRefractiveSurface(" in text_mutated
    # The untouched catalog elements are still recognized.
    assert "COASTLINE_20CM_MIRROR.to_position(" in text_mutated


def test_to_position_and_to_orientation_are_nonmutating_and_chainable():
    from cavity_design import EDMUND_4p03MM_ASPHERIC

    lens = copy.deepcopy(EDMUND_4p03MM_ASPHERIC)
    assert not lens.positions_defined  # catalog elements are floating
    # The internal geometry of a floating element is encoded as a relative (imaginary) offset from its first surface.
    internal_offset = np.imag(lens.surfaces[1].center)
    assert np.allclose(internal_offset, [3.1e-3, 0.0, 0.0], atol=1e-15)
    target = np.array([7e-3, 0.0, 0.0])

    placed = lens.to_position(target)
    assert placed is not lens
    assert not lens.positions_defined  # original untouched (still floating)
    assert placed.positions_defined
    assert np.allclose(placed.surfaces[0].center, target, atol=1e-15)
    assert np.allclose(
        placed.surfaces[1].center - placed.surfaces[0].center,
        internal_offset,
        atol=1e-12,
    )  # rigid

    up = np.array([0.0, 1.0, 0.0])
    rotated = placed.to_orientation(up)
    assert np.allclose(
        placed.surfaces[0].outwards_normal, lens.surfaces[0].outwards_normal, atol=1e-15
    )  # untouched
    assert np.allclose(rotated.surfaces[0].outwards_normal, up, atol=1e-12)
    assert np.allclose(
        rotated.surfaces[0].center, target, atol=1e-12
    )  # the rotation pivots on the first surface
    # Rigid rotation: internal distances preserved, and the second face's normal (RIGHT) rotates to -up.
    assert np.isclose(
        np.linalg.norm(rotated.surfaces[1].center - rotated.surfaces[0].center),
        np.linalg.norm(internal_offset),
        atol=1e-12,
    )
    assert np.allclose(rotated.surfaces[1].outwards_normal, -up, atol=1e-12)

    # Single surface: to_orientation preserves the vertex and recomputes the sphere origin.
    mirror = SphericalMirror(
        radius=5e-3,
        outwards_normal=np.array([1.0, 0, 0]),
        center=np.array([1e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    moved = mirror.to_position(np.array([2e-3, 0.0, 0.0]))
    assert np.allclose(mirror.center, [1e-3, 0, 0], atol=1e-15) and np.allclose(
        moved.center, [2e-3, 0, 0], atol=1e-15
    )
    turned = moved.to_orientation(np.array([0.0, 0.0, 1.0]))
    assert np.allclose(turned.center, [2e-3, 0, 0], atol=1e-15)
    assert np.allclose(turned.outwards_normal, [0, 0, 1], atol=1e-15)
    assert np.allclose(
        turned.origin,
        turned.center - turned.radius * turned.outwards_normal,
        atol=1e-15,
    )


def test_params_machinery_deprecation_warnings():
    from cavity_design import Surface

    cavity = _make_fabry_perot()
    with pytest.warns(DeprecationWarning):
        params = cavity.to_params
    with pytest.warns(DeprecationWarning):
        _ = cavity.formatted_textual_params
    with pytest.warns(DeprecationWarning):
        surface_params = cavity[0].to_params
    with pytest.warns(DeprecationWarning):
        _ = Surface.from_params(surface_params)
    with pytest.warns(DeprecationWarning):
        _ = Cavity.from_params(
            params, standing_wave=True, lambda_0_laser=LAMBDA_0_LASER
        )


def test_object_workflow_emits_no_deprecation_warnings():
    import warnings as warnings_module

    # The live-object workflow (construction, placement, repr/init_syntax, specs) must never trigger the params
    # deprecation warnings from inside the library.
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        cavity = _make_standard_catalog_cavity()
        _ = cavity.init_syntax
        _ = repr(cavity)
        _ = str(cavity)
        # specs() on the aspheric cavity hits an unrelated pre-existing NotImplementedError (incidence angle for
        # aspheric surfaces), so the rebuilt element table is exercised on a Fabry-Perot cavity instead (whose
        # mirrors need material properties, for the finesse/losses rows).
        from cavity_design import PHYSICAL_SIZES_DICT

        mirror_properties = PHYSICAL_SIZES_DICT["material_properties_ULE"]
        radius, u = 5e-3, 1e-5
        fp = Cavity(
            elements=[
                SphericalMirror(
                    radius=radius,
                    outwards_normal=np.array([0, 0, -1.0]),
                    center=np.array([0, 0, -radius]),
                    curvature_sign=CurvatureSigns.concave,
                    diameter=0.01,
                    material_properties=mirror_properties,
                ),
                SphericalMirror(
                    radius=radius,
                    outwards_normal=np.array([0, 0, 1.0]),
                    center=np.array([0, 0, radius - u]),
                    curvature_sign=CurvatureSigns.concave,
                    diameter=0.01,
                    material_properties=mirror_properties,
                ),
            ],
            standing_wave=True,
            lambda_0_laser=LAMBDA_0_LASER,
            power=1e3,
            use_paraxial_ray_tracing=False,
            t_is_trivial=True,
            p_is_trivial=True,
        )
        # A precomputed (dummy) tolerance matrix skips the expensive tolerance computation inside specs().
        _ = fp.specs(tolerance_dataframe=np.zeros((2, 5)))
    params_warnings = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "OpticalSurfaceParams" in str(w.message)
    ]
    assert not params_warnings


def test_aspheric_surface_floating_center():
    # AsphericSurface follows the same floating-position convention as the other surface types:
    # center may be omitted / None (stored as a size-3 nan array, positions not defined).
    from cavity_design import RIGHT

    coeffs = np.array([0.0, 55.1, 7.27e4])
    s = AsphericRefractiveSurface(
        outwards_normal=RIGHT,
        polynomial_coefficients=coeffs,
        n_1=1.45,
        n_2=1,
        diameter=6.35e-3,
    )
    assert s.center.shape == (3,) and np.all(np.isnan(s.center))
    assert not s.positions_defined

    s.center = np.array([0.01, 0.0, 0.0])
    assert s.positions_defined
    s.center = None
    assert s.center.shape == (3,) and np.all(np.isnan(s.center))
    assert not s.positions_defined

    # The shape itself is not optional - only the pose is.
    with pytest.raises(TypeError):
        AsphericRefractiveSurface(outwards_normal=RIGHT)


def test_aspheric_surface_relative_center_resolution():
    # An imaginary center is a relative offset from the previous surface, resolved when the
    # containing OpticalSystem is placed.
    from cavity_design import LEFT, RIGHT

    coeffs = np.array([0.0, 55.1, 7.27e4])
    T_c = 3.4e-3
    flat = FlatRefractiveSurface(
        outwards_normal=LEFT,
        n_1=1,
        n_2=1.45,
        diameter=6.35e-3,
        name="floating lens - flat side",
    )
    asph = AsphericRefractiveSurface(
        center=T_c * RIGHT * 1j,
        outwards_normal=RIGHT,
        polynomial_coefficients=coeffs,
        n_1=1.45,
        n_2=1,
        diameter=6.35e-3,
        curvature_sign=CurvatureSigns.concave,
        name="floating lens - aspheric side",
    )
    lens = OpticalSystem(elements=[flat, asph], use_paraxial_ray_tracing=True)
    assert not lens.positions_defined

    anchor = np.array([0.01, 0.0, 0.0])
    placed = lens.to_position(anchor)
    assert placed.positions_defined
    np.testing.assert_allclose(placed.surfaces[0].center, anchor, atol=1e-12)
    np.testing.assert_allclose(
        placed.surfaces[1].center, anchor + T_c * RIGHT, atol=1e-12
    )
    # The original floating lens is untouched (to_position is non-mutating).
    assert not lens.positions_defined


def test_invert_optical_system_carries_and_verifies_mode():
    from cavity_design import (
        LASER_OPTIK_MIRROR_REFRACTIVE,
        EKSMA_LENS_20MM_ASPHERIC,
        match_a_mode_to_mirror,
        LEFT,
    )

    mirror = LASER_OPTIK_MIRROR_REFRACTIVE.to_position(5e-3 * LEFT)
    lens = EKSMA_LENS_20MM_ASPHERIC.to_position(
        mirror.surfaces[1].center + 5.5e-3 * LEFT
    )
    system = OpticalSystem(
        elements=[mirror, lens],
        use_paraxial_ray_tracing=True,
        p_is_trivial=True,
        t_is_trivial=True,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    mode_initial = match_a_mode_to_mirror(
        lambda_0_laser=LAMBDA_0_LASER,
        mirror=mirror.surfaces[0],
        NA=0.02,
        mode_going_away_from_mirror=False,
    )
    system.set_given_mode_parameters(mode_parameters_after_first_surface=mode_initial)

    inverted = system.invert()  # raises internally if the reversed mode mismatches
    # Elements are preserved as elements (no flattening to single surfaces):
    assert [type(e).__name__ for e in inverted.elements] == [
        "OpticalSystem",
        "OpticalSystem",
    ]
    # Surface order reversed:
    assert inverted.surfaces[0].name == system.surfaces[-1].name
    # Central line flipped:
    assert np.allclose(
        inverted.central_line[0].k_vector, -system.central_line[-1].k_vector
    )
    # The reversed-propagated mode at the originally-first surface is the direction-reverse of the forward mode:
    q_forward = system.arms[0].mode_parameters_on_surface_0.q
    q_inverted = inverted.arms[-1].mode_parameters_on_surface_1.q
    assert np.allclose(q_inverted, -np.conj(q_forward), rtol=1e-9)


def test_invert_floating_element_preserves_floating_convention():
    from cavity_design import EKSMA_LENS_20MM_ASPHERIC

    inverted = EKSMA_LENS_20MM_ASPHERIC.invert()
    assert not inverted.positions_defined
    # New first surface (originally last) is undefined; the internal step flipped sign and stayed relative:
    assert np.all(np.isnan(inverted.surfaces[0].center))
    offset = np.imag(inverted.surfaces[1].center)
    assert np.allclose(offset, [-3.434e-3, 0.0, 0.0], atol=1e-15)
    placed = inverted.to_position(np.array([0.01, 0.0, 0.0]))
    assert placed.positions_defined
    assert np.allclose(placed.surfaces[1].center, [0.01 - 3.434e-3, 0.0, 0.0])


def test_matching_a_too_small_NA_reports_the_smallest_usable_one():
    from cavity_design import match_a_mode_to_mirror, NA_of_z_R, COASTLINE_20CM_REFRACTIVE, LEFT

    mirror = COASTLINE_20CM_REFRACTIVE.to_position(0.2 * LEFT).surfaces[0]
    # A mode only reaches the mirror's curvature while z_R <= radius / 2, so NA has a hard floor.
    NA_min = NA_of_z_R(z_R=mirror.radius / 2, lambda_0_laser=LAMBDA_0_LASER)

    with pytest.raises(ValueError, match=r"smallest NA this mirror supports is 0.00184033"):
        match_a_mode_to_mirror(
            lambda_0_laser=LAMBDA_0_LASER, mirror=mirror, NA=0.001, mode_going_away_from_mirror=False
        )

    # The advertised minimum must itself be usable - the boundary round-trips through z_R_of_NA to within an ulp,
    # so a naive strict comparison would reject the very value the message tells the user to substitute.
    limiting_mode = match_a_mode_to_mirror(
        lambda_0_laser=LAMBDA_0_LASER, mirror=mirror, NA=NA_min, mode_going_away_from_mirror=False
    )
    assert np.asarray(limiting_mode.z_R)[0] == pytest.approx(mirror.radius / 2, rel=1e-9)
    # At the limit the waist sits radius / 2 from the mirror (the confocal case).
    assert np.asarray(limiting_mode.center)[0, 0] == pytest.approx(mirror.center[0] + mirror.radius / 2, abs=1e-12)

    # The tolerance is not a loophole: anything meaningfully below the floor is still rejected.
    with pytest.raises(ValueError, match="smallest NA"):
        match_a_mode_to_mirror(
            lambda_0_laser=LAMBDA_0_LASER, mirror=mirror, NA=NA_min * (1 - 1e-6), mode_going_away_from_mirror=False
        )

    # Passing z_R directly hits the same guard, and the message still names the NA to substitute.
    with pytest.raises(ValueError, match="smallest NA this mirror supports"):
        match_a_mode_to_mirror(
            lambda_0_laser=LAMBDA_0_LASER, mirror=mirror, z_R=0.5, mode_going_away_from_mirror=False
        )


def test_flip_turns_a_floating_lens_around():
    from cavity_design import THOLABS_200MM_PLANO_CONVEX_LENS

    lens = THOLABS_200MM_PLANO_CONVEX_LENS
    flipped = lens.flip()

    # The catalogued lens is convex-first, flat-second; turned around it is flat-first, convex-second.
    assert [type(s).__name__ for s in lens.surfaces] == ["SphericalRefractiveSurface", "FlatRefractiveSurface"]
    assert [type(s).__name__ for s in flipped.surfaces] == ["FlatRefractiveSurface", "SphericalRefractiveSurface"]
    # Optical inversion: the indices are swapped and the curvature sign flips ...
    assert (flipped.surfaces[0].n_1, flipped.surfaces[0].n_2) == (lens.surfaces[0].n_1, lens.surfaces[0].n_2)
    assert flipped.surfaces[1].curvature_sign == -lens.surfaces[0].curvature_sign
    # ... and the physical turn-around negates every outwards normal, so the cap now bulges the other way.
    assert np.allclose(flipped.surfaces[1].outwards_normal, -lens.surfaces[0].outwards_normal)
    # It stays floating, and the centre thickness keeps its sign (the spacing is walked in the reversed order).
    assert not flipped.positions_defined
    assert np.all(np.isnan(flipped.surfaces[0].center))
    assert np.allclose(np.imag(flipped.surfaces[1].center), [2.8e-3, 0.0, 0.0], atol=1e-15)

    # flip() is an involution.
    assert all(surfaces_are_equivalent(a, b) for a, b in zip(lens.flip().flip().surfaces, lens.surfaces))


def test_flip_preserves_focal_length_but_moves_the_principal_planes():
    from cavity_design import THOLABS_200MM_PLANO_CONVEX_LENS

    def trace_collimated(lens, height=1e-3):
        """Exact-trace a ray parallel to the axis at `height`; return its back focal distance."""
        placed = copy.deepcopy(lens).to_position(ORIGIN)
        ray = Ray(origin=np.array([[-5e-3, height, 0.0]]), k_vector=np.array([RIGHT]))
        for surface in placed.surfaces:
            ray = surface.propagate_ray(ray)
        crossing = ray.origin[..., 0] - ray.origin[..., 1] * ray.k_vector[..., 0] / ray.k_vector[..., 1]
        return float(crossing[0] - placed.surfaces[-1].center[0])

    lens = THOLABS_200MM_PLANO_CONVEX_LENS
    back_focal_catalogued = trace_collimated(lens)
    back_focal_flipped = trace_collimated(lens.flip())
    # Same lens, so the same power (~200 mm nominal) either way round ...
    assert 0.195 < back_focal_catalogued < 0.21
    assert 0.195 < back_focal_flipped < 0.21
    # ... but turning a plano-convex lens around shifts the principal planes, so the back focal distance moves by
    # roughly the centre thickness / n.
    assert back_focal_flipped - back_focal_catalogued == pytest.approx(2.8e-3 / 1.507, rel=0.05)


def test_flip_of_a_placed_element_turns_it_around_in_place():
    from cavity_design import THOLABS_200MM_PLANO_CONVEX_LENS, EDMUND_4p5MM_ASPHERIC_83580

    position = np.array([0.2, 0.0, 0.0])
    for lens in (THOLABS_200MM_PLANO_CONVEX_LENS, EDMUND_4p5MM_ASPHERIC_83580):
        placed = lens.to_position(position)
        flipped = placed.flip()
        # Turned around in its mount: the element keeps the span it occupied, faces reversed.
        assert np.allclose(flipped.surfaces[0].center, placed.surfaces[0].center)
        assert np.allclose(flipped.surfaces[-1].center, placed.surfaces[-1].center)
        assert [type(s).__name__ for s in flipped.surfaces] == [type(s).__name__ for s in placed.surfaces][::-1]
        # Mirroring about the midpoint makes flipping commute with placement ...
        assert all(
            surfaces_are_equivalent(a, b) for a, b in zip(lens.flip().to_position(position).surfaces, flipped.surfaces)
        )
        # ... and it is still an involution.
        assert all(surfaces_are_equivalent(a, b) for a, b in zip(placed.surfaces, flipped.flip().surfaces))


def test_flip_of_a_placed_system_reverses_the_spacings_and_retraces():
    from cavity_design import THOLABS_200MM_PLANO_CONVEX_LENS, EDMUND_4p5MM_ASPHERIC_83580

    system = OpticalSystem(
        elements=[THOLABS_200MM_PLANO_CONVEX_LENS, EDMUND_4p5MM_ASPHERIC_83580],
        use_paraxial_ray_tracing=True,
        p_is_trivial=True,
        t_is_trivial=True,
    )
    system.place_element(element=system[0], position=np.array([0.2, 0.0, 0.0]), recalculate_optic=False)
    system.place_element(element=system[1], position=1e-2 * RIGHT, reference_center=system[0], recalculate_optic=True)
    flipped = system.flip()

    spacings = np.diff([s.center[0] for s in system.surfaces])
    assert np.allclose(np.diff([s.center[0] for s in flipped.surfaces]), spacings[::-1])
    assert np.allclose(flipped.surfaces[0].center, system.surfaces[0].center)
    assert np.allclose(flipped.surfaces[-1].center, system.surfaces[-1].center)
    # The geometry moved, so the optics are re-derived rather than carried over stale.
    assert np.allclose(flipped.central_line.k_vector[0], RIGHT)


def test_flip_of_a_cavity_is_rejected_in_favour_of_invert():
    from cavity_design import LASER_OPTIK_MIRROR, EKSMA_LENS_20MM_ASPHERIC, COASTLINE_20CM_MIRROR, LEFT

    cavity = Cavity(
        elements=[
            LASER_OPTIK_MIRROR.to_position(5e-3 * LEFT),
            EKSMA_LENS_20MM_ASPHERIC.to_position(np.array([0.0176, 0.0, 0.0])),
            COASTLINE_20CM_MIRROR.to_position(np.array([0.25, 0.0, 0.0])).to_orientation(RIGHT),
        ],
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        t_is_trivial=True,
        p_is_trivial=True,
        use_paraxial_ray_tracing=True,
    )
    with pytest.raises(NotImplementedError, match="Cavity"):
        cavity.flip()


def test_flip_of_a_tilted_element_mirrors_about_its_own_axis():
    from cavity_design import THOLABS_200MM_PLANO_CONVEX_LENS

    # The mirror plane must be orthogonal to the element's own first->last line, not to x.
    axis = np.array([1.0, 0.3, 0.0]) / np.linalg.norm([1.0, 0.3, 0.0])
    tilted = OpticalSystem(
        elements=[THOLABS_200MM_PLANO_CONVEX_LENS.to_position(ORIGIN).to_orientation(-axis)],
        use_paraxial_ray_tracing=True,
        p_is_trivial=False,
        t_is_trivial=False,
    )
    flipped = tilted.flip()
    for original, turned in zip(tilted.surfaces, reversed(flipped.surfaces)):
        assert np.allclose(turned.outwards_normal, -original.outwards_normal)
    assert np.allclose(flipped.surfaces[0].center, tilted.surfaces[0].center)
    assert np.allclose(flipped.surfaces[-1].center, tilted.surfaces[-1].center)


def test_invert_cavity_preserves_structure_and_mode():
    from cavity_design import (
        LASER_OPTIK_MIRROR,
        EKSMA_LENS_20MM_ASPHERIC,
        DUMMY_LENS,
        SphericalMirror,
    )

    cavity = Cavity(
        elements=[
            LASER_OPTIK_MIRROR.to_position(np.array([-0.005, 0.0, 0.0])),
            EKSMA_LENS_20MM_ASPHERIC.to_position(
                np.array([0.017623230771841976, 0.0, 0.0])
            ),
            DUMMY_LENS.to_position(np.array([0.03305723077184197, 0.0, 0.0])),
            SphericalMirror(
                name="End mirror",
                radius=0.2,
                outwards_normal=np.array([1.0, 0.0, 0.0]),
                center=np.array([0.4511688875799871, 0.0, 0.0]),
                curvature_sign=-1,
                diameter=0.0254,
            ),
        ],
        standing_wave=True,
        lambda_0_laser=1.064e-06,
        t_is_trivial=True,
        p_is_trivial=True,
        use_paraxial_ray_tracing=False,
    )
    assert cavity.resonating_mode_successfully_traced

    inverted = cavity.invert()  # re-derives and verifies the mode internally
    # Still a Cavity, with the one-way element structure preserved (no flattening, no round-trip in elements):
    assert type(inverted) is Cavity
    assert len(inverted.elements) == len(cavity.elements)
    assert len(inverted.elements_ordered) == len(cavity.elements_ordered)
    assert inverted.resonating_mode_successfully_traced
    # The re-derived mode is the direction-reverse of the original at the originally-first surface:
    q_forward = cavity.arms[0].mode_parameters_on_surface_0.q
    comparison_arm_index = len(inverted.surfaces) - 2
    q_inverted = inverted.arms[comparison_arm_index].mode_parameters_on_surface_1.q
    assert np.allclose(q_inverted, -np.conj(q_forward), rtol=1e-6)


# ----------------------------------------------------------------------------------------------------
# Cartesian ovals
# ----------------------------------------------------------------------------------------------------

# (n_1, n_2, E_1, E_2) covering every combination of real/virtual object and real/virtual image.
CARTESIAN_OVAL_CONJUGATES = [
    (1.0, 1.5, 1.0, 1.0),  # real object    -> real image,    n_2 > n_1
    (1.5, 1.0, 1.0, 2.0),  # real object    -> real image,    n_2 < n_1
    (1.0, 1.5, 1.0, -2.5),  # real object    -> virtual image
    (1.0, 1.5, -2.0, 1.0),  # virtual object -> real image
    (1.5, 1.0, -2.0, -0.5),  # virtual object -> virtual image
]


def _cartesian_oval_incoming_fan(surface, half_angle=0.14, n_rays=9):
    """A fan of rays diverging from / converging on the object focus, whichever E_1 calls for."""
    optical_axis = surface.propagation_direction
    transverse = np.cross(optical_axis, np.array([0.0, 0.0, 1.0]))
    angles = np.linspace(-half_angle, half_angle, n_rays)
    k_vector = np.stack([np.cos(t) * optical_axis + np.sin(t) * transverse for t in angles])
    if surface.E_1 > 0:
        # Real object: the rays leave focus_1.
        origin = np.tile(surface.focus_1, (n_rays, 1))
    else:
        # Virtual object: the rays arrive from upstream, aimed at focus_1, and are intercepted before it.
        origin = surface.focus_1 - 2 * abs(surface.E_1) * k_vector
    return Ray(origin=origin, k_vector=k_vector, n=surface.n_1)


def _distance_from_point_to_ray_lines(ray, point):
    """Perpendicular distance from a point to each ray's (infinite) line.

    Using the line rather than the forward half-line makes this work for a virtual image too, where the
    outgoing rays only meet the focus when extended backwards."""
    delta = point - ray.origin
    along = np.sum(delta * ray.k_vector, axis=-1)
    return np.linalg.norm(delta - along[..., np.newaxis] * ray.k_vector, axis=-1)


@pytest.mark.parametrize("n_1, n_2, E_1, E_2", CARTESIAN_OVAL_CONJUGATES)
def test_cartesian_oval_perfect_focus(n_1, n_2, E_1, E_2):
    # The defining property, and the one a polynomial asphere cannot satisfy: every ray of a wide fan is
    # refracted exactly through the image focus, with no spherical aberration whatsoever.
    surface = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=1.2
    )
    assert not np.allclose(surface.focus_1, surface.focus_2), "degenerate conjugates make this test vacuous"

    outgoing = surface.propagate_ray(_cartesian_oval_incoming_fan(surface))
    assert np.all(np.isfinite(outgoing.origin)), "some rays failed to intersect the surface"
    # A genuinely non-paraxial fan: the marginal ray must be well off the axis.
    assert surface.radial_distance_from_axis(outgoing.origin).max() > 0.1

    misses = _distance_from_point_to_ray_lines(outgoing, surface.focus_2)
    assert np.all(misses < 1e-12), f"rays missed focus_2 by up to {misses.max()}"


def test_cartesian_oval_tilted_axis_perfect_focus():
    # The same, on an optical axis that is neither along x nor inside a coordinate plane, so that no
    # accidental alignment can hide a frame error.
    outwards_normal = normalize_vector(np.array([-1.0, 0.3, 0.2]))
    center = np.array([0.011, -0.004, 0.007])
    surface = RefractiveCartesianOval(
        center=center, outwards_normal=outwards_normal, E_1=0.02, E_2=0.05, n_1=1.0, n_2=1.45, diameter=6.35e-3
    )
    outgoing = surface.propagate_ray(_cartesian_oval_incoming_fan(surface, half_angle=0.075))
    assert np.all(np.isfinite(outgoing.origin))
    misses = _distance_from_point_to_ray_lines(outgoing, surface.focus_2)
    assert np.all(misses < 1e-15), f"rays missed focus_2 by up to {misses.max()}"


@pytest.mark.parametrize("n_1, n_2, E_1, E_2", CARTESIAN_OVAL_CONJUGATES)
def test_cartesian_oval_pose_conventions(n_1, n_2, E_1, E_2):
    surface = RefractiveCartesianOval(
        center=np.array([0.1, 0.0, 0.0]), outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=0.2
    )
    signed_radius = signed_vertex_radius_of_a_cartesian_oval(n_1=n_1, n_2=n_2, E_1=E_1, E_2=E_2)

    assert surface.radius >= 0
    assert np.isclose(surface.radius, abs(signed_radius))
    assert surface.curvature_sign == np.sign(signed_radius)
    # origin is the center of curvature, on the far side from outwards_normal.
    np.testing.assert_allclose(surface.origin, surface.center - surface.outwards_normal * surface.radius, atol=1e-15)
    # The surface bulges towards outwards_normal, so the sag along inwards_normal is non-negative.
    rho = np.linspace(0, surface.diameter / 2, 11)
    assert np.all(surface.local_sag(rho) >= 0)
    # Near the axis the sag is the parabola of the matching sphere.
    assert np.isclose(surface.local_sag(1e-4), 1e-8 / (2 * surface.radius), rtol=1e-6)
    # The foci sit on the optical axis at the signed distances they were given.
    np.testing.assert_allclose(surface.focus_1, surface.center - E_1 * surface.propagation_direction, atol=1e-15)
    np.testing.assert_allclose(surface.focus_2, surface.center + E_2 * surface.propagation_direction, atol=1e-15)


def test_cartesian_oval_matches_the_quartic_polynomial():
    # Ties the implementation back to the Cartesian oval polynomial. Squaring
    #     n_1*sign(E_1)*L_1 + n_2*sign(E_2)*L_2 = C,   C = n_1*E_1 + n_2*E_2
    # once gives the grouped form below. Note the sign of the radical term, which is what selects the
    # branch: a plain '+' there describes a surface that does not image the two foci at all.
    n_1, n_2, E_1, E_2 = 1.0, 1.45, 0.02, 0.05
    surface = RefractiveCartesianOval(
        center=np.array([0.1, 0.0, 0.0]), outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=6.35e-3
    )
    C = surface.C
    assert np.isclose(C, n_1 * E_1 + n_2 * E_2)

    optical_axis = surface.propagation_direction
    transverse = np.cross(optical_axis, np.array([0.0, 0.0, 1.0]))
    rho = np.linspace(0, surface.diameter / 2, 17)
    points = surface.center + np.outer(rho, transverse) + np.outer(surface.local_sag(rho), surface.inwards_normal)
    x = (points - surface.center) @ optical_axis
    rho_squared = np.sum((points - surface.center) ** 2, axis=-1) - x**2

    left_hand_side = (
        (n_1**2 - n_2**2) * (x**2 + rho_squared)
        + 2 * x * (n_1**2 * E_1 + n_2**2 * E_2)
        + (n_1**2 * E_1**2 - n_2**2 * E_2**2 - C**2)
    )
    right_hand_side = -2 * C * n_2 * np.sign(E_2) * np.sqrt((x - E_2) ** 2 + rho_squared)
    np.testing.assert_allclose(left_hand_side, right_hand_side, atol=1e-15)
    # And the un-squared residual that the solvers actually drive to zero.
    assert np.abs(surface.defining_equation(points)).max() < 1e-15


def test_cartesian_oval_normal_satisfies_snells_law():
    # The normal is the gradient of the optical path residual; check it against Snell's law directly,
    # independently of the refraction code path.
    n_1, n_2 = 1.0, 1.45
    surface = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=0.02, E_2=0.05, n_1=n_1, n_2=n_2, diameter=6.35e-3
    )
    incoming = _cartesian_oval_incoming_fan(surface, half_angle=0.075)
    outgoing = surface.propagate_ray(incoming)
    normal = surface.normal_at_a_point(outgoing.origin)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=1e-14)

    # The tangential component of n*k is continuous across the surface.
    def tangential(k_vector):
        return k_vector - np.sum(k_vector * normal, axis=-1)[..., np.newaxis] * normal

    np.testing.assert_allclose(n_1 * tangential(incoming.k_vector), n_2 * tangential(outgoing.k_vector), atol=1e-14)


def test_cartesian_oval_departs_from_its_matching_sphere():
    # The oval is not secretly its own vertex sphere: at this numerical aperture that sphere shows plain
    # spherical aberration, so the higher-order shape is doing real work.
    n_1, n_2, E_1, E_2 = 1.0, 1.5, 1.0, 1.0
    oval = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=0.4
    )
    matching_sphere = SphericalRefractiveSurface(
        radius=oval.radius,
        outwards_normal=LEFT,
        center=ORIGIN,
        n_1=n_1,
        n_2=n_2,
        curvature_sign=oval.curvature_sign,
    )
    fan = _cartesian_oval_incoming_fan(oval, half_angle=0.14)
    assert _distance_from_point_to_ray_lines(oval.propagate_ray(fan), oval.focus_2).max() < 1e-12
    sphere_misses = _distance_from_point_to_ray_lines(matching_sphere.propagate_ray(fan), oval.focus_2)
    assert np.nanmax(sphere_misses) > 1e-4


def test_cartesian_oval_beats_a_fitted_polynomial_asphere():
    # The reason this surface type exists. An AsphericRefractiveSurface can only approximate the perfect
    # profile with a truncated even polynomial; fitting one to the oval's own sag - with as many
    # coefficients as the lens in test_aspheric_lens - still leaves a focus residual many orders of
    # magnitude above the oval's, which is exact by construction.
    n_1, n_2, E_1, E_2 = 1.0, 1.5, 1.0, 1.0
    diameter = 0.4
    oval = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=diameter
    )

    rho = np.linspace(0, diameter / 2, 400)
    polynomial_coefficients = np.polyfit(rho**2, oval.local_sag(rho), 4)[::-1]
    polynomial_coefficients[0] = 0.0  # the profile passes through the vertex
    fitted_asphere = AsphericRefractiveSurface(
        center=ORIGIN,
        outwards_normal=LEFT,
        polynomial_coefficients=polynomial_coefficients,
        n_1=n_1,
        n_2=n_2,
        curvature_sign=oval.curvature_sign,
        diameter=diameter,
    )
    # The fit really is a good one - this is not a straw man.
    assert np.abs(Polynomial(polynomial_coefficients)(rho**2) - oval.local_sag(rho)).max() < 1e-6

    fan = _cartesian_oval_incoming_fan(oval, half_angle=0.14)
    oval_misses = _distance_from_point_to_ray_lines(oval.propagate_ray(fan), oval.focus_2)
    asphere_misses = _distance_from_point_to_ray_lines(fitted_asphere.propagate_ray(fan), oval.focus_2)
    assert oval_misses.max() < 1e-12
    assert np.nanmax(asphere_misses) > 1e-7
    assert np.nanmax(asphere_misses) > 1e6 * oval_misses.max()


def test_cartesian_oval_inverse():
    surface = RefractiveCartesianOval(
        center=np.array([0.1, 0.0, 0.0]),
        outwards_normal=LEFT,
        E_1=0.02,
        E_2=0.05,
        n_1=1.0,
        n_2=1.45,
        diameter=6.35e-3,
        name="oval",
    )
    inverted = surface.inverse

    # Same shape in space, opposite illumination.
    np.testing.assert_allclose(inverted.center, surface.center, atol=1e-15)
    np.testing.assert_allclose(inverted.outwards_normal, surface.outwards_normal, atol=1e-15)
    np.testing.assert_allclose(inverted.origin, surface.origin, atol=1e-15)
    assert np.isclose(inverted.radius, surface.radius)
    assert inverted.curvature_sign == -surface.curvature_sign
    # The foci swap roles without moving.
    np.testing.assert_allclose(inverted.focus_1, surface.focus_2, atol=1e-15)
    np.testing.assert_allclose(inverted.focus_2, surface.focus_1, atol=1e-15)
    assert (inverted.n_1, inverted.n_2) == (surface.n_2, surface.n_1)

    # Tracing the inverse focuses just as exactly, and inverting twice is the identity.
    outgoing = inverted.propagate_ray(_cartesian_oval_incoming_fan(inverted, half_angle=0.05))
    assert np.all(_distance_from_point_to_ray_lines(outgoing, inverted.focus_2) < 1e-15)
    assert surfaces_are_equivalent(surface, inverted.inverse)


def test_cartesian_oval_paraxial_agrees_with_the_matching_sphere():
    n_1, n_2, E_1, E_2 = 1.0, 1.45, 0.02, 0.05
    surface = RefractiveCartesianOval(
        center=np.array([0.1, 0.0, 0.0]), outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=6.35e-3
    )
    # The vertex radius is the textbook paraxial refraction result for these conjugates.
    assert np.isclose(surface.radius, abs((n_2 - n_1) / (n_1 / E_1 + n_2 / E_2)))

    matching_sphere = SphericalRefractiveSurface(
        radius=surface.radius,
        outwards_normal=LEFT,
        center=surface.center,
        n_1=n_1,
        n_2=n_2,
        curvature_sign=surface.curvature_sign,
    )
    np.testing.assert_allclose(
        surface.ABCD_matrix(cos_theta_incoming=np.array(1.0)),
        matching_sphere.ABCD_matrix(cos_theta_incoming=np.array(1.0)),
    )
    # Close to the axis the exact intersection converges onto the sphere's.
    for height, tolerance in ((1e-6, 1e-9), (1e-5, 1e-8)):
        ray = Ray(origin=np.array([0.0, height, 0.0]), k_vector=np.array([1.0, 0.0, 0.0]), n=n_1)
        np.testing.assert_allclose(
            surface.find_intersection_with_ray_exact(ray),
            matching_sphere.find_intersection_with_ray_exact(ray),
            atol=tolerance,
        )


def test_cartesian_oval_ray_shapes_and_aperture():
    surface = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=0.02, E_2=0.05, n_1=1.0, n_2=1.45, diameter=6.35e-3
    )
    # A single ray keeps the bare (3,) shape, and a grid of rays keeps its leading shape.
    single = Ray(origin=np.array([-0.01, 1e-3, 0.0]), k_vector=np.array([1.0, 0.0, 0.0]), n=1.0)
    assert surface.find_intersection_with_ray_exact(single).shape == (3,)

    grid_origin = np.zeros((3, 4, 3))
    grid_origin[..., 0] = -0.01
    grid_origin[..., 1] = np.linspace(-1e-3, 1e-3, 12).reshape(3, 4)
    grid_k_vector = np.zeros((3, 4, 3))
    grid_k_vector[..., 0] = 1.0
    grid = Ray(origin=grid_origin, k_vector=grid_k_vector, n=1.0)
    intersections = surface.find_intersection_with_ray_exact(grid)
    assert intersections.shape == (3, 4, 3)
    assert np.all(np.isfinite(intersections))
    assert surface.normal_at_a_point(intersections).shape == (3, 4, 3)

    # A ray outside the clear aperture misses, and shows up as nan rather than as a spurious hit.
    heights = np.array([3.0e-3, 4.0e-3])  # the aperture radius is 3.175e-3
    outside = Ray(
        origin=np.stack([np.full_like(heights, -0.01), heights, np.zeros_like(heights)], axis=-1),
        k_vector=np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)),
        n=1.0,
    )
    hit, miss = surface.find_intersection_with_ray_exact(outside)
    assert np.all(np.isfinite(hit))
    assert np.all(np.isnan(miss))


def test_cartesian_oval_init_syntax_round_trip():
    surface = RefractiveCartesianOval(
        center=np.array([0.1, 0.0, 0.0]),
        outwards_normal=LEFT,
        E_1=0.02,
        E_2=0.05,
        n_1=1.0,
        n_2=1.45,
        diameter=6.35e-3,
        material_properties=MaterialProperties(refractive_index=1.45),
        name="round trip oval",
    )
    assert "RefractiveCartesianOval(" in surface.init_syntax
    assert surfaces_are_equivalent(surface, eval(surface.init_syntax))
    # An inverted oval round-trips too - its curvature_sign is the one that differs from the default.
    assert surfaces_are_equivalent(surface.inverse, eval(surface.inverse.init_syntax))
    # The bare geometry class carries n_1/n_2 as well, since they define its shape.
    bare = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=0.02, E_2=0.05, n_1=1.0, n_2=1.45, diameter=6.35e-3)
    assert surfaces_are_equivalent(bare, eval(bare.init_syntax))


def test_cartesian_oval_floating_center():
    # Same floating-position convention as the other surface types: center may be omitted, and the shape
    # (unlike the pose) is mandatory.
    surface = RefractiveCartesianOval(outwards_normal=RIGHT, E_1=0.02, E_2=0.05, n_1=1.0, n_2=1.45, diameter=6.35e-3)
    assert surface.center.shape == (3,) and np.all(np.isnan(surface.center))
    assert not surface.positions_defined
    # Intrinsic geometry is available even while floating.
    assert np.isfinite(surface.radius) and np.isfinite(surface.thickness_center)

    surface.center = np.array([0.01, 0.0, 0.0])
    assert surface.positions_defined
    surface.center = None
    assert not surface.positions_defined

    with pytest.raises(TypeError):
        RefractiveCartesianOval(outwards_normal=RIGHT)


def test_cartesian_oval_relative_center_resolution():
    # An imaginary center is a relative offset from the previous surface, resolved when the containing
    # OpticalSystem is placed.
    T_c = 3.4e-3
    flat = FlatRefractiveSurface(
        outwards_normal=LEFT, n_1=1, n_2=1.45, diameter=6.35e-3, name="floating oval lens - flat side"
    )
    oval = RefractiveCartesianOval(
        center=T_c * RIGHT * 1j,
        outwards_normal=RIGHT,
        E_1=0.02,
        E_2=0.05,
        n_1=1.45,
        n_2=1,
        diameter=6.35e-3,
        name="floating oval lens - oval side",
    )
    lens = OpticalSystem(elements=[flat, oval], use_paraxial_ray_tracing=True)
    assert not lens.positions_defined

    anchor = np.array([0.01, 0.0, 0.0])
    placed = lens.to_position(anchor)
    assert placed.positions_defined
    np.testing.assert_allclose(placed.surfaces[0].center, anchor, atol=1e-12)
    np.testing.assert_allclose(placed.surfaces[1].center, anchor + T_c * RIGHT, atol=1e-12)
    assert not lens.positions_defined


def test_cartesian_oval_rejects_inconsistent_parameters():
    base = dict(center=ORIGIN, outwards_normal=LEFT, n_1=1.0, n_2=1.45, diameter=6.35e-3)
    with pytest.raises(ValueError, match="non-zero"):
        RefractiveCartesianOval(E_1=0.0, E_2=0.05, **base)
    with pytest.raises(ValueError, match="must differ"):
        RefractiveCartesianOval(
            center=ORIGIN, outwards_normal=LEFT, n_1=1.45, n_2=1.45, E_1=0.02, E_2=0.05, diameter=6.35e-3
        )
    # curvature_sign is fixed by the optics, not free as it is for an asphere.
    with pytest.raises(ValueError, match="contradicts the optics"):
        RefractiveCartesianOval(E_1=0.02, E_2=0.05, curvature_sign=CurvatureSigns.concave, **base)
    # An afocal oval has a flat vertex, so its illumination direction has to be stated explicitly.
    with pytest.raises(ValueError, match="flat"):
        RefractiveCartesianOval(E_1=0.02, E_2=-0.02 * 1.45, **base)
    afocal = RefractiveCartesianOval(E_1=0.02, E_2=-0.02 * 1.45, curvature_sign=CurvatureSigns.convex, **base)
    assert np.isinf(afocal.radius)

@pytest.mark.parametrize("n_1, n_2, E_1, E_2", CARTESIAN_OVAL_CONJUGATES)
@pytest.mark.parametrize("method", ["taylor", "fit"])
def test_cartesian_oval_sag_expansion_matches_the_exact_sag(n_1, n_2, E_1, E_2, method):
    # The polynomial coefficients must reproduce the oval's own local_sag, and do so better and better as the
    # degree grows - that is the whole contract of the expansion.
    # The aperture is set as a fraction of the vertex radius rather than to a fixed size, so that every conjugate
    # pair here is sampled at a comparable steepness - and well inside the radius of convergence of the series.
    radius = abs(signed_vertex_radius_of_a_cartesian_oval(n_1=n_1, n_2=n_2, E_1=E_1, E_2=E_2))
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=0.4 * radius)
    rho = np.linspace(0, oval.diameter / 2, 41)
    exact_sag = oval.local_sag(rho)

    errors = []
    for degree in (2, 4, 6, 10):
        coefficients = oval.sag_polynomial_coefficients(degree=degree, method=method)
        assert len(coefficients) == degree // 2 + 1
        assert coefficients[0] == 0, "the vertex of the expansion must sit on the oval's center"
        errors.append(np.max(np.abs(Polynomial(coefficients)(rho**2) - exact_sag)))

    assert np.all(np.diff(errors) < 0), f"the expansion did not improve with degree: {errors}"
    assert errors[-1] < 1e-5 * radius


@pytest.mark.parametrize("n_1, n_2, E_1, E_2", CARTESIAN_OVAL_CONJUGATES)
def test_cartesian_oval_expansion_starts_at_the_matching_sphere(n_1, n_2, E_1, E_2):
    # The quadratic term of any sag polynomial is 1/(2R), so the leading term of the expansion has to be the
    # sphere that osculates the oval at its vertex - and hence the asphere built from it reports the same radius.
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=0.5)
    longitudinal = cartesian_oval_longitudinal_expansion(n_1=n_1, n_2=n_2, E_1=E_1, E_2=E_2, n_coefficients=4)
    signed_radius = signed_vertex_radius_of_a_cartesian_oval(n_1=n_1, n_2=n_2, E_1=E_1, E_2=E_2)

    assert longitudinal[0] == 0
    assert np.isclose(longitudinal[1], 1 / (2 * signed_radius))
    assert np.isclose(oval.sag_polynomial_coefficients(degree=6)[1], 1 / (2 * oval.radius))


def test_aplanatic_cartesian_oval_expands_into_a_sphere():
    # When C == 0 the defining equation collapses to L_1/L_2 = const - a sphere of Apollonius, i.e. the aplanatic
    # points. The expansion then has to come out as the sphere's own Taylor series, coefficient for coefficient.
    # This is also the case that a solve based on the squared (quartic) form of the oval would divide by zero on.
    n_1, n_2, E_1 = 1.0, 1.5, 0.03
    E_2 = -n_1 * E_1 / n_2  # makes C = n_1*E_1 + n_2*E_2 vanish
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=E_1, E_2=E_2, n_1=n_1, n_2=n_2, diameter=0.012)
    assert oval.C == 0

    sphere = AsphericRefractiveSurface.pseudo_spherical(
        radius=oval.radius,
        outwards_normal=LEFT,
        center=ORIGIN,
        diameter=oval.diameter,
        curvature_sign=oval.curvature_sign,
    )
    assert np.allclose(oval.sag_polynomial_coefficients(degree=10), sphere.polynomial.coef, rtol=1e-11, atol=0)


def test_cartesian_oval_sag_expansion_rejects_a_bad_degree():
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=1.0, E_2=2.0, n_1=1.0, n_2=1.5, diameter=0.5)
    for degree in (0, 1, 5, -4):
        with pytest.raises(ValueError, match="degree"):
            oval.sag_polynomial_coefficients(degree=degree)
    with pytest.raises(ValueError, match="method"):
        oval.sag_polynomial_coefficients(degree=6, method="chebyshev")


@pytest.mark.parametrize("method", ["taylor", "fit"])
def test_pseudo_cartesian_oval_converges_to_the_oval(method):
    # The point of the factory: as the degree grows the asphere's focus approaches the oval's exact one, and even
    # at a modest degree it beats the sphere that matches the same vertex.
    oval = RefractiveCartesianOval(
        center=ORIGIN, outwards_normal=LEFT, E_1=0.03, E_2=0.06, n_1=1.0, n_2=1.5, diameter=0.01
    )
    incoming = _cartesian_oval_incoming_fan(oval, half_angle=0.15, n_rays=15)

    def worst_focus_miss(surface):
        outgoing = surface.propagate_ray(incoming)
        assert np.all(np.isfinite(outgoing.origin)), "some rays failed to intersect the surface"
        return _distance_from_point_to_ray_lines(outgoing, oval.focus_2).max()

    misses = [
        worst_focus_miss(AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=degree, expansion_method=method))
        for degree in (2, 4, 6, 10)
    ]
    assert np.all(np.diff(misses) < 0), f"the asphere did not improve with degree: {misses}"
    assert worst_focus_miss(oval) < 1e-12 < misses[-1], "the oval itself must still be the exact one"

    matching_sphere = AsphericRefractiveSurface.pseudo_spherical(
        radius=oval.radius,
        outwards_normal=LEFT,
        center=ORIGIN,
        n_1=oval.n_1,
        n_2=oval.n_2,
        diameter=oval.diameter,
        curvature_sign=oval.curvature_sign,
    )
    assert misses[-1] < 0.01 * worst_focus_miss(matching_sphere)


def test_pseudo_cartesian_oval_takes_its_parameters_from_the_oval():
    material_properties = PHYSICAL_SIZES_DICT["material_properties_fused_silica"]
    oval = RefractiveCartesianOval(
        center=np.array([0.011, -0.004, 0.0]),
        outwards_normal=normalize_vector(np.array([-1.0, 0.3, 0.0])),
        E_1=0.02,
        E_2=0.05,
        n_1=1.0,
        n_2=1.45,
        diameter=6.35e-3,
        name="oval",
        material_properties=material_properties,
    )
    asphere = AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=6)

    assert np.allclose(asphere.center, oval.center)
    assert np.allclose(asphere.outwards_normal, oval.outwards_normal)
    assert asphere.curvature_sign == oval.curvature_sign  # derived from the optics, never guessed
    assert np.isclose(asphere.radius, oval.radius)
    assert (asphere.n_1, asphere.n_2) == (oval.n_1, oval.n_2)
    assert asphere.diameter == oval.diameter and asphere.name == oval.name
    assert asphere.material_properties is material_properties

    # The same surface described by its parameters instead of by an object.
    from_parameters = AsphericRefractiveSurface.pseudo_cartesian_oval(
        E_1=oval.E_1,
        E_2=oval.E_2,
        n_1=oval.n_1,
        n_2=oval.n_2,
        center=oval.center,
        outwards_normal=oval.outwards_normal,
        diameter=oval.diameter,
        degree=6,
    )
    assert np.allclose(from_parameters.polynomial.coef, asphere.polynomial.coef)

    # An explicit argument overrides the oval it came from.
    overridden = AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=6, n_2=1.6, name="overridden")
    assert (overridden.n_2, overridden.name) == (1.6, "overridden")
    assert not np.allclose(overridden.polynomial.coef, asphere.polynomial.coef)


def test_pseudo_cartesian_oval_adds_corrections_on_top():
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=0.03, E_2=0.06, n_1=1.0, n_2=1.5, diameter=0.01)
    corrections = [0, 0, 1.5e3, -2e5]
    asphere = AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=10, polynomial_coefficients=corrections)

    expected = oval.sag_polynomial_coefficients(degree=10) + np.pad(corrections, (0, 6 - len(corrections)))
    assert np.allclose(asphere.polynomial.coef, expected)

    # Corrections beyond the requested degree are trimmed, with a warning - as in pseudo_spherical.
    with pytest.warns(UserWarning, match="trimming"):
        trimmed = AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=4, polynomial_coefficients=corrections)
    assert np.allclose(trimmed.polynomial.coef, oval.sag_polynomial_coefficients(degree=4) + corrections[:3])


def test_pseudo_cartesian_oval_of_a_floating_oval():
    # A floating oval has no center yet; the expansion is pure local geometry, so it must still work.
    oval = CartesianOval(outwards_normal=LEFT, E_1=0.03, E_2=0.06, n_1=1.0, n_2=1.5, diameter=0.01)
    asphere = AsphericRefractiveSurface.pseudo_cartesian_oval(oval, degree=6)
    assert not asphere.positions_defined
    assert np.allclose(asphere.polynomial.coef, oval.sag_polynomial_coefficients(degree=6))

    asphere.center = np.array([0.002, 0.0, 0.0])
    assert asphere.positions_defined


def test_pseudo_spherical_from_a_spherical_surface():
    material_properties = PHYSICAL_SIZES_DICT["material_properties_fused_silica"]
    sphere = SphericalRefractiveSurface(
        radius=8e-3,
        outwards_normal=LEFT,
        center=np.array([1e-3, 0.0, 0.0]),
        n_1=1.0,
        n_2=1.45,
        curvature_sign=CurvatureSigns.convex,
        diameter=7.75e-3,
        name="a catalog surface",
        material_properties=material_properties,
    )
    asphere = AsphericRefractiveSurface.pseudo_spherical(sphere)

    # Identical to spelling every parameter out by hand, which is how this was called before.
    by_hand = AsphericRefractiveSurface.pseudo_spherical(
        radius=8e-3,
        outwards_normal=LEFT,
        center=np.array([1e-3, 0.0, 0.0]),
        n_1=1.0,
        n_2=1.45,
        curvature_sign=CurvatureSigns.convex,
        diameter=7.75e-3,
        name="a catalog surface",
        material_properties=material_properties,
    )
    assert surfaces_are_equivalent(asphere, by_hand)
    assert np.allclose(asphere.polynomial.coef, by_hand.polynomial.coef)
    assert asphere.name == by_hand.name and asphere.material_properties is material_properties

    # It really is the same sphere: a fan of rays hits both within the paraxial residual of the expansion.
    ray = Ray(
        origin=np.array([[-0.02, y, 0.0] for y in np.linspace(-2e-3, 2e-3, 9)]),
        k_vector=np.tile(RIGHT, (9, 1)),
        n=1.0,
    )
    assert np.allclose(
        asphere.find_intersection_with_ray_exact(ray), sphere.find_intersection_with_ray_exact(ray), atol=1e-9
    )

    # An explicit argument still wins over the surface it came from.
    overridden = AsphericRefractiveSurface.pseudo_spherical(sphere, n_2=1.6, diameter=5e-3)
    assert (overridden.n_2, overridden.diameter) == (1.6, 5e-3)
    assert overridden.n_1 == sphere.n_1 and overridden.curvature_sign == sphere.curvature_sign


def test_pseudo_spherical_from_a_plain_spherical_surface():
    # A mirror carries no refractive indices, so those fall back to their defaults rather than blowing up.
    mirror = SphericalMirror(
        radius=5e-3,
        outwards_normal=LEFT,
        center=np.array([-5e-3, 0.0, 0.0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=7.75e-3,
    )
    asphere = AsphericRefractiveSurface.pseudo_spherical(mirror)
    assert (asphere.n_1, asphere.n_2) == (1, 1)
    assert asphere.curvature_sign == mirror.curvature_sign
    assert np.isclose(asphere.radius, mirror.radius)


def test_pseudo_spherical_needs_a_radius():
    with pytest.raises(TypeError, match="radius"):
        AsphericRefractiveSurface.pseudo_spherical(outwards_normal=LEFT, center=ORIGIN, diameter=7.75e-3)

# (back_focal_length, front_focal_length, T_c, n) of a few two-oval lenses worth checking.
CARTESIAN_OVAL_LENSES = [
    (5e-3, 50e-3, 4e-3, 1.5),  # a fast collector: strongly asymmetric conjugates
    (50e-3, 5e-3, 4e-3, 1.5),  # the same, run backwards
    (8e-3, 30e-3, 6e-3, 1.45),  # a genuinely thick element
    (12e-3, -40e-3, 3e-3, 1.5),  # a virtual image
    (-30e-3, 20e-3, 3e-3, 1.5),  # a virtual object
]


def _cartesian_oval_lens_fan(lens, back_focal_length, marginal_ray_height, n_rays=15):
    """A fan of rays leaving (or aimed at) the object point of a placed two-oval lens.

    The fan is specified by the height its marginal ray reaches at the back face rather than by an angle, so that
    lenses with very different object distances are all sampled at a comparable fraction of the clear aperture."""
    back_surface = lens.surfaces[0]
    optical_axis = back_surface.propagation_direction
    transverse = np.cross(optical_axis, np.array([0.0, 0.0, 1.0]))
    half_angle = np.arctan(marginal_ray_height / abs(back_focal_length))
    angles = np.linspace(-half_angle, half_angle, n_rays)
    k_vector = np.stack([np.cos(t) * optical_axis + np.sin(t) * transverse for t in angles])
    object_point = back_surface.center - back_focal_length * optical_axis
    if back_focal_length > 0:
        origin = np.tile(object_point, (n_rays, 1))
    else:  # A virtual object: the rays are intercepted on their way to it.
        origin = object_point - 2 * abs(back_focal_length) * k_vector
    return Ray(origin=origin, k_vector=k_vector, n=back_surface.n_1)


def _trace_through_cartesian_oval_lens(lens, incoming):
    inside = lens.surfaces[0].propagate_ray(incoming)
    return inside, lens.surfaces[1].propagate_ray(inside)


@pytest.mark.parametrize("back_focal_length, front_focal_length, T_c, n", CARTESIAN_OVAL_LENSES)
@pytest.mark.parametrize("split", CARTESIAN_OVAL_LENS_SPLITS)
def test_cartesian_oval_lens_images_its_conjugate_pair_exactly(back_focal_length, front_focal_length, T_c, n, split):
    # The defining property of the element: both faces are exact ovals, so the pair is stigmatic to machine
    # precision - and it is so for *every* split, since the split only moves the (perfect) intermediate image.
    lens = generate_cartesian_oval_lens(
        back_focal_length=back_focal_length,
        front_focal_length=front_focal_length,
        T_c=T_c,
        n=n,
        diameter=4e-3,
        split=split,
    ).to_position(ORIGIN)
    incoming = _cartesian_oval_lens_fan(lens, back_focal_length, marginal_ray_height=0.6e-3)
    inside, outgoing = _trace_through_cartesian_oval_lens(lens, incoming)
    assert np.all(np.isfinite(outgoing.origin)), "some rays failed to cross the lens"

    image_point = lens.surfaces[1].center + front_focal_length * lens.surfaces[1].propagation_direction
    misses = _distance_from_point_to_ray_lines(outgoing, image_point)
    assert np.all(misses < 1e-12), f"rays missed the image point by up to {misses.max()}"


def test_cartesian_oval_lens_is_floating_and_placeable():
    T_c = 4e-3
    lens = generate_cartesian_oval_lens(back_focal_length=5e-3, front_focal_length=50e-3, T_c=T_c, n=1.5, diameter=3e-3)
    back, front = lens.surfaces
    assert np.all(np.isnan(back.center)), "the back face should be left undefined"
    assert np.allclose(front.center, T_c * RIGHT * 1j), "the front face should be a relative offset of T_c"

    placed = lens.to_position(ORIGIN)
    assert np.allclose(placed.surfaces[0].center, ORIGIN)
    assert np.allclose(placed.surfaces[1].center, T_c * RIGHT)
    assert np.all(np.isnan(lens.surfaces[0].center)), "to_position must not mutate the original"
    # Both faces have to be traversed in the same direction for the lens to be a lens at all.
    assert np.allclose(placed.surfaces[0].propagation_direction, RIGHT)
    assert np.allclose(placed.surfaces[1].propagation_direction, RIGHT)
    assert (placed.surfaces[0].n_1, placed.surfaces[0].n_2) == (1.0, 1.5)
    assert (placed.surfaces[1].n_1, placed.surfaces[1].n_2) == (1.5, 1.0)


@pytest.mark.parametrize("back_focal_length, front_focal_length, T_c, n", CARTESIAN_OVAL_LENSES)
def test_cartesian_oval_lens_split_rules(back_focal_length, front_focal_length, T_c, n):
    K = 1 / back_focal_length - 1 / front_focal_length

    def distance_of(split, thickness=T_c):
        return cartesian_oval_lens_intermediate_image_distance(
            back_focal_length=back_focal_length, front_focal_length=front_focal_length, T_c=thickness, split=split
        )

    assert np.isclose(distance_of("thin"), -2 / K)
    assert np.isclose(distance_of("equal_deviation"), -(2 + T_c / front_focal_length) / K)

    # equal_curvature_step is defined by a quadratic rather than a formula, so check it actually solves it.
    a = distance_of("equal_curvature_step")
    assert np.isclose(K * a**2 + (2 - K * T_c) * a - T_c, 0, atol=1e-12 * abs(a))

    # All three coincide in the thin limit, which is the only regime where the thickness cannot matter.
    thin_limit = [distance_of(split, thickness=0.0) for split in CARTESIAN_OVAL_LENS_SPLITS]
    assert np.allclose(thin_limit, -2 / K)


@pytest.mark.parametrize("back_focal_length, front_focal_length, T_c, n", CARTESIAN_OVAL_LENSES[:3])
def test_equal_deviation_split_balances_the_incidence_angles(back_focal_length, front_focal_length, T_c, n):
    # The reason to prefer this split: the two faces see the same angle of incidence, which is what minimises the
    # larger of the two - and with it the Fresnel loss and the margin before total internal reflection on the way
    # out. The claim is paraxial, so it is checked with a narrow fan.
    def air_side_incidence_angles(split):
        lens = generate_cartesian_oval_lens(
            back_focal_length=back_focal_length,
            front_focal_length=front_focal_length,
            T_c=T_c,
            n=n,
            diameter=6e-3,
            split=split,
        ).to_position(ORIGIN)
        incoming = _cartesian_oval_lens_fan(lens, back_focal_length, marginal_ray_height=0.2e-3)
        inside, outgoing = _trace_through_cartesian_oval_lens(lens, incoming)
        assert np.all(np.isfinite(outgoing.origin))

        def angle(surface, ray, hit_point):
            cosine = np.abs(np.sum(surface.normal_at_a_point(hit_point) * ray.k_vector, axis=-1))
            return np.max(np.arccos(np.clip(cosine, -1, 1)))

        at_the_back = angle(lens.surfaces[0], incoming, inside.origin)  # already outside the glass
        in_the_glass = angle(lens.surfaces[1], inside, outgoing.origin)
        return at_the_back, np.arcsin(np.clip(n * np.sin(in_the_glass), -1, 1))  # refract back out to air

    balanced = air_side_incidence_angles("equal_deviation")
    assert np.isclose(balanced[0], balanced[1], rtol=0.02), f"angles not balanced: {balanced}"
    for split in ("thin", "equal_curvature_step"):
        assert max(balanced) < 0.99 * max(air_side_incidence_angles(split)), f"{split} beat equal_deviation"


def test_cartesian_oval_lens_accepts_an_explicit_split():
    shared = dict(back_focal_length=5e-3, front_focal_length=50e-3, T_c=4e-3, n=1.5, diameter=3e-3)
    chosen = -13e-3
    lens = generate_cartesian_oval_lens(intermediate_image_distance=chosen, **shared).to_position(ORIGIN)
    assert np.isclose(lens.surfaces[0].E_2, chosen)
    assert np.isclose(lens.surfaces[1].E_1, shared["T_c"] - chosen)
    # The two faces must agree on where the intermediate image is, in absolute terms.
    assert np.allclose(lens.surfaces[0].focus_2, lens.surfaces[1].focus_1)
    # ... and it is still a perfect imager, which is the whole point of the split being free.
    incoming = _cartesian_oval_lens_fan(lens, shared["back_focal_length"], marginal_ray_height=0.5e-3)
    _, outgoing = _trace_through_cartesian_oval_lens(lens, incoming)
    image_point = lens.surfaces[1].center + shared["front_focal_length"] * RIGHT
    assert np.all(_distance_from_point_to_ray_lines(outgoing, image_point) < 1e-12)


def test_cartesian_oval_lens_rejects_impossible_designs():
    shared = dict(back_focal_length=5e-3, front_focal_length=50e-3, T_c=2e-3, n=1.5, diameter=2e-3)
    with pytest.raises(ValueError, match="T_c"):
        generate_cartesian_oval_lens(**{**shared, "T_c": 0.0})
    with pytest.raises(ValueError, match="differ"):
        generate_cartesian_oval_lens(**{**shared, "n": 1.0})
    with pytest.raises(ValueError, match="diameter"):
        generate_cartesian_oval_lens(**{**shared, "diameter": 0.0})
    with pytest.raises(ValueError, match="split"):
        generate_cartesian_oval_lens(**shared, split="whatever_feels_right")
    # Equal conjugates leave the beam collimated in the glass, which no finite-conjugate oval can express.
    with pytest.raises(ValueError, match="collimated"):
        generate_cartesian_oval_lens(**{**shared, "front_focal_length": shared["back_focal_length"]})
    # An aperture the oval never reaches.
    with pytest.raises(ValueError, match="clear aperture"):
        generate_cartesian_oval_lens(**{**shared, "diameter": 40e-3})


def test_cartesian_oval_local_sag_is_nan_past_the_widest_point():
    # An oval closes on itself, so beyond its widest point there is no sag to report. The Newton iteration would
    # happily return an arbitrary number there, so the result is verified against the defining equation.
    oval = CartesianOval(center=ORIGIN, outwards_normal=LEFT, E_1=5e-3, E_2=50e-3, n_1=1.0, n_2=1.5, diameter=2e-3)
    rho = np.array([0.0, 0.5e-3, 1e-3, 1.0, 1e3])
    sag = oval.local_sag(rho)
    assert np.all(np.isfinite(sag[:3])) and sag[0] == 0
    assert np.all(np.isnan(sag[3:])), f"unreachable radii should be nan, got {sag[3:]}"
def _two_surface_lens(radius_back, sign_back, radius_front, sign_front, T_c=3e-3, n=1.5):
    """A floating thick lens from two spherical faces, described by magnitudes and curvature signs."""

    def face(radius, curvature_sign, center, n_1, n_2, name):
        if not np.isfinite(radius):  # A flat face has no curvature sign to orient it by.
            return FlatRefractiveSurface(outwards_normal=RIGHT, center=center, n_1=n_1, n_2=n_2, name=name)
        # propagation is along +RIGHT, and curvature_sign is taken with respect to the incoming ray, so a convex
        # face (+1) is one the light reaches from its outwards_normal side.
        return SphericalRefractiveSurface(
            radius=radius,
            outwards_normal=-curvature_sign * RIGHT,
            center=center,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=curvature_sign,
            name=name,
        )

    surfaces = [
        face(radius_back, sign_back, None, 1.0, n, "back"),
        face(radius_front, sign_front, T_c * RIGHT * 1j, n, 1.0, "front"),
    ]
    return OpticalSystem(elements=surfaces, use_paraxial_ray_tracing=True, t_is_trivial=True, p_is_trivial=True)


def test_lensmaker_radius_of_a_surface():
    # radius is a non-negative magnitude and the direction the face bends lives in curvature_sign, so the signed
    # radius the lensmaker formula wants is their product.
    convex = SphericalRefractiveSurface(
        radius=5e-3, outwards_normal=LEFT, center=ORIGIN, curvature_sign=CurvatureSigns.convex
    )
    concave = SphericalRefractiveSurface(
        radius=5e-3, outwards_normal=RIGHT, center=ORIGIN, curvature_sign=CurvatureSigns.concave
    )
    assert lensmaker_radius_of_a_surface(convex, fallback_curvature_sign=CurvatureSigns.convex) == 5e-3
    assert lensmaker_radius_of_a_surface(concave, fallback_curvature_sign=CurvatureSigns.convex) == -5e-3

    # A flat face carries no curvature sign, so the fallback decides - and 1/R is zero for either choice.
    flat = FlatRefractiveSurface(outwards_normal=LEFT, center=ORIGIN, n_1=1.0, n_2=1.5)
    assert lensmaker_radius_of_a_surface(flat, fallback_curvature_sign=CurvatureSigns.convex) == np.inf
    assert lensmaker_radius_of_a_surface(flat, fallback_curvature_sign=CurvatureSigns.concave) == -np.inf


def test_focal_length_of_a_biconvex_lens_object():
    # The ordinary case: signs (+1, -1), which is what the two helpers used to assume unconditionally.
    T_c, n = 3e-3, 1.5
    lens = _two_surface_lens(10e-3, CurvatureSigns.convex, 20e-3, CurvatureSigns.concave, T_c=T_c, n=n)
    assert np.isclose(focal_length_of_lens_object(lens), focal_length_of_lens_formula(10e-3, -20e-3, n, T_c))
    assert np.isclose(
        back_focal_length_of_lens_object(lens), back_focal_length_of_lens_formula(R_1=10e-3, R_2=-20e-3, n=n, T_c=T_c)
    )
    assert focal_length_of_lens_object(lens) > 0  # a converging lens


def test_focal_length_of_a_meniscus_lens_object():
    # Both faces bending the same way. Taking the second radius as -R regardless, as the helpers once did, turns a
    # meniscus into a biconvex lens and gets the focal length badly wrong.
    T_c, n = 3e-3, 1.5
    lens = _two_surface_lens(10e-3, CurvatureSigns.concave, 20e-3, CurvatureSigns.concave, T_c=T_c, n=n)
    assert np.isclose(focal_length_of_lens_object(lens), focal_length_of_lens_formula(-10e-3, -20e-3, n, T_c))
    assert focal_length_of_lens_object(lens) < 0, "concave towards the light first: this one diverges"

    # A concentric meniscus (equal radii, same sign) has no power at all but for its thickness: the 1/R_1 - 1/R_2
    # terms cancel and only the (n-1)^2 T_c / (n R_1 R_2) term is left, so it is a far weaker lens than the
    # biconvex one the old hard-coded radii would have mistaken it for.
    concentric = _two_surface_lens(5e-3, CurvatureSigns.concave, 5e-3, CurvatureSigns.concave, T_c=T_c, n=n)
    mistaken_for = _two_surface_lens(5e-3, CurvatureSigns.convex, 5e-3, CurvatureSigns.concave, T_c=T_c, n=n)
    assert np.isclose(focal_length_of_lens_object(concentric), focal_length_of_lens_formula(-5e-3, -5e-3, n, T_c))
    assert focal_length_of_lens_object(concentric) > 5 * focal_length_of_lens_object(mistaken_for)


def test_focal_length_of_a_plano_concave_lens_object():
    # A flat entrance and a concave exit: a diverging lens, which the hard-coded +R/-R pair reported as converging.
    T_c, n = 3e-3, 1.5
    lens = _two_surface_lens(np.inf, CurvatureSigns.flat, 20e-3, CurvatureSigns.convex, T_c=T_c, n=n)
    assert np.isclose(focal_length_of_lens_object(lens), focal_length_of_lens_formula(np.inf, 20e-3, n, T_c))
    assert focal_length_of_lens_object(lens) < 0


def test_focal_length_of_a_cartesian_oval_lens_object():
    # The case that turned this up: a two-oval lens goes meniscus for some conjugate pairs, and its placement in a
    # cavity is measured from the back focal length, so the sign has to follow the ovals rather than be assumed.
    shared = dict(back_focal_length=4e-3, T_c=3.83e-3, n=1.45, diameter=7.75e-3)
    biconvex = generate_cartesian_oval_lens(front_focal_length=0.4, **shared)
    meniscus = generate_cartesian_oval_lens(front_focal_length=-0.01, **shared)
    assert [surface.curvature_sign for surface in biconvex.surfaces] == [
        CurvatureSigns.convex,
        CurvatureSigns.concave,
    ]
    assert [surface.curvature_sign for surface in meniscus.surfaces] == [
        CurvatureSigns.concave,
        CurvatureSigns.concave,
    ]
    for lens in (biconvex, meniscus):
        back, front = lens.surfaces
        expected = back_focal_length_of_lens_formula(
            R_1=back.radius * back.curvature_sign,
            R_2=front.radius * front.curvature_sign,
            n=back.n_2,
            T_c=lens.T_c,
        )
        assert np.isclose(back_focal_length_of_lens_object(lens), expected)

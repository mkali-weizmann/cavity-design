import copy

import numpy as np
import pytest
from numpy.polynomial import Polynomial


from cavity_design import (
    CurvedMirror,
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
)


def test_fabry_perot_mode_finding():
    # Compares the numerical result from the analytical solution for a simple Fabry-Perot cavity
    R_1 = 5e-3
    R_2 = 5e-3
    u = 1e-5
    L = R_1 + R_2 - u
    surface_1 = CurvedMirror(
        radius=R_1,
        outwards_normal=np.array([0, 0, -1]),
        center=np.array([0, 0, -R_1]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    surface_2 = CurvedMirror(
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
    from cavity_design import CurvedRefractiveSurface

    T_c = 3e-3
    R = 20e-3
    n_glass = 1.5
    normal_in = -forward
    normal_out = forward
    s1 = CurvedRefractiveSurface(
        radius=R,
        outwards_normal=normal_in,
        center=np.array([center_x - T_c / 2, 0.0, 0.0]),
        n_1=1.0,
        n_2=n_glass,
        curvature_sign=1,
        name="lens_front",
    )
    s2 = CurvedRefractiveSurface(
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
    from cavity_design import CurvedMirror

    R = 50e-3
    L = 100e-3
    m1 = CurvedMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=-1,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = CurvedMirror(
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
    from cavity_design import CurvedMirror

    R = 50e-3
    L = 100e-3
    m1 = CurvedMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=-1,
        name="m1",
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = CurvedMirror(
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
    m1 = CurvedMirror(
        radius=R_mirror,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m1",
        diameter=25e-3,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = CurvedMirror(
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
    m1 = CurvedMirror(
        radius=R_mirror,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-L / 2, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        name="m1",
        diameter=25e-3,
    )
    lens = _make_lens_group(center_x=0.0)
    m2 = CurvedMirror(
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
    m = CurvedMirror(
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
    m1 = CurvedMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-R, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m2 = CurvedMirror(
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
    m1 = CurvedMirror(
        radius=R,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-R, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m2 = CurvedMirror(
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
    m = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=R,
        outwards_normal=np.array([0, 0, -1.0]),
        center=np.array([0, 0, -R]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=None,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=np.array([-5e-3, 0, 0]),
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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

    m0 = CurvedMirror(
        radius=5e-3,
        outwards_normal=np.array([-1.0, 0, 0]),
        center=None,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    m1 = CurvedMirror(
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
    mirror = CurvedMirror(
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
                CurvedMirror(
                    radius=radius,
                    outwards_normal=np.array([0, 0, -1.0]),
                    center=np.array([0, 0, -radius]),
                    curvature_sign=CurvatureSigns.concave,
                    diameter=0.01,
                    material_properties=mirror_properties,
                ),
                CurvedMirror(
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

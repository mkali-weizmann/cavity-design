import numpy as np
from numpy.polynomial import Polynomial


from cavity import (
    CurvedMirror,
    CurvatureSigns,
    Cavity,
    LAMBDA_0_LASER,
    FlatRefractiveSurface,
    AsphericRefractiveSurface,
    Ray,
    widget_convenient_exponent,
    perturb_cavity,
    evaluate_cavities_modes_on_surface,
    mirror_lens_mirror_cavity_generator,
)
from utils import (
    OpticalElementParams,
    MaterialProperties,
    PerturbationPointer,
    gaussians_overlap_integral,
    LensParams,
    solve_aspheric_profile,
)

from simple_analysis_scripts.analyze_potential import (
    choose_source_position_for_desired_focus_analytic,
    generate_input_parameters_for_lenses, generate_one_lens_optical_system, initialize_rays, analyze_potential,
)


def test_fabry_perot_mode_finding():
    # Compares the numerical result from the analytical solution for a simple Fabry-Perot cavity
    R_1 = 5e-3
    R_2 = 5e-3
    u = 1e-5
    L = R_1 + R_2 - u
    surface_1 = CurvedMirror(
        center=np.array([0, 0, -R_1]),
        outwards_normal=np.array([0, 0, -1]),
        radius=R_1,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    surface_2 = CurvedMirror(
        center=np.array([0, 0, -R_1 + L]),
        outwards_normal=np.array([0, 0, 1]),
        radius=R_2,
        curvature_sign=CurvatureSigns.concave,
        diameter=0.01,
    )
    cavity = Cavity(
        surfaces=[surface_1, surface_2],
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
    params = [
        OpticalElementParams(
            name="Small Mirror",
            surface_type="curved_mirror",
            x=-4.999961263669513e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            r_1=5e-03,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
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
        ),
        OpticalElementParams(
            name="Lens",
            surface_type="thick_lens",
            x=6.387599281689135e-03,
            y=0,
            z=0,
            theta=0,
            phi=0,
            r_1=2.422e-02,
            r_2=5.488e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=2.913797540986543e-03,
            n_inside_or_after=1.76e00,
            n_outside_or_before=1e00,
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
        ),
        OpticalElementParams(
            name="Big Mirror",
            surface_type="curved_mirror",
            x=4.078081462362321e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            r_1=2e-01,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
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
    s_1 = FlatRefractiveSurface(outwards_normal=optical_axis, center=back_center, n_1=n_1, n_2=n_2, diameter=diameter)

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

    rays_are_collimated = np.allclose(ray_output.k_vector @ optical_axis, 1.0, atol=1e-7)

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
        origin=np.array([[0, 2, -6], [0, 0, -6]]), k_vector=np.array([[0, 0, 1], [0, np.sqrt(2) / 2, np.sqrt(2) / 2]])
    )
    intersections, normals = s.enrich_intersection_geometries(ray_initial)
    expected_intersections = np.array([[0, 2, -4], [0, 2, -4]])
    assert np.allclose(
        intersections, expected_intersections, atol=1e-6
    ), f"Aspheric intersection test failed: expected {expected_intersections}, got {intersections}"


def test_aspheric_comparison_to_edmunds():
    params = LensParams(n=1.5168, f=45.23, T_c=7.24)

    coefficients, y, x = solve_aspheric_profile(params, y_max=12.5, n_points=1500, return_raw_values=True)
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
    assert max_residual < 1e-4, f"Aspheric comparison to Edmunds failed: max residual {max_residual}"


def test_perturbation():
    power_laser = 5.0000000000e04
    element_index_0 = 1
    param_name_0 = "y"
    perturbation_value_special_log_0 = -2.0894916580e00
    perturbation_value_special_log_0_fine = -1.5773755980e00
    element_index_1 = 0
    param_name_1 = "x"
    perturbation_value_special_log_1 = 1.7763568394e-15
    perturbation_value_special_log_1_fine = 1.7763568394e-15

    params = [
        OpticalElementParams(
            name="Small Mirror",
            surface_type="curved_mirror",
            x=-4.999961263669513e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            r_1=5e-03,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
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
        ),
        OpticalElementParams(
            name="Lens",
            surface_type="thick_lens",
            x=6.458249990515623e-03,
            y=0,
            z=0,
            theta=0,
            phi=0,
            r_1=2.422e-02,
            r_2=5.488e-03,
            curvature_sign=CurvatureSigns.concave,
            T_c=2.913797540986543e-03,
            n_inside_or_after=1.76e00,
            n_outside_or_before=1e00,
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
        ),
        OpticalElementParams(
            name="Big Mirror",
            surface_type="curved_mirror",
            x=3.565787616476249e-01,
            y=0,
            z=0,
            theta=0,
            phi=0,
            r_1=2e-01,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
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
        ),
    ]

    perturbation_value_0 = widget_convenient_exponent(perturbation_value_special_log_0, base=10, scale=-10)
    perturbation_value_1 = widget_convenient_exponent(perturbation_value_special_log_1, base=10, scale=-10)

    perturbation_value_0_fine = widget_convenient_exponent(perturbation_value_special_log_0_fine, base=10, scale=-10)
    perturbation_value_1_fine = widget_convenient_exponent(perturbation_value_special_log_1_fine, base=10, scale=-10)

    perturbation_value_0 += perturbation_value_0_fine
    perturbation_value_1 += perturbation_value_1_fine

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
            element_index=element_index_0, parameter_name=param_name_0, perturbation_value=perturbation_value_0
        ),
        PerturbationPointer(
            element_index=element_index_1, parameter_name=param_name_1, perturbation_value=perturbation_value_1
        ),
    ]
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointers)

    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_mode = evaluate_cavities_modes_on_surface(
        cavity, perturbed_cavity, arm_index=0
    )
    overlap = gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)

    assert (
        overlap - 0.9082902582542912
    ) < 1e-6, f"Perturbation test failed: Expected overlap of 0.9082902582542912 but got {overlap}"
    assert np.all(
        np.isclose(perturbed_cavity.central_line[0].k_vector, np.array([0.99991744, 0.0128496, 0.0]))
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
            np.array([[0, 0.00000000e00, 0.00000000e00], [0, 0.00000000e00, 0.00000000e00]]),
        )
    ), f"cavity_smart_generation_test failed: center should be approximately [[8.67361738e-19, 0.00000000e+00, 0.00000000e+00], [8.67361738e-19, 0.00000000e+00, 0.00000000e+00]], instead got {cavity.mode_parameters[0].center}"
    assert np.all(
        np.isclose(cavity.mode_parameters[0].z_R, np.array([1.50525208e-05, 1.50525208e-05]))
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
    params = [
        OpticalElementParams(
            name="None",
            surface_type="curved_mirror",
            x=-4.999964994473332e-03,
            y=0,
            z=0,
            theta=0,
            phi=-1e00 * np.pi,
            r_1=5e-03,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=np.nan,
            material_properties=MaterialProperties(
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
            ),
        ),
        OpticalElementParams(
            name="None",
            surface_type="curved_mirror",
            x=4.999964994473332e-03,
            y=0,
            z=0,
            theta=0,
            phi=0,
            r_1=5e-03,
            r_2=np.nan,
            curvature_sign=CurvatureSigns.concave,
            T_c=np.nan,
            n_inside_or_after=1e00,
            n_outside_or_before=1e00,
            diameter=np.nan,
            material_properties=MaterialProperties(
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
            ),
        ),
    ]
    perturbation_value_0 = widget_convenient_exponent(perturbation_value_special_log_0, base=10, scale=-10)
    perturbation_value_1 = widget_convenient_exponent(perturbation_value_special_log_1, base=10, scale=-10)

    perturbation_value_0_fine = widget_convenient_exponent(perturbation_value_special_log_0_fine, base=10, scale=-10)
    perturbation_value_1_fine = widget_convenient_exponent(perturbation_value_special_log_1_fine, base=10, scale=-10)

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
            element_index=element_index_0, parameter_name=param_name_0, perturbation_value=perturbation_value_0
        ),
        PerturbationPointer(
            element_index=element_index_1, parameter_name=param_name_1, perturbation_value=perturbation_value_1
        ),
    ]
    perturbed_cavity = perturb_cavity(cavity=cavity, perturbation_pointer=perturbation_pointers)
    if eval_box != "":
        try:
            exec(f"print({eval_box})")
        except (NameError, AttributeError) as e:
            print(f"invalid expression: {e}")
    u = np.linalg.norm(perturbed_cavity.physical_surfaces[0].origin - perturbed_cavity.physical_surfaces[1].origin)
    NA_analytical = np.sqrt(2 * LAMBDA_0_LASER / np.pi) * (perturbed_cavity.arms[0].central_line.length * u) ** (-1 / 4)
    NA_numerical = perturbed_cavity.mode_parameters[0].NA[0]
    assert np.isclose(
        NA_numerical, NA_analytical, rtol=0.0001
    ), f"Fabry-Perot perturbation test failed: expected NA of approximately {NA_analytical} but got {NA_numerical}"


def test_potential_single_lens():
    dn = 0
    lens_types = ["aspheric - lab", "spherical - like labs aspheric", "avantier", "aspheric - like avantier"]
    lens_type = lens_types[2]
    n_actual, n_design, T_c, back_focal_length, R_1, R_2, R_2_signed, diameter = generate_input_parameters_for_lenses(
        lens_type=lens_type, dn=dn
    )
    n_rays = 400
    unconcentricity = 2.24255506e-3  # np.float64(0.007610344827586207)  # ,  np.float64(0.007268965517241379)
    phi_max = 0.04
    desired_focus = 200e-3
    print_tests = True

    defocus = choose_source_position_for_desired_focus_analytic(
        desired_focus=desired_focus,
        T_c=T_c,
        n_design=n_design,
        diameter=diameter,
        back_focal_length=back_focal_length,
        R_1=R_1,
        R_2=R_2_signed,
    )

    # defocus = back_focal_length - 4.9307005112e-3

    # results_dict = generate_system_and_analyze_potential(R_1=R_1, R_2=R_2_signed, back_focal_length=back_focal_length,
    #                                                      defocus=defocus, T_c=T_c, n_design=n_design, diameter=diameter,
    #                                                      unconcentricity=unconcentricity, n_actual=n_actual,
    #                                                      n_rays=n_rays, phi_max=phi_max, extract_R_analytically=True,
    #                                                      print_tests=print_tests)

    optical_system, optical_axis = generate_one_lens_optical_system(R_1=R_1, R_2=R_2_signed,
                                                                    back_focal_length=back_focal_length,
                                                                    defocus=defocus, T_c=T_c, n_design=n_design,
                                                                    diameter=diameter, n_actual=n_actual, )
    rays_0 = initialize_rays(defocus=defocus, n_rays=n_rays, phi_max=phi_max, diameter=diameter,
                             back_focal_length=back_focal_length)
    results_dict = analyze_potential(optical_system=optical_system, rays_0=rays_0, unconcentricity=unconcentricity,
                                     print_tests=print_tests)
    assert np.isclose(
        np.abs(results_dict["zero_derivative_points"] * 1e3), 0.15342637331775477
    ), f"Potential single lens test failed: expected zero derivative point at approximately 0.15342637331775477 mm but got {results_dict['zero_derivative_points']*1e3} mm"

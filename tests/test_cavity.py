import numpy as np
from numpy.polynomial import Polynomial

from cavity import (
    CurvedMirror,
    CurvatureSigns,
    Cavity,
    LAMBDA_0_LASER, FlatRefractiveSurface, AsphericRefractiveSurface, Ray,
)
from utils import OpticalElementParams, MaterialProperties


def test_fabry_perot_mode_finding():
    # Compares the numerical result from the analytical solution for a simple Fabry-Perot cavity
    R_1 = 5e-3
    R_2 = 5e-3
    u = 1e-5
    L = R_1 + R_2 - u
    surface_1 = CurvedMirror(center=np.array([0, 0, -R_1]),
                                    outwards_normal=np.array([0, 0, -1]),
                                    radius=R_1,
                                    curvature_sign=CurvatureSigns.concave,
                                    diameter=0.01)
    surface_2 = CurvedMirror(center=np.array([0, 0, -R_1 + L]),
                                    outwards_normal=np.array([0, 0, 1]),
                                    radius=R_2,
                                    curvature_sign=CurvatureSigns.concave,
                                    diameter=0.01)
    cavity = Cavity(physical_surfaces=[surface_1, surface_2],
                    standing_wave=True,
                    lambda_0_laser=LAMBDA_0_LASER,
                    power=1e3,
                    use_paraxial_ray_tracing=False)
    theoretical_reighly_range = np.sqrt(u * L) / 2
    actual_reighly_range = cavity.arms[0].mode_parameters.z_R[0]

    theoretical_waist = np.sqrt(LAMBDA_0_LASER * theoretical_reighly_range / np.pi)
    actual_waist = cavity.arms[0].mode_parameters.w_0[0]
    # print(f'Theoretical Reighly range: {theoretical_reighly_range}, Actual Reighly range: {actual_reighly_range}')
    # print(f'Theoretical Waist: {theoretical_waist}, Actual Waist: {actual_waist}')
    assert (theoretical_reighly_range / actual_reighly_range - 1) < 1e-6, f'Reighly range mismatch: theoretical {theoretical_reighly_range}, actual {actual_reighly_range}'
    assert (theoretical_waist / actual_waist - 1) < 1e-6, f'Waist mismatch: theoretical {theoretical_waist}, actual {actual_waist}'


def test_mirror_lens_mirror_design():
    params = [
        OpticalElementParams(name='Small Mirror', surface_type='curved_mirror', x=-4.999961263669513e-03, y=0, z=0,
                             theta=0, phi=-1e+00 * np.pi, r_1=5e-03, r_2=np.nan, curvature_sign=CurvatureSigns.concave,
                             T_c=np.nan, n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                             material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                    beta_surface_absorption=1e-06,
                                                                    kappa_conductivity=1.31e+00, dn_dT=None,
                                                                    nu_poisson_ratio=1.7e-01,
                                                                    alpha_volume_absorption=None,
                                                                    intensity_reflectivity=9.99889e-01,
                                                                    intensity_transmittance=1e-04, temperature=np.nan)),
        OpticalElementParams(name='Lens', surface_type='thick_lens', x=6.387599281689135e-03, y=0, z=0, theta=0, phi=0,
                             r_1=2.422e-02, r_2=5.488e-03, curvature_sign=CurvatureSigns.concave,
                             T_c=2.913797540986543e-03, n_inside_or_after=1.76e+00, n_outside_or_before=1e+00,
                             material_properties=MaterialProperties(refractive_index=1.76e+00, alpha_expansion=5.5e-06,
                                                                    beta_surface_absorption=1e-06,
                                                                    kappa_conductivity=4.606e+01, dn_dT=1.17e-05,
                                                                    nu_poisson_ratio=3e-01,
                                                                    alpha_volume_absorption=1e-02,
                                                                    intensity_reflectivity=1e-04,
                                                                    intensity_transmittance=9.99899e-01,
                                                                    temperature=np.nan)),
        OpticalElementParams(name='Big Mirror', surface_type='curved_mirror', x=4.078081462362321e-01, y=0, z=0,
                             theta=0, phi=0, r_1=2e-01, r_2=np.nan, curvature_sign=CurvatureSigns.concave, T_c=np.nan,
                             n_inside_or_after=1e+00, n_outside_or_before=1e+00,
                             material_properties=MaterialProperties(refractive_index=None, alpha_expansion=7.5e-08,
                                                                    beta_surface_absorption=1e-06,
                                                                    kappa_conductivity=1.31e+00, dn_dT=None,
                                                                    nu_poisson_ratio=1.7e-01,
                                                                    alpha_volume_absorption=None,
                                                                    intensity_reflectivity=9.99889e-01,
                                                                    intensity_transmittance=1e-04, temperature=np.nan))
    ]

    cavity = Cavity.from_params(params=params,
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

    assert cavity.mode_parameters[0].NA[0] / 0.156 - 1 < 1e-4, f'Numerical NA mismatch: expected 0.156, got {cavity.mode_parameters[0].NA}'
    assert cavity.arms[2].central_line.length / 0.4 - 1 < 1e-4, f'Numerical cavity length mismatch: expected 0.4, got {cavity.arms[2].central_line.length}'


def test_aspheric_lens():
    phi = 0.2
    theta = 0.2

    f = 20.0
    T_c = 3.0
    n_1 = 1
    n_2 = 1.5
    polynomial_coefficients = [0, 4.54546675e-02, -2.23050041e-05,
                               1.88752450e-08]  # generated for f=20, Tc=3 in aspheric_lens_generator.py
    polynomial = Polynomial(polynomial_coefficients)

    optical_axis = np.array([np.cos(phi), np.sin(phi), 0])

    diameter = 15
    back_center = f * optical_axis
    front_center = back_center + T_c * optical_axis
    s_1 = FlatRefractiveSurface(outwards_normal=optical_axis, center=back_center, n_1=n_1, n_2=n_2, diameter=diameter)

    s_2 = AsphericRefractiveSurface(center=front_center,
                                    outwards_normal=optical_axis,
                                    diameter=diameter,
                                    polynomial_coefficients=polynomial,
                                    n_1=n_2,
                                    n_2=n_1)

    ray_initial = Ray(origin=np.array([[0, 0, 0], [0, 0, 0]]),
                      k_vector=np.array([[np.cos(-theta + phi), np.sin(-theta + phi), 0],
                                         # [np.cos(phi), np.sin(phi), 0],
                                         [np.cos(theta / 2 + phi), np.sin(theta / 2 + phi), 0]]))

    ray_inner = s_1.interact_with_ray(ray_initial)

    ray_output = s_2.interact_with_ray(ray_inner)

    rays_are_collimated = np.allclose(ray_output.k_vector @ optical_axis, 1.0, atol=1e-4)

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

    assert rays_are_collimated, f'Aspheric lens test failed: output rays are not collimated, dot products: {ray_output.k_vector @ optical_axis}'


def test_aspheric_intersection():
    polynomial_coefficients = [0, 1]
    polynomial = Polynomial(polynomial_coefficients)
    optical_axis = np.array([0, 0, 1])
    diameter = 4
    center = np.array([0, 0, 0])
    s = AsphericRefractiveSurface(center=center,
                                  outwards_normal=optical_axis,
                                  diameter=diameter,
                                  polynomial_coefficients=polynomial,
                                  n_1=1,
                                  n_2=1.5)
    ray_initial = Ray(origin=np.array([[0, 2, -6],
                                       [0, 0, -6]]),
                        k_vector=np.array([[0, 0, 1],
                                           [0, np.sqrt(2)/2, np.sqrt(2)/2]]))
    intersections, normals = s.enrich_intersection_geometries(ray_initial)
    expected_intersections = np.array([[0, 2, -4],
                                       [0, 2, -4]])
    assert np.allclose(intersections, expected_intersections, atol=1e-6), f'Aspheric intersection test failed: expected {expected_intersections}, got {intersections}'
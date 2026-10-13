"""Unit tests for critical utils.py and cavity.py functions not covered by integration tests."""

import numpy as np
import pytest

from utils import (
    SurfacesTypes,
    SURFACE_TYPES_DICT,
    SURFACE_TYPES_DICT_INVERSE,
    MaterialProperties,
    OpticalElementParams,
    CurvatureSigns,
    PerturbationPointer,
    ABCD_free_space,
    normalize_vector,
    rotation_matrix_around_n,
    unit_vector_of_angles,
    angles_of_unit_vector,
    angles_distance,
    angles_difference,
    focal_length_of_lens,
    principal_planes_of_lens,
    image_of_a_point_with_thick_lens,
    back_focal_length_of_lens,
    radius_of_f_and_n,
    w_0_of_z_R,
    z_R_of_w_0,
    z_R_of_NA,
    w_0_of_NA,
    R_of_q,
    w_of_q,
    spot_size,
    dT_c_of_a_lens,
    maximal_lens_height,
    plane_name_to_xy_indices,
    pretty_print_number,
    stable_sqrt,
    signif,
    green_function_free_space,
    normal_to_a_sphere,
    generalized_snells_law,
    generalized_mirror_law,
    convert_material_to_mirror_or_lens,
    LAMBDA_0_LASER,
    gaussian_integral_2d_log,
    gaussians_overlap_integral,
    angles_difference,
)
from cavity import (
    LocalModeParameters,
    ModeParameters,
    Ray,
    decompose_ABCD_matrix,
    propagate_local_mode_parameter_through_ABCD,
    local_mode_parameters_of_round_trip_ABCD,
    complete_orthonormal_basis,
    IdealLens,
    INWARD,
    OUTWARD,
)


# ─── SurfacesTypes ───────────────────────────────────────────────────────────


def test_surfaces_types_from_integer_representation():
    for integer, name in SURFACE_TYPES_DICT_INVERSE.items():
        result = SurfacesTypes.from_integer_representation(integer_representation=integer)
        expected = SurfacesTypes.__dict__[name]
        assert result == expected


def test_surfaces_types_has_refractive_index_true():
    for surface_type in [
        SurfacesTypes.curved_refractive_surface,
        SurfacesTypes.thick_lens,
        SurfacesTypes.ideal_thick_lens,
        SurfacesTypes.aspheric_surface,
        SurfacesTypes.thick_aspheric_lens,
    ]:
        assert SurfacesTypes.has_refractive_index(surface_type=surface_type) is True


def test_surfaces_types_has_refractive_index_false():
    for surface_type in [
        SurfacesTypes.curved_mirror,
        SurfacesTypes.flat_mirror,
        SurfacesTypes.ideal_lens,
        SurfacesTypes.flat_surface,
    ]:
        assert SurfacesTypes.has_refractive_index(surface_type=surface_type) is False


# ─── MaterialProperties ──────────────────────────────────────────────────────


def test_material_properties_repr_contains_fields():
    mat = MaterialProperties(refractive_index=1.5, alpha_expansion=5e-6)
    text = repr(mat)
    assert "MaterialProperties(" in text
    assert "refractive_index" in text


def test_material_properties_to_array_shape():
    mat = MaterialProperties(
        refractive_index=1.76,
        alpha_expansion=5.5e-6,
        beta_surface_absorption=1e-6,
        kappa_conductivity=46.06,
        dn_dT=11.7e-6,
        nu_poisson_ratio=0.3,
        alpha_volume_absorption=0.01,
        intensity_reflectivity=1e-4,
        intensity_transmittance=0.9998,
    )
    arr = mat.to_array
    assert arr.shape == (10,)
    assert arr[0] == pytest.approx(1.76)


# ─── convert_material_to_mirror_or_lens ─────────────────────────────────────


def test_convert_material_to_lens():
    mat = MaterialProperties(refractive_index=1.5, beta_surface_absorption=1e-6)
    result = convert_material_to_mirror_or_lens(
        material_properties=mat, convert_to_type="lens"
    )
    assert result.intensity_reflectivity == pytest.approx(100e-6)
    assert result.intensity_transmittance == pytest.approx(1 - 1e-6 - 100e-6)


def test_convert_material_to_mirror():
    mat = MaterialProperties(refractive_index=1.5, beta_surface_absorption=1e-6)
    result = convert_material_to_mirror_or_lens(
        material_properties=mat, convert_to_type="mirror"
    )
    assert result.intensity_transmittance == pytest.approx(100e-6)
    assert result.intensity_reflectivity == pytest.approx(1 - 10e-6 - 1e-6 - 100e-6)


def test_convert_material_invalid_type():
    mat = MaterialProperties(refractive_index=1.5, beta_surface_absorption=1e-6)
    with pytest.raises(ValueError):
        convert_material_to_mirror_or_lens(material_properties=mat, convert_to_type="prism")


# ─── PerturbationPointer ─────────────────────────────────────────────────────


def test_perturbation_pointer_call():
    pp = PerturbationPointer(element_index=0, parameter_name="x")
    pp2 = pp(perturbation_value=1e-4)
    assert pp2.perturbation_value == pytest.approx(1e-4)
    assert pp2.element_index == 0
    assert pp2.parameter_name == "x"


def test_perturbation_pointer_len_scalar():
    pp = PerturbationPointer(element_index=0, parameter_name="x", perturbation_value=1e-4)
    assert len(pp) == 1


def test_perturbation_pointer_len_array():
    pp = PerturbationPointer(
        element_index=0, parameter_name="x", perturbation_value=np.array([1e-4, 2e-4, 3e-4])
    )
    assert len(pp) == 3


def test_perturbation_pointer_getitem():
    pp = PerturbationPointer(
        element_index=1, parameter_name="y", perturbation_value=np.array([0.1, 0.2])
    )
    pp0 = pp[0]
    assert pp0.perturbation_value == pytest.approx(0.1)


def test_perturbation_pointer_iter():
    values = np.array([0.1, 0.2, 0.3])
    pp = PerturbationPointer(element_index=0, parameter_name="z", perturbation_value=values)
    extracted = [p.perturbation_value for p in pp]
    assert extracted == pytest.approx(values.tolist())


def test_perturbation_pointer_iter_scalar():
    pp = PerturbationPointer(element_index=0, parameter_name="z", perturbation_value=0.5)
    extracted = list(pp)
    assert len(extracted) == 1
    assert extracted[0].perturbation_value == pytest.approx(0.5)


# ─── pretty_print_number ─────────────────────────────────────────────────────


def test_pretty_print_number_none():
    result = pretty_print_number(number=None)
    assert "None" in result


def test_pretty_print_number_nan():
    result = pretty_print_number(number=float("nan"))
    assert "np.nan" in result


def test_pretty_print_number_zero():
    result = pretty_print_number(number=0)
    assert "0" in result


def test_pretty_print_number_regular():
    result = pretty_print_number(number=3.14)
    assert "3.14" in result


def test_pretty_print_number_angle():
    result = pretty_print_number(number=np.pi / 2, represents_angle=True)
    assert "np.pi" in result


# ─── plane_name_to_xy_indices ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "plane,expected",
    [
        ("xy", (0, 1)),
        ("yx", (0, 1)),
        ("xz", (0, 2)),
        ("zx", (0, 2)),
        ("yz", (1, 2)),
        ("zy", (1, 2)),
    ],
)
def test_plane_name_to_xy_indices(plane, expected):
    assert plane_name_to_xy_indices(plane=plane) == expected


def test_plane_name_to_xy_indices_invalid():
    with pytest.raises(ValueError):
        plane_name_to_xy_indices(plane="ab")


# ─── Geometric helpers ───────────────────────────────────────────────────────


def test_dT_c_of_a_lens():
    # For a sphere of radius R, with a chord at height h, the sagitta is R - sqrt(R^2 - h^2)
    R, h = 10.0, 3.0
    result = dT_c_of_a_lens(R=R, h=h)
    expected = R * (1 - np.sqrt(1 - h**2 / R**2))
    assert result == pytest.approx(expected)


def test_maximal_lens_height():
    R, w = 10.0, 2.0
    result = maximal_lens_height(R=R, w=w)
    expected = R * np.sqrt(1 - ((R - w / 2) / R) ** 2)
    assert result == pytest.approx(expected)


def test_normalize_vector_unit():
    v = np.array([3.0, 4.0, 0.0])
    result = normalize_vector(vector=v)
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(0.8)


def test_normalize_vector_ignore_null():
    v = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = normalize_vector(vector=v, ignore_null_vectors=True)
    assert np.linalg.norm(result[0]) == pytest.approx(1.0)
    assert np.all(result[1] == 0.0)


def test_rotation_matrix_around_n_zero_rotation():
    n = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_around_n(n=n, theta=0.0)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_rotation_matrix_around_n_quarter_turn():
    # Rotate x-axis around z-axis by 90 degrees
    n = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_around_n(n=n, theta=np.pi / 2)
    x_rotated = R @ np.array([1.0, 0.0, 0.0])
    # The result should lie in the xy plane, z=0, and be a unit vector
    assert np.allclose(x_rotated[2], 0.0, atol=1e-10)
    assert np.linalg.norm(x_rotated) == pytest.approx(1.0, abs=1e-10)


def test_unit_vector_of_angles_along_x():
    v = unit_vector_of_angles(theta=0.0, phi=0.0)
    assert np.allclose(v, [1.0, 0.0, 0.0], atol=1e-12)


def test_unit_vector_of_angles_along_z():
    v = unit_vector_of_angles(theta=np.pi / 2, phi=0.0)
    assert np.allclose(v, [0.0, 0.0, 1.0], atol=1e-12)


def test_angles_of_unit_vector_roundtrip():
    theta_in, phi_in = 0.3, 1.1
    v = unit_vector_of_angles(theta=theta_in, phi=phi_in)
    theta_out, phi_out = angles_of_unit_vector(unit_vector=v)
    assert theta_out == pytest.approx(theta_in, abs=1e-12)
    assert phi_out == pytest.approx(phi_in, abs=1e-12)


def test_angles_distance_same_vector():
    v = np.array([1.0, 0.0, 0.0])
    assert angles_distance(direction_vector_1=v, direction_vector_2=v) == pytest.approx(0.0)


def test_angles_distance_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert angles_distance(direction_vector_1=v1, direction_vector_2=v2) == pytest.approx(np.pi / 2)


def test_angles_difference_wrapping():
    # 350 degrees - 10 degrees = 340 degrees, but shortest path is -20 degrees
    d = angles_difference(angle_1=10 * np.pi / 180, angle_2=350 * np.pi / 180)
    assert d == pytest.approx(-20 * np.pi / 180, abs=1e-10)


def test_angles_difference_basic():
    d = angles_difference(angle_1=0.0, angle_2=np.pi / 4)
    assert d == pytest.approx(np.pi / 4)


# ─── Optics helpers ──────────────────────────────────────────────────────────


def test_focal_length_of_lens_biconvex():
    # Lensmaker's equation: for biconvex R_1>0, R_2<0 (sign convention: R_1 measured from left)
    R_1, R_2, n, T_c = 0.05, -0.05, 1.5, 0.005
    f = focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
    assert f > 0  # converging lens


def test_principal_planes_of_lens_symmetric():
    # Symmetric biconvex lens: h1 and h2 should be equal in magnitude
    R_1, R_2, n, T_c = 0.1, -0.1, 1.5, 0.01
    h_1, h_2 = principal_planes_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
    assert h_1 == pytest.approx(-h_2, rel=1e-6)


def test_image_of_a_point_thin_lens_approximation():
    # For thin lens (T_c ~ 0): image equation 1/v - 1/u = 1/f
    R_1, R_2, n, T_c = 0.1, -0.1, 1.5, 1e-6
    f = focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
    distance_to_face_1 = 2 * f  # object at 2f
    d_2 = image_of_a_point_with_thick_lens(
        distance_to_face_1=distance_to_face_1, R_1=R_1, R_2=R_2, n=n, T_c=T_c
    )
    # For object at 2f, image is at 2f
    assert d_2 == pytest.approx(2 * f, rel=0.01)


def test_back_focal_length_of_lens():
    R_1, R_2, n, T_c = 0.1, -0.1, 1.5, 0.01
    bfl = back_focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
    f = focal_length_of_lens(R_1=R_1, R_2=R_2, n=n, T_c=T_c)
    # BFL should be close to f for thin lenses
    assert abs(bfl) < 2 * abs(f)


def test_radius_of_f_and_n():
    f, n = 0.05, 1.5
    R = radius_of_f_and_n(f=f, n=n)
    assert R == pytest.approx(2 * f * (n - 1))


# ─── Gaussian beam math ──────────────────────────────────────────────────────


def test_w_0_of_z_R_roundtrip():
    z_R = np.array([1e-3, 1e-3])
    lambda_0 = 1064e-9
    n = 1.0
    w_0 = w_0_of_z_R(z_R=z_R, lambda_0_laser=lambda_0, n=n)
    z_R_back = z_R_of_w_0(w_0=w_0, lambda_laser=lambda_0 / n)
    assert np.allclose(z_R_back, z_R, rtol=1e-10)


def test_z_R_of_NA():
    NA = 0.1
    lambda_0 = 1064e-9
    z_R = z_R_of_NA(NA=NA, lambda_laser=lambda_0)
    w_0 = w_0_of_NA(NA=NA, lambda_laser=lambda_0)
    expected_z_R = np.pi * w_0**2 / lambda_0
    assert z_R == pytest.approx(expected_z_R, rel=1e-10)


def test_w_0_of_NA():
    NA = 0.15
    lambda_0 = 1064e-9
    w_0 = w_0_of_NA(NA=NA, lambda_laser=lambda_0)
    assert w_0 == pytest.approx(lambda_0 / (np.pi * NA), rel=1e-10)


def test_R_of_q_at_waist():
    # At the waist (z=0), q = i*z_R, so R should be infinity
    z_R = 1e-3
    q = np.array([0 + 1j * z_R, 0 + 1j * z_R])
    R = R_of_q(q=q)
    assert np.all(np.isinf(R))


def test_R_of_q_away_from_waist():
    z = 1e-3
    z_R = 1e-3
    q = np.array([z + 1j * z_R, z + 1j * z_R])
    R = R_of_q(q=q)
    expected = (z**2 + z_R**2) / z
    assert np.allclose(R, expected, rtol=1e-8)


def test_w_of_q():
    z_R = 1e-3
    lambda_0 = 1064e-9
    q = np.array([0 + 1j * z_R])
    w = w_of_q(q=q, lambda_laser=lambda_0)
    expected_w0 = w_0_of_z_R(z_R=np.array([z_R]), lambda_0_laser=lambda_0, n=1.0)
    assert np.allclose(w, expected_w0, rtol=1e-8)


def test_spot_size_at_waist():
    z_R = 1e-3
    lambda_0 = 1064e-9
    n = 1.0
    w_0_expected = w_0_of_z_R(z_R=np.array([z_R]), lambda_0_laser=lambda_0, n=n)
    w = spot_size(z=np.array([0.0]), z_R=np.array([z_R]), lambda_0_laser=lambda_0, n=n)
    assert np.allclose(w, w_0_expected, rtol=1e-10)


def test_spot_size_at_rayleigh_range():
    z_R = 1e-3
    lambda_0 = 1064e-9
    n = 1.0
    w_0_val = w_0_of_z_R(z_R=np.array([z_R]), lambda_0_laser=lambda_0, n=n)
    w = spot_size(z=np.array([z_R]), z_R=np.array([z_R]), lambda_0_laser=lambda_0, n=n)
    assert np.allclose(w, w_0_val * np.sqrt(2), rtol=1e-10)


# ─── ABCD_free_space ─────────────────────────────────────────────────────────


def test_ABCD_free_space_identity_at_zero():
    M = ABCD_free_space(length=0.0)
    assert np.allclose(M, np.eye(4), atol=1e-15)


def test_ABCD_free_space_structure():
    L = 0.5
    M = ABCD_free_space(length=L)
    assert M.shape == (4, 4)
    assert M[0, 1] == pytest.approx(L)
    assert M[2, 3] == pytest.approx(L)
    assert M[0, 0] == pytest.approx(1.0)


# ─── stable_sqrt ─────────────────────────────────────────────────────────────


def test_stable_sqrt_positive():
    assert stable_sqrt(x=4.0) == pytest.approx(2.0)


def test_stable_sqrt_negative():
    assert np.isnan(stable_sqrt(x=-1.0))


def test_stable_sqrt_nan():
    assert np.isnan(stable_sqrt(x=float("nan")))


def test_stable_sqrt_array():
    arr = np.array([4.0, -1.0, 9.0])
    result = stable_sqrt(x=arr)
    assert result[0] == pytest.approx(2.0)
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(3.0)


# ─── signif ──────────────────────────────────────────────────────────────────


def test_signif_rounds_to_significant_figures():
    assert signif(x=3.14159, p=3) == pytest.approx(3.14, rel=1e-6)


def test_signif_large_number():
    assert signif(x=123456, p=3) == pytest.approx(123000, rel=1e-6)


# ─── Green's function / EM helpers ───────────────────────────────────────────


def test_green_function_free_space_scalar():
    r_source = np.array([0.0, 0.0, 0.0])
    r_observer = np.array([1.0, 0.0, 0.0])
    k = 2 * np.pi / 1064e-9
    G = green_function_free_space(r_source=r_source, r_observer=r_observer, k=k)
    expected = np.exp(1j * k * 1.0) / (4 * np.pi * 1.0)
    assert G == pytest.approx(expected)


def test_normal_to_a_sphere_outward():
    center = np.array([0.0, 0.0, 0.0])
    r_surface = np.array([1.0, 0.0, 0.0])
    normal = normal_to_a_sphere(r_surface=r_surface, o_center=center, sign=1)
    assert np.allclose(normal, [1.0, 0.0, 0.0], atol=1e-12)


def test_normal_to_a_sphere_inward():
    center = np.array([0.0, 0.0, 0.0])
    r_surface = np.array([0.0, 1.0, 0.0])
    normal = normal_to_a_sphere(r_surface=r_surface, o_center=center, sign=-1)
    assert np.allclose(normal, [0.0, -1.0, 0.0], atol=1e-12)


# ─── Ray optics functions ────────────────────────────────────────────────────


def test_generalized_mirror_law_normal_incidence():
    k = np.array([0.0, 0.0, 1.0])
    n = np.array([0.0, 0.0, 1.0])
    reflected = generalized_mirror_law(k_vector=k, n_forwards=n)
    assert np.allclose(reflected, [0.0, 0.0, -1.0], atol=1e-12)


def test_generalized_mirror_law_grazing():
    k = np.array([1.0, 0.0, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    reflected = generalized_mirror_law(k_vector=k, n_forwards=n)
    assert np.allclose(reflected, [1.0, 0.0, 0.0], atol=1e-12)


def test_generalized_snells_law_normal_incidence():
    k = np.array([0.0, 0.0, 1.0])
    n_fwd = np.array([0.0, 0.0, 1.0])
    result = generalized_snells_law(k_vector=k, n_forwards=n_fwd, n_1=1.0, n_2=1.5)
    # Normal incidence: direction unchanged, only changes medium
    assert np.allclose(result, [0.0, 0.0, 1.0], atol=1e-10)


def test_generalized_snells_law_45_degrees():
    # Ray at 45 degrees in xz plane, going into denser medium
    theta_in = np.pi / 4
    k = normalize_vector(np.array([np.sin(theta_in), 0.0, np.cos(theta_in)]))
    n_fwd = np.array([0.0, 0.0, 1.0])
    n_1, n_2 = 1.0, 1.5
    result = generalized_snells_law(k_vector=k, n_forwards=n_fwd, n_1=n_1, n_2=n_2)
    # Check Snell's law: n1*sin(theta_in) = n2*sin(theta_out)
    sin_theta_out = result[0]  # x-component is sin(theta_out)
    assert sin_theta_out == pytest.approx(n_1 * np.sin(theta_in) / n_2, rel=1e-6)


# ─── gaussian_integral_2d_log ────────────────────────────────────────────────


def test_gaussian_integral_2d_log_identity():
    # For a standard 2D Gaussian: exp(-x^2/2 - y^2/2), A = -[[1,0],[0,1]], b=0, c=0
    # Integral = 2*pi
    A = -np.eye(2)
    b = np.zeros(2)
    c = 0.0
    log_integral = gaussian_integral_2d_log(A=A, b=b, c=c)
    assert np.real(log_integral) == pytest.approx(np.log(2 * np.pi), rel=1e-6)


# ─── LocalModeParameters ─────────────────────────────────────────────────────


def test_local_mode_parameters_from_q():
    z_R = 1e-3
    q = np.array([0.0 + 1j * z_R, 0.0 + 1j * z_R])
    lmp = LocalModeParameters(q=q, lambda_0_laser=LAMBDA_0_LASER)
    assert np.allclose(lmp.z_minus_z_0, [0.0, 0.0])
    assert np.allclose(lmp.z_R, [z_R, z_R])


def test_local_mode_parameters_from_z_minus_z0_and_zR():
    z_R = 2e-3
    z0 = 1e-3
    lmp = LocalModeParameters(z_minus_z_0=z0, z_R=z_R, lambda_0_laser=LAMBDA_0_LASER)
    assert np.allclose(lmp.z_minus_z_0, [z0, z0])
    assert np.allclose(lmp.z_R, [z_R, z_R])


def test_local_mode_parameters_missing_args():
    with pytest.raises(ValueError):
        LocalModeParameters(lambda_0_laser=LAMBDA_0_LASER)


def test_local_mode_parameters_w_0():
    z_R = np.array([1e-3, 1e-3])
    lmp = LocalModeParameters(q=0 + 1j * z_R, lambda_0_laser=LAMBDA_0_LASER)
    expected_w0 = w_0_of_z_R(z_R=z_R, lambda_0_laser=LAMBDA_0_LASER, n=1.0)
    assert np.allclose(lmp.w_0, expected_w0, rtol=1e-10)


def test_local_mode_parameters_spot_size_at_waist():
    z_R = np.array([1e-3, 1e-3])
    lmp = LocalModeParameters(q=0 + 1j * z_R, lambda_0_laser=LAMBDA_0_LASER)
    # At the waist z_minus_z_0=0, so spot_size = w_0
    w = lmp.spot_size
    expected = w_0_of_z_R(z_R=z_R, lambda_0_laser=LAMBDA_0_LASER, n=1.0)
    assert np.allclose(w, expected, rtol=1e-8)


def test_local_mode_parameters_radius_of_curvature_at_waist():
    z_R = np.array([1e-3, 1e-3])
    lmp = LocalModeParameters(q=0 + 1j * z_R, lambda_0_laser=LAMBDA_0_LASER)
    R = lmp.radius_of_curvature
    assert np.all(np.isinf(R))


def test_local_mode_parameters_radius_of_curvature_away():
    z = 1e-3
    z_R = 1e-3
    q = np.array([z + 1j * z_R, z + 1j * z_R])
    lmp = LocalModeParameters(q=q, lambda_0_laser=LAMBDA_0_LASER)
    R = lmp.radius_of_curvature
    expected = (z**2 + z_R**2) / z
    assert np.allclose(R, expected, rtol=1e-8)


# ─── ModeParameters ──────────────────────────────────────────────────────────


def test_mode_parameters_from_w_0():
    w_0 = np.array([1e-4, 1e-4])
    mp = ModeParameters(
        center=np.array([0.0, 0.0, 0.0]),
        k_vector=np.array([0.0, 0.0, 1.0]),
        w_0=w_0,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    assert mp.z_R is not None
    expected_z_R = z_R_of_w_0(w_0=w_0, lambda_laser=LAMBDA_0_LASER)
    assert np.allclose(mp.z_R, expected_z_R, rtol=1e-10)


def test_mode_parameters_from_z_R():
    z_R = np.array([1e-3, 1e-3])
    mp = ModeParameters(
        center=np.array([0.0, 0.0, 0.0]),
        k_vector=np.array([0.0, 0.0, 1.0]),
        z_R=z_R,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    assert mp.w_0 is not None
    expected_w0 = w_0_of_z_R(z_R=z_R, lambda_0_laser=LAMBDA_0_LASER, n=1.0)
    assert np.allclose(mp.w_0, expected_w0, rtol=1e-10)


def test_mode_parameters_missing_w0_and_zR():
    with pytest.raises(ValueError):
        ModeParameters(
            center=np.array([0.0, 0.0, 0.0]),
            k_vector=np.array([0.0, 0.0, 1.0]),
            lambda_0_laser=LAMBDA_0_LASER,
        )


def test_mode_parameters_NA():
    z_R = np.array([1e-3, 1e-3])
    mp = ModeParameters(
        center=np.array([0.0, 0.0, 0.0]),
        k_vector=np.array([0.0, 0.0, 1.0]),
        z_R=z_R,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    expected_NA = np.sqrt(LAMBDA_0_LASER / (np.pi * z_R))
    assert np.allclose(mp.NA, expected_NA, rtol=1e-10)


def test_mode_parameters_invert_direction():
    z_R = np.array([1e-3, 1e-3])
    k = np.array([0.0, 0.0, 1.0])
    mp = ModeParameters(
        center=np.array([0.0, 0.0, 0.0]),
        k_vector=k,
        z_R=z_R,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    inv = mp.invert_direction()
    assert np.allclose(inv.k_vector, -k)
    assert np.allclose(inv.z_R, z_R)


def test_mode_parameters_R_of_z_at_waist():
    z_R = np.array([1e-3, 1e-3])
    center = np.array([0.0, 0.0, 0.0])
    mp = ModeParameters(
        center=center,
        k_vector=np.array([0.0, 0.0, 1.0]),
        z_R=z_R,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    R = mp.R_of_z(p=0.0)
    assert R == np.inf


def test_mode_parameters_local_mode_at_point():
    z_R = np.array([1e-3, 1e-3])
    center = np.array([0.0, 0.0, 0.0])
    mp = ModeParameters(
        center=center,
        k_vector=np.array([0.0, 0.0, 1.0]),
        z_R=z_R,
        lambda_0_laser=LAMBDA_0_LASER,
    )
    point = np.array([0.0, 0.0, 1e-3])
    lmp = mp.local_mode_parameters_at_a_point(p=point)
    # z_minus_z_0 should be 1e-3
    assert np.allclose(lmp.z_minus_z_0, [1e-3, 1e-3], rtol=1e-8)


# ─── decompose_ABCD_matrix ───────────────────────────────────────────────────


def test_decompose_ABCD_2x2():
    M = np.array([[1.0, 0.5], [-2.0, 1.0]])
    A, B, C, D = decompose_ABCD_matrix(ABCD=M)
    assert A == pytest.approx(1.0)
    assert B == pytest.approx(0.5)
    assert C == pytest.approx(-2.0)
    assert D == pytest.approx(1.0)


def test_decompose_ABCD_4x4():
    M = np.zeros((4, 4))
    M[0, 0] = 1.0
    M[0, 1] = 0.3
    M[1, 0] = -10.0
    M[1, 1] = 1.0
    M[2, 2] = 1.0
    M[2, 3] = 0.5
    M[3, 2] = -20.0
    M[3, 3] = 1.0
    A, B, C, D = decompose_ABCD_matrix(ABCD=M)
    assert np.allclose(A, [1.0, 1.0])
    assert np.allclose(B, [0.3, 0.5])
    assert np.allclose(C, [-10.0, -20.0])
    assert np.allclose(D, [1.0, 1.0])


# ─── propagate_local_mode_parameter_through_ABCD ─────────────────────────────


def test_propagate_through_free_space_ABCD():
    # Free-space propagation: q_new = q + L
    L = 0.1
    z_R = 1e-3
    q = np.array([0.0 + 1j * z_R, 0.0 + 1j * z_R])
    lmp = LocalModeParameters(q=q, lambda_0_laser=LAMBDA_0_LASER)
    M = ABCD_free_space(length=L)
    lmp_out = propagate_local_mode_parameter_through_ABCD(local_mode_parameters=lmp, ABCD=M)
    assert np.allclose(lmp_out.q, q + L, rtol=1e-10)


def test_propagate_through_thin_lens():
    # Thin lens: q_out = q / (1 - q/f)
    f = 0.05
    z_R = 1e-3
    z = f  # object at focal plane
    q = np.array([z + 1j * z_R, z + 1j * z_R])
    lmp = LocalModeParameters(q=q, lambda_0_laser=LAMBDA_0_LASER)
    M = np.array([[1, 0, 0, 0], [-1 / f, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1 / f, 1]])
    lmp_out = propagate_local_mode_parameter_through_ABCD(local_mode_parameters=lmp, ABCD=M)
    # q_out = (A*q + B) / (C*q + D) = q / (-q/f + 1)
    expected_q = q / (-q / f + 1)
    assert np.allclose(lmp_out.q, expected_q, rtol=1e-10)


# ─── local_mode_parameters_of_round_trip_ABCD ────────────────────────────────


def test_round_trip_ABCD_stable_cavity():
    # Fabry-Perot cavity: for stable cavity q should be purely imaginary at center
    L = 0.05
    R_mirror = 0.1  # radius of curvature
    # Round-trip: flat mirror → free space L → curved mirror (f=R/2) → free space L → flat mirror
    f = R_mirror / 2
    M_lens = np.array([[1, 0, 0, 0], [-1 / f, 1, 0, 0], [0, 0, 1, 0], [0, 0, -1 / f, 1]])
    M_free = ABCD_free_space(length=L)
    M_round_trip = M_free @ M_lens @ M_free
    lmp = local_mode_parameters_of_round_trip_ABCD(
        round_trip_ABCD=M_round_trip, n=1.0, lambda_0_laser=LAMBDA_0_LASER
    )
    # z_R should be positive (stable cavity)
    assert np.all(lmp.z_R > 0)


# ─── complete_orthonormal_basis ──────────────────────────────────────────────


def test_complete_orthonormal_basis_z_axis():
    v = np.array([0.0, 0.0, 1.0])
    basis = complete_orthonormal_basis(v=v)
    # Should return x and y as spanning vectors
    assert basis.shape == (2, 3)
    # Both vectors should be perpendicular to v
    for b in basis:
        assert np.dot(b, v) == pytest.approx(0.0, abs=1e-10)
    # Vectors should be orthonormal
    assert np.dot(basis[0], basis[1]) == pytest.approx(0.0, abs=1e-10)
    assert np.linalg.norm(basis[0]) == pytest.approx(1.0)
    assert np.linalg.norm(basis[1]) == pytest.approx(1.0)


def test_complete_orthonormal_basis_x_axis():
    v = np.array([1.0, 0.0, 0.0])
    basis = complete_orthonormal_basis(v=v)
    assert basis.shape == (2, 3)
    for b in basis:
        assert np.dot(b, v) == pytest.approx(0.0, abs=1e-10)


def test_complete_orthonormal_basis_arbitrary():
    v = normalize_vector(np.array([1.0, 1.0, 1.0]))
    basis = complete_orthonormal_basis(v=v)
    assert basis.shape == (2, 3)
    for b in basis:
        assert np.dot(b, v) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(basis[0], basis[1]) == pytest.approx(0.0, abs=1e-10)


# ─── Ray ─────────────────────────────────────────────────────────────────────


def test_ray_init_normalizes_k_vector():
    origin = np.array([0.0, 0.0, 0.0])
    k = np.array([3.0, 4.0, 0.0])
    ray = Ray(origin=origin, k_vector=k)
    assert np.linalg.norm(ray.k_vector) == pytest.approx(1.0)


def test_ray_parameterization():
    origin = np.array([1.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    ray = Ray(origin=origin, k_vector=k)
    point = ray.parameterization(t=2.0)
    assert np.allclose(point, [1.0, 0.0, 2.0])


def test_ray_parameterization_with_n():
    origin = np.array([0.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    n = 1.5
    ray = Ray(origin=origin, k_vector=k, n=n)
    point = ray.parameterization(t=1.5, optical_path_length=True)
    assert np.allclose(point, [0.0, 0.0, 1.0])


def test_ray_optical_path_length():
    origin = np.array([0.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    ray = Ray(origin=origin, k_vector=k, length=3.0, n=1.5)
    assert ray.optical_path_length == pytest.approx(4.5)


def test_ray_broadcast_single_k_multiple_origins():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    k = np.array([0.0, 0.0, 1.0])
    ray = Ray(origin=origins, k_vector=k)
    assert ray.k_vector.shape == (3, 3)
    assert ray.origin.shape == (3, 3)


# ─── IdealLens.ABCD_matrix ───────────────────────────────────────────────────


def test_ideal_lens_ABCD_matrix():
    f = 0.05
    lens = IdealLens(
        outwards_normal=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        focal_length=f,
    )
    M = lens.ABCD_matrix()
    assert M.shape == (4, 4)
    # The (1,0) and (3,2) elements should be -1/f (thin lens formula)
    assert M[1, 0] == pytest.approx(-1.0 / f)
    assert M[3, 2] == pytest.approx(-1.0 / f)
    # Diagonal should be 1
    assert M[0, 0] == pytest.approx(1.0)
    assert M[1, 1] == pytest.approx(1.0)

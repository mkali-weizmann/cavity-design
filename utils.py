import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Any
import pickle as pkl
import warnings
# import dataclass:
from dataclasses import dataclass

# Every optical element has a np.ndarray representation, and those two dictionaries defines the order and meaning of
# the array columns.
PRETTY_INDICES_NAMES = {'x': 'x [m]',
                        'y': 'y [m]',
                        't': 'Elevation angle [rads]',
                        'p': 'Azimuthal angle [rads]',
                        'r_1': 'Radius of curvature 1 [m]',
                        'r_2': 'Radius of curvature 2 [m]',
                        'n_outside_or_before': 'Index of refraction (before the surface)',
                        'T_c': 'Center thickness [m]',
                        'n_inside_or_after': 'Index of refraction (after the surface)',
                        'z': 'z [m]',
                        'curvature_sign': 'Curvature sign',
                        'material_refractive_index': 'Material refractive index',
                        'alpha_expansion': 'Thermal expansion coefficient [1/K]',
                        'beta_surface_absorption': 'Surface power absorption coefficient',
                        'kappa_conductivity': 'Thermal conductivity coefficient [W/(m*K)]',
                        'dn_dT': 'dn_dT [1/K]',
                        'nu_poisson_ratio': 'Poisson ratio',
                        'alpha_volume_absorption': 'Volume power absorption coefficient [1/m]',
                        'intensity_reflectivity': 'Intensity reflectivity',
                        'intensity_transmittance': 'Transmissivity',
                        'temperature': 'Temperature increase [K]',
                        'surface_type': 'Surface type'}
INDICES_DICT = {name: i for i, name in enumerate(PRETTY_INDICES_NAMES.keys())}

INDICES_DICT_INVERSE = {v: k for k, v in INDICES_DICT.items()}
# set numpy to raise an error on warnings:
SURFACE_TYPES_DICT = {'curved_mirror': 0, 'thick_lens': 1, 'curved_refractive_surface': 2, 'ideal_lens': 3,
                      'flat_mirror': 4, 'ideal_thick_lens': 5}
SURFACE_TYPES_DICT_INVERSE = {v: k for k, v in SURFACE_TYPES_DICT.items()}

@dataclass
class SurfacesTypes:
    thick_lens = 'thick_lens'
    curved_mirror = 'curved_mirror'
    curved_refractive_surface = 'curved_refractive_surface'
    ideal_lens = 'ideal_lens'
    flat_mirror = 'flat_mirror'
    ideal_thick_lens = 'ideal_thick_lens'

    @staticmethod
    def from_integer_representation(integer_representation: int) -> str:
        return SurfacesTypes.__dict__[SURFACE_TYPES_DICT_INVERSE[integer_representation]]

@dataclass
class CurvatureSigns:
    convex = -1
    concave = 1

    @staticmethod
    def from_integer_representation(integer_representation: int) -> int:
        if integer_representation == 1:
            return CurvatureSigns.convex
        elif integer_representation == -1:
            return CurvatureSigns.concave
        else:
            raise ValueError("Curvature sign must be either 1 or -1")


# with open('data/params_dict.pkl', 'rb') as f:
#     params_dict = pkl.load(f)

CENTRAL_LINE_TOLERANCE = 1
STRETCH_FACTOR = 1  # 0.001
C_LIGHT_SPEED = 299792458  # [m/s]
ROOM_TEMPERATURE = 293  # [K]


def pretty_print_array(array: np.ndarray):
    # Prints an array in a way that can be copy-pasted into the code.
    print(f"np.{array=}".replace('array=', "").replace("nan", "np.nan").replace("\n", "").replace("],", "],\n").replace(",        ", ", ").replace(",      ", ", ").replace("       [", "          ["))


def pretty_print_number(number: Optional[float], represents_angle: bool = False):
    if number is None:
        pre_padded = 'None'
    elif np.isnan(number):
        pre_padded = 'np.nan'
    elif number == 0:
        pre_padded = '0'
    else:
        if represents_angle:
            number /= np.pi
        formatted = f"{number:.15e}"
        # Remove trailing zeros and the decimal point if not necessary
        parts = formatted.split('e')
        parts[0] = parts[0].rstrip('0').rstrip('.')
        pre_padded = 'e'.join(parts)
        if represents_angle:
            pre_padded += ' * np.pi'
    final_string = pre_padded.ljust(21, ' ')
    return final_string


def nvl(var, val: Any = np.nan):
    if var is None:
        return val
    return var


def plane_name_to_xy_indices(plane: str) -> Tuple[int, int]:
    if plane in ['xy', 'yx']:
        x_index = 0
        y_index = 1
    elif plane in ['xz', 'zx']:
        x_index = 0
        y_index = 2
    elif plane in ['yz', 'zy']:
        x_index = 1
        y_index = 2
    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
    return x_index, y_index


def params_to_perturbable_params_indices(params_array: np.ndarray, remove_one_of_the_angles: bool = False) -> List[int]:
    # Associates the cavity parameters with the number of parameters needed to describe the cavity.
    # If there is a lens, then the number of parameters is 7 (x, y, t, p, r, n_2):
    if (np.any(params_array[:, INDICES_DICT['surface_type']] == SURFACE_TYPES_DICT['curved_refractive_surface'])
     or np.any(params_array[:, INDICES_DICT['surface_type']] == SURFACE_TYPES_DICT['thick_lens'])):
        params_indices = [INDICES_DICT['x'], INDICES_DICT['y'], INDICES_DICT['t'], INDICES_DICT['p'],
                          INDICES_DICT['r_1'], INDICES_DICT['n_inside_or_after']]
    # If there is no lens but there is a curved mirror: (x, y, t, p, r)
    else:
        params_indices = [INDICES_DICT['x'], INDICES_DICT['y'], INDICES_DICT['t'], INDICES_DICT['p'],
                          INDICES_DICT['r_1']]
    if remove_one_of_the_angles:
        params_indices.remove(INDICES_DICT['t'])
    return params_indices


def maximal_lens_height(R: float, w: float) -> float:
    return R * np.sqrt(1 - ((R - w/2) / R) ** 2)


def safe_exponent(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    a_safe = np.clip(a=np.real(a), a_min=-200, a_max=None) + 1j * np.imag(a)  # ARBITRARY
    return np.exp(a_safe)


def ABCD_free_space(length: float) -> np.ndarray:
    return np.array([[1, length, 0, 0],
                     [0, 1,      0, 0],
                     [0, 0,      1, length],  # / cos_theta_between_planes
                     [0, 0,      0, 1]])


def normalize_vector(vector: Union[np.ndarray, list]) -> np.ndarray:
    if isinstance(vector, list):
        vector = np.array(vector)
    return vector / np.linalg.norm(vector, axis=-1)[..., np.newaxis]


def rotation_matrix_around_n(n, theta):
    # Rotates a vector around the axis n by theta radians.
    # This funny stacked syntax is to allow theta to be of any dimension
    A = np.stack([np.stack([np.cos(theta) + n[0] ** 2 * (1 - np.cos(theta)),
                            n[0] * n[1] * (1 - np.cos(theta)) - n[2] * np.sin(theta),
                            n[0] * n[2] * (1 - np.cos(theta)) + n[1] * np.sin(theta)], axis=-1),
                  np.stack([n[1] * n[0] * (1 - np.cos(theta)) + n[2] * np.sin(theta),
                            np.cos(theta) + n[1] ** 2 * (1 - np.cos(theta)),
                            n[1] * n[2] * (1 - np.cos(theta)) - n[0] * np.sin(theta)], axis=-1),
                  np.stack([n[2] * n[0] * (1 - np.cos(theta)) - n[1] * np.sin(theta),
                            n[2] * n[1] * (1 - np.cos(theta)) + n[0] * np.sin(theta),
                            np.cos(theta) + n[2] ** 2 * (1 - np.cos(theta))], axis=-1)], axis=-1)
    return A


def focal_length_of_lens(R_1, R_2, n, width):
    # for R_1 = R_2 = 0.05, d=0.01, n=1.5, this function returns 1.5 [m]
    one_over_f = (n - 1) * ((1 / R_1) + (1 / R_2) + ((n - 1) * width) / (n * R_1 * R_2))
    return 1 / one_over_f


def sin_without_trailing_epsilon(phi: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    pi_multiplications = np.mod(phi, np.pi)  # This is to avoid numerical trailing epsilon.
    sin_phi = np.sin(phi)
    if isinstance(sin_phi, (float, int)):
        if pi_multiplications == 0:
            sin_phi = 0
    else:
        sin_phi[pi_multiplications == 0] = 0
    return sin_phi


def cos_without_trailing_epsilon(phi: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    pi_half_multiplications = np.mod(phi - np.pi / 2, np.pi)  # This is to avoid numerical trailing epsilon.
    cos_phi = np.cos(phi)
    if isinstance(cos_phi, (float, int)):
        if pi_half_multiplications == 0:
            cos_phi = 0
    else:
        cos_phi[pi_half_multiplications == 0] = 0
    return cos_phi


def unit_vector_of_angles(theta: Union[np.ndarray, float], phi: Union[np.ndarray, float]) -> np.ndarray:
    # Those are the angles of the unit vector in spherical coordinates, with respect to the global system of coordinates
    # theta and phi are assumed to be in radians
    sin_phi = sin_without_trailing_epsilon(phi)
    cos_phi = cos_without_trailing_epsilon(phi)
    sin_theta = sin_without_trailing_epsilon(theta)
    cos_theta = cos_without_trailing_epsilon(theta)

    return np.stack([cos_theta * cos_phi, cos_theta * sin_phi, sin_theta], axis=-1)


def angles_of_unit_vector(unit_vector: Union[np.ndarray, float]) -> Union[Tuple[np.ndarray, np.ndarray],
                                                                          Tuple[float, float]]:
    # theta and phi are returned in radians
    theta = np.arcsin(unit_vector[..., 2])
    phi = np.arctan2(unit_vector[..., 1], unit_vector[..., 0])
    return theta, phi


def angles_distance(direction_vector_1: np.ndarray, direction_vector_2: np.ndarray):
    inner_product = np.sum(direction_vector_1 * direction_vector_2, axis=-1)
    inner_product = np.clip(inner_product, -1, 1)
    return np.arccos(inner_product)


def angles_difference(angle_1: Union[np.ndarray, float], angle_2: Union[np.ndarray, float]) -> np.ndarray:
    diff = angle_2 - angle_1
    result = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return result


def radius_of_f_and_n(f: float, n: float) -> float:
    return 2 * f * (n - 1)


def w_0_of_z_R(z_R: np.ndarray, lambda_0_laser: float, n: float) -> np.ndarray:
    # z_R_reduced is an array because of two transverse dimensions
    return np.sqrt(z_R * lambda_0_laser / (np.pi * n ** 2))


def z_R_of_w_0(w_0: np.ndarray, lambda_laser: float) -> np.ndarray:
    # lambda_laser is the wavelength of the laser in the medium = lambda_0 / n
    return np.pi * w_0 ** 2 / lambda_laser


def spot_size(z: np.ndarray, z_R: np.ndarray, lambda_0_laser: float, n: float) -> np.ndarray:  # AKA w(z)
    # lambda_laser is the wavelength of the laser in the medium = lambda_0 / n
    w_0 = w_0_of_z_R(z_R, lambda_0_laser, n)
    w_z = w_0 * np.sqrt(1 + (z / z_R) ** 2)
    return w_z


def stack_df_for_print(df):
    stacked_df = df.stack().reset_index()
    stacked_df.columns = ['Parameter', 'Element', 'Value']
    stacked_df = stacked_df[['Element', 'Parameter', 'Value']]
    return stacked_df


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def gaussian_integral_2d_log(A: np.ndarray, b: np.ndarray, c):
    # The integral over exp( x.T A_2 x + b.T x + c):
    eigen_values = np.linalg.eigvals(A)
    A_inv = np.linalg.inv(A)
    dim = A.shape[0]
    try:
        log_integral = np.log(np.sqrt((2 * np.pi) ** dim / np.linalg.det(A))) + 0.5 * b.T @ A_inv @ b + c
    except FloatingPointError:
        log_integral = np.nan
    return log_integral


def gaussian_norm_log(A: np.ndarray, b: np.ndarray, c: float):
    return 1 / 2 * gaussian_integral_2d_log(A + np.conjugate(A), b + np.conjugate(b), c + np.conjugate(c))


def gaussians_overlap_integral(A_1: np.ndarray, A_2: np.ndarray,
                               # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
                               b_1: np.ndarray, b_2: np.ndarray,
                               c_1: float, c_2: float) -> float:
    A_1_conjugate = np.conjugate(A_1)
    b_1_conjugate = np.conjugate(b_1)
    c_1_conjugate = np.conjugate(c_1)

    A = A_1_conjugate + A_2
    b = b_1_conjugate + b_2
    c = c_1_conjugate + c_2
    # b = mu_1.T @ A_1_conjugate + mu_2.T @ A_2 + b_1_conjugate + b_2
    # c = (-1/2) * (mu_1.T @ A_1_conjugate @ mu_1 + mu_2.T @ A_2 @ mu_2) + c_1_conjugate + c_2
    normalization_factor_log = gaussian_norm_log(A_1, b_1, c_1) + gaussian_norm_log(A_2, b_2, c_2)
    integral_normalized_log = gaussian_integral_2d_log(A, b, c) - normalization_factor_log
    return safe_exponent(integral_normalized_log)


def interval_parameterization(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def functions_first_crossing(f: Callable, initial_step: float, crossing_value: float = 0.9,
                             accuracy: float = 0.001, max_f_eval: int = 100) -> float:
    # assumes f(0) == 1 and a decreasing function.
    stopping_flag = False
    increasing_ratio = 2
    n = 10
    last_n_evaluations = np.zeros(n)
    last_n_xs = np.zeros(n)
    borders_min = 0
    borders_max = np.nan
    loop_counter = 0
    f_0 = f(0)
    if np.isnan(f_0):
        # warnings.warn('Function has no value at x_input=0, returning nan')
        f(0)
        raise ValueError('Function has no value at x_input=0, returning nan')
        return np.nan
    f_borders_min = f_0
    f_borders_max = np.nan
    last_n_evaluations[0] = f_0
    x = initial_step
    while not stopping_flag:
        loop_counter += 1
        f_x = f(x)
        last_n_xs[np.mod(loop_counter, n)] = x
        last_n_evaluations[np.mod(loop_counter, n)] = f_x
        if loop_counter == max_f_eval and not np.isnan(borders_max):  # if it wasn't found but we know it's value
            # approximately, then interpolate it:
            x = (crossing_value - f_borders_min) / (f_borders_max - f_borders_min) * (
                    borders_max - borders_min) + borders_min
            # warnings.warn(
            #     f"Did not find crossing value, interpolated it between ({borders_min:.8e}, {f_borders_min:.5e}) and ({borders_max:.8e}, {f_borders_max:.5e}) to be ({x:.8e}, {crossing_value:.5e})")
            stopping_flag = True
        elif not np.any(np.invert(
                np.abs(last_n_evaluations - 1) < 1e-18)) or loop_counter > max_f_eval:  # if the function is not
            # decreasing or we reached the max number of function evaluations:
            x = np.nan
            stopping_flag = True
        elif f_x > crossing_value + accuracy:
            borders_min = x
            f_borders_min = f_x
            if np.isnan(borders_max):
                x *= increasing_ratio
            else:
                x = interval_parameterization(x, borders_max, 0.5)
        elif f_x < crossing_value - accuracy:
            increasing_ratio = 1.1
            borders_max = x
            f_borders_max = f_x
            x = interval_parameterization(borders_min, x, 0.5)
        elif not np.any(np.invert(np.isnan(last_n_evaluations))):  # If all the last n evaluations were nan:
            borders_max = np.min(last_n_xs)  # Set borders_max to be the first value from which
            # onwards it seems to always be nan.
            x = interval_parameterization(borders_min, borders_max, 0.5)
        elif np.isnan(f_x):
            if np.isnan(borders_max):
                increasing_ratio = 1.1
                x *= increasing_ratio
                f_borders_max = f_x
            else:
                # randomize new x with higher probability to be closer to borders_min (the pdf is p(x)=2(1-x)), the cdf
                # is F(x)=2(x-x^2) and the inverse cdf is F^-1(y)=1-sqrt(1-y)
                y = np.random.uniform()
                x_normalized = 1 - np.sqrt(1 - y)
                x = interval_parameterization(borders_min, borders_max, x_normalized)
        elif crossing_value - accuracy < f_x < crossing_value + accuracy:
            stopping_flag = True
        else:
            raise ValueError('This should not happen')
    return x


def dT_c_of_a_lens(R, h):
    dT_c = R * (1- np.sqrt(1 - h ** 2 / R ** 2))
    return dT_c


def thick_lens_focal_length(T_c: float, R_1: float, R_2: float, n: float) -> float:
    f_inverse = (n - 1) * (1 / R_1 - 1 / R_2 + (n - 1) * T_c / (n * R_1 * R_2))
    f = 1 / f_inverse
    return f


def working_distance_of_a_lens(R_1, R_2, n, T_c):
    f = thick_lens_focal_length(T_c, R_1, R_2, n)
    h_2 = -f * (n-1) * T_c / (n * R_1)
    working_distance = f - h_2
    return working_distance


def stable_sqrt(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    # like square root, but returns nan if the parts of the input is negative, instead of throwing a FloatingPointError.
    if isinstance(x, np.ndarray):
        x[np.isnan(x)] = -1  # If it is already nan, then it will stay nan.
        s = np.sqrt(x + 0j)
        s[np.imag(s) != 0] = np.nan
        s = np.real(s)
        return s
    else:
        if np.isnan(x) or x < 0:
            return np.nan
        else:
            return np.sqrt(x)

def generate_initial_parameters_grid(center: np.ndarray,
                                     range_limit: float,
                                     N_resolution: int,
                                     p_is_trivial: bool,
                                     t_is_trivial: bool):
    base_grid = np.linspace(-range_limit, range_limit, N_resolution)
    angle_factor = 100
    if p_is_trivial or t_is_trivial:
        if p_is_trivial:
            POS, ANGLE = np.meshgrid(base_grid, base_grid * angle_factor + center[1], indexing='ij')
            TRIVIAL_GRID = np.zeros_like(POS)
            initial_parameters = np.stack([POS + center[0], ANGLE + center[1], TRIVIAL_GRID + center[2], TRIVIAL_GRID + center[3]], axis=-1)
        else:  # (if t is trivial)
            POS, ANGLE = np.meshgrid(base_grid, base_grid * angle_factor, indexing='ij')
            TRIVIAL_GRID = np.zeros_like(POS)
            initial_parameters = np.stack([TRIVIAL_GRID + center[0], TRIVIAL_GRID + center[1], POS + center[2], ANGLE + center[3]], axis=-1)
    else:
        X, T, Y, P = np.meshgrid(base_grid + center[0],
                                 base_grid * angle_factor + center[1],
                                 base_grid + center[2],
                                 base_grid * angle_factor + center[3], indexing='ij')
        initial_parameters = np.stack([X, T, Y, P], axis=-1)
    return initial_parameters


def reverse_elements_order_of_mirror_lens_mirror(params: np.ndarray) -> np.ndarray:
    # swap first and third rows of params:
    params[[0, 2]] = params[[2, 0]]
    params[1, [4, 5]] = params[1, [5, 4]]
    params[1, 3] += 1j
    return params
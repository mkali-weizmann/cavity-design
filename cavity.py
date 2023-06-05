import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable
from scipy import optimize
import warnings
from dataclasses import dataclass
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

INDICES_DICT = {'x': 0, 'y': 1, 't': 2, 'p': 3, 'r': 4, 'w': 5, 'n_1': 6, 'n_2': 7, 'z': 8, 'curvature_sign': 9,
                'surface_type': 10}
INDICES_DICT_INVERSE = {v: k for k, v in INDICES_DICT.items()}
# set numpy to raise an error on warnings:
np.seterr(all='raise')
N = 50
ROOT_ERRORS = np.zeros(N)
I = 0
# Throughout the code, all tensors can take any number of dimensions, but the last dimension is always the coordinate
# dimension. this allows a Ray to be either a single ray, a list of rays, or a list of lists of rays, etc.
# For example, a Ray could be a set of rays with a starting point for every combination of x, y, z. in this case, the
# ray.origin tensor will be of the size N_x | N_y | N_z | 3.

# The ray is always traced starting from the last surface of the cavity, such that the first mirror is the first mirror
# the ray hits. in the initial state of the cavity it means that the ray starts from cavity.physical_surfaces[-1].center and hits
# first the cavity.physical_surfaces[0] mirror. After the plane that is perpendicular to the central line and between the two
# physical_surfaces is calculated, then the ray starts at cavity.surfaces[-1].center (which is that plane) and hits first the
# cavity.surfaces[0] which is the first mirror.

# As a convention, the locations (parameterized usually by t and p) always appear before the angles (parameterized by
# theta and phi). also, t and theta appear before p and phi.
# If for example there is a parameter q both for t axis and p axis, then the first element of q will be the q of t,
# and the second element of q will be the q of p.

class LocalModeParameters:
    def __init__(self, z_minus_z_0: Optional[np.ndarray] = None,
                 z_R: Optional[np.ndarray] = None,
                 q: Optional[np.ndarray] = None):
        if q is not None:
            self.q = q
        elif z_minus_z_0 is not None and z_R is not None:
            self.q = z_minus_z_0 + 1j * z_R
        else:
            raise ValueError('Either q or z_minus_z_0 and z_R must be provided')

    @property
    def z_minus_z_0(self):
        return self.q.real

    @property
    def z_R(self):
        return self.q.imag


@dataclass
class ModeParameters:
    center: np.ndarray  # First dimension is t or p, second dimension is x, y, z
    k_vector: np.ndarray
    z_R: np.ndarray
    principle_axes: np.ndarray  # First dimension is t or p, second dimension is x, y, z
    lambda_laser: Optional[float]

    @property
    def ray(self):
        return Ray(self.center, self.k_vector)

    @property
    def w_0(self):
        if self.lambda_laser is None:
            return None
        else:
            return np.sqrt(self.lambda_laser * self.z_R / np.pi)

    @property
    def NA(self):
        if self.lambda_laser is None:
            return None
        else:
            if self.z_R[0] == 0 or self.z_R[1] == 0:
                return np.array([np.nan, np.nan])
            else:
                return np.sqrt(self.lambda_laser / (np.pi * self.z_R))


def safe_exponent(a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    a_safe = np.clip(a=np.real(a), a_min=-200, a_max=None) + 1j * np.imag(a)  # ARBITRARY
    return np.exp(a_safe)


def ABCD_free_space(length: float) -> np.ndarray:
    return np.array([[1, length, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, length],  # / cos_theta_between_planes
                     [0, 0, 0, 1]])


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


def unit_vector_of_angles(theta: Union[np.ndarray, float], phi: Union[np.ndarray, float]) -> np.ndarray:
    # theta and phi are assumed to be in radians
    return np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)], axis=-1)


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


def decompose_ABCD_matrix(ABCD: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ABCD.shape == (4, 4):
        A, B, C, D = ABCD[(0, 2), (0, 2)], ABCD[(0, 2), (1, 3)], \
                     ABCD[(1, 3), (0, 2)], ABCD[(1, 3), (1, 3)]
    else:
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], \
                     ABCD[1, 0], ABCD[1, 1]
    return A, B, C, D


def propagate_local_mode_parameter_through_ABCD(local_mode_parameters: LocalModeParameters,
                                                ABCD: np.ndarray) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(ABCD)
    q_new = (A * local_mode_parameters.q + B) / (C * local_mode_parameters.q + D)
    return LocalModeParameters(q=q_new)


def local_mode_parameters_of_round_trip_ABCD(round_trip_ABCD: np.ndarray) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(round_trip_ABCD)
    q_z = (A - D + np.sqrt(A ** 2 + 2 * C * B + D ** 2 - 2 + 0j)) / (2 * C)
    q_z = np.real(q_z) + 1j * np.abs(np.imag(q_z))  # ATTENTION - make sure this line is justified.

    return LocalModeParameters(q=q_z)  # First dimension is theta or phi,second dimension is z_minus_z_0 or
    # z_R.


def radius_of_f_and_n(f: float, n: float) -> float:
    return 2 * f * (n - 1)


def w_0_of_z_R(z_R: np.ndarray, lambda_laser: float) -> np.ndarray:
    return np.sqrt(z_R * lambda_laser / np.pi)


def w_of_z_R(z: np.ndarray, z_R: np.ndarray, lambda_laser) -> np.ndarray:
    w_0 = w_0_of_z_R(z_R, lambda_laser)
    w_z = w_0 * np.sqrt(1 + (z / z_R) ** 2)
    return w_z


def spot_size(z: np.ndarray, z_R: np.ndarray, lambda_laser) -> np.ndarray:
    w_0 = w_0_of_z_R(z_R, lambda_laser)
    w_z = w_0 * np.sqrt(1 + (z / z_R) ** 2)
    return w_z


class Ray:
    def __init__(self, origin: np.ndarray, k_vector: np.ndarray,
                 length: Optional[Union[np.ndarray, float]] = None):
        if k_vector.ndim == 1 and origin.shape[0] > 1:
            k_vector = np.tile(k_vector, (*origin.shape[:-1], 1))
        elif origin.ndim == 1 and k_vector.shape[0] > 1:
            origin = np.tile(origin, (*k_vector.shape[:-1], 1))

        self.origin = origin  # m_rays | 3
        self.k_vector = normalize_vector(k_vector)  # m_rays | 3
        if length is not None and isinstance(length, float):
            length = np.ones(origin.shape[0]) * length
        self.length = length  # m_rays or None

    def parameterization(self, t: Union[np.ndarray, float]) -> np.ndarray:
        # Currently this function allows only one t per ray. if needed it can be extended to allow multiple t per ray.
        # t needs to be either a float or a numpy array with dimensions m_rays
        return self.origin + t[..., np.newaxis] * self.k_vector

    def plot(self, ax: Optional[plt.Axes] = None, dim=3, color='r', linewidth: float = 1):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        if self.length is None:
            length = np.ones_like(self.origin[..., 0])
        else:
            length = self.length
        ray_origin_reshaped = self.origin.reshape(-1, 3)
        ray_k_vector_reshaped = self.k_vector.reshape(-1, 3)
        lengths_reshaped = length.reshape(-1)
        if dim == 3:
            [ax.plot(
                [ray_origin_reshaped[i, 0],
                 ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0]],
                [ray_origin_reshaped[i, 1],
                 ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1]],
                [ray_origin_reshaped[i, 2],
                 ray_origin_reshaped[i, 2] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 2]],
                color=color, linewidth=linewidth)
                for i in range(ray_origin_reshaped.shape[0])]
        else:
            [ax.plot(
                [ray_origin_reshaped[i, 0],
                 ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0]],
                [ray_origin_reshaped[i, 1],
                 ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1]],
                color=color, linewidth=linewidth)
                for i in range(ray_origin_reshaped.shape[0])]

        return ax


class Surface:
    def __init__(self, outwards_normal: np.ndarray, name: Optional['str'] = None, **kwargs):
        self.outwards_normal = normalize_vector(outwards_normal)
        self.name = name

    @property
    def center(self):
        raise NotImplementedError

    @property
    def inwards_normal(self):
        return -self.outwards_normal

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        # Take parameters and return points on the surface
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        # takes a point on the surface and returns the parameters
        raise NotImplementedError

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None, dim: int = 2, length=0.6):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        if dim == 3:
            t = np.linspace(-length / 2, length / 2, 100)
        else:
            t = 0
        s = np.linspace(-length / 2, length / 2, 100)
        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        if isinstance(self, CurvedRefractiveSurface):
            color = 'grey'
        elif isinstance(self, PhysicalSurface):
            color = 'b'
        else:
            color = 'black'

        if dim == 3:
            ax.plot_surface(x, y, z, color=color, alpha=0.25)
        else:
            ax.plot(x, y, color=color)
        if name is not None:
            name_position = self.parameterization(0.4, 0)
            if dim == 3:
                ax.text(name_position[0], name_position[1], name_position[2], s=name)
            else:
                if ax.get_xlim()[0] < name_position[0] < ax.get_xlim()[1] and ax.get_ylim()[0] < name_position[1] < \
                        ax.get_ylim()[1]:
                    ax.text(name_position[0], name_position[1], s=name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if dim == 3:
            ax.set_zlabel('z')

        # center_plus_normal = self.center + self.inwards_normal * length
        # if dim == 3:
        #     ax.plot([self.center[0], center_plus_normal[0]],
        #             [self.center[1], center_plus_normal[1]],
        #             [self.center[2], center_plus_normal[2]], 'g-')
        # else:
        #     ax.plot([self.center[0], center_plus_normal[0]],
        #             [self.center[1], center_plus_normal[1]], 'g-')
        return ax

    def generate_ray_from_parameters(self, t: float, p: float, theta: float, phi: float) -> Ray:
        k_vector = unit_vector_of_angles(theta, phi)
        origin = self.parameterization(t, p)
        return Ray(origin=origin, k_vector=k_vector)

    def spanning_vectors(self):
        pseudo_y = normalize_vector(np.cross(np.array([0, 0, 1]), self.inwards_normal))
        pseudo_z = normalize_vector(np.cross(self.inwards_normal, pseudo_y))
        return pseudo_z, pseudo_y


class PhysicalSurface(Surface):
    def __init__(self, outwards_normal: np.ndarray, name: Optional['str'] = None, **kwargs):
        super().__init__(outwards_normal=outwards_normal, name=name, **kwargs)

    @property
    def center(self):
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        raise NotImplementedError

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def reflect_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError

    def ABCD_matrix(self, cos_theta_incoming: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def to_params(self):
        raise NotImplementedError


class FlatSurface(Surface):
    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center: Optional[np.ndarray] = None,
                 name: Optional['str'] = None,
                 **kwargs):
        super().__init__(outwards_normal=outwards_normal, name=name, **kwargs)
        if distance_from_origin is None and center is None:
            raise ValueError('Either distance_from_origin or center must be specified')
        if distance_from_origin is not None and center is not None:
            raise ValueError('Only one of distance_from_origin or center must be specified')
        if distance_from_origin is not None:
            self.distance_from_origin = distance_from_origin
            self.center_of_mirror_private = self.outwards_normal * distance_from_origin
        if center is not None:
            self.center_of_mirror_private = center
            self.distance_from_origin = center @ self.outwards_normal

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        A_vec = self.outwards_normal * self.distance_from_origin
        BA_vec = A_vec - ray.origin
        BC = BA_vec @ self.outwards_normal
        cos_theta = ray.k_vector @ self.outwards_normal
        t = BC / cos_theta
        intersection_point = ray.parameterization(t)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return intersection_point

    @property
    def center(self):
        # The reason for this property is that in other PhysicalSurface classes it is a property.
        return self.center_of_mirror_private

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]):
        pseudo_z, pseudo_y = self.spanning_vectors()
        if isinstance(t, (float, int)):
            t = np.array(t)
        if isinstance(p, (float, int)):
            p = np.array(p)
        points = self.center + t[..., np.newaxis] * pseudo_z + p[..., np.newaxis] * pseudo_y
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_z, pseudo_y = self.spanning_vectors()
        t = (points - self.center) @ pseudo_z
        p = (points - self.center) @ pseudo_y
        return t, p


class FlatMirror(FlatSurface, PhysicalSurface):

    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center: Optional[np.ndarray] = None,
                 name: Optional['str'] = None):
        super().__init__(outwards_normal=outwards_normal, distance_from_origin=distance_from_origin, center=center,
                         name=name)

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None, dim: int = 3, length=0.6):
        return super().plot(ax, name, dim, length)

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        return super().find_intersection_with_ray(ray)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    def to_params(self):
        raise NotImplementedError

    def reflect_direction(self, ray: Ray) -> np.ndarray:
        dot_product = ray.k_vector @ self.outwards_normal  # m_rays
        k_projection_on_normal = dot_product[..., np.newaxis] * self.outwards_normal
        reflected_direction = ray.k_vector - 2 * k_projection_on_normal
        return reflected_direction

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # Assumes the ray is in the x-y plane, and the mirror is in the z-x plane
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, -1]])

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray)
        return Ray(intersection_point, reflected_direction_vector)


class IdealLens(FlatSurface, PhysicalSurface):

    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center: Optional[np.ndarray] = None,
                 focal_length: Optional[float] = None,
                 name: Optional['str'] = None, ):
        super().__init__(outwards_normal=outwards_normal, distance_from_origin=distance_from_origin, center=center,
                         name=name)
        self.focal_length = focal_length

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None, dim: int = 3, length=0.6):
        return super().plot(ax, name, dim, length)

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        return super().find_intersection_with_ray(ray)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    def to_params(self):
        raise NotImplementedError

    def reflect_direction(self, ray: Ray) -> np.ndarray:
        intersection_point = self.find_intersection_with_ray(ray)
        pseudo_z, pseudo_y = self.spanning_vectors()
        t, p = self.get_parameterization(intersection_point)
        t_projection, p_projection = ray.k_vector @ pseudo_z, ray.k_vector @ pseudo_y
        theta, phi = np.pi / 2 - np.arccos(t_projection), np.pi / 2 - np.arccos(p_projection)
        input_vector = np.array([t, theta, p, phi])
        if len(input_vector.shape) > 1:
            input_vector = np.swapaxes(input_vector, 0, 1)
        output_vector = self.ABCD_matrix(cos_theta_incoming=0) @ input_vector
        if len(input_vector.shape) > 1:
            output_vector = np.swapaxes(output_vector, 0, 1)
        t_projection_out, p_projection_out = np.cos(np.pi / 2 - output_vector[1, ...]), np.cos(
            np.pi / 2 - output_vector[3, ...])
        # ABCD_MATRIX METHOD
        # Here I assume all rays come from the same direction to the lens
        if ray.k_vector.reshape(-1)[0:3] @ self.outwards_normal > 0:
            forwards_normal = self.outwards_normal
        else:
            forwards_normal = - self.outwards_normal
        component_t = np.multiply.outer(t_projection_out, pseudo_z)
        component_p = np.multiply.outer(p_projection_out, pseudo_y)
        component_n = np.multiply.outer((1 - t_projection_out ** 2 - p_projection_out ** 2) ** 0.5, forwards_normal)
        output_direction_vector = component_t + component_p + component_n

        return output_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # THIS CURRENTLY DOES NOT HOLD FOR THE CASE WHERE THE RAY IS NOT PERPENDICULAR TO THE LENS!
        return np.array([[1, 0, 0, 0],
                         [-1 / self.focal_length, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, -1 / self.focal_length, 1]])

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray)
        return Ray(intersection_point, reflected_direction_vector)


class CurvedSurface(Surface):
    def __init__(self, radius: float,
                 outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
                 center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
                 # the plate.
                 origin: Optional[np.ndarray] = None,  # The center of the sphere.
                 curvature_sign: int = 1,
                 # 1 for concave (where the ray is hitting the sphere from the inside) and -1 for convex
                 # (where the ray is hitting the sphere from the outside). this is used to find the correct intersection
                 # point of a ray with the surface
                 name: Optional['str'] = None, ):
        super().__init__(outwards_normal=outwards_normal, name=name)
        self.radius = radius
        self.curvature_sign = curvature_sign

        if origin is None and center is None:
            raise ValueError('Either origin or center must be provided.')
        elif origin is not None and center is not None:
            raise ValueError('Only one of origin and center must be provided.')
        elif origin is None:
            self.origin = center + radius * self.inwards_normal
        else:
            self.origin = origin

    @staticmethod
    def from_params(params: np.ndarray, name: Optional[str] = None):
        x, y, t, p, r, w, n_1, n_2, z, curvature_sign, surface_type = params
        center = np.array([x, y, z])
        outwards_normal = unit_vector_of_angles(t, p)
        if surface_type == 0:
            surface = CurvedMirror(radius=r,
                                   outwards_normal=outwards_normal,
                                   center=center,
                                   curvature_sign=curvature_sign,
                                   name=name)
        elif surface_type == 1:
            surface = CurvedRefractiveSurface(radius=r,
                                              outwards_normal=outwards_normal,
                                              center=center,
                                              curvature_sign=curvature_sign,
                                              n_1=n_1,
                                              n_2=n_2,
                                              name=name)
        elif surface_type == 2:
            if name is None:
                names = None
            else:
                names = [name + '_1', name + '_2']
            surface_1, surface_2 = generate_lens_from_params(params, names=names)
            return surface_1, surface_2
        else:
            raise ValueError(f'Unknown surface type {surface_type}')
        return surface

    def to_params(self):
        x, y, z = self.center
        r = self.radius
        t, p = angles_of_unit_vector(self.outwards_normal)
        if isinstance(self, CurvedMirror):
            surface_type = 0
            n_1 = 0
            n_2 = 0
        elif isinstance(self, CurvedRefractiveSurface):
            surface_type = 1
            n_1 = self.n_1
            n_2 = self.n_2
        else:
            raise ValueError(f'Unknown surface type {type(self)}')
        return x, y, t, p, r, n_1, n_2, z, self.curvature_sign, surface_type

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        # Result of the next line of mathematica to find the intersection:
        # Solve[(x0 + kx * t - xc) ^ 2 + (y0 + ky * t - yc) ^ 2 + (z0 + kz * t - zc) ^ 2 == R ^ 2, t]
        l = (-ray.k_vector[..., 0] * ray.origin[..., 0] + ray.k_vector[..., 0] * self.origin[0] - ray.k_vector[
            ..., 1] *
             ray.origin[..., 1] + ray.k_vector[..., 1] * self.origin[1] - ray.k_vector[..., 2] * ray.origin[..., 2] +
             ray.k_vector[..., 2] * self.origin[2] + self.curvature_sign * np.sqrt(
                    -4 * (ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2) * (
                            -self.radius ** 2 + (ray.origin[..., 0] - self.origin[0]) ** 2 + (
                            ray.origin[..., 1] - self.origin[1]) ** 2 + (
                                    ray.origin[..., 2] - self.origin[2]) ** 2) + 4 * (
                            ray.k_vector[..., 0] * (ray.origin[..., 0] - self.origin[0]) + ray.k_vector[..., 1] * (
                            ray.origin[..., 1] - self.origin[1]) + ray.k_vector[..., 2] * (
                                    ray.origin[..., 2] - self.origin[2])) ** 2) / 2) / (
                    ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2)
        ray.length = l
        return ray.parameterization(l)

    def parameterization(self, t: Union[np.ndarray, float],  # the length of arc to travel on the sphere from the center
                         # of the mirror to the point of interest, in the direction "pseudo_z". pseudo_z is
                         # described in the get_spanning_vectors method. it is analogous to theta / R in the
                         # classical parameterization.
                         p: Union[np.ndarray, float]  # The same as t but in the direction of pseudo_y. It is analogous
                         # to phi / R in the classical parameterization.
                         ) -> np.ndarray:
        # This parameterization treats the sphere as if as the center of the mirror was on the x-axis.
        # The conceptual difference between this parameterization and the classical one of [sin(theta)cos(phi),
        # sin(theta)sin(phi), cos(theta)]] is that here there is barely any Jacobian determinant.
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        # Notice how the order of rotations matters. First we rotate around the z axis, then around the y-axis.
        # Doing it the other way around would give parameterization that is not aligned with the conventional theta, phi
        # parameterization. This is important for the get_parameterization method.
        rotation_matrix = rotation_matrix_around_n(pseudo_y, -t / self.radius) @ \
                          rotation_matrix_around_n(pseudo_z, p / self.radius)  # The minus sign is because of the
        # orientation of the pseudo_y axis.

        points = self.origin + self.radius * rotation_matrix @ self.outwards_normal
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        normalized_points = (points - self.origin) / self.radius
        p = np.arctan2(normalized_points @ pseudo_y, normalized_points @ self.outwards_normal) * self.radius
        # Notice that t is like theta but instead of ranging in [0, pi] it ranges in [-pi/2, pi/2].
        t = np.arcsin(np.clip(normalized_points @ pseudo_z, -1, 1)) * self.radius
        return t, p

    @property
    def center(self):
        return self.origin - self.radius * self.inwards_normal

    def get_spanning_vectors(self):
        # For the case of the sphere with normal on the x-axis, those will be the y and z axis.
        # For the case of the sphere with normal on the y-axis, those will be the x and z axis.
        pseudo_y = np.cross(np.array([0, 0, 1]), self.inwards_normal)
        pseudo_z = np.cross(self.inwards_normal, pseudo_y)  # Should be approximately equal to \hat{z}, and exactly
        # equal if the outwards_normal is in the x-y plane.
        return pseudo_y, pseudo_z

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray, intersection_point)
        return Ray(intersection_point, reflected_direction_vector)

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None, dim: int = 3, length=0.6):
        super().plot(ax, name, dim, length=0.6 * self.radius)


class CurvedMirror(CurvedSurface, PhysicalSurface):
    def __init__(self, radius: float,
                 outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
                 center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
                 # the plate.
                 origin: Optional[np.ndarray] = None,  # The center of the sphere.
                 curvature_sign: int = 1,
                 name: Optional['str'] = None, ):
        super().__init__(radius=radius, outwards_normal=outwards_normal, center=center, origin=origin,
                         curvature_sign=curvature_sign, name=name)

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        # Notice that this function does not reflect along the normal of the mirror but along the normal projection
        # of the ray on the mirror.
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        mirror_normal_vector = (self.origin - intersection_point) * self.curvature_sign  # m_rays | 3
        mirror_normal_vector = normalize_vector(mirror_normal_vector)
        dot_product = np.sum(ray.k_vector * mirror_normal_vector, axis=-1)  # m_rays  # This dot product is written
        # like so because both tensors have the same shape and the dot product is calculated along the last axis.
        # you could also perform this product by transposing the second tensor and then dot multiplying the two tensors,
        # but this it would be cumbersome to do so.
        reflected_direction_vector = ray.k_vector - 2 * dot_product[
            ..., np.newaxis] * mirror_normal_vector  # m_rays | 3
        return reflected_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None):
        # order of rows/columns elements is [t, theta, p, phi]
        # An approximation is done here (beyond the small angles' approximation) by assuming that the central line
        # lives in the x,y plane, such that the plane of incidence is the x,y plane (parameterized by p and phi)
        # and the sagittal plane is its transverse (parameterized by t and theta).
        # This is justified for small perturbations of a cavity whose central line actually lives in the x,y plane.
        # It is not really justified for bigger perturbations and should be corrected.
        # It should be corrected by first finding the real axes, # And then apply a rotation matrix to this matrix on
        # both sides.
        ABCD = np.array([[1, 0, 0, 0],
                         [-2 * cos_theta_incoming / self.radius, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 2 / (self.radius * cos_theta_incoming), -1]])
        return ABCD

    def plot_2d(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        if ax is None:
            fig, ax = plt.subplots()
        d_theta = 0.3
        p = np.linspace(-d_theta, d_theta, 50)
        p_grey = np.linspace(d_theta, -d_theta + 2 * np.pi, 100)
        points = self.parameterization(0, p)
        grey_points = self.parameterization(0, p_grey)
        ax.plot(points[:, 0], points[:, 1], 'b-')
        ax.plot(grey_points[:, 0], grey_points[:, 1], color=(0.81, 0.81, 0.81), linestyle='-.', linewidth=0.5,
                label=None)
        ax.plot(self.origin[0], self.origin[1], 'bo')


class CurvedRefractiveSurface(CurvedSurface, PhysicalSurface):
    def __init__(self,
                 radius: float,
                 outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
                 center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
                 # the plate.
                 origin: Optional[np.ndarray] = None,  # The center of the sphere.
                 n_1: float = 1,
                 n_2: float = 1.5,
                 curvature_sign: int = 1,
                 name: Optional['str'] = None, ):
        super().__init__(radius=radius, outwards_normal=outwards_normal, center=center, origin=origin,
                         curvature_sign=curvature_sign, name=name)
        self.n_1 = n_1
        self.n_2 = n_2

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        n_backwards = (self.origin - intersection_point) * self.curvature_sign  # m_rays | 3
        n_backwards = normalize_vector(n_backwards)
        n_forwards = -n_backwards
        cos_theta_incoming = np.sum(ray.k_vector * n_forwards, axis=-1)  # m_rays
        n_orthogonal = ray.k_vector - cos_theta_incoming[..., np.newaxis] * n_forwards  # m_rays | 3
        if np.linalg.norm(n_orthogonal) < 1e-14:
            reflected_direction_vector = n_forwards
        else:
            n_orthogonal = normalize_vector(n_orthogonal)
            sin_theta_outgoing = np.sqrt((self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming ** 2))  # m_rays
            reflected_direction_vector = n_forwards * np.sqrt(
                1 - sin_theta_outgoing ** 2) + n_orthogonal * sin_theta_outgoing
        return reflected_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        cos_theta_outgoing = np.sqrt(1 - (self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming ** 2))
        R_signed = self.radius * self.curvature_sign
        delta_n_e_out_of_plane = self.n_2 * cos_theta_outgoing - self.n_1 * cos_theta_incoming
        delta_n_e_in_plane = delta_n_e_out_of_plane / (cos_theta_incoming * cos_theta_outgoing)

        # See the comment in the ABCD_matrix method of the CurvedSurface class for an explanation of the approximation.
        ABCD = np.array([[1, 0, 0, 0],  # t
                         [delta_n_e_out_of_plane / R_signed, 1, 0, 0],  # theta
                         [0, 0, cos_theta_outgoing / cos_theta_incoming, 0],  # p
                         [0, 0, delta_n_e_in_plane / R_signed, cos_theta_incoming / cos_theta_outgoing]])  # phi
        return ABCD


def generate_lens_from_params(params: np.ndarray, names: Optional[List[str]] = None):
    x, y, t, p, r, w, n_in, n_out, z, curvature_sign, surface_type = params
    if names is None:
        names = [None, None]
    center = np.array([x, y, z])
    direction = unit_vector_of_angles(t, p)
    center_1 = center - (1 / 2) * w * direction
    center_2 = center + (1 / 2) * w * direction
    surface_1 = CurvedRefractiveSurface(radius=r,
                                        outwards_normal=-direction,
                                        center=center_1,
                                        n_1=n_out,
                                        n_2=n_in,
                                        curvature_sign=-1,
                                        name=names[0])
    surface_2 = CurvedRefractiveSurface(radius=r,
                                        outwards_normal=direction,
                                        center=center_2,
                                        n_1=n_in,
                                        n_2=n_out,
                                        curvature_sign=1,
                                        name=names[1])
    return surface_1, surface_2


class Arm:
    def __init__(self,
                 surface_1: Surface,
                 surface_2: Surface,
                 central_line: Optional[Ray] = None,
                 mode_parameters_on_surface_1: Optional[LocalModeParameters] = None,
                 mode_parameters_on_surface_2: Optional[LocalModeParameters] = None,
                 mode_principle_axes: Optional[np.ndarray] = None,
                 lambda_laser: Optional[float] = None):
        self.surface_1 = surface_1
        self.surface_2 = surface_2
        self.mode_parameters_on_surface_1 = mode_parameters_on_surface_1
        self.mode_parameters_on_surface_2 = mode_parameters_on_surface_2
        self.central_line = central_line
        self.mode_principle_axes = mode_principle_axes
        self.lambda_laser = lambda_laser

    def propagate(self, ray: Ray):
        if isinstance(self.surface_2, PhysicalSurface):
            ray = self.surface_2.reflect_ray(ray)
        else:
            new_position = self.surface_2.find_intersection_with_ray(ray)
            ray = Ray(new_position, ray.k_vector)
        return ray

    @property
    def ABCD_matrix_free_space(self):
        if self.central_line is None:
            raise ValueError('Central line not set')
        matrix = ABCD_free_space(self.central_line.length)
        return matrix

    @property
    def ABCD_matrix_reflection(self):
        if self.central_line is None:
            raise ValueError('Central line not set')
        cos_theta = np.abs(self.central_line.k_vector @ self.surface_2.outwards_normal)  # ABS because we want the
        # angle between the ray and the normal to be positive
        if isinstance(self.surface_2, PhysicalSurface):
            matrix = self.surface_2.ABCD_matrix(cos_theta)
        else:
            matrix = np.eye(4)
        return matrix

    @property
    def ABCD_matrix(self):
        matrix = self.ABCD_matrix_reflection @ self.ABCD_matrix_free_space
        return matrix

    def propagate_local_mode_parameters(self):
        if self.mode_parameters_on_surface_1 is None:
            raise ValueError('Mode parameters on surface 1 not set')
        self.mode_parameters_on_surface_2 = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_1,
            self.ABCD_matrix_free_space)
        next_mode_parameters = propagate_local_mode_parameter_through_ABCD(self.mode_parameters_on_surface_2,
                                                                           self.ABCD_matrix_reflection)
        return next_mode_parameters

    # @property
    # def mode_principle_axes(self):
    #     if self.central_line is None:
    #         raise ValueError('Central line not set')
    #     z_hat = np.array([0, 0, 1])
    #     pseudo_x = np.cross(z_hat, self.central_line.k_vector)
    #     mode_principle_axes = np.stack([z_hat, pseudo_x], axis=-1).T  # [z_x, z_y, z_z], [x_x, x_y, x_z]
    #     return mode_principle_axes

    @property
    def mode_parameters(self):
        if self.mode_parameters_on_surface_1 is None:
            return None
        center = self.surface_1.center - \
                 self.mode_parameters_on_surface_1.z_minus_z_0[..., np.newaxis] * \
                 self.central_line.k_vector
        mode_parameters = ModeParameters(center=center,
                                         k_vector=self.central_line.k_vector,
                                         z_R=self.mode_parameters_on_surface_1.z_R,
                                         principle_axes=self.mode_principle_axes,
                                         lambda_laser=self.lambda_laser)
        return mode_parameters

    def local_mode_parameters_on_a_point(self, point: np.ndarray):
        if self.central_line is None:
            raise ValueError('Central line not set')
        if self.mode_parameters_on_surface_1 is None:
            return None

        point_plane_distance_from_surface_1 = (point - self.central_line.origin) @ self.central_line.k_vector
        propagation_ABCD = ABCD_free_space(point_plane_distance_from_surface_1)
        local_mode_parameters = propagate_local_mode_parameter_through_ABCD(self.mode_parameters_on_surface_1,
                                                                            propagation_ABCD)
        return local_mode_parameters

    @property
    def names(self):
        return [self.surface_1.name, self.surface_2.name]


class Cavity:
    def __init__(self,
                 physical_surfaces: List[PhysicalSurface],
                 standing_wave: bool = False,
                 lambda_laser: Optional[float] = None,
                 params: Optional[np.ndarray] = None,
                 names: Optional[List[str]] = None,
                 set_central_line: bool = True,
                 set_mode_parameters: bool = True,
                 set_initial_surface: bool = False):
        self.standing_wave = standing_wave
        self.physical_surfaces = physical_surfaces
        self.arms: List[Arm] = [
            Arm(self.physical_surfaces_ordered[i],
                self.physical_surfaces_ordered[np.mod(i + 1, len(self.physical_surfaces_ordered))],
                lambda_laser=lambda_laser) for i in range(len(self.physical_surfaces_ordered))]
        self.central_line_successfully_traced: Optional[bool] = None
        self.lambda_laser: Optional[float] = lambda_laser
        self.params = params
        self.names_memory = names

        if set_central_line:
            self.find_central_line()
        if set_mode_parameters:
            self.set_mode_parameters()
        if set_initial_surface:
            self.set_initial_surface()

    @staticmethod
    def from_params(params: np.ndarray,
                    standing_wave: bool = True,
                    lambda_laser: Optional[float] = None,
                    names: Optional[List[str]] = None,
                    set_central_line: bool = True,
                    set_mode_parameters: bool = True,
                    set_initial_surface: bool = False):
        mirrors = []
        if names is None:
            names = [None for i in range(len(params))]
        for i in range(len(params)):
            surface_temp = CurvedSurface.from_params(params[i, :], name=names[i])
            if isinstance(surface_temp, tuple):
                mirrors.extend(surface_temp)
            else:
                mirrors.append(surface_temp)
        cavity = Cavity(mirrors, standing_wave, lambda_laser, params=params, names=names,
                        set_central_line=set_central_line, set_mode_parameters=set_mode_parameters,
                        set_initial_surface=set_initial_surface)
        return cavity

    @property
    def to_params(self):
        if self.params is None:
            return np.array([surface.to_params() for surface in self.physical_surfaces])
        else:
            return self.params

    @property
    def physical_surfaces_ordered(self):
        if self.standing_wave:
            backwards_list = copy.deepcopy(self.physical_surfaces[-2:0:-1])
            for surface in backwards_list:
                if isinstance(surface, CurvedRefractiveSurface):
                    surface.curvature_sign = -surface.curvature_sign
                    n_1, n_2 = surface.n_1, surface.n_2
                    surface.n_1 = n_2
                    surface.n_2 = n_1
            return self.physical_surfaces + backwards_list
        else:
            return self.physical_surfaces

    @property
    def central_line(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return [arm.central_line for arm in self.arms]

    @property
    def ABCD_matrices(self):
        if self.arms[0].central_line is None:
            return None
        else:
            ABCD_list = [arm.ABCD_matrix for arm in self.arms]
            return ABCD_list

    @property
    def ABCD_round_trip(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return np.linalg.multi_dot(self.ABCD_matrices[::-1])

    @property
    def mode_parameters(self):
        if self.arms[0].central_line is None:
            return None
        else:
            return [arm.mode_parameters for arm in self.arms]

    @property
    def surfaces(self):
        return [arm.surface_1 for arm in self.arms]

    @property
    def default_initial_k_vector(self) -> np.ndarray:
        if self.central_line is not None and self.central_line_successfully_traced:
            initial_k_vector = self.central_line[0].k_vector
        else:
            initial_k_vector = self.arms[0].surface_2.center - self.arms[0].surface_1.center
            initial_k_vector = normalize_vector(initial_k_vector)
        return initial_k_vector

    @property
    def default_initial_angles(self) -> Tuple[float, float]:
        initial_k_vector = self.default_initial_k_vector
        theta, phi = angles_of_unit_vector(initial_k_vector)
        return theta, phi

    @property
    def default_initial_ray(self) -> Ray:
        if self.central_line_successfully_traced:
            return self.central_line[0]
        else:
            initial_k_vector = self.default_initial_k_vector
            initial_ray = Ray(origin=self.arms[0].surface_1.center, k_vector=initial_k_vector)
            return initial_ray

    @property
    def names(self):
        if self.names_memory is None:
            return [surface.name for surface in self.physical_surfaces]
        else:
            return self.names_memory

    def trace_ray(self, ray: Ray) -> List[Ray]:
        ray_history = [ray]
        for arm in self.arms:
            ray = arm.propagate(ray)
            ray_history.append(ray)
        return ray_history

    def trace_ray_parametric(self,
                             starting_position_and_angles: np.ndarray) -> Tuple[np.ndarray, List[Ray]]:
        # Like trace ray, but works as a function of the starting position and angles as parameters on the starting
        # surface, instead of the starting position and angles as a vector in 3D space.

        initial_ray = self.ray_of_initial_parameters(starting_position_and_angles)
        ray_history = self.trace_ray(initial_ray)
        final_intersection_point = ray_history[-1].origin
        t_o, p_o = self.arms[0].surface_1.get_parameterization(final_intersection_point)  # Here it is the initial
        # surface on purpose.
        theta_o, phi_o = angles_of_unit_vector(ray_history[-1].k_vector)
        final_position_and_angles = np.array([t_o, theta_o, p_o, phi_o])
        return final_position_and_angles, ray_history

    def f_roots(self,
                starting_position_and_angles: np.ndarray) -> np.ndarray:
        # The roots of this function are the initial parameters for the central line.
        try:
            final_position_and_angles, _ = self.trace_ray_parametric(starting_position_and_angles)
            diff = np.zeros_like(starting_position_and_angles)
            diff[[0, 2]] = final_position_and_angles[[0, 2]] - starting_position_and_angles[[0, 2]]
            diff[[1, 3]] = angles_difference(final_position_and_angles[[1, 3]], starting_position_and_angles[[1, 3]])
        except FloatingPointError:
            diff = np.array([np.nan, np.nan, np.nan, np.nan])
        return diff

    def find_central_line(self, override_existing=False) -> Tuple[np.ndarray, bool]:
        if self.central_line_successfully_traced is not None and not override_existing:
            # I never debugged those two lines:
            initial_theta, initial_phi = angles_of_unit_vector(self.central_line[0].k_vector)
            initial_t, initial_p = self.arms[0].surface_1.get_parameterization(self.central_line[0].origin)
            return np.array([initial_t, initial_theta, initial_p, initial_phi]), self.central_line_successfully_traced

        theta_initial_guess, phi_initial_guess = self.default_initial_angles
        # global I
        initial_guess = np.array([0, theta_initial_guess, 0, phi_initial_guess])

        # In the documentation it says optimize.fsolve returns a solution, together with some flags, and also this is
        # how pycharm suggests to use it. But in practice it returns only the solution, not sure why.
        central_line_initial_parameters: np.ndarray = optimize.fsolve(self.f_roots, initial_guess, maxfev=10000)
        # optimize.fsolve(self.f_roots, initial_guess, maxfev=1000)



        root_error = np.linalg.norm(self.f_roots(central_line_initial_parameters))
        # ROOT_ERRORS[I] = root_error
        # I += 1
        # I = 0
        # shift = np.linspace(0, 3e-8, N)
        # overlaps, _ = cavity.calculated_shifted_cavity_overlap_integral(parameter_index=(1, 1), shift=shift)
        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(shift, overlaps)
        # ax[1].plot(shift, ROOT_ERRORS)
        # plt.show()
        if root_error > 1e-9:
            central_line_successfully_traced = False
        else:
            central_line_successfully_traced = True

        origin_solution = self.arms[0].surface_1.parameterization(central_line_initial_parameters[0],
                                                                  central_line_initial_parameters[2])  # t, p
        k_vector_solution = unit_vector_of_angles(central_line_initial_parameters[1],
                                                  central_line_initial_parameters[3])  # theta, phi
        central_line = Ray(origin_solution, k_vector_solution)
        # This line is to save the central line in the ray history, so that it can be plotted later.
        central_line = self.trace_ray(central_line)
        for i, arm in enumerate(self.arms):
            arm.central_line = central_line[i]
        self.central_line_successfully_traced = central_line_successfully_traced
        return central_line_initial_parameters, central_line_successfully_traced

    def set_mode_parameters(self):
        if self.central_line_successfully_traced is None:
            self.find_central_line()

        mode_parameters_current = local_mode_parameters_of_round_trip_ABCD(self.ABCD_round_trip)
        for arm in self.arms:
            arm.mode_parameters_on_surface_1 = mode_parameters_current
            mode_parameters_current = arm.propagate_local_mode_parameters()
            arm.mode_principle_axes = self.principle_axes(arm.central_line.k_vector)

    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        if self.central_line_successfully_traced is None:
            self.find_central_line()
        # ATTENTION! THIS ASSUMES THAT ALL THE CENTRAL LINE arms ARE IN THE SAME PLANE.
        # I find the biggest psuedo z because if the first two k_vector are parallel, the cross product is zero and the
        # result of the cross product will be determined by arbitrary numerical errors.
        possible_pseudo_zs = [np.cross(self.central_line[0].k_vector, self.central_line[i].k_vector) for i in
                              range(1, len(self.central_line))]  # Points to the positive
        biggest_psuedo_z = possible_pseudo_zs[np.argmax([np.linalg.norm(pseudo_z) for pseudo_z in possible_pseudo_zs])]
        # biggest_psuedo_z = np.cross(self.central_line[0].k_vector, self.central_line[1].k_vector)
        pseudo_z = normalize_vector(biggest_psuedo_z)
        pseudo_x = np.cross(pseudo_z, k_vector)
        principle_axes = np.stack([pseudo_z, pseudo_x], axis=-1).T  # [z_x, z_y, z_z], [x_x, x_y, x_z]
        return principle_axes

    def ray_of_initial_parameters(self, initial_parameters: np.ndarray):
        k_vector_i = unit_vector_of_angles(theta=initial_parameters[1], phi=initial_parameters[3])
        origin_i = self.arms[0].surface_1.parameterization(t=initial_parameters[0], p=initial_parameters[2])
        input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
        return input_ray

    def generate_spot_size_lines(self, dim=2) :
        if self.arms[0].mode_parameters is None:
            self.set_mode_parameters()
        list_of_spot_size_lines = []
        for arm in self.arms:
            t = np.linspace(0, arm.central_line.length, 100)
            ray_points = arm.central_line.parameterization(t=t)
            z_minus_z_0 = np.linalg.norm(ray_points[:, np.newaxis, :] - arm.mode_parameters.center, axis=2)  # Before
            # the norm the size is 100 | 2 | 3 and after it is 100 | 2 (100 points for in_plane and out_of_plane
            # dimensions)
            principle_axes = arm.mode_principle_axes
            sign = np.array([1, -1])
            spot_size_value = spot_size(z_minus_z_0, arm.mode_parameters.z_R, self.lambda_laser)
            spot_size_lines = ray_points[:, np.newaxis, np.newaxis, :] + \
                              spot_size_value[:, :, np.newaxis, np.newaxis] * \
                              principle_axes[np.newaxis, :, np.newaxis, :] * \
                              sign[np.newaxis, np.newaxis, :,
                              np.newaxis]  # The size is 100 (n_points) | 2 (axis) | 2 (sign) | 3 (coordinate)
            if dim == 2:
                spot_size_lines = spot_size_lines[:, 1, :, 0:2]  # Drop the z axis, and drop the lines of the
                # transverse axis
                list_of_spot_size_lines.extend(
                    [spot_size_lines[:, 0, :], spot_size_lines[:, 1, :]])  # Each element is a
                # 100 | 3 array

            else:
                list_of_spot_size_lines.extend([spot_size_lines[:, 0, 0, :], spot_size_lines[:, 0, 1, :],
                                                spot_size_lines[:, 1, 0, :], spot_size_lines[:, 1, 1, :]])  # Each
                # element is a  100 | 3 array.

        return list_of_spot_size_lines

    def set_initial_surface(self) -> Optional[Surface]:
        # adds a virtual surface on the first arm that is perpendicular to the beam and centered between the first two
        # physical_surfaces.
        if not isinstance(self.arms[0].surface_1, PhysicalSurface):
            return self.arms[0].surface_1
        # gets a surface that sits between the first two physical_surfaces, centered and perpendicular to the central line.
        if self.central_line is None:
            final_position_and_angles, success = self.find_central_line()
            if not success:
                warnings.warn("Could not find central line, so no initial surface could be set.")
                return None
        middle_point = (self.central_line[0].origin + self.central_line[1].origin) / 2
        initial_surface = FlatSurface(outwards_normal=-self.central_line[0].k_vector, center=middle_point)

        first_leg = self.arms[0]
        first_leg_first_sub_leg = Arm(first_leg.surface_1, initial_surface)
        first_leg_second_sub_leg = Arm(initial_surface, first_leg.surface_2)
        if self.standing_wave:
            last_leg = self.arms[-1]
            last_leg_first_sub_leg = Arm(last_leg.surface_1, initial_surface)
            last_leg_second_sub_leg = Arm(initial_surface, last_leg.surface_2)
            legs_list = [first_leg_second_sub_leg] + self.arms[1:-1] + [last_leg_first_sub_leg,
                                                                        last_leg_second_sub_leg,
                                                                        first_leg_first_sub_leg]
        else:
            legs_list = [first_leg_second_sub_leg] + self.arms[1:] + [first_leg_first_sub_leg]
        self.arms = legs_list
        # Now, after you found the initial_surface, we can retrace the central line, but now let it out from the
        # initial surface, instead of the first mirror.
        self.find_central_line(override_existing=True)
        return initial_surface

    def ABCD_round_trip_matrix_numeric(self,
                                       central_line_initial_parameters: Optional[
                                           np.ndarray] = None) -> np.ndarray:
        if central_line_initial_parameters is None:
            central_line_initial_parameters, success = self.find_central_line()
            if not success:
                raise ValueError("Could not find central line")

        if isinstance(self.arms[0].surface_1, PhysicalSurface):
            self.set_initial_surface()

        dr = 1e-9

        # The i'th, j'th element of optimize.approx_fprime is the derivative of the i'th component of the output with
        # respect to the j'th component of the input, which is exactly the definition of the i'th j'th element of the
        # ABCD matrix.

        def trace_ray_parametric_parameters_only(parameters_initial):
            parameters_final, _ = self.trace_ray_parametric(parameters_initial)
            return parameters_final

        ABCD_matrix = optimize.approx_fprime(central_line_initial_parameters, trace_ray_parametric_parameters_only, dr)
        return ABCD_matrix

    def plot(self,
             ax: Optional[plt.Axes] = None,
             axis_span: Optional[float] = None,
             camera_center: Union[float, int] = -1,
             ray_list: Optional[List[Ray]] = None,
             dim: int = 2,
             laser_color: str = 'r') -> plt.Axes:

        if axis_span is None:
            axis_span = 1.1 * max([m.center[0] for m in self.physical_surfaces] +
                                  [m.center[1] for m in self.physical_surfaces] +
                                  [m.center[2] for m in self.physical_surfaces])

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

        if camera_center == -1:
            origin_camera = np.mean(np.stack([m.center for m in self.physical_surfaces]), axis=0)
        else:
            camera_center_int = np.floor(camera_center)
            if np.mod(camera_center, 1) == 0.5:
                origin_camera = (self.arms[camera_center_int].surface_1.center + self.arms[camera_center_int].surface_2.center) / 2
            else:
                origin_camera = self.surfaces[camera_center_int].center

        ax.set_xlim(origin_camera[0] - axis_span, origin_camera[0] + axis_span)
        ax.set_ylim(origin_camera[1] - axis_span, origin_camera[1] + axis_span)

        if ray_list is None and self.central_line is not None:
            ray_list = self.central_line

            for ray in ray_list:
                ray.plot(ax=ax, dim=dim, color=laser_color)
        for i, surface in enumerate(self.surfaces):
            surface.plot(ax=ax, dim=dim)  # , name=str(i)

        if self.lambda_laser is not None:
            try:
                spot_size_lines = self.generate_spot_size_lines(dim=dim)
                for line in spot_size_lines:
                    if dim == 2:
                        ax.plot(line[:, 0], line[:, 1], color=laser_color, linestyle='--', alpha=0.8, linewidth=0.5)
                    else:
                        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=laser_color, linestyle='--', alpha=0.8,
                                linewidth=0.5)
            except FloatingPointError:
                pass
        return ax

    def calculated_shifted_cavity_overlap_integral(self, parameter_index: Tuple[int, int],
                                                   shift: Union[float, np.ndarray] = np.linspace(-1e-6, 1e-6, 50)) -> \
            Tuple[np.ndarray, np.ndarray]:
        shift_input_is_float = isinstance(shift, (float, int))
        if shift_input_is_float:
            shift = np.array([shift])
        overlaps = np.zeros_like(shift, dtype=np.float64)
        NAs = np.zeros_like(shift)
        for i, shift_value in enumerate(shift):
            new_cavity = perturb_cavity(self, parameter_index, shift_value)
            try:
                overlap = calculate_cavities_overlap_matrices(cavity_1=self, cavity_2=new_cavity)
            except np.linalg.LinAlgError:
                continue
            overlaps[i] = np.abs(overlap)
            if new_cavity.arms[0].mode_parameters is not None:
                NAs[i] = new_cavity.arms[0].mode_parameters.NA[0]
        if shift_input_is_float:
            overlaps = overlaps[0]
        return overlaps, NAs

    def calculate_parameter_critical_tolerance(self,
                                               parameter_index: Tuple[int, int],
                                               initial_step: float = 1e-6,
                                               overlap_threshold: float = 0.9,
                                               accuracy: float = 1e-3) -> float:
        def f(shift):
            return self.calculated_shifted_cavity_overlap_integral(parameter_index, shift)[0]

        return functions_first_crossing_both_directions(f=f, initial_step=initial_step,
                                                        crossing_value=overlap_threshold, accuracy=accuracy)

    def generate_tolerance_threshold_matrix(self,
                                  initial_step: float = 1e-6,
                                  overlap_threshold: float = 0.9,
                                  accuracy: float = 1e-3, print_progress: bool = True) -> np.ndarray:
        tolerance_matrix = np.zeros((self.params.shape[0], 7))
        for i in range(self.params.shape[0]):
            if print_progress:
                print(i)
            for j in range(7):
                if print_progress:
                    print("  ", j)
                tolerance_matrix[i, j] = self.calculate_parameter_critical_tolerance(parameter_index=(i, j),
                                                                                     initial_step=initial_step,
                                                                                     overlap_threshold=overlap_threshold,
                                                                                     accuracy=accuracy)
        return tolerance_matrix

    def generate_overlap_series(self,
                                    shifts: Union[np.ndarray, float],  # Float is interpreted as linspace's limits,
                                    # np.ndarray means that the i'th j'th element of shifts is the linspace limits of
                                    # the i'th j'th parameter.
                                    shift_size: int = 30,
                                    print_progress: bool = True,) -> np.ndarray:
        overlaps = np.zeros((self.params.shape[0], 7, shift_size))
        for i in range(self.params.shape[0]):
            if print_progress:
                print(i)
            for j in range(7):
                if print_progress:
                    print("  ", j)
                if isinstance(shifts, (float, int)):
                    shift_series = np.linspace(-shifts, shifts, shift_size)
                else:
                    shift_series = np.linspace(-shifts[i, j], shifts[i, j], shift_size)
                overlaps[i, j, :], _ = self.calculated_shifted_cavity_overlap_integral(parameter_index=(i, j),
                                                                                        shift=shift_series)
        return overlaps

    def generate_overlaps_graphs(self,
                                  initial_step: float = 1e-6,
                                  overlap_threshold: float = 0.9,
                                  accuracy: float = 1e-3,
                                  print_progress: bool = True,
                                  arm_index_for_NA: int = 0,
                                  tolerance_matrix: Optional[np.ndarray] = None,
                                  overlaps_series: Optional[np.ndarray] = None):
        fig, ax = plt.subplots(self.params.shape[0], 7, figsize=(7*5, self.params.shape[0]*2))
        if tolerance_matrix is None:
            tolerance_matrix = self.generate_tolerance_threshold_matrix(initial_step=initial_step,
                                                                        overlap_threshold=overlap_threshold,
                                                                        accuracy=accuracy, print_progress=print_progress)
        if overlaps_series is None:
            overlaps_series = self.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix),
                                                           shift_size=30,
                                                           print_progress=True)
        plt.suptitle(f"NA={self.arms[arm_index_for_NA].mode_parameters.NA[0]:.3e}")
        for i in range(self.params.shape[0]):
            print(i)
            for j in range(7):
                print("  ", j)
                tolerance = tolerance_matrix[i, j]
                tolerance_abs = np.abs(tolerance)
                shifts = np.linspace(-2 * tolerance_abs, 2 * tolerance_abs, 30)
                ax[i, j].plot(shifts, overlaps_series[i, j, :])
                title = f"{self.names[i]}, {INDICES_DICT_INVERSE[j]}, tolerance: {tolerance_abs:.2e}"
                ax[i, j].set_title(title)
                if i == self.params.shape[0] - 1:
                    ax[i, j].set_xlabel("Shift")
                if j == 0:
                    ax[i, j].set_ylabel("Overlap")
                ax[i, j].axvline(tolerance, color='g', linestyle='--')
                ax[i, j].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
                ax[i, j].axhline(overlap_threshold, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        return fig, ax


def generate_tolerance_of_NA(
                             params: np.ndarray,
                             parameter_index_for_NA_control: Tuple[int, int],
                             arm_index_for_NA: int,
                             parameter_values: np.ndarray,
                             initial_step: float = 1e-6,
                             overlap_threshold: float = 0.9,
                             accuracy: float = 1e-3,
                             lambda_laser: float = 532e-9,
                             standing_wave: bool = True
):
    tolerance_matrix = np.zeros((params.shape[0], 7, parameter_values.shape[0]))
    NAs = np.zeros(parameter_values.shape[0])
    for k, parameter_value in enumerate(parameter_values):
        print(k)
        params_temp = params.copy()
        params_temp[parameter_index_for_NA_control] = parameter_value
        cavity = Cavity.from_params(params=params_temp, set_mode_parameters=True, lambda_laser=lambda_laser,
                                    standing_wave=standing_wave)
        if np.any(np.isnan(cavity.mode_parameters[arm_index_for_NA].NA)) or np.any(cavity.mode_parameters[arm_index_for_NA].NA == 0):
            continue
        NAs[k] = cavity.mode_parameters[arm_index_for_NA].NA[0]  # ARBITRARY
        tolerance_matrix[:, :, k] = cavity.generate_tolerance_threshold_matrix(initial_step=initial_step,
                                                                                             overlap_threshold=overlap_threshold,
                                                                                             accuracy=accuracy)
    return NAs, tolerance_matrix


def plot_tolerance_of_NA(params: Optional[np.ndarray] = None,
                         parameter_index_for_NA_control: Optional[Tuple[int, int]] = None,
                         arm_index_for_NA: Optional[int] = None,
                         parameter_values: Optional[np.ndarray] = None,
                         initial_step: Optional[float] = 1e-6,
                         overlap_threshold: Optional[float] = 0.9,
                         accuracy: Optional[float] = 1e-3,
                         names: Optional[List[str]] = None,
                         lambda_laser: Optional[float] = 5.32e-6,
                         standing_wave: Optional[bool] = True,
                         NAs: Optional[np.ndarray] = None,
                         tolerance_matrix: Optional[np.ndarray] = None):
    if tolerance_matrix is None:
        NAs, tolerance_matrix = generate_tolerance_of_NA(params=params,
                                                         parameter_index_for_NA_control=parameter_index_for_NA_control,
                                                         arm_index_for_NA=arm_index_for_NA,
                                                         parameter_values=parameter_values,
                                                         initial_step=initial_step,
                                                         overlap_threshold=overlap_threshold,
                                                         accuracy=accuracy,
                                                         lambda_laser=lambda_laser,
                                                         standing_wave=standing_wave
                                                         )
    tolerance_matrix = np.abs(tolerance_matrix)
    fig, ax = plt.subplots(tolerance_matrix.shape[0], 7, figsize=(7*5, tolerance_matrix.shape[0]*2))
    if names is None:
        names = [None for _ in range(params.shape[0])]
    for i in range(tolerance_matrix.shape[0]):
        for j in range(7):
            ax[i, j].plot(NAs, tolerance_matrix[i, j, :], color='g')
            title = f"{names[i]}, {INDICES_DICT_INVERSE[j]}"
            ax[i, j].set_title(title)
            if i == tolerance_matrix.shape[0] - 1:
                ax[i, j].set_xlabel("NA")
            if j == 0:
                ax[i, j].set_ylabel("Tolerance")
            ax[i, j].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax[i, j].set_yscale('log')
    fig.tight_layout()
    return fig, ax


def calculate_gaussian_parameters_on_surface(surface: FlatSurface, mode_parameters: ModeParameters):
    intersection_point = surface.find_intersection_with_ray(mode_parameters.ray)
    intersection_point = intersection_point[0, :]
    z_minus_z_0 = np.linalg.norm(intersection_point - mode_parameters.center, axis=1)
    q_u, q_v = z_minus_z_0 + 1j * mode_parameters.z_R
    k = 2 * np.pi / mode_parameters.lambda_laser

    # Those are the vectors that define the mode and the surface: r_0 is the surface's center with respect to the mode's
    # center, t_hat and p_hat are the unit vectors that span the surface, k_hat is the mode's k vector,
    # u_hat and v_hat are the principle axes of the mode.
    r_0 = surface.center - mode_parameters.center[0, :]  # Techinically there are two centers, but their difference is
    # only in the k_hat direction, which doesn't make a difference on the projection on the two principle axes of the
    # mode, and for the projection of the k_hat vector we anyway need to set an arbitrary 0, so we can just take the
    # first center.
    t_hat, p_hat = normalize_vector(surface.parameterization(1, 0) - surface.parameterization(0, 0)), \
                   normalize_vector(surface.parameterization(0, 1) - surface.parameterization(0, 0))
    k_hat = mode_parameters.k_vector
    u_hat_v_hat = mode_parameters.principle_axes
    u_hat = u_hat_v_hat[0, :]
    v_hat = u_hat_v_hat[1, :]

    # The mode as a function of the surface's parameterization:
    # exp([t,p] @ A_2 @ [t,p] + b @ [t,p] + c

    A = 1j * k * np.array([[(t_hat @ u_hat) ** 2 / q_u + (t_hat @ v_hat) ** 2 / q_v,
                            (t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v],
                           [(t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v,
                            (p_hat @ u_hat) ** 2 / q_u + (p_hat @ v_hat) ** 2 / q_v]])
    b = - (1 / 2) * 1j * k * np.array(
        [(k_hat @ t_hat) + 2 * (r_0 @ u_hat) * (t_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (t_hat @ v_hat) / q_v,
         (k_hat @ p_hat) + 2 * (r_0 @ u_hat) * (p_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (p_hat @ v_hat) / q_v])
    c = - (1 / 2) * 1j * k * ((k_hat @ r_0) + (r_0 @ u_hat) ** 2 / q_u + (r_0 @ v_hat) ** 2 / q_v)

    return A, b, c


def evaluate_cavities_modes_on_surface(cavity_1: Cavity, cavity_2: Cavity):
    correct_modes = True
    for cavity in [cavity_1, cavity_2]:
        if cavity.arms[0].mode_parameters is None:
            try:
                cavity.set_mode_parameters()
            except FloatingPointError:
                correct_modes = False
                break

    mode_parameters_1 = cavity_1.arms[0].mode_parameters
    mode_parameters_2 = cavity_2.arms[0].mode_parameters

    NAs = np.concatenate((mode_parameters_1.NA, mode_parameters_2.NA))
    if cavity_1.central_line_successfully_traced is False or cavity_2.central_line_successfully_traced is False or\
            correct_modes is False or np.any(np.isnan(NAs)):
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
        return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes



    cavity_1_waist_pos = mode_parameters_1.center[0, :]
    P1 = FlatSurface(center=cavity_1_waist_pos, outwards_normal=mode_parameters_1.k_vector)
    try:
        A_1, b_1, c_1 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_1)
        A_2, b_2, c_2 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_2)
    except FloatingPointError:
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
    return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes


def calculate_cavities_overlap_matrices(cavity_1: Cavity, cavity_2: Cavity):
    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = evaluate_cavities_modes_on_surface(cavity_1, cavity_2)
    if correct_modes is False:
        return np.nan
    else:
        return gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)


def gaussian_norm_log(A: np.ndarray, b: np.ndarray, c: float):
    return 1 / 2 * gaussian_integral_2d_log(A + np.conjugate(A), b + np.conjugate(b), c + np.conjugate(c))


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


def gaussians_overlap_integral(A_1: np.ndarray, A_2: np.ndarray,
                               # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
                               b_1: np.ndarray, b_2: np.ndarray,
                               c_1: float, c_2: float):
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


def evaluate_gaussian(A: np.ndarray, b: np.ndarray, c: complex, axis_span: float, N: int=100):
    x = np.linspace(-axis_span, axis_span, N)
    y = np.linspace(-axis_span, axis_span, N)
    X, Y = np.meshgrid(x, y)
    R = np.stack([X, Y], axis=2)
    # mu = np.array([x_2, y_2])
    # R_shifted = R - mu[None, None, :]
    R_normed_squared = np.einsum('ijk,kl,ijl->ij', R, A, R)
    functions_values = safe_exponent(-(1 / 2) * R_normed_squared + np.einsum('k,ijk->ij', b, R) + c)
    return functions_values


def plot_gaussian_subplot(A: np.ndarray, b: np.ndarray, c: float, axis_span: float = 0.0005,
                          fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    functions_values = evaluate_gaussian(A, b, c, axis_span)
    im = ax.imshow(np.real(functions_values))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    return fig, ax


def plot_2_gaussians_subplots(A_1: np.ndarray, A_2: np.ndarray,
                              # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
                              b_1: np.ndarray, b_2: np.ndarray,
                              c_1: float, c_2: float, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None,
                              axis_span: float = 0.0005,
                              title: Optional[str] = ''):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    plot_gaussian_subplot(A_1, b_1, c_1, axis_span, fig, ax[0])
    plot_gaussian_subplot(A_2, b_2, c_2, axis_span, fig, ax[1])
    if title is not None:
        fig.suptitle(title)


def plot_2_gaussians_colors(A_1: np.ndarray, A_2: np.ndarray,
                            b_1: np.ndarray, b_2: np.ndarray,
                            c_1: float, c_2: float, ax: Optional[plt.Axes] = None, axis_span: float = 0.0005,
                            title: Optional[str] = '',
                            real_or_abs: str = 'abs'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    first_gaussian_values = evaluate_gaussian(A_1, b_1, c_1, axis_span)
    second_gaussian_values = evaluate_gaussian(A_2, b_2, c_2, axis_span)
    first_gaussian_values = first_gaussian_values / np.max(np.abs(first_gaussian_values))
    second_gaussian_values = second_gaussian_values / np.max(np.abs(second_gaussian_values))
    third_color_channel = np.zeros_like(first_gaussian_values)
    rgb_image = np.stack([first_gaussian_values, second_gaussian_values, third_color_channel], axis=2)
    if real_or_abs == 'abs':
        rgb_image = np.abs(rgb_image)
    else:
        rgb_image = np.real(rgb_image)
    ax.imshow(rgb_image)
    ax.set_title(title)


def perturb_cavity(cavity: Cavity, parameter_index: Tuple[int, int], shift_value: float):
    params = cavity.to_params
    new_params = copy.copy(params)
    new_params[parameter_index] = params[parameter_index] + shift_value
    new_cavity = Cavity.from_params(params=new_params, standing_wave=cavity.standing_wave,
                                    lambda_laser=cavity.lambda_laser)
    return new_cavity


def plot_2_cavity_perturbation_overlap(cavity: Cavity, parameter_index: Tuple[int, int], shift_value: float,
                                       ax: Optional[plt.Axes] = None, axis_span: float = 0.0005):
    new_cavity = perturb_cavity(cavity, parameter_index, shift_value)

    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_mode = evaluate_cavities_modes_on_surface(cavity, new_cavity)
    if correct_mode:
        plot_2_gaussians_colors(A_1, A_2, b_1, b_2, c_1, c_2, ax=ax, axis_span=axis_span,
                                title='Cavity perturbation overlap',
                                real_or_abs='abs')


def evaluate_gaussian_3d(points: np.ndarray, mode_parameters: ModeParameters):
    center = mode_parameters.center[0, :]
    points_relative = points - center
    u_vec = mode_parameters.principle_axes[0, :]
    v_vec = mode_parameters.principle_axes[1, :]
    k_vec = mode_parameters.k_vector
    k_projection = points_relative @ k_vec
    u_projection = points_relative @ u_vec
    v_projection = points_relative @ v_vec
    q = k_projection[:, :, None] + 1j * mode_parameters.z_R[None, None, :]
    q_u = q[:, :, 0]
    q_v = q[:, :, 1]
    k = 2 * np.pi / mode_parameters.lambda_laser
    integrand = -1j * k / 2 * (u_projection ** 2 / q_u + v_projection ** 2 / q_v + k_projection)
    gaussian = safe_exponent(integrand)
    return gaussian


def find_distance_to_first_crossing(shifts: np.ndarray, overlaps: np.ndarray, crossing_value: float = 0.9):
    # Assumes shifts is ascending and that overlaps[i] is the overlap of shift[i]
    positive_shifts = shifts[shifts > 0]
    negative_shifts = shifts[shifts < 0]
    positive_shifts_overlaps = overlaps[shifts > 0]
    negative_shifts_overlaps = overlaps[shifts < 0]
    negative_shifts_overlaps = negative_shifts_overlaps[::-1]
    negative_shifts = negative_shifts[::-1]
    first_positive_overlap_crossing = np.argmax(positive_shifts_overlaps < crossing_value)
    first_negative_overlap_crossing = np.argmax(negative_shifts_overlaps < crossing_value)
    crossing_positive_shift = positive_shifts[first_positive_overlap_crossing]
    crossing_negative_shift = negative_shifts[first_negative_overlap_crossing]
    minimal_distance = np.min(np.abs(np.array([crossing_positive_shift, crossing_negative_shift])))
    return minimal_distance


def interval_parameterization(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def functions_first_crossing(f: Callable, initial_step: float, crossing_value: float = 0.9,
                             accuracy: float = 0.001, max_f_eval: int = 100) -> float:
    # assumes f(0) == 1 and a decreasing function.
    stopping_flag = False
    increasing_ratio = 1.5
    n = 10
    last_n_evaluations = np.zeros(n)
    last_n_xs = np.zeros(n)
    borders_min = 0
    borders_max = np.nan
    loop_counter = 0
    f_0 = f(0)
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
            warnings.warn(f"Did not find crossing value, interpolated it between ({borders_min:.8e}, {f_borders_min:.5e}) and ({borders_max:.8e}, {f_borders_max:.5e}) to be ({x:.8e}, {crossing_value:.5e})")
            stopping_flag = True
        elif not np.any(np.invert(np.abs(last_n_evaluations - 1) < 1e-18)) or loop_counter > max_f_eval:  # if the function is not
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
            borders_max = x
            f_borders_max = f_x
            x = interval_parameterization(borders_min, x, 0.5)
        elif not np.any(np.invert(np.isnan(last_n_evaluations))):  # If all the last n evaluations were nan:
            borders_max = np.min(last_n_xs)  # Set borders_max to be the first value from which
            # onwards it seems to always be nan.
            x = interval_parameterization(borders_min, borders_max, 0.5)
        elif np.isnan(f_x):
            if np.isnan(borders_max):
                x *= increasing_ratio
                f_borders_max = f_x
            else:
                flip_coin = np.random.randint(2)
                if flip_coin == 0:
                    x = interval_parameterization(x, borders_max, 2**(-0.5))  # This square root of two is to assure the
                    # algorithm will not get stuck in a loop.
                else:
                    x = interval_parameterization(borders_min, x, 2**(-0.5))
        elif crossing_value - accuracy < f_x < crossing_value + accuracy:
            stopping_flag = True
        else:
            raise ValueError('This should not happen')
    print(loop_counter)
    return x


def functions_first_crossing_both_directions(f: Callable, initial_step: float, crossing_value: float = 0.9,
                                             accuracy: float = 0.001) -> float:
    positive_step = functions_first_crossing(f, initial_step, crossing_value, accuracy)
    negative_step = functions_first_crossing(lambda x: f(-x), initial_step, crossing_value, accuracy)
    if positive_step < negative_step:
        return positive_step
    else:
        return -negative_step
# %%

if __name__ == '__main__':
    x_1 = 3.4934200000e-01
    x_1_b = 6.5789500000e-04
    x_1_c = 0.0000000000e+00
    y_1 = 0.0000000000e+00
    t_1 = 0.0000000000e+00
    p_1 = 0.0000000000e+00
    r_1 = 2.0000000000e-01
    x_3 = -5.0000000000e-02
    y_3 = 0.0000000000e+00
    t_3 = 0.0000000000e+00
    p_3 = 0.0000000000e+00
    r_3 = 2.0000000000e-01
    lambda_laser = 5.3200e-07

    x_1 += x_1_b
    x_1 += x_1_c
    p_3 += np.pi
    dim = 2
    names = ['Right Mirror', 'Left Mirror']

    params = np.array([[x_1, y_1, t_1, p_1, r_1, 0, 0, 0, 0, 1, 0],
                       # [x_2, y_2, t_2, p_2, r_2, w_2, n_in, n_out, 0, 0, 2],
                       [x_3, y_3, t_3, p_3, r_3, 0, 0, 0, 0, 1, 0]])
    ratios = 1 + np.concatenate((np.array([0]), np.logspace(-7, -3, 5, endpoint=True)))
    x_1_values = x_1 * ratios
    cavity = Cavity.from_params(params=params, standing_wave=True,
                                lambda_laser=lambda_laser, names=names)
    ax = cavity.plot()
    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(-0.07, 0.07)
    ax.set_title(f"NA={cavity.arms[0].mode_parameters.NA[0]:.3e}")
    plt.savefig('figures/systems/mirror-mirror.png', dpi=300, bbox_inches='tight')
    plt.show()

#     NAs, tolerance_matrix = generate_tolerance_of_NA(params, parameter_index_for_NA_control=(0, 0), arm_index_for_NA=2,
#                                                      parameter_values=x_1_values)
#     overlaps_series = cavity.generate_overlap_series(shifts=2 * np.abs(tolerance_matrix[:, :, 0]),
#                                                      shift_size=30,
#                                                      print_progress=True)
# # %%
#     cavity.generate_overlaps_graphs(overlap_threshold=0.9, arm_index_for_NA=2,
#                                     tolerance_matrix=tolerance_matrix[:, :, 0], overlaps_series=overlaps_series)
#     plt.savefig('figures/NA tolerance/mirror_lens_mirror_left_control_overlaps.png', dpi=300, bbox_inches='tight')
#     plt.show()
# # %%
#     fig, ax = plot_tolerance_of_NA(NAs=NAs, tolerance_matrix=tolerance_matrix, names=names)
#     plt.savefig('figures/NA tolerance/mirror_lens_mirror_left_control_tolerance.png', dpi=300, bbox_inches='tight')
#     plt.show()



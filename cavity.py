import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy import optimize
import warnings
from dataclasses import dataclass
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable


# set numpy to raise an error on warnings:
np.seterr(all='raise')

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
            return np.sqrt(self.lambda_laser / (np.pi * self.z_R))


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
    # This funny stack synatx is to allow theta to be of any dimension
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
    one_over_f = (n - 1) * ((1/R_1) + (1/R_2) + ((n - 1) * width) / (n * R_1 * R_2))
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
    result = np.mod(diff + np.pi, 2*np.pi) - np.pi
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
    def __init__(self, outwards_normal: np.ndarray, **kwargs):
        self.outwards_normal = normalize_vector(outwards_normal)

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

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None, dim: int = 3, length=0.6):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111, projection='3d')
        if dim == 3:
            t = np.linspace(-length / 2, length / 2, 100)
        else:
            t = 0
        s = np.linspace(-length / 2, length / 2, 100)
        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        if isinstance(self, PhysicalSurface):
            color = 'b'
        elif isinstance(self, CurvedRefractiveSurface):
            color = 'grey'
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


class PhysicalSurface(Surface):
    def __init__(self, outwards_normal: np.ndarray, **kwargs):
        super().__init__(outwards_normal=outwards_normal, **kwargs)

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

    def ABCD_matrix(self, cos_theta_incoming: float) -> np.ndarray:
        raise NotImplementedError

    def to_params(self):
        raise NotImplementedError


class FlatSurface(Surface):
    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center: Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(outwards_normal=outwards_normal, **kwargs)
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

    def get_spanning_vectors(self):
        pseudo_y = np.cross(np.array([0, 0, 1]), self.inwards_normal)
        pseudo_z = np.cross(self.inwards_normal, pseudo_y)
        return pseudo_y, pseudo_z

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        if isinstance(t, (float, int)):
            t = np.array(t)
        if isinstance(p, (float, int)):
            p = np.array(p)
        points = self.center + t[..., np.newaxis] * pseudo_z + p[..., np.newaxis] * pseudo_y
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        t = (points - self.center) @ pseudo_z
        p = (points - self.center) @ pseudo_y
        return t, p


class FlatMirror(FlatSurface, PhysicalSurface):

    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center: Optional[np.ndarray] = None):
        super().__init__(outwards_normal=outwards_normal, distance_from_origin=distance_from_origin, center=center)

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

    def ABCD_matrix(self, cos_theta_incoming: float) -> np.ndarray:
        # Assumes the ray is in the x-y plane, and the mirror is in the z-x plane
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, -1]])

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
                 curvature_sign: int = 1  # 1 for concave (where the ray is hitting the sphere from the inside) and -1 for convex
                 # (where the ray is hitting the sphere from the outside). this is used to find the correct intersection
                 # point of a ray with the surface
                 ):
        super().__init__(outwards_normal=outwards_normal)
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
    def from_params(params: np.ndarray):
        x, y, z, t, p, r, curvature_sign, surface_type, n_1, n_2 = params
        center = np.array([x, y, z])
        outwards_normal = unit_vector_of_angles(t, p)
        if surface_type == 0:
            surface = CurvedMirror(radius=r,
                                  outwards_normal=outwards_normal,
                                  center=center,
                                  curvature_sign=curvature_sign)
        elif surface_type == 1:
            surface = CurvedRefractiveSurface(radius=r,
                                             outwards_normal=outwards_normal,
                                             center=center,
                                             curvature_sign=curvature_sign,
                                             n_1=n_1,
                                             n_2=n_2)
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
        return x, y, z, t, p, r, self.curvature_sign, surface_type, n_1, n_2

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
                 curvature_sign: int = 1):
        super().__init__(radius, outwards_normal, center, origin, curvature_sign)

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

    def ABCD_matrix(self, cos_theta_incoming: float):
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
                 curvature_sign: int = 1):
        super().__init__(radius, outwards_normal, center, origin, curvature_sign)
        self.n_1 = n_1
        self.n_2 = n_2

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        n_backwards = (self.origin - intersection_point) * self.curvature_sign  # m_rays | 3
        n_backwards = normalize_vector(n_backwards)
        n_forwards = -n_backwards
        cos_theta_incoming = np.sum(ray.k_vector * n_forwards, axis=-1)   # m_rays
        n_orthogonal = ray.k_vector - cos_theta_incoming[..., np.newaxis] * n_forwards  # m_rays | 3
        if np.linalg.norm(n_orthogonal) < 1e-14:
            reflected_direction_vector = n_forwards
        else:
            n_orthogonal = normalize_vector(n_orthogonal)
            sin_theta_outgoing = np.sqrt((self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming ** 2))  # m_rays
            reflected_direction_vector = n_forwards * np.sqrt(1 - sin_theta_outgoing ** 2) + n_orthogonal * sin_theta_outgoing
        return reflected_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float) -> np.ndarray:
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


def generate_lens(radius: float, thickness: float, n_out: float, n_in: float, center: Optional, direction: np.ndarray):
    surface_1 = CurvedRefractiveSurface(radius=radius,
                                        outwards_normal=-direction,
                                        center=center - (1/2) * direction * thickness,
                                        n_1=n_out,
                                        n_2=n_in,
                                        curvature_sign=-1)
    surface_2 = CurvedRefractiveSurface(radius=radius,
                                        outwards_normal=direction,
                                        center=center + (1/2) * direction * thickness,
                                        n_1=n_in,
                                        n_2=n_out,
                                        curvature_sign=1)
    return surface_1, surface_2


class Leg:
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
        cos_theta = self.central_line.k_vector @ self.surface_2.outwards_normal
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


class Cavity:
    def __init__(self,
                 physical_surfaces: List[PhysicalSurface],
                 set_initial_surface: bool = False,
                 standing_wave: bool = False,
                 lambda_laser: Optional[float] = None):
        self.standing_wave = standing_wave
        self.physical_surfaces = physical_surfaces
        self.legs: List[Leg] = [
            Leg(self.physical_surfaces_ordered[i],
                self.physical_surfaces_ordered[np.mod(i + 1, len(self.physical_surfaces_ordered))],
                lambda_laser=lambda_laser) for i in range(len(self.physical_surfaces_ordered))]
        self.central_line_successfully_traced: Optional[bool] = None
        self.lambda_laser: Optional[float] = lambda_laser

        # Find central line and add the initial surface at the beginning of the surfaces list
        if set_initial_surface:
            self.set_initial_surface()

    @staticmethod
    def from_params(params: np.ndarray,
                    set_initial_surface: bool = False,
                    standing_wave: bool = False,
                    lambda_laser: Optional[float] = None):
        mirrors = [CurvedMirror.from_params(params[i, :]) for i in range(len(params))]
        return Cavity(mirrors, set_initial_surface, standing_wave, lambda_laser)

    @property
    def to_params(self):
        return np.array([surface.to_params() for surface in self.physical_surfaces])

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
        if self.legs[0].central_line is None:
            return None
        else:
            return [leg.central_line for leg in self.legs]

    @property
    def ABCD_matrices(self):
        if self.legs[0].central_line is None:
            return None
        else:
            return [leg.ABCD_matrix for leg in self.legs]

    @property
    def ABCD_round_trip(self):
        if self.legs[0].central_line is None:
            return None
        else:
            return np.linalg.multi_dot(self.ABCD_matrices[::-1])

    @property
    def mode_parameters(self):
        if self.legs[0].central_line is None:
            return None
        else:
            return [leg.mode_parameters for leg in self.legs]

    @property
    def surfaces(self):
        return [leg.surface_1 for leg in self.legs]

    @property
    def default_initial_k_vector(self) -> np.ndarray:
        if self.central_line is not None and self.central_line_successfully_traced:
            initial_k_vector = self.central_line[0].k_vector
        else:
            initial_k_vector = self.legs[0].surface_2.center - self.legs[0].surface_1.center
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
            initial_ray = Ray(origin=self.legs[0].surface_1.center, k_vector=initial_k_vector)
            return initial_ray

    def trace_ray(self, ray: Ray) -> List[Ray]:
        ray_history = [ray]
        for leg in self.legs:
            ray = leg.propagate(ray)
            ray_history.append(ray)
        return ray_history

    def trace_ray_parametric(self,
                             starting_position_and_angles: np.ndarray) -> Tuple[np.ndarray, List[Ray]]:
        # Like trace ray, but works as a function of the starting position and angles as parameters on the starting
        # surface, instead of the starting position and angles as a vector in 3D space.

        initial_ray = self.ray_of_initial_parameters(starting_position_and_angles)
        ray_history = self.trace_ray(initial_ray)
        final_intersection_point = ray_history[-1].origin
        t_o, p_o = self.legs[0].surface_1.get_parameterization(final_intersection_point)  # Here it is the initial
        # surface on purpose.
        theta_o, phi_o = angles_of_unit_vector(ray_history[-1].k_vector)
        final_position_and_angles = np.array([t_o, theta_o, p_o, phi_o])
        return final_position_and_angles, ray_history

    def f_roots(self,
                starting_position_and_angles: np.ndarray) -> np.ndarray:
        # The roots of this function are the initial parameters for the central line.
        final_position_and_angles, _ = self.trace_ray_parametric(starting_position_and_angles)
        diff = np.zeros_like(starting_position_and_angles)
        diff[[0, 2]] = final_position_and_angles[[0, 2]] - starting_position_and_angles[[0, 2]]
        diff[[1, 3]] = angles_difference(final_position_and_angles[[1, 3]], starting_position_and_angles[[1, 3]])
        return diff

    def find_central_line(self, override_existing=False) -> Tuple[np.ndarray, bool]:
        if self.central_line_successfully_traced is not None and not override_existing:
            # I never debugged those two lines:
            initial_theta, initial_phi = angles_of_unit_vector(self.central_line[0].k_vector)
            initial_t, initial_p = self.legs[0].surface_1.get_parameterization(self.central_line[0].origin)
            return np.array([initial_t, initial_theta, initial_p, initial_phi]), self.central_line_successfully_traced

        theta_initial_guess, phi_initial_guess = self.default_initial_angles

        initial_guess = np.array([0, theta_initial_guess, 0, phi_initial_guess])

        # In the documentation it says optimize.fsolve returns a solution, together with some flags, and also this is
        # how pycharm suggests to use it. But in practice it returns only the solution, not sure why.
        central_line_initial_parameters: np.ndarray = optimize.fsolve(self.f_roots, initial_guess, epsfcn=1e-10)

        if np.linalg.norm(self.f_roots(central_line_initial_parameters)) > 1e-10:
            central_line_successfully_traced = False
        else:
            central_line_successfully_traced = True

        origin_solution = self.legs[0].surface_1.parameterization(central_line_initial_parameters[0],
                                                                  central_line_initial_parameters[2])  # t, p
        k_vector_solution = unit_vector_of_angles(central_line_initial_parameters[1],
                                                  central_line_initial_parameters[3])  # theta, phi
        central_line = Ray(origin_solution, k_vector_solution)
        # This line is to save the central line in the ray history, so that it can be plotted later.
        central_line = self.trace_ray(central_line)
        for i, leg in enumerate(self.legs):
            leg.central_line = central_line[i]
        self.central_line_successfully_traced = central_line_successfully_traced
        return central_line_initial_parameters, central_line_successfully_traced

    def set_mode_parameters(self):
        if self.central_line_successfully_traced is None:
            self.find_central_line()

        mode_parameters_current = local_mode_parameters_of_round_trip_ABCD(self.ABCD_round_trip)
        for leg in self.legs:
            leg.mode_parameters_on_surface_1 = mode_parameters_current
            mode_parameters_current = leg.propagate_local_mode_parameters()
            leg.mode_principle_axes = self.principle_axes(leg.central_line.k_vector)

    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        if self.central_line_successfully_traced is None:
            self.find_central_line()
        # ATTENTION! THIS ASSUMES THAT ALL THE CENTRAL LINE LEGS ARE IN THE SAME PLANE.
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
        origin_i = self.legs[0].surface_1.parameterization(t=initial_parameters[0], p=initial_parameters[2])
        input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
        return input_ray

    def generate_spot_size_lines(self, lambda_laser, dim=2):
        if self.legs[0].mode_parameters is None:
            self.set_mode_parameters()
        list_of_spot_size_lines = []
        for leg in self.legs:
            t = np.linspace(-0.2 * leg.central_line.length, 1.2 * leg.central_line.length, 100)
            ray_points = leg.central_line.parameterization(t=t)
            z_minus_z_0 = np.linalg.norm(ray_points[:, np.newaxis, :] - leg.mode_parameters.center, axis=2)  # Before
            # the norm the size is 100 | 2 | 3 and after it is 100 | 2 (100 points for in_plane and out_of_plane
            # dimensions)
            principle_axes = leg.mode_principle_axes
            sign = np.array([1, -1])
            spot_size_value = spot_size(z_minus_z_0, leg.mode_parameters.z_R, lambda_laser)
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

    def set_initial_surface(self):
        # adds a virtual surface on the first leg that is perpendicular to the beam and centered between the first two
        # physical_surfaces.
        if not isinstance(self.legs[0].surface_1, PhysicalSurface):
            return self.legs[0].surface_1
        # gets a surface that sits between the first two physical_surfaces, centered and perpendicular to the central line.
        if self.central_line is None:
            final_position_and_angles, success = self.find_central_line()
            if not success:
                warnings.warn("Could not find central line, so no initial surface could be set.")
                return None
        middle_point = (self.central_line[0].origin + self.central_line[1].origin) / 2
        initial_surface = FlatSurface(outwards_normal=-self.central_line[0].k_vector, center=middle_point)

        first_leg = self.legs[0]
        first_leg_first_sub_leg = Leg(first_leg.surface_1, initial_surface)
        first_leg_second_sub_leg = Leg(initial_surface, first_leg.surface_2)
        if self.standing_wave:
            last_leg = self.legs[-1]
            last_leg_first_sub_leg = Leg(last_leg.surface_1, initial_surface)
            last_leg_second_sub_leg = Leg(initial_surface, last_leg.surface_2)
            legs_list = [first_leg_second_sub_leg] + self.legs[1:-1] + [last_leg_first_sub_leg,
                                                                        last_leg_second_sub_leg,
                                                                        first_leg_first_sub_leg]
        else:
            legs_list = [first_leg_second_sub_leg] + self.legs[1:] + [first_leg_first_sub_leg]
        self.legs = legs_list
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

        if isinstance(self.legs[0].surface_1, PhysicalSurface):
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
             camera_center: int = -1,
             ray_list: Optional[List[Ray]] = None,
             dim: int = 3):

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
            origin_camera = self.legs[camera_center].surface_1.center

        ax.set_xlim(origin_camera[0] - axis_span, origin_camera[0] + axis_span)
        ax.set_ylim(origin_camera[1] - axis_span, origin_camera[1] + axis_span)

        if ray_list is None and self.central_line is not None:
            ray_list = self.central_line

        if ray_list is not None:
            if dim == 3:
                ax.set_zlim(origin_camera[2] - axis_span, origin_camera[2] + axis_span)
                ax.plot(ray_list[0].origin[0], ray_list[0].origin[1], ray_list[0].origin[2], 'go',
                        label='beginning')
                ax.plot(ray_list[-1].origin[0], ray_list[-1].origin[1], ray_list[-1].origin[2], 'ro',
                        label='end')
            else:
                ax.plot(ray_list[0].origin[0], ray_list[0].origin[1], 'go', label='beginning')
                ax.plot(ray_list[-1].origin[0], ray_list[-1].origin[1], 'ro', label='end')

            plt.legend()

            for ray in ray_list:
                ray.plot(ax=ax, dim=dim)
        for i, surface in enumerate(self.surfaces):
            surface.plot(ax=ax, name=str(i), dim=dim)

        if self.lambda_laser is not None:
            spot_size_lines = self.generate_spot_size_lines(lambda_laser=self.lambda_laser, dim=dim)
            for line in spot_size_lines:
                if dim == 2:
                    ax.plot(line[:, 0], line[:, 1], 'r--', alpha=0.8, linewidth=0.5)
                else:
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r--', alpha=0.8, linewidth=0.5)

        return ax

    def calculated_shifted_cavity_overlap_integral(self, parameter_index: Tuple[int, int],
                                                   shift: Union[float, np.ndarray]):
        params = self.to_params
        shift_input_is_float = isinstance(shift, float)
        if shift_input_is_float:
            shift = np.array([shift])
        overlaps = np.zeros_like(shift)
        for i, shift_value in enumerate(shift):
            new_params = copy.copy(params)
            new_params[parameter_index] = params[parameter_index] + shift_value
            new_cavity = Cavity.from_params(params=new_params, standing_wave=self.standing_wave,
                                            lambda_laser=self.lambda_laser)
            try:
                overlap = calculate_cavities_overlap_matrices(cavity_1=self, cavity_2=new_cavity)
            except np.linalg.LinAlgError:
                continue
            overlaps[i] = np.abs(overlap)
        if shift_input_is_float:
            overlaps = overlaps[0]
        return overlaps

    def calculate_shift_threshold(self, parameter_index: Tuple[int, int], lambda_laser: float, threshold: float):
        def overlap_of_shift(shift: float):
            return threshold - self.calculated_shifted_cavity_overlap_integral(parameter_index=parameter_index,
                                                                               shift=shift)

        return optimize.root_scalar(overlap_of_shift, x0=0, x1=0.001)


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
    b = 1j * k * np.array(
        [(k_hat @ t_hat) + 2 * (r_0 @ u_hat) * (t_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (t_hat @ v_hat) / q_v,
         (k_hat @ p_hat) + 2 * (r_0 @ u_hat) * (p_hat @ u_hat) / q_u + 2 * (r_0 @ v_hat) * (p_hat @ v_hat) / q_v])
    c = 1j * k * (k_hat @ r_0) + (r_0 @ u_hat) ** 2 / q_u + (r_0 @ v_hat) ** 2 / q_v

    return A, b, c


def calculate_cavities_overlap_matrices(cavity_1: Cavity, cavity_2: Cavity):
    for cavity in [cavity_1, cavity_2]:
        if cavity.legs[0].mode_parameters is None:
            cavity.set_mode_parameters()
    mode_parameters_1 = cavity_1.legs[0].mode_parameters
    mode_parameters_2 = cavity_2.legs[0].mode_parameters

    cavity_1_waist_pos = mode_parameters_1.center[0, :]
    P1 = FlatSurface(center=cavity_1_waist_pos, outwards_normal=mode_parameters_1.k_vector)

    A_1, b_1, c_1 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_1)
    A_2, b_2, c_2 = calculate_gaussian_parameters_on_surface(P1, mode_parameters_2)

    return gaussians_overlap_integral_matrices(A_1, A_2, b_1, b_2, c_1, c_2)


def calculate_cavities_overlap(cavity_1: Cavity, cavity_2: Cavity, lambda_laser: float):
    # Initialize all required parameters:
    cavity_1.find_central_line()
    cavity_2.find_central_line()
    cavity_1.set_mode_parameters()
    cavity_2.set_mode_parameters()

    # The two modes of the laser in the first leg of the cavity:
    mode_parameter_1 = cavity_1.legs[0].mode_parameters
    mode_parameter_2 = cavity_2.legs[0].mode_parameters

    cavity_1_waist_pos = mode_parameter_1.center[0, :]

    # The local mode parameters of the laser in the first leg of the cavity at the waist the cavity_1:
    local_mode_parameters_1 = cavity_1.legs[0].local_mode_parameters_on_a_point(cavity_1_waist_pos)
    local_mode_parameters_2 = cavity_2.legs[0].local_mode_parameters_on_a_point(cavity_1_waist_pos)

    # The plane that is perpendicular to the laser in cavity 1 and centered at its waist:
    P1 = FlatSurface(center=cavity_1_waist_pos, outwards_normal=mode_parameter_1.k_vector)

    # The widths in both dimensions of the lasers at the waist of cavity_1:
    width_t_1, width_p_1 = w_0_of_z_R(local_mode_parameters_1.z_R, lambda_laser=lambda_laser)
    width_t_2, width_p_2 = w_of_z_R(z=(cavity_1_waist_pos - cavity_2.legs[0].central_line.origin) @
                                      mode_parameter_2.k_vector,
                                    z_R=local_mode_parameters_2.z_R,
                                    lambda_laser=lambda_laser)

    # The intersection of the laser with the plane P1:
    # t_1, p_1 = 0, 0
    t_2, p_2 = P1.get_parameterization(P1.find_intersection_with_ray(mode_parameter_2.ray))
    # Each one of those returns as an array of two identical values (because there are different origins for the two
    # axes, thus there are two rays) but they coincide and so the values are identical and we can drop one of them:):
    t_2 = t_2[0]
    p_2 = p_2[0]

    # The vectors that are perpendicular to the lasers, the first is perpendicular to the central line plane,
    # and the other is inside it: (in the first cavity they are orthonormal basis to P1)
    t_1_vec_and_p_1_vec = cavity_1.legs[0].mode_principle_axes
    t_2_vec_and_p_2_vec = cavity_2.legs[0].mode_principle_axes
    t_1_vec = t_1_vec_and_p_1_vec[0, :]
    p_1_vec = t_1_vec_and_p_1_vec[1, :]
    t_2_vec = t_2_vec_and_p_2_vec[0, :]
    p_2_vec = t_2_vec_and_p_2_vec[1, :]

    # The projection of t_2 on P1:
    t_2_vec_projected_on_P1 = t_2_vec - np.dot(t_2_vec, mode_parameter_1.k_vector) * mode_parameter_1.k_vector
    p_2_vec_projected_on_P1 = p_2_vec - np.dot(p_2_vec, mode_parameter_1.k_vector) * mode_parameter_1.k_vector

    # the angle between t_1 and t_2:
    cos_theta_rotation = normalize_vector(t_2_vec_projected_on_P1) @ t_1_vec
    theta_rotation = np.arccos(cos_theta_rotation)

    # The projection of k_2 the principal axes of P1:  # CHANGE IT - REQUIRES K_VECTOR TO BE NOT NORMALIZED
    k_p = mode_parameter_2.k_vector @ p_1_vec * 2 * np.pi / lambda_laser
    k_t = mode_parameter_2.k_vector @ t_1_vec * 2 * np.pi / lambda_laser

    # The effective widths of the second lase in the plane P1: (since this plane is not necessarily perpendicular to the
    # laser, the widths appear shrank in that plane
    width_t_2_eff = width_t_2 * np.linalg.norm(t_2_vec_projected_on_P1)
    width_p_2_eff = width_p_2 * np.linalg.norm(p_2_vec_projected_on_P1)

    return gaussians_overlap_integral_v2(width_t_1, width_p_1, width_t_2_eff, width_p_2_eff, t_2, p_2, k_t, k_p,
                                      theta_rotation)


def gaussian_integral_2d(a_x, b_x, k_x, a_y, b_y, k_y, a, c):
    repetitive_denomenator = 4 * a_x * a_y - a ** 2

    root = 2 * np.pi * repetitive_denomenator ** (-1 / 2)
    exponent = np.exp((a_y * (b_x ** 2 - k_x ** 2) - a * (b_x * b_y - k_x * k_y) + a_x * (
            b_y ** 2 - k_y ** 2)) / repetitive_denomenator + c)
    cosine = np.cos(
        (2 * a_y * b_x * k_x - a * b_x * k_y - a * b_y * k_x + 2 * a_x * b_y * k_y) / repetitive_denomenator)

    return root * exponent * cosine


def gaussians_overlap_integral_v2(w_x_1, w_y_1, w_x_2, w_y_2, x_2, y_2, k_x, k_y, theta):
    a_x = 1 / (w_x_1 ** 2) + (np.cos(theta) ** 2) / (w_x_2 ** 2) + (np.sin(theta) ** 2) / (w_y_2 ** 2)
    b_x = -2 * x_2 / (w_x_1 ** 2)
    a_y = 1 / (w_y_1 ** 2) + (np.sin(theta) ** 2) / (w_x_2 ** 2) + (np.cos(theta) ** 2) / (w_y_2 ** 2)
    b_y = -2 * y_2 / (w_y_1 ** 2)
    a = (1 / (w_x_2 ** 2) - 1 / (w_y_2 ** 2)) * np.sin(2 * theta)
    c = -(x_2 ** 2) / (w_x_1 ** 2) - (y_2 ** 2) / (w_y_1 ** 2)

    normalization_factor = gaussian_norm(w_x_1, w_y_1, 0, 0, 0) * gaussian_norm(w_x_2, w_y_2, k_x, k_y, theta)

    return gaussian_integral_2d(a_x, b_x, k_x, a_y, b_y, k_y, a, c) / normalization_factor


def gaussian_norm(w_x, w_y, k_x, k_y, theta):
    a_x = 2 * (np.cos(theta) ** 2 / w_x ** 2 + np.sin(theta) ** 2 / w_y ** 2)
    a_y = 2 * (np.sin(theta) ** 2 / w_x ** 2 + np.cos(theta) ** 2 / w_y ** 2)
    a = 2 * np.sin(2 * theta) * (1 / w_x ** 2 - 1 / w_y ** 2)
    return np.sqrt((1 / 2) * (gaussian_integral_2d(a_x, 0, 0, a_y, 0, 0, a, 0) +
                              gaussian_integral_2d(a_x, 0, 2 * k_x, a_y, 0, 2 * k_y, a, 0)))


def gaussian_norm_matrices_log(A: np.ndarray, b: np.ndarray, c: float):
    return 1/2 * gaussian_integral_2d_matrices_log(A + np.conjugate(A), b + np.conjugate(b), c + np.conjugate(c))


def gaussian_integral_2d_matrices_log(A: np.ndarray, b: np.ndarray, c):
    # The integral over exp( x.T A_2 x + b.T x + c):
    eigen_values = np.linalg.eigvals(A)
    A_inv = np.linalg.inv(A)
    dim = A.shape[0]
    log_integral = np.log(np.sqrt((2 * np.pi) ** dim / (np.prod(eigen_values)))) + 0.5 * b.T @ A_inv @ b + c
    return log_integral


def gaussians_overlap_integral_matrices(A_1: np.ndarray, A_2: np.ndarray,
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
    normalization_factor_log = gaussian_norm_matrices_log(A_1, b_1, c_1) + gaussian_norm_matrices_log(A_2, b_2, c_2)
    integral_normalized_log = gaussian_integral_2d_matrices_log(A, b, c) - normalization_factor_log
    return np.exp(integral_normalized_log)


def evaluate_gaussian(A: np.ndarray, b: np.ndarray, c: float, axis_span: float):
    N = 100
    x = np.linspace(-axis_span, axis_span, N)
    y = np.linspace(-axis_span, axis_span, N)
    X, Y = np.meshgrid(x, y)
    R = np.stack([X, Y], axis=2)
    mu = np.array([x_2, y_2])
    R_shifted = R - mu[None, None, :]
    R_normed_squared = np.einsum('ijk,kl,ijl->ij', R_shifted, A, R_shifted)
    functions_values = np.exp(-(1 / 2) * R_normed_squared + np.einsum('k,ijk->ij', b, R) + c)
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
                     c_1: float, c_2: float, axis_span: float = 0.0005, title: Optional[str] = ''):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    plot_gaussian_subplot(A_1, b_1, c_1, axis_span, fig, ax[0])
    plot_gaussian_subplot(A_2, b_2, c_2, axis_span, fig, ax[1])
    if title is not None:
        fig.suptitle(title)


def plot_2_gaussians_colors(A_1: np.ndarray, A_2: np.ndarray,
                     # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
                     b_1: np.ndarray, b_2: np.ndarray,
                     c_1: float, c_2: float, axis_span: float = 0.0005, title: Optional[str] = '', real_or_abs: str = 'real'):
    first_gaussian_values = evaluate_gaussian(A_1, b_1, c_1, axis_span)
    second_gaussian_values = evaluate_gaussian(A_2, b_2, c_2, axis_span)
    third_color_channel = np.zeros_like(first_gaussian_values)
    rgb_image = np.stack([first_gaussian_values, second_gaussian_values, third_color_channel], axis=2)
    if real_or_abs == 'abs':
        rgb_image = np.abs(rgb_image)
    else:
        rgb_image = np.real(rgb_image)
    plt.imshow(rgb_image)
    plt.title(title)


if __name__ == '__main__':
    x_1 = 0.12  # 0.124167242
    y_1 = 0.000e+00
    t_1 = 0.000e+00
    p_1 = 0.000e+00
    r_1 = 0.05
    x_2 = 0.000e+00
    y_2 = 0.000e+00
    t_2 = 0.000e+00
    p_2 = 0.000e+00
    r_2 = 1.1e-1
    w_2 = 3e-3
    n_out = 1e+00
    n_in = 1.500e+00
    x_3 = -0.12
    y_3 = 0.000e+00
    t_3 = 0.000e+00
    p_3 = 0.000e+00
    r_3 = 0.05
    lambda_laser = 5.32e-07
    elev = 38.00
    azim = 168.00
    axis_span = 1.1*x_1
    focus_point = -1
    set_initial_surface = False
    dim = 2

    y_1 += 0
    t_1 += 0
    p_1 += 0
    x_2 += 0
    y_2 += 0
    t_2 += 0
    p_2 += np.pi
    x_3 += 0
    y_3 += 0
    t_3 += 0
    p_3 += np.pi

    mirror_1 = CurvedMirror.from_params(np.array([x_1, y_1, 0, t_1, p_1, r_1, 1, 0, 0, 0]))
    mirror_3 = CurvedMirror.from_params(np.array([x_3, y_3, 0, t_3, p_3, r_3, 1, 0, 0, 0]))
    lens_2_right, lens_2_left = generate_lens(radius=r_2,
                                              thickness=w_2,
                                              n_out=n_out,
                                              n_in=n_in,
                                              center=np.array([x_2, y_2, 0]),
                                              direction=unit_vector_of_angles(t_2, p_2))

    cavity = Cavity([mirror_1, lens_2_right, lens_2_left, mirror_3], set_initial_surface=False, standing_wave=True,
                    lambda_laser=lambda_laser)
    # cavity.find_central_line()
    central_line = cavity.trace_ray(cavity.default_initial_ray)
    for i, leg in enumerate(cavity.legs):
        leg.central_line = central_line[i]
    cavity.central_line_successfully_traced = True
    cavity.set_mode_parameters()

    ax = cavity.plot(dim=dim, axis_span=axis_span, camera_center=focus_point)
    # ax.set_xlim(- axis_span, axis_span)
    # ax.set_ylim(-0.0025, 0.0025)

    if dim == 3:
        ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"NA_out of plane = {cavity.legs[0].mode_parameters.NA[0]:.2e} , NA_in plane = {cavity.legs[0].mode_parameters.NA[1]:.2e}")

    # for mirror in cavity.physical_surfaces:
    #     ax.plot(mirror.origin[0], mirror.origin[1], 'ro')

    plt.show()
    fig_2, ax_2 = plt.subplots(2, 1)
    shifts_r_1 = np.linspace(-0.01, 0.02, 100)
    overlaps_r_1 = cavity.calculated_shifted_cavity_overlap_integral((0, 5), shifts_r_1)

    shifts_p_1 = np.linspace(-0.02, 0.02, 100)
    overlaps_p_1 = cavity.calculated_shifted_cavity_overlap_integral((0, 4), shifts_p_1)

    ax_2[0].plot(shifts_r_1, overlaps_r_1, label="r_1 stability")
    ax_2[1].plot(shifts_p_1, overlaps_p_1, label="p_1 stability")
    ax_2[0].legend()
    ax_2[1].legend()
    plt.show()


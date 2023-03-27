import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy import optimize
import warnings


rotation_matrix_aournd_z = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                                                   [np.sin(theta), np.cos(theta), 0],
                                                   [0, 0, 1]])  # 3 | 3

rotation_matrix_aournd_x = lambda theta: np.array([[1, 0, 0],
                                                   [0, np.cos(theta), -np.sin(theta)],
                                                   [0, np.sin(theta), np.cos(theta)]])  # 3 | 3

rotation_matrix_aournd_n = lambda theta, n: np.array([[np.cos(theta) + n[0] ** 2 * (1 - np.cos(theta)),
                                                       n[0] * n[1] * (1 - np.cos(theta)) - n[2] * np.sin(theta),
                                                       n[0] * n[2] * (1 - np.cos(theta)) + n[1] * np.sin(theta)],
                                                      [n[1] * n[0] * (1 - np.cos(theta)) + n[2] * np.sin(theta),
                                                       np.cos(theta) + n[1] ** 2 * (1 - np.cos(theta)),
                                                       n[1] * n[2] * (1 - np.cos(theta)) - n[0] * np.sin(theta)],
                                                      [n[2] * n[0] * (1 - np.cos(theta)) - n[1] * np.sin(theta),
                                                       n[2] * n[1] * (1 - np.cos(theta)) + n[0] * np.sin(theta),
                                                       np.cos(theta) + n[2] ** 2 * (1 - np.cos(theta))]])  # 3 | 3


def unit_vector_of_angles(theta: Union[np.ndarray, float], phi: Union[np.ndarray, float]) -> np.ndarray:
    # theta and phi are assumed to be in radians
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)

def angles_of_unit_vector(unit_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # theta and phi are returned in radians
    theta = np.arccos(unit_vector[..., 2])
    phi = np.arctan2(unit_vector[..., 1], unit_vector[..., 0])
    return theta, phi

def angles_distance(direction_vector_1: np.ndarray, direction_vector_2: np.ndarray):
    inner_product = np.sum(direction_vector_1 * direction_vector_2, axis=-1)
    inner_product = np.clip(inner_product, -1, 1)
    return np.arccos(inner_product)


class Ray:
    def __init__(self, origin: np.ndarray, k_vector: np.ndarray,
                 length: Optional[Union[np.ndarray, float]] = None):
        # if origin.ndim == 1:
        #     origin = origin[np.newaxis, :]

        if k_vector.ndim == 1 and origin.shape[0] > 1:
            k_vector = np.tile(k_vector, (*origin.shape[:-1], 1))
        elif origin.ndim == 1 and k_vector.shape[0] > 1:
            origin = np.tile(origin, (*k_vector.shape[:-1], 1))

        self.origin = origin  # m_rays | 3
        self.k_vector = k_vector / np.linalg.norm(k_vector, axis=-1)[..., np.newaxis]  # m_rays | 3
        if length is not None and isinstance(length, float):
            length = np.ones(origin.shape[0]) * length
        self.length = length  # m_rays or None

    def parameterization(self, t: Union[np.ndarray, float]) -> np.ndarray:
        # Currently this function allows only one t per ray. if needed it can be extended to allow multiple t per ray.
        # t needs to be either a float or a numpy array with dimensions m_rays
        return self.origin + t[..., np.newaxis] * self.k_vector

    def plot(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if self.length is None:
            length = np.ones_like(self.origin[..., 0])
        else:
            length = self.length
        ray_origin_reshaped = self.origin.reshape(-1, 3)
        ray_k_vector_reshaped = self.k_vector.reshape(-1, 3)
        lengths_reshaped = length.reshape(-1)
        [ax.plot([ray_origin_reshaped[i, 0], ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0]],
                    [ray_origin_reshaped[i, 1], ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1]],
                    [ray_origin_reshaped[i, 2], ray_origin_reshaped[i, 2] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 2]],
                    color='red', linewidth=0.4)
            for i in range(ray_origin_reshaped.shape[0])]


class Mirror:
    def reflect_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError

    def plot(self, ax: Optional[plt.Axes] = None):
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        raise NotImplementedError

    @property
    def center_of_mirror(self):
        raise NotImplementedError

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        raise NotImplementedError


class CurvedMirror(Mirror):
    def __init__(self, radius: float,
                 outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
                 center_of_mirror: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
                 # the plate.
                 origin: Optional[np.ndarray] = None  # The center of the sphere.
                 ):
        self.outwards_normal = outwards_normal / np.linalg.norm(outwards_normal)
        self.radius = radius

        if origin is None and center_of_mirror is None:
            raise ValueError('Either origin or center_of_mirror must be provided.')
        elif origin is not None and center_of_mirror is not None:
            raise ValueError('Only one of origin and center_of_mirror must be provided.')
        elif origin is None:
            self.origin = center_of_mirror + radius * self.outwards_normal
        else:
            self.origin = origin

    @property
    def inwards_normal(self):
        return - self.outwards_normal  # Pointing from the center of the mirror to the origin of the sphere.

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        # Result of the next line of mathematica to find the intersection:
        # Solve[(x0 + kx * t - xc) ^ 2 + (y0 + ky * t - yc) ^ 2 + (z0 + kz * t - zc) ^ 2 == R ^ 2, t]
        t = (-ray.k_vector[..., 0] * ray.origin[..., 0] + ray.k_vector[..., 0] * self.origin[0] - ray.k_vector[
            ..., 1] *
             ray.origin[..., 1] + ray.k_vector[..., 1] * self.origin[1] - ray.k_vector[..., 2] * ray.origin[..., 2] +
             ray.k_vector[..., 2] * self.origin[2] + np.sqrt(
                    -4 * (ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2) * (
                            -self.radius ** 2 + (ray.origin[..., 0] - self.origin[0]) ** 2 + (
                            ray.origin[..., 1] - self.origin[1]) ** 2 + (
                                    ray.origin[..., 2] - self.origin[2]) ** 2) + 4 * (
                            ray.k_vector[..., 0] * (ray.origin[..., 0] - self.origin[0]) + ray.k_vector[..., 1] * (
                            ray.origin[..., 1] - self.origin[1]) + ray.k_vector[..., 2] * (
                                    ray.origin[..., 2] - self.origin[2])) ** 2) / 2) / (
                    ray.k_vector[..., 0] ** 2 + ray.k_vector[..., 1] ** 2 + ray.k_vector[..., 2] ** 2)
        return ray.parameterization(t)

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        # Notice that this function does not reflect along the normal of the mirror but along the normal projection
        # of the ray on the mirror.
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        mirror_normal_vector = self.origin - intersection_point  # m_rays | 3
        mirror_normal_vector = mirror_normal_vector / np.linalg.norm(mirror_normal_vector, axis=-1)[..., np.newaxis]
        dot_product = np.sum(ray.k_vector * mirror_normal_vector, axis=-1)  # m_rays  # This dot product is written
        # like so because both tensors have the same shape and the dot product is calculated along the last axis.
        # you could also perform this product by transposing the second tensor and then dot multiplying the two tensors,
        # but this it would be cumbersome to do so.
        reflected_direction_vector = ray.k_vector - 2 * dot_product[..., np.newaxis] * mirror_normal_vector  # m_rays | 3
        return reflected_direction_vector

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray, intersection_point)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return Ray(intersection_point, reflected_direction_vector)

    @property
    def center_of_mirror(self):
        return self.origin - self.radius * self.inwards_normal

    def get_spanning_vectors(self):
        # For the case of the sphere with normal on the x-axis, those will be the y and z axis.
        # For the case of the sphere with normal on the y-axis, those will be the x and z axis.
        pseudo_y = np.cross(np.array([0, 0, 1]), self.outwards_normal)
        pseudo_z = np.cross(self.outwards_normal, pseudo_y)  # Should be approximately equal to \hat{z}, and exactly
        # equal if the normal is in the x-y plane.
        return pseudo_y, pseudo_z

    def parameterization(self, t: np.ndarray,  # the same as p but with theta, and it ranges in [-pi/2, pi/2]
                         # instead of [0, pi]. This is because I want to treat the center of the mirror as the origin
                         # of the parameterization.
                         p: np.ndarray  # this is the spherical phi measured as if as the center of the mirror
                         # was on the x-axis
                         ) -> np.ndarray:
        # This parameterization treats the sphere as if as the center of the mirror was on the x-axis.
        # The conceptual difference between this parameterization and the classical one of [sin(theta)cos(phi),
        # sin(theta)sin(phi), cos(theta)]] is that here there is barely any Jacobian determinant.
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        # Notice how the order of rotations matters. First we rotate around the z axis, then around the y-axis.
        # Doing it the other way around would give parameterization that is not aligned with the conventional theta, phi
        # parameterization. This is important for the get_parameterization method.
        rotation_matrix = rotation_matrix_aournd_n(pseudo_y, p) @ rotation_matrix_aournd_n(pseudo_z, t)
        return self.origin + self.radius * rotation_matrix @ self.outwards_normal

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        normalized_points = (points - self.origin) / self.radius
        p = np.arctan2(points @ pseudo_y, points @ self.outwards_normal)
        # Notice that t is like theta but instead of ranging in [0, pi] it ranges in [-pi/2, pi/2].
        t = np.arcsin(np.clip(normalized_points @ pseudo_z, -1, 1))
        return t, p

    def plot(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(-0.3, 0.3, 100)
        p = np.linspace(-0.3, 0.3, 100)
        T, P = np.meshgrid(t, p)
        points = self.parameterization(T, P)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        ax.plot_surface(x, y, z, color='b')
        ax.plot([self.center_of_mirror[0], self.origin[0]], [self.center_of_mirror[1], self.origin[1]],
                [self.center_of_mirror[2], self.origin[2]], 'g-')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


class FlatMirror(Mirror):
    def __init__(self, normal: np.ndarray, distance_from_origin: float):
        self.normal = normal / np.linalg.norm(normal)
        self.distance_from_origin = distance_from_origin

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        A_vec = self.normal * self.distance_from_origin
        BA_vec = A_vec - ray.origin
        BC = BA_vec @ self.normal
        cos_theta = ray.k_vector @ self.normal
        t = BC / cos_theta
        intersection_point = ray.parameterization(t)
        return intersection_point

    def reflect_direction(self, ray: Ray) -> np.ndarray:
        dot_product = ray.k_vector @ self.normal  # m_rays
        k_projection_on_normal = dot_product[..., np.newaxis] * self.normal
        reflected_direction = ray.k_vector - 2 * k_projection_on_normal
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return Ray(intersection_point, reflected_direction_vector)

    @property
    def center_of_mirror(self):
        return self.normal * self.distance_from_origin

    def get_spanning_vectors(self):
        pseudo_y = np.cross(np.array([0, 0, 1]), self.normal)
        pseudo_z = np.cross(self.normal, pseudo_y)
        return pseudo_y, pseudo_z

    def parameterization(self, t, p):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        points = self.center_of_mirror + t[..., np.newaxis] * pseudo_y + p[..., np.newaxis] * pseudo_z
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        t = (points - self.center_of_mirror) @ pseudo_y
        p = (points - self.center_of_mirror) @ pseudo_z
        return t, p


    def plot(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(-2, 2, 100)
        s = np.linspace(-2, 2, 100)
        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        ax.plot_surface(x, y, z, color='b')
        ax.plot([0, self.center_of_mirror[0]],
                [0, self.center_of_mirror[1]],
                [0, self.center_of_mirror[2]], 'g-')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


class Cavity:
    def __init__(self, mirrors: List[Mirror]):
        mirrors.append(mirrors[0])
        self.mirrors = mirrors
        self.ray_history: List[Ray] = []

    def trace_ray(self, ray: Ray) -> Ray:
        self.ray_history = [ray]
        for mirror in self.mirrors[1:]:
            ray = mirror.reflect_ray(ray)
            self.ray_history.append(ray)
        return ray

    def find_central_line(self) -> Tuple[np.ndarray, bool]:
        def f_roots(starting_position_and_angles: np.ndarray) -> np.ndarray:
            t_i, p_i, theta_i, phi_i = starting_position_and_angles
            k_vector_i = unit_vector_of_angles(theta_i, phi_i)
            origin_i = self.mirrors[0].parameterization(t_i, p_i)
            input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
            output_ray = self.trace_ray(input_ray)
            final_intersection_point = self.mirrors[0].find_intersection_with_ray(self.ray_history[-2])
            t_o, p_o = self.mirrors[0].get_parameterization(final_intersection_point)
            theta_o, phi_o = angles_of_unit_vector(output_ray.k_vector)
            return np.array([t_o - t_i, p_o-p_i, theta_o, phi_o])

        initial_k_vector = self.mirrors[1].center_of_mirror - self.mirrors[0].center_of_mirror
        initial_k_vector /= np.linalg.norm(initial_k_vector)
        theta_initial_guess, phi_initial_guess = angles_of_unit_vector(initial_k_vector)

        initial_guess = np.array([0, 0, theta_initial_guess, phi_initial_guess])

        final_position_and_angles: np.ndarray = optimize.fsolve(f_roots, initial_guess)

        if np.linalg.norm(f_roots(final_position_and_angles)) > 1e-6:
            success = False
        else:
            success = True
        k_vector_solution = unit_vector_of_angles(final_position_and_angles[2], final_position_and_angles[3])
        origin_solution = self.mirrors[0].parameterization(final_position_and_angles[0], final_position_and_angles[1])
        central_line = Ray(origin_solution, k_vector_solution)
        self.trace_ray(central_line)
        return final_position_and_angles, success

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for ray in self.ray_history:
            ray.plot(ax)
        for mirror in self.mirrors:
            mirror.plot(ax)
        ax.set_xlim3d(-3, 3)
        ax.set_ylim3d(-3, 3)
        ax.set_zlim3d(-3, 3)

    @property
    def final_ray(self):
        if len(self.ray_history) == 0:
            return None
        else:
            return self.ray_history[-1]

if __name__ == '__main__':
    x_1 = 1
    y_1 = 0.00
    r_1 = 2
    t_1 = -np.pi / 6
    x_2 = 0
    y_2 = np.sqrt(3)
    r_2 = 2
    t_2 = np.pi / 2
    x_3 = -1
    y_3 = 0
    r_3 = 2
    t_3 = 7 * np.pi / 6

    global_origin = np.array([0, 0, 0])
    normal_1 = unit_vector_of_angles(np.pi/2, t_1)
    center_1 = np.array([x_1, y_1, 0])
    normal_2 = unit_vector_of_angles(np.pi/2, t_2)
    center_2 = np.array([x_2, y_2, 0])
    normal_3 = unit_vector_of_angles(np.pi/2, t_3)
    center_3 = np.array([x_3, y_3, 0])

    mirror_1 = CurvedMirror(radius=2, outwards_normal=normal_1, center_of_mirror=center_1)
    mirror_2 = CurvedMirror(radius=2, outwards_normal=normal_2, center_of_mirror=center_2)
    mirror_3 = CurvedMirror(radius=2, outwards_normal=normal_3, center_of_mirror=center_3)

    mirror_1.plot()
    plt.show()
    # cavity = Cavity([mirror_1, mirror_2, mirror_3])
    # t_and_theta_central_line, success = cavity.find_central_line()

    # cavity.plot()
    # plt.show()


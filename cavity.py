import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy import optimize
import warnings
import time


# Throughout the code, all tensors can take any number of dimensions, but the last dimension is always the coordinate
# dimension. this allows a Ray to be either a single ray, a list of rays, or a list of lists of rays, etc.
# For example, a Ray could be a set of rays with a starting point for every combination of x, y, z. in this case, the
# ray.origin tensor will be of the size N_x | N_y | N_z | 3.


def ABCD_free_space(length: float) -> np.ndarray:
    return np.array([[1, length, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, length],  #  / cos_theta_between_planes
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


def unit_vector_of_angles(theta: Union[np.ndarray, float], phi: Union[np.ndarray, float]) -> np.ndarray:
    # theta and phi are assumed to be in radians
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def angles_of_unit_vector(unit_vector: Union[np.ndarray, float]) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
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
        self.k_vector = normalize_vector(k_vector)  # m_rays | 3
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
        [ax.plot(
            [ray_origin_reshaped[i, 0], ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0]],
            [ray_origin_reshaped[i, 1], ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1]],
            [ray_origin_reshaped[i, 2], ray_origin_reshaped[i, 2] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 2]],
            color='red', linewidth=1)
            for i in range(ray_origin_reshaped.shape[0])]


class Mirror:
    def __init__(self, outwards_normal: np.ndarray):
        self.outwards_normal = normalize_vector(outwards_normal)

    def reflect_ray(self, ray: Ray) -> Ray:
        raise NotImplementedError

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        raise NotImplementedError

    @property
    def center_of_mirror(self):
        raise NotImplementedError

    @property
    def inwards_normal(self):
        raise -self.outwards_normal

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        raise NotImplementedError

    def ABCD_matrix(self, cos_theta_incoming: float) -> np.ndarray:
        raise NotImplementedError


class FlatMirror(Mirror):
    def __init__(self,
                 outwards_normal: np.ndarray,
                 distance_from_origin: Optional[float] = None,
                 center_of_mirror: Optional[np.ndarray] = None):
        super().__init__(outwards_normal)
        if distance_from_origin is None and center_of_mirror is None:
            raise ValueError('Either distance_from_origin or center_of_mirror must be specified')
        if distance_from_origin is not None and center_of_mirror is not None:
            raise ValueError('Only one of distance_from_origin or center_of_mirror must be specified')
        if distance_from_origin is not None:
            self.distance_from_origin = distance_from_origin
            self.center_of_mirror_private = self.outwards_normal * distance_from_origin
        if center_of_mirror is not None:
            self.center_of_mirror_private = center_of_mirror
            self.distance_from_origin = center_of_mirror @ self.outwards_normal

    def ABCD_matrix(self, cos_theta_incoming: float) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, -1]])

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        A_vec = self.outwards_normal * self.distance_from_origin
        BA_vec = A_vec - ray.origin
        BC = BA_vec @ self.outwards_normal
        cos_theta = ray.k_vector @ self.outwards_normal
        t = BC / cos_theta
        intersection_point = ray.parameterization(t)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        return intersection_point

    def reflect_direction(self, ray: Ray) -> np.ndarray:
        dot_product = ray.k_vector @ self.outwards_normal  # m_rays
        k_projection_on_normal = dot_product[..., np.newaxis] * self.outwards_normal
        reflected_direction = ray.k_vector - 2 * k_projection_on_normal
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray)
        return Ray(intersection_point, reflected_direction_vector)

    @property
    def center_of_mirror(self):
        # The reason for this property is that in other Mirror classes it is a property.
        return self.center_of_mirror_private

    def get_spanning_vectors(self):
        pseudo_y = np.cross(np.array([0, 0, 1]), self.outwards_normal)
        pseudo_z = np.cross(self.outwards_normal, pseudo_y)
        return pseudo_y, pseudo_z

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        if isinstance(t, (float, int)):
            t = np.array(t)
        if isinstance(p, (float, int)):
            p = np.array(p)
        points = self.center_of_mirror + t[..., np.newaxis] * pseudo_z + p[..., np.newaxis] * pseudo_y
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        t = (points - self.center_of_mirror) @ pseudo_y
        p = (points - self.center_of_mirror) @ pseudo_z
        return t, p

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(-0.3, 0.3, 100)
        s = np.linspace(-0.3, 0.3, 100)
        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        ax.plot_surface(x, y, z, color='b')
        ax.plot([0, self.center_of_mirror[0]],
                [0, self.center_of_mirror[1]],
                [0, self.center_of_mirror[2]], 'g-')
        if name is not None:
            name_position = self.parameterization(0.4, 0)
            ax.text(name_position[0], name_position[1], name_position[2], s=name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')


class CurvedMirror(Mirror):
    def __init__(self, radius: float,
                 outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
                 center_of_mirror: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
                 # the plate.
                 origin: Optional[np.ndarray] = None  # The center of the sphere.
                 ):
        super().__init__(outwards_normal)
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
        ray.length = t
        return ray.parameterization(t)

    def reflect_direction(self, ray: Ray, intersection_point: Optional[np.ndarray] = None) -> np.ndarray:
        # Notice that this function does not reflect along the normal of the mirror but along the normal projection
        # of the ray on the mirror.
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray)
        mirror_normal_vector = self.origin - intersection_point  # m_rays | 3
        mirror_normal_vector = normalize_vector(mirror_normal_vector)
        dot_product = np.sum(ray.k_vector * mirror_normal_vector, axis=-1)  # m_rays  # This dot product is written
        # like so because both tensors have the same shape and the dot product is calculated along the last axis.
        # you could also perform this product by transposing the second tensor and then dot multiplying the two tensors,
        # but this it would be cumbersome to do so.
        reflected_direction_vector = ray.k_vector - 2 * dot_product[
            ..., np.newaxis] * mirror_normal_vector  # m_rays | 3
        return reflected_direction_vector

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.find_intersection_with_ray(ray)
        reflected_direction_vector = self.reflect_direction(ray, intersection_point)
        return Ray(intersection_point, reflected_direction_vector)

    @property
    def center_of_mirror(self):
        return self.origin - self.radius * self.inwards_normal

    def get_spanning_vectors(self):
        # For the case of the sphere with normal on the x-axis, those will be the y and z axis.
        # For the case of the sphere with normal on the y-axis, those will be the x and z axis.
        pseudo_y = np.cross(np.array([0, 0, 1]), self.outwards_normal)
        pseudo_z = np.cross(self.outwards_normal, pseudo_y)  # Should be approximately equal to \hat{z}, and exactly
        # equal if the outwards_normal is in the x-y plane.
        return pseudo_y, pseudo_z

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
        rotation_matrix = rotation_matrix_around_n(pseudo_y, t / self.radius) @ \
                          rotation_matrix_around_n(pseudo_z, p / self.radius)
        return self.origin + self.radius * rotation_matrix @ self.outwards_normal

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        normalized_points = (points - self.origin) / self.radius
        p = np.arctan2(normalized_points @ pseudo_y, normalized_points @ self.outwards_normal) * self.radius
        # Notice that t is like theta but instead of ranging in [0, pi] it ranges in [-pi/2, pi/2].
        t = np.arcsin(np.clip(normalized_points @ pseudo_z, -1, 1)) * self.radius
        return t, p

    def plot(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(-0.3, 0.3, 100) * self.radius
        p = np.linspace(-0.3, 0.3, 100) * self.radius
        T, P = np.meshgrid(t, p)
        points = self.parameterization(T, P)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        ax.scatter(x, y, z, color='b', s=0.001)
        ax.plot([self.center_of_mirror[0], self.origin[0]], [self.center_of_mirror[1], self.origin[1]],
                [self.center_of_mirror[2], self.origin[2]], 'g-')
        if name is not None:
            name_position = self.parameterization(0.4, 0)
            ax.text(name_position[0], name_position[1], name_position[2], s=name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def ABCD_matrix(self, cos_theta_incoming: float):
        # order of rows/columns elements is [t, theta, p, phi]

        return np.array([[1, 0, 0, 0],
                         [- 2 * cos_theta_incoming / self.radius, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, -2 / (self.radius * cos_theta_incoming), 1]])


class Cavity:
    def __init__(self, mirrors: List[Mirror]):
        self.mirrors = mirrors
        self.ray_history: List[Ray] = []
        self.ABCD_matrices: List[np.ndarray] = [np.eye(4) for _ in range((len(mirrors) - 1) * 2)]

    def trace_ray(self, ray: Ray, final_surface=None) -> Ray:
        self.ray_history = [ray]

        for mirror in self.mirrors[1:] + [self.mirrors[0]]:
            ray = mirror.reflect_ray(ray)
            self.ray_history.append(ray)

        if final_surface is not None:
            final_position = final_surface.find_intersection_with_ray(ray)
            ray = Ray(final_position, ray.k_vector)
            self.ray_history.append(ray)
        return ray

    def trace_ray_parametric(self,
                             starting_position_and_angles: np.ndarray,
                             initial_and_final_surface: Optional[Mirror] = None) -> np.ndarray:
        # Like trace ray, but works as a function of the starting position and angles as parameters on the starting
        # surface, instead of the starting position and angles as a vector in 3D space.
        if initial_and_final_surface is None:
            initial_surface = self.mirrors[0]
            final_surface = None
        else:
            initial_surface = initial_and_final_surface
            final_surface = initial_and_final_surface

        t_i, theta_i, p_i, phi_i = starting_position_and_angles  # Those correspond to x, theta_x, y, theta_y in the
        # ABCD matrix notation.
        k_vector_i = unit_vector_of_angles(theta_i, phi_i)
        origin_i = initial_surface.parameterization(t_i, p_i)
        input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
        output_ray = self.trace_ray(input_ray, final_surface=final_surface)
        final_intersection_point = output_ray.origin
        t_o, p_o = initial_surface.get_parameterization(final_intersection_point)  # Here it is the initial surface
        # on purpose.
        theta_o, phi_o = angles_of_unit_vector(output_ray.k_vector)
        final_position_and_angles = np.array([t_o, theta_o, p_o, phi_o])
        return final_position_and_angles

    def f_roots(self,
                starting_position_and_angles: np.ndarray,
                initial_and_final_surface: Optional[Mirror] = None) -> np.ndarray:
        final_position_and_angles = self.trace_ray_parametric(starting_position_and_angles, initial_and_final_surface)
        return final_position_and_angles - starting_position_and_angles

    def find_central_line(self, initial_and_final_surface: Optional[Mirror] = None) -> Tuple[np.ndarray, bool]:

        if initial_and_final_surface is None:
            initial_and_final_surface = self.mirrors[0]

        theta_initial_guess, phi_initial_guess = self.default_initial_angles(initial_and_final_surface)

        initial_guess = np.array([0, theta_initial_guess, 0, phi_initial_guess])

        # In the documentation it says optimize.fsolve returns a solution, together with some flags, and also this is
        # how pycharm suggests to use it. But in practice it returns only the solution, not sure why.
        final_position_and_angles: np.ndarray = optimize.fsolve(self.f_roots, initial_guess,
                                                                args=(initial_and_final_surface))

        if np.linalg.norm(self.f_roots(final_position_and_angles)) > 1e-6:
            success = False
        else:
            success = True

        origin_solution = self.mirrors[0].parameterization(final_position_and_angles[0],
                                                           final_position_and_angles[2])  # t, p
        k_vector_solution = unit_vector_of_angles(final_position_and_angles[1],
                                                  final_position_and_angles[3])  # theta, phi
        central_line = Ray(origin_solution, k_vector_solution)
        # This line is to save the central line in the ray history, so that it can be plotted later.
        self.trace_ray(central_line, initial_and_final_surface)
        return final_position_and_angles, success

    # def find_central_line_grid_search(self,
    #                                   dx: float = 0.02,
    #                                   n_resolution: int = 30,
    #                                   center_guess: np.ndarray = np.array([0, 0, 0, 0]),
    #                                   n_iterations: int = 3,
    #                                   angles_initial_guess: Optional[Tuple[float, float]] = None) -> np.ndarray:
    #     # Not working
    #     dummy_linspace = np.linspace(-dx, dx, n_resolution)
    #     t = dummy_linspace + center_guess[0]
    #     p = dummy_linspace + center_guess[1]
    #
    #     if angles_initial_guess is None:
    #         initial_k_vector = self.mirrors[1].center_of_mirror - self.mirrors[0].center_of_mirror
    #         initial_k_vector = normalize_vector(initial_k_vector)
    #         theta_initial_guess, phi_initial_guess = angles_of_unit_vector(initial_k_vector)
    #     else:
    #         theta_initial_guess, phi_initial_guess = angles_initial_guess
    #
    #     theta = dummy_linspace + theta_initial_guess + center_guess[2]
    #     phi = dummy_linspace + phi_initial_guess + center_guess[3]
    #     T_i, P_i, Theta_i, Phi_i = np.meshgrid(t, p, theta, phi)
    #     k_vector_i = unit_vector_of_angles(Theta_i, Phi_i)
    #     origin_i = self.mirrors[0].parameterization(T_i, P_i)
    #     input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
    #     output_ray = self.trace_ray(input_ray)
    #     final_intersection_point = self.mirrors[0].find_intersection_with_ray(self.ray_history[-2])
    #     T_o, P_o = self.mirrors[0].get_parameterization(final_intersection_point)
    #     Theta_o, Phi_o = angles_of_unit_vector(output_ray.k_vector)
    #     f_roots = np.stack([T_o - T_i, P_o - P_i, Theta_o - Theta_i, Phi_o - Phi_i], axis=-1)
    #     f_roots_reshaped = f_roots.reshape((-1, 4))
    #     f_roots_no_nans = f_roots_reshaped[~np.isnan(f_roots_reshaped).any(axis=1), :]
    #     f_roots_var = np.std(f_roots_no_nans, axis=0)
    #     f_roots_normalized = f_roots / f_roots_var[None, None, None, None, :]
    #     f_roots_norm = np.linalg.norm(f_roots_normalized, axis=-1)
    #     f_roots_norm[np.isnan(f_roots_norm)] = np.inf
    #     min_index = np.unravel_index(np.nanargmin(f_roots_norm), f_roots_norm.shape)
    #     best_solution = np.array([T_i[min_index], P_i[min_index], Theta_i[min_index] - theta_initial_guess, Phi_i[min_index] - phi_initial_guess])
    #     print(f"Best solution:\n"
    #           f"t=     {best_solution[0]:.2e} in [{t[0]:.2e}, {t[-1]:.2e}]\n"
    #           f"p=     {best_solution[1]:.2e} in [{p[0]:.2e}, {p[-1]:.2e}]\n"
    #           f"theta= {best_solution[2]:.2e} in [{theta[0] - theta_initial_guess:.2e}, {theta[-1] - theta_initial_guess:.2e}]\n"
    #           f"phi=   {best_solution[3]:.2e} in [{phi[0] - phi_initial_guess:.2e}, {phi[-1] - phi_initial_guess:.2e}]\n\n"
    #           f"error_t=     {f_roots[min_index][0]:.2e}\n"
    #           f"error_p=     {f_roots[min_index][1]:.2e}\n"
    #           f"error_theta= {f_roots[min_index][2]:.2e}\n"
    #           f"error_phi=   {f_roots[min_index][3]:.2e}")
    #     if n_iterations == 0:
    #         return best_solution
    #     else:
    #         return self.find_central_line_grid_search(dx=2*dx / n_resolution, n_resolution=n_resolution,
    #                                                   center_guess=best_solution, n_iterations=n_iterations - 1,
    #                                                   angles_initial_guess=(theta_initial_guess, phi_initial_guess))

    def get_initial_surface(self):
        # gets a surface that sits between the first two mirrors, centered and perpendicular to the central line.
        if len(self.ray_history) == 0:
            self.find_central_line()
        middle_point = (self.ray_history[0].origin + self.ray_history[1].origin) / 2
        return FlatMirror(outwards_normal=self.ray_history[0].k_vector, center_of_mirror=middle_point)

    def generate_ABCD_matrix_numeric(self,
                                     central_line_parameters: Optional[np.ndarray] = None,
                                     initial_and_final_surface: Optional[Mirror] = None) -> np.ndarray:
        if central_line_parameters is None:
            central_line_parameters, success = self.find_central_line(initial_and_final_surface)
            if not success:
                raise ValueError("Could not find central line")
        dr = 1e-9
        # The i'th, j'th element of optimize.approx_fprime is the derivative of the i'th component of the output with
        # respect to the j'th component of the input, which is exactly the definition of the i'th j'th element of the
        # ABCD matrix.
        ABCD_matrix = optimize.approx_fprime(central_line_parameters, self.trace_ray_parametric, dr,
                                             initial_and_final_surface)
        return ABCD_matrix

    def generate_ABCD_matrix_analytic(self, central_line_parameters: Optional[np.ndarray] = None,
                                      initial_and_final_surface: Optional[Mirror] = None) -> np.ndarray:
        if initial_and_final_surface is None:
            initial_and_final_surface = self.get_initial_surface()
        if central_line_parameters is None:
            central_line_parameters, success = self.find_central_line(initial_and_final_surface)
            if not success:
                raise ValueError("Could not find central line")

        cos_thetas_incidents = [self.ray_history[i].k_vector @ self.mirrors[i + 1].outwards_normal
                                for i in range(len(self.mirrors) - 1)]
        for i in range(len(self.mirrors) - 1):
            self.ABCD_matrices[2 * i] = ABCD_free_space(self.ray_history[i].length)
            self.ABCD_matrices[2 * i + 1] = self.mirrors[i].ABCD_matrix(cos_thetas_incidents[i])

        return np.linalg.multi_dot(self.ABCD_matrices[::-1])

    def generate_ray_from_parameters(self, t: float, p: float, theta: float, phi: float,
                                     initial_surface: Optional[Mirror] = None) -> Ray:
        if initial_surface is None:
            initial_surface = self.mirrors[0]
        k_vector = unit_vector_of_angles(theta, phi)
        origin = initial_surface.parameterization(t, p)
        return Ray(origin=origin, k_vector=k_vector)

    def default_initial_k_vector(self, initial_surface: Optional[Mirror] = None) -> np.ndarray:
        if initial_surface is None:
            initial_surface = self.mirrors[0]
        initial_k_vector = self.mirrors[1].center_of_mirror - initial_surface.center_of_mirror
        initial_k_vector = normalize_vector(initial_k_vector)
        return initial_k_vector

    def default_initial_angles(self,
                               initial_surface: Optional[Mirror] = None) -> Tuple[float, float]:
        initial_k_vector = self.default_initial_k_vector(initial_surface)
        theta, phi = angles_of_unit_vector(initial_k_vector)
        return theta, phi

    def plot(self,
             ax: Optional[plt.Axes] = None,
             axis_span: float = 3,
             center_on_ray_beginning: bool = True):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for ray in self.ray_history:
            ray.plot(ax)
        for i, mirror in enumerate(self.mirrors):
            mirror.plot(ax, name=str(i))

        self.get_initial_surface().plot(ax, name="initial surface")

        if center_on_ray_beginning:
            origin_camera = self.ray_history[0].origin
        else:
            origin_camera = np.mean(np.stack([m.center_of_mirror for m in self.mirrors]), axis=0)
        ax.set_xlim(origin_camera[0] - axis_span, origin_camera[0] + axis_span)
        ax.set_ylim(origin_camera[1] - axis_span, origin_camera[1] + axis_span)
        ax.set_zlim(origin_camera[2] - axis_span, origin_camera[2] + axis_span)

        return ax

    @property
    def final_ray(self):
        if len(self.ray_history) == 0:
            return None
        else:
            return self.ray_history[-1]


if __name__ == '__main__':
    # print(f"{x_1=:.2f}\n{y_1=:.2f}\n{t_1=:.2f}\n{p_1=:.2f}\n{x_2=:.2f}\n{y_2=:.2f}\n{t_2=:.2f}\n{p_2=:.2f}\n{x_3=:.2f}\n{y_3=:.2f}\n{t_3=:.2f}\n{p_3=:.2f}\n\n{t_ray=:.2f}\n{theta_ray=:.2f}\n{p_ray=:.2f}\n{phi_ray=:.2f}\n{elev=:.2f}\n{azim=:.2f}\n{zoom=:.2f}\n{center_first_mirror=}")

    x_1 = 0.00
    y_1 = 0.00
    t_1 = 0.00
    p_1 = 0.00
    x_2 = 0.00
    y_2 = 0.00
    t_2 = 0.00
    p_2 = 0.00
    x_3 = 0.00
    y_3 = 0.00
    t_3 = 0.00
    p_3 = 0.00
    t_ray = 0.00
    theta_ray = 0.00
    p_ray = 0.00
    phi_ray = 0.00
    elev = 38.00
    azim = 168.00
    axis_span = 1.00
    center_on_ray_beginning = True

    x_1 += 1
    y_1 += 0.00
    t_1 += np.pi / 2
    p_1 += -np.pi / 6
    x_2 += 0
    y_2 += np.sqrt(3)
    t_2 += np.pi / 2
    p_2 += np.pi / 2
    x_3 += -1
    y_3 += 0
    t_3 += np.pi / 2
    p_3 += 7 * np.pi / 6

    normal_1 = unit_vector_of_angles(t_1, p_1)
    center_1 = np.array([x_1, y_1, 0])
    normal_2 = unit_vector_of_angles(t_2, p_2)
    center_2 = np.array([x_2, y_2, 0])
    normal_3 = unit_vector_of_angles(t_3, p_3)
    center_3 = np.array([x_3, y_3, 0])

    mirror_1 = FlatMirror(outwards_normal=normal_1, distance_from_origin=1)
    mirror_2 = FlatMirror(outwards_normal=normal_2, distance_from_origin=1)
    mirror_3 = FlatMirror(outwards_normal=normal_3, distance_from_origin=1)

    cavity = Cavity(mirrors=[mirror_1, mirror_2, mirror_3])
    initial_surface = cavity.get_initial_surface()
    central_line = cavity.find_central_line(initial_surface)


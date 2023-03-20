import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy import optimize


def unit_vector_of_angle(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def angles_distance(theta_1, theta_2):
    simple_distance = theta_2 - theta_1
    d_1 = np.mod(simple_distance, 2*np.pi)
    d_2 = np.mod(simple_distance, 2*np.pi)
    return np.min([d_1, d_2])

class Ray:
    def __init__(self, origin: np.ndarray, theta: float, length: Optional[float] = None):
        self.origin = origin
        self.theta = theta
        self.length = length

    @property
    def direction_vector(self):
        return unit_vector_of_angle(self.theta)

    def plot(self):
        if self.length is None or np.isnan(self.length):
            length_plot = 3
        else:
            length_plot = self.length
        ray_end = self.origin + length_plot * self.direction_vector
        plt.plot([self.origin[0], ray_end[0]], [self.origin[1], ray_end[1]], 'r-')


class Mirror:
    def reflect_ray(self, ray):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    @property
    def center_of_mirror(self):
        raise NotImplementedError

    def mirror_parameterization(self, t):
        raise NotImplementedError

    def find_intersection_with_ray(self, ray: Ray) -> float:
        raise NotImplementedError


class CurvedMirror(Mirror):
    def __init__(self, radius: float, center_angle: float, center_of_mirror: Optional[np.ndarray] = None,
                 origin: Optional[np.ndarray] = None,):
        if origin is None and center_of_mirror is None:
            raise ValueError('Either origin or center_of_mirror must be provided.')
        if origin is not None and center_of_mirror is not None:
            raise ValueError('Only one of origin and center_of_mirror must be provided.')
        if origin is not None:
            self.origin = origin
        else:
            self.origin = center_of_mirror - radius * unit_vector_of_angle(center_angle)
        self.radius = radius
        self.center_angle = center_angle

    def find_intersection_with_ray(self, ray: Ray) -> float:
        if np.abs(ray.theta - np.pi / 2) < 1e-6:
            x = ray.origin[0]
            y = self.origin[1] + np.sqrt(self.radius**2 - (x - self.origin[0])**2)
        else:
            # ray is y=ax+b
            a = np.tan(ray.theta)
            b = ray.origin[1] - a * ray.origin[0]
            o_x = self.origin[0]
            o_y = self.origin[1]
            r = self.radius
            # The next two lines are from Mathematica, and are used to solve the quadratic equation.
            # Simplify[Solve[{(x - Ox)^2 + (y - Oy)^2 == r^2, y == a*x + b}, {x, y}]]
            pre_determinanta = -a * b + o_x + a * o_y
            determinanta = np.sqrt(-b**2 + 2*a*o_x*o_y - o_y**2 + b * (-2*a*o_x+2*o_y) + r**2 + a**2 * (r**2 - o_x**2))

            x_1 = (pre_determinanta + determinanta) / (1 + a**2)
            x_2 = (pre_determinanta - determinanta) / (1 + a**2)
            y_1 = a * x_1 + b
            y_2 = a * x_2 + b

            inner_product_1 = np.dot(ray.direction_vector, np.array([x_1 - ray.origin[0], y_1 - ray.origin[1]]))
            inner_product_2 = np.dot(ray.direction_vector, np.array([x_2 - ray.origin[0], y_2 - ray.origin[1]]))

            if inner_product_1 > inner_product_2:  # Choose the incidence that is further down the ray.
                x = x_1
                y = y_1
            else:
                x = x_2
                y = y_2
        t = np.arctan2(y - self.origin[1], x - self.origin[0])
        return t

    def mirror_parameterization(self, t):
        return self.origin + self.radius * unit_vector_of_angle(t).T

    def reflect_along_normal(self, ray: Ray, t: Optional[float] = None) -> float:
        if t is None:
            t = self.find_intersection_with_ray(ray)
        ray_direction_vector = ray.direction_vector
        mirror_line_direction_vector = -unit_vector_of_angle(t)
        reflected_direction_vector = ray_direction_vector - 2 * np.dot(ray_direction_vector,
                                                                       mirror_line_direction_vector) * mirror_line_direction_vector
        reflected_direction = np.arctan2(reflected_direction_vector[1], reflected_direction_vector[0])
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        t = self.find_intersection_with_ray(ray)
        intersection_point = self.mirror_parameterization(t)
        reflected_direction = self.reflect_along_normal(ray, t)
        ray.length = np.linalg.norm(intersection_point - ray.origin)
        return Ray(intersection_point, reflected_direction)

    @property
    def center_of_mirror(self):
        return self.origin + self.radius * unit_vector_of_angle(self.center_angle)

    def plot(self):
        d_theta = 0.3
        t = np.linspace(self.center_angle-d_theta, self.center_angle+d_theta, 50)
        t_grey = np.linspace(self.center_angle+d_theta, self.center_angle-d_theta + 2*np.pi, 100)
        points = self.mirror_parameterization(t)
        grey_points = self.mirror_parameterization(t_grey)
        plt.plot(points[:, 0], points[:, 1], 'b-')
        plt.plot(grey_points[:, 0], grey_points[:, 1], color=(0.8, 0.8, 0.8), linestyle='-.', linewidth=0.5)
        plt.plot(self.center_of_mirror[0], self.center_of_mirror[1], 'bo')


class FlatMirror(Mirror):
    def __init__(self, origin: np.array, theta_normal: float):
        self.origin = origin
        self.theta_normal = theta_normal

    @property
    def direction_vector(self):
        return unit_vector_of_angle(self.theta_normal)

    @property
    def mirror_line_direction(self):
        return np.mod(self.theta_normal + np.pi / 2, 2 * np.pi)

    @property
    def mirror_line_direction_vector(self):
        return unit_vector_of_angle(self.mirror_line_direction)

    def find_intersection_with_ray(self, ray: Ray) -> float:
        ray_direction_vector = ray.direction_vector
        ray_origin = ray.origin
        mirror_line_direction_vector = self.mirror_line_direction_vector
        mirror_line_origin = self.origin
        t = np.cross(ray_direction_vector, mirror_line_origin - ray_origin) / np.cross(mirror_line_direction_vector,
                                                                                       ray_direction_vector)
        return t

    def reflect_along_normal(self, ray: Ray) -> float:
        ray_direction_vector = ray.direction_vector
        mirror_line_direction_vector = self.direction_vector
        reflected_direction_vector = ray_direction_vector - 2 * np.dot(ray_direction_vector,
                                                                       mirror_line_direction_vector) * mirror_line_direction_vector
        reflected_direction = np.arctan2(reflected_direction_vector[1], reflected_direction_vector[0])
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.mirror_parameterization(self.find_intersection_with_ray(ray))
        reflected_direction = self.reflect_along_normal(ray)
        ray.length = np.linalg.norm(intersection_point - ray.origin)
        return Ray(intersection_point, reflected_direction)

    @property
    def center_of_mirror(self):
        return self.origin

    def mirror_parameterization(self, t):
        return self.origin + t * self.mirror_line_direction_vector

    def plot(self):
        mirror_line_origin = self.origin - self.mirror_line_direction_vector
        mirror_line_end = self.origin + self.mirror_line_direction_vector
        plt.plot([mirror_line_origin[0], mirror_line_end[0]], [mirror_line_origin[1], mirror_line_end[1]], 'k-')
        normal_vector_end = self.origin + self.direction_vector
        plt.plot([self.origin[0], normal_vector_end[0]], [self.origin[1], normal_vector_end[1]], 'k--')
        plt.plot(self.origin[0], self.origin[1], 'ko')


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

    def find_central_line(self) -> Ray:
        def f_roots(t_and_theta_i: np.ndarray) -> np.ndarray:
            t_i, theta_i = t_and_theta_i[0], t_and_theta_i[1]
            ray = Ray(self.mirrors[0].mirror_parameterization(t_i), theta_i)
            output_ray = self.trace_ray(ray)
            t_o = self.mirrors[0].find_intersection_with_ray(self.ray_history[-2])
            return np.array([t_o - t_i, angles_distance(output_ray.theta, ray.theta)])

        initial_guess = np.array([0,
                                  np.arctan2(self.mirrors[1].center_of_mirror[1] - self.mirrors[0].center_of_mirror[1],
                                             self.mirrors[1].center_of_mirror[0] - self.mirrors[0].center_of_mirror[
                                                 0])])
        t_and_theta = optimize.fsolve(f_roots, initial_guess)
        central_line = Ray(self.mirrors[0].mirror_parameterization(t_and_theta[0]), t_and_theta[1])
        self.trace_ray(central_line)
        return central_line

    def plot(self):
        for ray in self.ray_history:
            ray.plot()
        for mirror in self.mirrors:
            mirror.plot()

    @property
    def final_ray(self):
        if len(self.ray_history) == 0:
            return None
        else:
            return self.ray_history[-1]


if __name__ == '__main__':
    x_1 = 2.00
    y_1 = 0.00
    r_1 = 1.00
    t_1 = 0.00
    x_2 = -1.00
    y_2 = 0.45
    r_2 = 0.20
    t_2 = 2.28
    x_3 = -1.00
    y_3 = -0.45
    r_3 = 0.20
    t_3 = 4.04

    center_mirror_1 = np.array([x_1, y_1])
    center_mirror_2 = np.array([x_2, y_2])
    center_mirror_3 = np.array([x_3, y_3])

    mirror_1 = CurvedMirror(r_1, t_1, center_mirror_1)
    mirror_2 = CurvedMirror(r_2, t_2, center_mirror_2)
    mirror_3 = CurvedMirror(r_3, t_3, center_mirror_3)
    ray = Ray(mirror_1.center_of_mirror, np.arctan2(mirror_2.center_of_mirror[1] - mirror_1.center_of_mirror[1], mirror_2.center_of_mirror[0] - mirror_1.center_of_mirror[0]))

    cavity = Cavity([mirror_1, mirror_2, mirror_3])

    central_line = cavity.find_central_line()

    # reflected_ray_1 = mirror_2.reflect_ray(ray)
    # reflected_ray_2 = mirror_3.reflect_ray(reflected_ray_1)
    # reflected_ray_3 = mirror_1.reflect_ray(reflected_ray_2)
    # fig = plt.figure(figsize=(10, 10))
    # ray.plot()
    # reflected_ray_1.plot()
    # reflected_ray_2.plot()
    # reflected_ray_3.plot()
    # mirror_1.plot()
    # mirror_2.plot()
    # mirror_3.plot()
    cavity.plot()
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()

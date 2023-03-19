import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Ray:
    def __init__(self, origin: np.ndarray, theta: float, length: Optional[float] = None):
        self.origin = origin
        self.theta = theta
        self.length = length

    @property
    def direction_vector(self):
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    def plot(self):
        if self.length is None:
            length_plot = 1
        else:
            length_plot = self.length
        ray_end = self.origin + length_plot * self.direction_vector
        plt.plot([self.origin[0], ray_end[0]], [self.origin[1], ray_end[1]], 'r-')


class Mirror:
    def deflect_ray(self, ray):
        raise NotImplementedError


class FlatMirror(Mirror):
    def __init__(self, origin: np.array, theta_normal: float):
        self.origin = origin
        self.theta_normal = theta_normal

    @property
    def direction_vector(self):
        return np.array([np.cos(self.theta_normal), np.sin(self.theta_normal)])

    @property
    def mirror_line_direction(self):
        return np.mod(self.theta_normal + np.pi / 2, 2 * np.pi)

    @property
    def mirror_line_direction_vector(self):
        return np.array([np.cos(self.mirror_line_direction), np.sin(self.mirror_line_direction)])

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        ray_direction_vector = ray.direction_vector
        ray_origin = ray.origin
        mirror_line_direction_vector = self.mirror_line_direction_vector
        mirror_line_origin = self.origin
        t = np.cross(mirror_line_origin - ray_origin, mirror_line_direction_vector) / np.cross(ray_direction_vector,
                                                                                               mirror_line_direction_vector)
        intersection = ray_origin + t * ray_direction_vector
        return intersection

    def reflect_along_normal(self, ray: Ray) -> float:
        ray_direction_vector = ray.direction_vector
        mirror_line_direction_vector = self.direction_vector
        reflected_direction_vector = ray_direction_vector - 2 * np.dot(ray_direction_vector,
                                                                       mirror_line_direction_vector) * mirror_line_direction_vector
        reflected_direction = np.arctan2(reflected_direction_vector[1], reflected_direction_vector[0])
        return reflected_direction

    def deflect_ray(self, ray: Ray) -> Ray:
        intersection = self.find_intersection_with_ray(ray)
        reflected_direction = self.reflect_along_normal(ray)
        ray.length = np.linalg.norm(intersection - ray.origin)
        return Ray(intersection, reflected_direction)

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
        self.ray_history.append(ray)
        for mirror in self.mirrors:
            ray = mirror.deflect_ray(ray)
            self.ray_history.append(ray)
        return ray

    @property
    def final_ray(self):
        if len(self.ray_history) == 0:
            return None
        else:
            return self.ray_history[-1]

if __name__ == '__main__':
    origin_1 = np.array([0, 0])
    theta_1 = np.pi / 2
    origin_2 = np.array([1.5, 1])
    theta_2 = np.pi
    origin_3 = np.array([-1, 3])
    theta_3 = 6.6*np.pi / 4
    theta_ray = np.pi / 4

    initial_ray = Ray(origin_1, theta_ray)
    mirror_1 = FlatMirror(origin_1, theta_1)
    mirror_2 = FlatMirror(origin_2, theta_2)
    mirror_3 = FlatMirror(origin_3, theta_3)
    deflected_ray_2 = mirror_2.deflect_ray(initial_ray)
    deflected_ray_3 = mirror_3.deflect_ray(deflected_ray_2)
    deflected_ray_4 = mirror_1.deflect_ray(deflected_ray_3)
    figure = plt.figure(figsize=(8, 8))
    initial_ray.plot()
    deflected_ray_2.plot()
    deflected_ray_3.plot()
    deflected_ray_4.plot()
    mirror_1.plot()
    mirror_2.plot()
    mirror_3.plot()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()

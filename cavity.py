import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import sympy as sp
import sympy.vector as spv

S = spv.CoordSys3D('S')

class Ray:
    def __init__(self, origin: spv.Vector, direction: spv.Vector, length: Optional[sp.Symbol] = None):
        self.origin = origin
        self.direction = direction.normalize()
        self.length = length

    def plot(self, substitution_dict: dict):
        origin_x_plot = self.origin.coeff(S.i).subs(substitution_dict).evalf()
        origin_y_plot = self.origin.coeff(S.j).subs(substitution_dict).evalf()
        direction_x_plot = self.direction.coeff(S.i).subs(substitution_dict).evalf()
        direction_y_plot = self.direction.coeff(S.j).subs(substitution_dict).evalf()
        if self.length is None:
            length_plot = 1
        else:
            length_plot = self.length.subs(substitution_dict).evalf()
        x = np.array([origin_x_plot, origin_x_plot + length_plot * direction_x_plot])
        y = np.array([origin_y_plot, origin_y_plot + length_plot * direction_y_plot])
        plt.plot(x, y, 'r-')


class Mirror:
    def deflect_ray(self, ray):
        raise NotImplementedError


class FlatMirror(Mirror):
    def __init__(self, origin: spv.Vector, normal_direction: spv.Vector):
        self.origin = origin
        self.normal_direction = normal_direction.normalize()

    @property
    def mirror_line_direction(self):
        return self.normal_direction.cross(S.k)

    def find_intersection_with_ray(self, ray: Ray) -> spv.Vector:
        t, s = sp.symbols('t s')
        x_ray = ray.origin.coeff(S.i) + ray.direction.coeff(S.i) * t
        y_ray = ray.origin.coeff(S.j) + ray.direction.coeff(S.j) * t
        x_mirror = self.origin.coeff(S.i) + self.mirror_line_direction.coeff(S.i) * s
        y_mirror = self.origin.coeff(S.j) + self.mirror_line_direction.coeff(S.j) * s
        s_and_t = sp.solve([sp.simplify(x_ray - x_mirror), sp.simplify(y_ray - y_mirror)], [t, s])
        return x_ray.subs(t, s_and_t[t]) * S.i + y_ray.subs(t, s_and_t[t]) * S.j

    def reflect_along_normal(self, ray: Ray) -> spv.Vector:
        normal = self.normal_direction.normalize()
        return ray.direction - 2 * normal.dot(ray.direction) * normal

    def deflect_ray(self, ray: Ray) -> Ray:
        intersection = self.find_intersection_with_ray(ray)
        reflected_direction = self.reflect_along_normal(ray)
        ray.length = (intersection - ray.origin).magnitude()
        return Ray(intersection, reflected_direction)

    def plot(self, substitution_dict: dict):
        origin_x_plot = self.origin.coeff(S.i).subs(substitution_dict).evalf()
        origin_y_plot = self.origin.coeff(S.j).subs(substitution_dict).evalf()
        direction_x_plot = self.mirror_line_direction.coeff(S.i).subs(substitution_dict).evalf()
        direction_y_plot = self.mirror_line_direction.coeff(S.j).subs(substitution_dict).evalf()
        length_plot = 1
        x = np.array([origin_x_plot - length_plot * direction_x_plot, origin_x_plot + length_plot * direction_x_plot])
        y = np.array([origin_y_plot - length_plot * direction_y_plot, origin_y_plot + length_plot * direction_y_plot])
        plt.plot(x, y, 'b-')


class Cavity:
    def __init__(self, mirrors: List[Mirror]):
        mirrors.append(mirrors[0])
        self.mirrors = mirrors
        self.ray_history: List[Ray] = []

    def trace_ray(self, ray:Ray) -> Ray:
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

# %%
# import time
# start_time = time.time()
# x_1, y_1, direction_x_1, direction_y_1, x_2, y_2, direction_x_2, direction_y_2, x_3, y_3, direction_x_3, direction_y_3, direction_x_ray, direction_y_ray = sp.symbols('x_1, y_1, direction_x_1, direction_y_1, x_2, y_2, direction_x_2, direction_y_2, x_3, y_3, direction_x_3, direction_y_3, direction_x_ray, direction_y_ray')
# origin_1 = x_1 * S.i + y_1 * S.j
# origin_2 = x_2 * S.i + y_2 * S.j
# origin_3 = x_3 * S.i + y_3 * S.j
# direction_1 = direction_x_1 * S.i + direction_y_1 * S.j
# direction_2 = direction_x_2 * S.i + direction_y_2 * S.j
# direction_3 = direction_x_3 * S.i + direction_y_3 * S.j
# direction_ray = direction_x_ray * S.i + direction_y_ray * S.j
# initial_ray = Ray(origin_1, direction_ray)
# mirror1 = FlatMirror(origin_1, direction_1)
# mirror2 = FlatMirror(origin_2, direction_2)
# mirror3 = FlatMirror(origin_3, direction_3)
# deflected_ray2 = mirror2.deflect_ray(initial_ray)
# deflected_ray3 = mirror3.deflect_ray(deflected_ray2)
# time_duration = time.time() - start_time
# print(f'Calculation took {time_duration} seconds')

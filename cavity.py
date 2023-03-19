import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sympy as sp


class Ray:
    def __init__(self, x: sp.Symbol, y: sp.Symbol, theta: sp.Symbol):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def ax_b(self) -> Tuple[sp.tan, sp.Symbol]:
        a = sp.tan(self.theta)
        b = self.y - a * self.x
        return a, b

    def plot(self, substitution_dict: dict):
        a, b = self.ax_b
        a_subs = a.subs(substitution_dict)
        b_subs = b.subs(substitution_dict)
        # x_plot = sp.Symbol('x_plot')
        # sp.plot(a_subs * x_plot + b_subs, (x_plot, 0, 1))
        x_plot = np.linspace(0, 1, 100)
        y_plot = a_subs * x_plot + b_subs
        plt.plot(x_plot, y_plot)


class Mirror:
    def deflect_ray(self, ray):
        raise NotImplementedError


class FlatMirror(Mirror):
    def __init__(self, x: sp.Symbol, y: sp.Symbol, theta: sp.Symbol):
        self.x = x
        self.y = y
        self.theta = theta  # Direction of the normal to the mirror

    @property
    def ax_b(self) -> Tuple[sp.Symbol, sp.Symbol]:
        a = - 1 / sp.tan(self.theta)
        b = self.y - a * self.x
        return a, b

    def deflect_ray(self, ray:Ray) -> Ray:
        mirror_a, mirror_b = self.ax_b
        ray_a, ray_b = ray.ax_b
        x_incidence = (mirror_b - ray_b) / (ray_a - mirror_a)
        y_incidence = mirror_a * x_incidence + mirror_b
        theta = 2 * self.theta - ray.theta - sp.pi
        return Ray(x_incidence, y_incidence, theta)

    def plot(self, substitution_dict: dict):
        a, b = self.ax_b
        a_subs = a.subs(substitution_dict).evalf()
        b_subs = b.subs(substitution_dict).evalf()
        # x_plot = sp.Symbol('x_plot')
        # sp.plot(a_subs * x_plot + b_subs, (x_plot, -1, 1))
        x = np.linspace(-1, 1, 100)
        y = a_subs * x + b_subs
        plt.plot(x, y, 'k-')
        plt.axis('equal')


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

    # def solve(self, ray: Ray) -> Ray:
    #     x_output, y_output, theta_output = sp.solve([])

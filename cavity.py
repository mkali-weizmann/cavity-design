import numpy as np
import matplotlib.pyplot as plt
from typing import List
# import sympy as sp


class Ray:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def ax_b(self):
        a = np.tan(self.theta)
        b = self.y - a * self.x
        return a, b

    def plot(self):
        x = np.linspace(0, 10, 100)
        y = self.ax_b[0] * x + self.ax_b[1]
        plt.plot(x, y)
        plt.show()


class Mirror:
    def deflect_ray(self, ray):
        raise NotImplementedError


class FlatMirror(Mirror):
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    @property
    def ax_b(self):
        a = - 1 / np.tan(self.theta)
        b = self.y - a * self.x
        return a, b

    def plot(self):
        x = np.linspace(0, 10, 100)
        y = self.ax_b[0] * x + self.ax_b[1]
        plt.plot(x, y)
        plt.show()

    def deflect_ray(self, ray):
        mirror_a, mirror_b = self.ax_b
        x_incidence = (mirror_b - ray.b) / (ray.a - mirror_a)
        y_incidence = mirror_a * x_incidence + mirror_b
        theta = 2 * self.theta - ray.theta - np.pi
        return Ray(x_incidence, y_incidence, theta)


class Cavity():
    def __init__(self, mirrors: List[Mirror]):
        self.mirrors = mirrors.append(mirrors[0])
        self.ray_history = []

    def trace_ray(self, ray):
        for mirror in self.mirrors:
            ray = mirror.deflect_ray(ray)
            self.ray_history.append(ray)
        return ray

    def plot(self):
        for mirror in self.mirrors:
            mirror.plot()
        for ray in self.ray_history:
            ray.plot()
        plt.show()




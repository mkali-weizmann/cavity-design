import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy import optimize
import warnings


def unit_vector_of_angle(theta: Union[np.ndarray, float]):
    return np.array([np.cos(theta), np.sin(theta)]).T  # N_theta | 2


def angles_distance(theta_2: np.ndarray, theta_1: np.ndarray):
    simple_distance = theta_2 - theta_1
    d_1 = np.mod(simple_distance, 2*np.pi)
    d_2 = np.mod(-simple_distance, 2*np.pi)
    return np.where(d_1 < d_2, d_1, -d_2)


class Ray:
    def __init__(self, origin: np.ndarray, theta: Union[np.ndarray, float],
                 length: Optional[Union[np.ndarray, float]] = None):
        if origin.ndim == 1:
            origin = origin[np.newaxis, :]
        if isinstance(theta, float):
            theta = np.ones(origin.shape[0]) * theta
        self.origin = origin  # m_rays | 2
        self.theta = theta  # m_rays
        if length is not None and isinstance(length, float):
            length = np.ones(origin.shape[0]) * length
        self.length = length  # m_rays or None

    @property
    def direction_vector(self):
        return unit_vector_of_angle(self.theta)  # m_rays | 2

    def plot(self):
        if self.length is None:
            length_plot = np.ones(self.origin.shape[0]) * 3
        else:
            length_plot = np.where(np.isnan(self.length), 3, self.length)
        ray_end = self.origin + length_plot[:, np.newaxis] * self.direction_vector  # m_rays | 2
        [plt.plot([self.origin[i, 0], ray_end[i, 0]], [self.origin[i, 1], ray_end[i, 1]], 'r-')
         for i in range(self.origin.shape[0])]


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

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError


class CurvedMirror(Mirror):
    def __init__(self, radius: float, center_angle: float, center_of_mirror: Optional[np.ndarray] = None,
                 origin: Optional[np.ndarray] = None,):
        if origin is None and center_of_mirror is None:
            raise ValueError('Either origin or center_of_mirror must be provided.')
        elif origin is not None and center_of_mirror is not None:
            raise ValueError('Only one of origin and center_of_mirror must be provided.')
        elif origin is None:
            self.origin = center_of_mirror - radius * unit_vector_of_angle(center_angle)
        else:
            self.origin = origin
        self.radius = radius
        self.center_angle = center_angle

    def find_intersection_with_ray(self, ray: Ray) -> np.ndarray:
        half_pi_elements = np.abs(np.mod(ray.theta, np.pi) - np.pi / 2) < 1e-6
        ray_theta_no_half_pi = np.where(half_pi_elements, 0, ray.theta)

        # For elements in which ray.theta is pi/2 or 3*np.pi/2 we need to find the intersection with the mirror one way:
        half_pi_elements_x = ray.origin[:, 0]
        # This extra line is to avoid the warning that comes from taking the square root of a negative number.
        y_distance_from_origin = np.sqrt(self.radius**2 - (half_pi_elements_x - self.origin[0])**2 +0j)
        y_distance_from_origin = np.where(np.abs(np.imag(y_distance_from_origin) > 1e-9), np.nan, np.real(y_distance_from_origin))
        half_pi_elements_y_1 = self.origin[1] + y_distance_from_origin
        half_pi_elements_y_2 = self.origin[1] - y_distance_from_origin

        # For elements in which ray.theta is not pi/2, we need to find the intersection with the mirror another way:
        # ray is y=ax+b
        a = np.tan(ray_theta_no_half_pi)
        b = ray.origin[:, 1] - a * ray.origin[:, 0]
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

        # Choose element-wise the correct calculation:
        x_1 = np.where(half_pi_elements, half_pi_elements_x, x_1)
        x_2 = np.where(half_pi_elements, half_pi_elements_x, x_2)
        y_1 = np.where(half_pi_elements, half_pi_elements_y_1, y_1)
        y_2 = np.where(half_pi_elements, half_pi_elements_y_2, y_2)

        # Choose the intersection that is further down the ray:
        inner_product_1 = np.sum(ray.direction_vector * np.stack([x_1 - ray.origin[:, 0], y_1 - ray.origin[:, 1]], axis=1), axis=1)
        inner_product_2 = np.sum(ray.direction_vector * np.stack([x_2 - ray.origin[:, 0], y_2 - ray.origin[:, 1]], axis=1), axis=1)

        condition = inner_product_1 > inner_product_2
        x = np.where(condition, x_1, x_2)
        y = np.where(condition, y_1, y_2)

        t = np.arctan2(y - self.origin[1], x - self.origin[0])
        return t

    def mirror_parameterization(self, t: np.ndarray) -> np.ndarray:
        return self.origin + self.radius * unit_vector_of_angle(t)

    def reflect_along_normal(self, ray: Ray, t: Optional[np.ndarray] = None) -> np.ndarray:
        if t is None:
            t = self.find_intersection_with_ray(ray)
        ray_direction_vector = ray.direction_vector
        mirror_normal_vector = -unit_vector_of_angle(t)
        dot_product = np.sum(ray_direction_vector * mirror_normal_vector, axis=1)
        reflected_direction_vector = ray_direction_vector - 2 * dot_product[:, np.newaxis] * mirror_normal_vector
        reflected_direction = np.arctan2(reflected_direction_vector[:, 1], reflected_direction_vector[:, 0])
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        t = self.find_intersection_with_ray(ray)
        intersection_point = self.mirror_parameterization(t)
        reflected_direction = self.reflect_along_normal(ray, t)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=1)
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
        reflected_direction = np.arctan2(reflected_direction_vector[:, 1], reflected_direction_vector[:, 0])
        return reflected_direction

    def reflect_ray(self, ray: Ray) -> Ray:
        intersection_point = self.mirror_parameterization(self.find_intersection_with_ray(ray))
        reflected_direction = self.reflect_along_normal(ray)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=1)
        return Ray(intersection_point, reflected_direction)

    @property
    def center_of_mirror(self):
        return self.origin

    def mirror_parameterization(self, t):
        if isinstance(t, float):
            return self.origin + t * self.mirror_line_direction_vector
        else:
            return self.origin + t[:, np.newaxis] * self.mirror_line_direction_vector

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

    def find_central_line(self) -> Tuple[np.ndarray, bool]:
        def f_roots(t_and_theta_i: np.ndarray) -> np.ndarray:
            t_i, theta_i = np.array([t_and_theta_i[0]]), np.array([t_and_theta_i[1]])
            ray = Ray(self.mirrors[0].mirror_parameterization(t_i), theta_i)
            output_ray = self.trace_ray(ray)
            t_o = self.mirrors[0].find_intersection_with_ray(self.ray_history[-2])
            return np.concatenate([t_o - t_i, angles_distance(output_ray.theta, ray.theta)])

            # error = (t_o - t_i)**2 + angles_distance(output_ray.theta, ray.theta)**2
            # print(f"{t_i=}, {theta_i=}\n{t_o=}, {output_ray.theta=}\n{error=}\n")
            # return error

        initial_angle_guess = np.arctan2(self.mirrors[1].center_of_mirror[1] - self.mirrors[0].center_of_mirror[1],
                                             self.mirrors[1].center_of_mirror[0] - self.mirrors[0].center_of_mirror[0])
        initial_guess = np.array([0, initial_angle_guess])
        # dt_max = 0.0001
        # dtheta_max = 0.0001
        t_and_theta = optimize.fsolve(f_roots, initial_guess)#, method='SLSQP',
                                        #bounds=((-dt_max, dt_max),
#                                                (initial_angle_guess-dtheta_max, initial_angle_guess+dtheta_max))).x
        if np.linalg.norm(f_roots(t_and_theta)) > 1e-6:
            success = False
        else:
            success = True
        central_line = Ray(self.mirrors[0].mirror_parameterization(t_and_theta[0]), t_and_theta[1])
        self.trace_ray(central_line)
        return t_and_theta, success

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

    x_1 = 1.90200e-01
    y_1 = 0.00000e+00
    r_1 = 0.00000e+00
    t_1 = 0.00000e+00
    x_2 = 0.00000e+00
    y_2 = 0.00000e+00
    r_2 = 0.00000e+00
    t_2 = 0.00000e+00
    x_3 = 0.00000e+00
    y_3 = 0.00000e+00
    r_3 = 0.00000e+00
    t_3 = 0.00000e+00
    default_t_2 = 2.00000e+00
    dtheta_initial_plot = -1.77636e-15
    ds = 9.80010e-02
    t_shift = 0.00000e+00
    theta_shift = 0.00000e+00
    dt_initial_plot = np.array([0.00000e+00])
    scale = 1.00


    x_1 += 1
    y_1 += 0.00
    r_1 += 2
    t_1 += -np.pi / 6
    x_2 += 0
    y_2 += np.sqrt(3)
    r_2 += 2
    t_2 += np.pi / 2
    x_3 += -1
    y_3 += 0
    r_3 += 2
    t_3 += 7 * np.pi / 6

    center_mirror_1 = np.array([x_1, y_1])
    center_mirror_2 = np.array([x_2, y_2])
    center_mirror_3 = np.array([x_3, y_3])

    mirror_1 = CurvedMirror(r_1, t_1, center_mirror_1)
    mirror_2 = CurvedMirror(r_2, t_2, center_mirror_2)
    mirror_3 = CurvedMirror(r_3, t_3, center_mirror_3)

    cavity = Cavity([mirror_1, mirror_2, mirror_3])
    t_and_theta_central_line, success = cavity.find_central_line()
    print(success)

    if success:
        plot_ray_t = t_and_theta_central_line[0] + dt_initial_plot
        plot_ray_theta = t_and_theta_central_line[1] + dtheta_initial_plot
    else:
        plot_ray_t = t_1 + dt_initial_plot
        plot_ray_theta = np.arctan2(mirror_2.center_of_mirror[1] - mirror_1.center_of_mirror[1],
                                    mirror_2.center_of_mirror[0] - mirror_1.center_of_mirror[0]) + dtheta_initial_plot

    ray = Ray(mirror_1.mirror_parameterization(plot_ray_t), plot_ray_theta)

    reflected_ray_1 = mirror_2.reflect_ray(ray)
    reflected_ray_2 = mirror_3.reflect_ray(reflected_ray_1)
    reflected_ray_3 = mirror_1.reflect_ray(reflected_ray_2)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    plt.sca(ax1)
    ray.plot()
    reflected_ray_1.plot()
    reflected_ray_2.plot()
    reflected_ray_3.plot()
    mirror_1.plot()
    mirror_2.plot()
    mirror_3.plot()
    plt.plot((x_1, x_2, x_3, x_1), (y_1, y_2, y_3, y_1), linewidth=0.5)
    x_lim_min = max(x_1 - r_1 - 0.2, x_2 - r_2 - 0.2, x_3 - r_3 - 0.2)
    x_lim_max = max(x_1 + r_1 + 0.2, x_2 + r_2 + 0.2, x_3 + r_3 + 0.2)
    y_lim_min = max(y_1 - r_1 - 0.2, y_2 - r_2 - 0.2, y_3 - r_3 - 0.2)
    y_lim_max = max(y_1 + r_1 + 0.2, y_2 + r_2 + 0.2, y_3 + r_3 + 0.2)
    plt.xlim(x_lim_min, x_lim_max)
    plt.axis('equal')

    t_initial = (np.linspace(-3 * ds, 3 * ds, 100) + t_shift * ds) * scale + t_1
    theta_0 = np.arctan2(mirror_2.center_of_mirror[1] - mirror_1.center_of_mirror[1],
                         mirror_2.center_of_mirror[0] - mirror_1.center_of_mirror[0])
    theta_initial = theta_0 + np.linspace(-3 * ds, 3 * ds, 100) + theta_shift * ds
    if success:
        t_initial += t_and_theta_central_line[0] - t_1
        theta_initial += t_and_theta_central_line[1] - theta_0

    all_permutations = np.transpose([np.tile(t_initial, len(theta_initial)), np.repeat(theta_initial, len(t_initial))])

    ray = Ray(mirror_1.mirror_parameterization(all_permutations[:, 0]), all_permutations[:, 1])

    final_ray = cavity.trace_ray(ray)

    t_final = cavity.mirrors[0].find_intersection_with_ray(cavity.ray_history[-2])

    t_initial_reshaped = all_permutations[:, 0].reshape(len(t_initial), len(theta_initial))
    theta_initial_reshaped = all_permutations[:, 1].reshape(len(t_initial), len(theta_initial))
    t_final_reshaped = t_final.reshape(len(t_initial), len(theta_initial))
    theta_final_reshaped = final_ray.theta.reshape(len(t_initial), len(theta_initial))

    delta_t = t_final_reshaped - t_initial_reshaped
    delta_theta = angles_distance(theta_final_reshaped, theta_initial_reshaped)

    # Plot the difference between the initial and final t on the left upper axis
    im = ax2.imshow(delta_t, extent=(
    t_initial[0] - t_1, t_initial[-1] - t_1, theta_initial[-1] - theta_0, theta_initial[0] - theta_0)
                    )  #
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs_2 = ax2.contour(t_initial - t_1, theta_initial - theta_0,
                           delta_t, levels=[0], colors=['white'])
        cs_2b = ax2.contour(t_initial - t_1, theta_initial - theta_0,
                            delta_theta, levels=[0], colors=['black'], linestyles='--')
        cbar = fig.colorbar(im, ax=ax2)

    try:
        cbar.add_lines(cs_2)
    except ValueError:
        print("kaki")

    ax3.set_xlabel('dt')
    ax3.set_ylabel('d_theta')
    # ax2.set_aspect(0.8)
    ax2.set_title('t_final -t_initial')
    ax2.set_aspect(1 / scale)
    ax2.plot(t_and_theta_central_line[1] - theta_0, t_and_theta_central_line[0] - t_1)

    # Plot the difference between the initial and final theta on the right upper axis
    # Plot the difference between the initial and final theta on the right upper axis
    im = ax3.imshow(angles_distance(theta_final_reshaped, theta_initial_reshaped), extent=(
    t_initial[0] - t_1, t_initial[-1] - t_1, theta_initial[-1] - theta_0, theta_initial[0] - theta_0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs_3 = ax3.contour(t_initial - t_1, theta_initial - theta_0,
                           delta_t, levels=[0], colors=['white'])
        cs_3b = ax3.contour(t_initial - t_1, theta_initial - theta_0,
                            delta_theta, levels=[0], colors=['black'], linestyles='--')

        cbar = fig.colorbar(im, ax=ax3)
    try:
        cbar.add_lines(cs_3b)
    except ValueError:
        print("kaki")

    ax3.set_xlabel('dt')
    ax3.set_ylabel('d_theta')
    ax3.set_aspect(1 / scale)

    ax3.set_title('theta_final - theta_initial')
    ax3.plot(t_and_theta_central_line[1] - theta_0, t_and_theta_central_line[0] - t_1)
    plt.show()

















from __future__ import annotations

from scipy.optimize import brentq

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from datetime import datetime
from hashlib import md5
from tqdm import tqdm
from utils import MaterialProperties
from numpy.polynomial import Polynomial

pd.set_option("display.max_rows", 500)
pd.options.display.float_format = "{:.3e}".format
# np.seterr(all="raise") #TODO: some functions need it on and some need it off.


def params_to_perturbable_params_names(
    params_list: List[OpticalElementParams], remove_one_of_the_angles: bool = False
) -> List[str]:
    # Associates the cavity parameters with the number of parameters needed to describe the cavity.
    # If there is a lens, then the number of parameters is 7 (x, y, theta, phi, r, n_2):

    perturbable_params = [
        ParamsNames.x,
        ParamsNames.y,
        ParamsNames.theta,
        ParamsNames.phi,
        ParamsNames.r_1,
        ParamsNames.n_inside_or_after,
    ]

    surface_types = [params.surface_type for params in params_list]
    if not (
        SurfacesTypes.curved_refractive_surface in surface_types
        or SurfacesTypes.thick_lens in surface_types
        or SurfacesTypes.ideal_thick_lens in surface_types
    ):
        perturbable_params.remove(ParamsNames.n_inside_or_after)
    if remove_one_of_the_angles:
        perturbable_params.remove(ParamsNames.theta)
    return perturbable_params


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
# theta and phi). also, theta and t appear before phi and p.
# If for example there is a gaussian parameter q both for theta axis and phi axis, then the first element of q will be the q of theta,
# and the second element of q will be the q of phi.


class LocalModeParameters:
    # The gaussian mode parameters at a point, without global coordinates information like where it is and where is it
    # pointing to.
    def __init__(
        self,
        z_minus_z_0: Optional[Union[np.ndarray, float]] = None,  # The actual distance should be multiplied by n
        z_R: Optional[Union[np.ndarray, float]] = None,
        q: Optional[Union[np.ndarray, float]] = None,
        lambda_0_laser: Optional[float] = None,  # the laser's wavelength in vacuum
        n: float = 1,  # refractive index
    ):
        if q is not None:
            if isinstance(q, float):
                q = np.array([q, q])
            self.q: np.ndarray = q
        elif z_minus_z_0 is not None and z_R is not None:
            if isinstance(z_R, float):
                z_R = np.array([z_R, z_R])
            if isinstance(q, float):
                z_minus_z_0 = np.array([z_minus_z_0, z_minus_z_0])
            self.q: np.ndarray = z_minus_z_0 + 1j * z_R
        else:
            raise ValueError("Either q or z_minus_z_0 and z_R must be provided")
        self.lambda_0_laser = lambda_0_laser
        self.n = n

    @property
    def z_minus_z_0(self):
        return self.q.real

    @property
    def z_R(self):
        if np.all(np.iscomplex(self.q)):
            return self.q.imag
        else:
            return np.ones(self.q.shape) * np.nan

    @property
    def w_0(self):
        return w_0_of_z_R(z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)

    @property
    def lambda_laser(self):
        return self.lambda_0_laser / self.n

    def to_mode_parameters(self, location_of_local_mode_parameter: np.ndarray, k_vector: np.ndarray):
        center = location_of_local_mode_parameter - self.z_minus_z_0[:, np.newaxis] * k_vector
        z_hat = np.array([0, 0, 1])
        if np.linalg.norm(k_vector - z_hat) < 1e-10:  # if the k_vector is almost parallel to z_hat, better take another
            # vector as z_hat to avoid numerical instability
            z_hat = np.array([0, 1, 0])
        pseudo_y = normalize_vector(np.cross(z_hat, k_vector))
        pseudo_z = normalize_vector(np.cross(k_vector, pseudo_y))
        principle_axes = np.stack([pseudo_z, pseudo_y], axis=0)

        return ModeParameters(
            center=center,
            k_vector=k_vector,
            w_0=self.w_0,
            principle_axes=principle_axes,
            lambda_0_laser=self.lambda_0_laser,
        )

    @property
    def spot_size(self):
        if np.any(self.z_R == 0):
            w_z = np.array([np.nan, np.nan])
        else:
            w_z = spot_size(z=self.z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)  # spot_size
        return w_z


@dataclass
class ModeParameters:
    # I have once spent a few hours unifying LocalModeParameters and ModeParameters into one class and at the end
    # saw it only makes the code more cumbersome and less readable, so I rolled back.
    center: (
        np.ndarray
    )  # First dimension is theta or phi (the two transversal axes of the mode), second dimension is x, y, z
    k_vector: np.ndarray
    w_0: np.ndarray
    principle_axes: np.ndarray  # First dimension is theta or phi, second dimension is x, y, z
    lambda_0_laser: Optional[float]
    n: float = 1  # refractive index

    def __post_init__(self):
        if not isinstance(self.w_0, np.ndarray):
            raise TypeError(f"waist must be np.ndarray, for both axes, got {type(self.w_0)}")

        if isinstance(self.center, np.ndarray):  # If it is not None
            if not np.isnan(self.center.flat[0]).item():  # If the mode is valid and is not nans
                if self.center.ndim == 1:  # If it has only one axis instead of two:
                    self.center = np.tile(self.center, (2, 1))  # Make it two...

    @property
    def ray(self):
        return Ray(self.center, self.k_vector)

    @property
    def z_R(self):  # The Rayleigh range in vacuum
        if self.lambda_0_laser is None:
            return None
        else:
            return np.pi * self.w_0**2 / self.lambda_0_laser

    @property
    def lambda_laser(self):
        return self.lambda_0_laser / self.n

    @property
    def NA(self):
        if self.lambda_0_laser is None:
            return None
        else:
            if self.z_R[0] == 0 or self.z_R[1] == 0:
                return np.array([np.nan, np.nan])
            else:
                return np.sqrt(self.lambda_0_laser / (np.pi * self.z_R))

    def local_mode_parameters(self, z_minus_z_0):
        return LocalModeParameters(z_minus_z_0=z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)

    def R_of_z(self, p: Union[float, np.ndarray]) -> float:
        if isinstance(p, np.ndarray):
            z_minus_z_0 = (p - self.center) @ self.k_vector
        else:
            z_minus_z_0 = p

        if z_minus_z_0 == 0:
            R_z = np.inf
        else:
            R_z = (z_minus_z_0**2 + self.z_R**2) / z_minus_z_0
        return R_z

    def z_of_R(self, R: float, output_type: type) -> Union[float, np.ndarray]:
        # negative R for negative z, positive R for positive z
        discriminant = 1 - 4 * self.z_R[0] ** 2 / R**2
        if discriminant < 0:
            raise ValueError("R is too small and is never achieved for that mode. R must be larger than 2 * z_R.")

        z_minus_z_0 = R * (1 + np.sqrt(1 - 4 * self.z_R[0] ** 2 / R**2)) / 2

        if output_type == np.ndarray:
            p = self.center[0, :] + z_minus_z_0 * self.k_vector
        elif output_type == float:
            p = z_minus_z_0
        else:
            raise ValueError("output_type must be either np.ndarray or float")

        return p


def decompose_ABCD_matrix(
    ABCD: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ABCD.shape == (4, 4):
        A, B, C, D = (
            ABCD[(0, 2), (0, 2)],
            ABCD[(0, 2), (1, 3)],
            ABCD[(1, 3), (0, 2)],
            ABCD[(1, 3), (1, 3)],
        )
    else:
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
    return A, B, C, D


def propagate_local_mode_parameter_through_ABCD(
    local_mode_parameters: LocalModeParameters, ABCD: np.ndarray, n_1: float = 1, n_2: float = 1
) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(ABCD)
    q_new = n_2 * (A * local_mode_parameters.q / n_1 + B) / (C * local_mode_parameters.q / n_1 + D)
    # q_new = (A * local_mode_parameters.q + B) / (C * local_mode_parameters.q + D)
    return LocalModeParameters(q=q_new, lambda_0_laser=local_mode_parameters.lambda_0_laser, n=n_2)


def local_mode_parameters_of_round_trip_ABCD(
    round_trip_ABCD: np.ndarray,
    n: float,  # refractive_index at the begining of the roundtrip
    lambda_0_laser: Optional[float] = None,
) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(round_trip_ABCD)
    q_z = (A - D + np.sqrt(A**2 + 2 * C * B + D**2 - 2 + 0j)) / (2 * C)
    q_z = np.real(q_z) + 1j * np.abs(np.imag(q_z))  # For the beam amplitude to decay with transverse radius and not
    # to grow, the term -ik * i*im(1/q_z) has to be negative. since in the simulation k is always positive, we take the
    # imaginary part of q_z to be positive (and the imaginary part of its inverse is negative).

    return LocalModeParameters(
        q=q_z, lambda_0_laser=lambda_0_laser, n=n
    )  # First dimension is theta or phi,second dimension is z_minus_z_0 or
    # z_R.


class Ray:
    def __init__(
        self,
        origin: np.ndarray,  # [m_rays..., 3]  # Last index is for x,y,z
        k_vector: np.ndarray,  # [m_rays..., 3]  # Last index is for x,y,z
        length: Union[np.ndarray, float] = np.nan,  # [m_rays...]
        n: float = np.nan,  # [m_rays...], the refractive index in the medium the ray is in. Assumes all rays are in
            # The same medium.
    ):
        if k_vector.ndim == 1 and origin.shape[0] > 1:
            k_vector = np.tile(k_vector, (*origin.shape[:-1], 1))
        elif origin.ndim == 1 and k_vector.shape[0] > 1:
            origin = np.tile(origin, (*k_vector.shape[:-1], 1))

        self.origin = origin  # m_rays | 3
        self.k_vector = normalize_vector(k_vector)  # m_rays | 3
        if length is not None and isinstance(length, float) and origin.ndim > 1:  # If there is one length for many rays
            length = np.ones(origin.shape[0]) * length
        self.length = length  # m_rays or None
        self.n = n  # m_rays or None

    def __getitem__(self, key):
        subscripted_ray = Ray(
            self.origin[key], self.k_vector[key], self.length[key] if self.length is not None else None
        )
        return subscripted_ray

    def parameterization(self, t: Union[np.ndarray, float], optical_path_length: bool = False) -> np.ndarray:
        # Currently this function allows only one t per ray. if needed it can be extended to allow multiple t per ray.
        # theta needs to be either a float or a numpy array with dimensions m_rays
        if isinstance(t, (float, int)):
            t = np.array(t)
        if optical_path_length:
            if np.isnan(self.n):
                raise ValueError("n is None, cannot use optical_path_length=True")
            else:
                n_temp = self.n
        else:
            n_temp = 1
        return self.origin + t[..., np.newaxis] * self.k_vector / n_temp

    @property
    def optical_path_length(self) -> Optional[np.ndarray]:
        if self.length is not None and self.n is not None:
            return self.length * self.n
        else:
            return np.nan

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        dim=2,
        plane: str = "xy",
        length: Union[np.ndarray, float] = np.nan,
        **kwargs,
    ):
        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        if not np.isnan(length):
            length = np.ones(self.origin.shape[:-1]) * length
        elif not np.any(np.isnan(self.length)):
            length = np.where(np.isinf(self.length), 1, self.length)  # If length is inf, we take it to be 0 for plotting purposes
        else:
            length = np.ones_like(self.origin[..., 0])
        ray_origin_reshaped = self.origin.reshape(-1, 3)
        ray_k_vector_reshaped = self.k_vector.reshape(-1, 3)
        lengths_reshaped = length.reshape(-1)
        if dim == 3:
            [
                ax.plot(
                    [
                        ray_origin_reshaped[i, 0],
                        ray_origin_reshaped[i, 0] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 0],
                    ],
                    [
                        ray_origin_reshaped[i, 1],
                        ray_origin_reshaped[i, 1] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 1],
                    ],
                    [
                        ray_origin_reshaped[i, 2],
                        ray_origin_reshaped[i, 2] + lengths_reshaped[i] * ray_k_vector_reshaped[i, 2],
                    ],
                    **kwargs,
                )
                for i in range(ray_origin_reshaped.shape[0])
            ]
        else:
            x_index, y_index = plane_name_to_xy_indices(plane)
            [
                ax.plot(
                    [
                        ray_origin_reshaped[i, x_index],
                        ray_origin_reshaped[i, x_index] + lengths_reshaped[i] * ray_k_vector_reshaped[i, x_index],
                    ],
                    [
                        ray_origin_reshaped[i, y_index],
                        ray_origin_reshaped[i, y_index] + lengths_reshaped[i] * ray_k_vector_reshaped[i, y_index],
                    ],
                    **kwargs,
                )
                for i in range(ray_origin_reshaped.shape[0])
            ]

        return ax

class RaySequence:
    def __init__(self, rays: List[Ray]):
        self.origin = np.stack([ray.origin for ray in rays], axis=0)  # [n_rays, m_rays..., 3]
        self.k_vector = np.stack([ray.k_vector for ray in rays], axis=0)  # [n_rays, m_rays..., 3]
        self.n = np.array([ray.n for ray in rays])  # [n_rays]
        length = np.stack([ray.length for ray in rays], axis=0)  # [n_rays, m_rays...]
        length[np.isnan(length)] = np.inf
        self.length = length  # [n_rays, m_rays...]

    def __getitem__(self, key):
        # If key is a tuple and the second-from-last element is a slice, this case is not implemented.
        if isinstance(key, tuple) and len(key) >= 2 and isinstance(key[0], slice):
            raise NotImplementedError("Slicing over the ray axis (key[-2]) is not implemented")
        subscripted_ray = Ray(
            self.origin[key], self.k_vector[key], self.length[key] if self.length is not None else None
        )
        return subscripted_ray

    def parameterization(self, t: float, optical_path_length: bool = False) -> np.ndarray:
        if isinstance(t, (float, int)):
            t = np.array(t)
        if optical_path_length:
            relevant_lengths_array = self.cumulative_optical_path_length
            relevant_n = self.n
        else:
            relevant_lengths_array = self.cumulative_length
            relevant_n = np.ones_like(self.n)
        output_points = np.zeros(self.origin.shape[1:])
        for i in range(np.prod(self.origin.shape[1:-1])):
            full_index = np.unravel_index(i, self.origin.shape[1:-1])
            first_step_before_t = np.searchsorted(relevant_lengths_array[:, *full_index], t)
            length_before_t = 0 if first_step_before_t == 0 else relevant_lengths_array[first_step_before_t-1, *full_index]
            remaining_t = t - length_before_t
            point_at_t = self.origin[first_step_before_t, *full_index] + remaining_t * self.k_vector[first_step_before_t, *full_index] / relevant_n[first_step_before_t]
            output_points[full_index, :] = point_at_t
        return output_points


    @property
    def optical_path_length(self) -> np.ndarray:
        return self.length * self.n.reshape((self.n.shape[0],) + (1,) * (self.length.ndim - 1))

    @property
    def cumulative_length(self) -> np.ndarray:
        return np.cumsum(self.length, axis=0)

    @property
    def cumulative_optical_path_length(self) -> np.ndarray:
        return np.cumsum(self.optical_path_length, axis=0)

class Surface:
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        self.outwards_normal = normalize_vector(outwards_normal)
        self.name = name
        self.radius = radius
        self.diameter = diameter
        self.material_properties = material_properties

    @property
    def center(self):
        raise NotImplementedError

    @property
    def inwards_normal(self):
        return -self.outwards_normal

    def normal_at_a_point(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_normal_at_a_point(self, point: np.ndarray, k_vector: Optional[np.ndarray]) -> np.ndarray:
        # Normal to a point, pointing forwards along the ray if k_vector is given
        normal = self.normal_at_a_point(point)
        if k_vector is None:
            return normal
        else:
            return normal * np.sign(np.sum(normal * k_vector, axis=-1))[..., np.newaxis]

    def find_intersection_with_ray(self, ray: Ray, paraxial: bool = False) -> np.ndarray:
        if paraxial:
            return self.find_intersection_with_ray_paraxial(ray)
        else:
            return self.find_intersection_with_ray_exact(ray)

    def enrich_intersection_geometries(
        self,
        ray: Ray,
        intersection_point: Optional[np.ndarray] = None,
        forward_normal: Optional[np.ndarray] = None,
        paraxial: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if intersection_point is None:
            intersection_point = self.find_intersection_with_ray(ray, paraxial=paraxial)
        if forward_normal is None:
            forward_normal = self.forward_normal_at_a_point(intersection_point, ray.k_vector)
        return intersection_point, forward_normal

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        # Take parameters and return points on the surface
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        # takes a point on the surface and returns the parameters
        raise NotImplementedError

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 2,
        plane: str = "xy",
        color: Optional[str] = None,
        diameter: Optional[float] = None,
        fine_resolution=False,
        **kwargs,
    ):
        diameter = nvl(nvl(diameter, self.diameter), 7.75e-3)
        if np.isinf(self.radius):
            half_spreading_length = nvl(diameter, 0.01) / 2
        else:
            half_spreading_angle = np.arcsin(min([diameter / (2 * self.radius), 1]))
            half_spreading_length = half_spreading_angle * self.radius
        if fine_resolution:
            N_points = 10000
        else:
            N_points = 100

        if ax is None:
            fig = plt.figure()
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        if dim == 3:
            s = np.linspace(-half_spreading_length, half_spreading_length, N_points)
            t = np.linspace(-half_spreading_length, half_spreading_length, N_points)
        else:
            if plane in ["xy", "yx"]:
                t = 0
                s = np.linspace(-half_spreading_length, half_spreading_length, N_points)
            elif plane in ["xz", "zx"]:
                s = 0
                t = np.linspace(-half_spreading_length, half_spreading_length, N_points)
            elif plane in ["yz", "zy"]:
                s = 0
                t = np.linspace(-half_spreading_length, half_spreading_length, N_points)
            else:
                raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

        T, S = np.meshgrid(t, s)
        points = self.parameterization(T, S)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        if color is None:
            if isinstance(self, CurvedRefractiveSurface):
                color = "grey"
            elif isinstance(self, PhysicalSurface):
                color = "b"
            else:
                color = "black"

        if dim == 3:
            ax.plot_surface(x, y, z, color=color, alpha=0.25, **kwargs)
        else:
            if plane in ["xy", "yx"]:
                x_dummy = points[:, 0, 0]
                y_dummy = points[:, 0, 1]
            elif plane in ["xz", "zx"]:
                x_dummy = points[0, :, 0]
                y_dummy = points[0, :, 2]
            elif plane in ["yz", "zy"]:
                x_dummy = points[0, :, 1]
                y_dummy = points[0, :, 2]
            else:
                raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
            ax.plot(x_dummy, y_dummy, color=color, **kwargs)
        if name is not None:
            name_position = self.parameterization(0.4, 0)
            if dim == 3:
                ax.text(name_position[0], name_position[1], name_position[2], s=name)
            else:
                if (
                    ax.get_xlim()[0] < name_position[0] < ax.get_xlim()[1]
                    and ax.get_ylim()[0] < name_position[1] < ax.get_ylim()[1]
                ):
                    ax.text(name_position[0], name_position[1], s=name)
        if plane in ["xy", "yx"]:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
        elif plane in ["xz", "zx"]:
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
        elif plane in ["yz", "zy"]:
            ax.set_xlabel("y [m]")
            ax.set_ylabel("z [m]")
        if dim == 3:
            ax.set_zlabel("z [m]")

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
        # Returns to vectors that are perpendicular to the inwards normal and to each other.
        # The optical axiMost optical elements, are approximately parallel to the x axis, and so pseudo_y and pseudo_z are approximately y and z.
        parallel_to_actual_z = np.abs(self.inwards_normal @ np.array([0, 0, 1])) > 0.9999
        if not parallel_to_actual_z:
            pseudo_x = np.array([0, 0, 1])
        else:
            pseudo_x = np.array([0, 1, 0])
        pseudo_y = normalize_vector(np.cross(pseudo_x, self.inwards_normal))
        pseudo_z = normalize_vector(np.cross(self.inwards_normal, pseudo_y))
        return pseudo_z, pseudo_y

    @staticmethod
    def from_params(params: OpticalElementParams, name: Optional[str] = None):
        p = params  # Just for brevity in the code
        center = np.array([p.x, p.y, p.z])
        outwards_normal = unit_vector_of_angles(p.theta, p.phi)
        if p.surface_type == SurfacesTypes.curved_mirror:  # Mirror
            surface = CurvedMirror(
                radius=p.r_1,
                outwards_normal=outwards_normal,
                center=center,
                curvature_sign=p.curvature_sign,
                name=p.name,
                diameter=p.diameter,
                thermal_properties=p.material_properties,
            )
        elif p.surface_type == SurfacesTypes.thick_lens:  # ThickLens
            surface = generate_lens_from_params(p)
        elif p.surface_type == SurfacesTypes.ideal_thick_lens:  # IdealThickLens
            surface = generate_thick_ideal_lens_from_params(p)
        elif p.surface_type == SurfacesTypes.curved_refractive_surface:  # Refractive surface (one side of a lens)
            surface = CurvedRefractiveSurface(
                radius=p.r_1,
                outwards_normal=outwards_normal,
                center=center,
                n_1=p.n_outside_or_before,
                n_2=p.n_inside_or_after,
                curvature_sign=p.curvature_sign,
                name=p.name,
                thermal_properties=p.material_properties,
                thickness=p.T_c,
                diameter=p.diameter,
            )
        elif p.surface_type == SurfacesTypes.ideal_lens:  # Ideal lens
            surface = IdealLens(
                outwards_normal=outwards_normal,
                center=center,
                focal_length=p.r_1,
                name=p.name,
                thermal_properties=p.material_properties,
                diameter=p.diameter,
            )
        elif p.surface_type == SurfacesTypes.flat_mirror:  # Flat mirror
            surface = FlatMirror(
                outwards_normal=outwards_normal,
                center=center,
                name=p.name,
                thermal_properties=p.material_properties,
                diameter=p.diameter,
            )
        elif p.surface_type == SurfacesTypes.aspheric_surface:
            return AsphericRefractiveSurface(
                name=p.name,
                center=center,
                outwards_normal=outwards_normal,
                diameter=p.diameter,
                n_1=p.n_outside_or_before,
                n_2=p.n_inside_or_after,
                material_properties=p.material_properties,
                polynomial_coefficients=p.polynomial_coefficients,
            )
        elif p.surface_type == SurfacesTypes.thick_aspheric_lens:
            return generate_aspheric_lens_from_params(p)
        elif p.surface_type == SurfacesTypes.flat_refractive_surface:
            surface = FlatRefractiveSurface(
                outwards_normal=outwards_normal,
                center=center,
                n_1=p.n_outside_or_before,
                n_2=p.n_inside_or_after,
                name=p.name,
                thermal_properties=p.material_properties,
                diameter=p.diameter,
            )
        else:
            raise ValueError(f"Unknown surface type {p.surface_type}")
        return surface

    @property
    def to_params(self) -> OpticalElementParams:
        x, y, z = self.center
        if isinstance(self, IdealLens):
            r_1 = self.focal_length
            r_2 = np.nan
        elif isinstance(self, CurvedSurface):
            r_1 = self.radius
            r_2 = np.nan
        else:
            r_1 = 0
            r_2 = 0
        theta, phi = angles_of_unit_vector(self.outwards_normal)
        n_1 = 1
        n_2 = 1
        if isinstance(self, CurvedMirror):
            surface_type = SurfacesTypes.curved_mirror
            curvature_sign = self.curvature_sign
        elif isinstance(self, CurvedRefractiveSurface):
            surface_type = SurfacesTypes.curved_refractive_surface
            n_1 = self.n_1
            n_2 = self.n_2
            curvature_sign = self.curvature_sign
        elif isinstance(self, IdealLens):
            surface_type = SurfacesTypes.ideal_lens
            curvature_sign = 0
        elif isinstance(self, FlatMirror):
            surface_type = SurfacesTypes.flat_mirror
            curvature_sign = 0
        elif isinstance(self, FlatRefractiveSurface):
            surface_type = SurfacesTypes.flat_refractive_surface
            n_1 = self.n_1
            n_2 = self.n_2
            curvature_sign = 0
        elif isinstance(self, FlatSurface):
            surface_type = SurfacesTypes.flat_surface
            curvature_sign = 0
        elif isinstance(self, AsphericRefractiveSurface):
            surface_type = SurfacesTypes.aspheric_surface
            n_1 = self.n_1
            n_2 = self.n_2
            curvature_sign = self.curvature_sign
        else:
            raise ValueError(f"Unknown surface type {type(self)}")
        if isinstance(self, AsphericSurface):
            polynomial_coefficients = self.polynomial.coef
        else:
            polynomial_coefficients = None
        if self.material_properties is None:
            self.material_properties = MaterialProperties()

        params = OpticalElementParams(
            name=self.name,
            surface_type=surface_type,
            x=x,
            y=y,
            z=z,
            theta=theta,
            phi=phi,
            r_1=r_1,
            r_2=r_2,
            curvature_sign=curvature_sign,
            T_c=np.nan,
            n_inside_or_after=n_2,
            n_outside_or_before=n_1,
            diameter=self.diameter,
            material_properties=self.material_properties,
            polynomial_coefficients=polynomial_coefficients
        )
        return params


class PhysicalSurface(Surface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: Optional[MaterialProperties] = None,
        **kwargs,
    ):

        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            radius=radius,
            diameter=diameter,
            material_properties=material_properties,
            **kwargs,
        )

    @property
    def center(self):
        raise NotImplementedError

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        raise NotImplementedError

    def propagate_ray(self, ray: Ray, paraxial: bool = False) -> Ray:
        # Scatters ray and updates it's length:
        intersection_point, forward_normal = self.enrich_intersection_geometries(ray, paraxial=paraxial)
        ray.length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        scattered_direction_vector = self.scatter_direction(ray, forward_normal, paraxial=paraxial)
        n_output = getattr(self, 'n_2', ray.n)
        return Ray(origin=intersection_point, k_vector=scattered_direction_vector, n=n_output)

    def scatter_direction(
        self, ray: Ray, forward_normal: Optional[np.ndarray] = None, paraxial: bool = False
    ) -> np.ndarray:
        if paraxial:
            return self.scatter_direction_paraxial(ray)
        else:
            return self.scatter_direction_exact(ray, forward_normal=forward_normal)

    def scatter_direction_paraxial(self, ray: Ray) -> np.ndarray:
        forwards_normal = self.forward_normal_at_a_point(self.center, ray.k_vector)

        flat_surface = FlatSurface(
            outwards_normal=forwards_normal,
            center=self.center,
        )
        intersection_point = flat_surface.find_intersection_with_ray(ray, paraxial=True)
        pseudo_z, pseudo_y = flat_surface.spanning_vectors()
        t, p = flat_surface.get_parameterization(
            intersection_point
        )  # Those are the coordinates of pseudo_z and pseudo_y
        t_projection, p_projection = ray.k_vector @ pseudo_z, ray.k_vector @ pseudo_y
        theta, phi = np.pi / 2 - np.arccos(t_projection), np.pi / 2 - np.arccos(p_projection)
        input_vector = np.array([t, theta, p, phi])
        if len(input_vector.shape) > 1:
            input_vector = np.swapaxes(input_vector, 0, 1)
        output_vector = self.ABCD_matrix(cos_theta_incoming=1) @ input_vector  # For the sake of ray tracing, we
        # reflect the ray with respect to the optical element's optical axis, and not with respect to the central line
        # that was even not calculate yet. therefore, the cos_theta_incoming used here is the trivial one.
        if len(input_vector.shape) > 1:
            output_vector = np.swapaxes(output_vector, 0, 1)
        t_projection_out, p_projection_out = cos_without_trailing_epsilon(
            np.pi / 2 - output_vector[1, ...]
        ), cos_without_trailing_epsilon(np.pi / 2 - output_vector[3, ...])
        # Those are the components of the output direction vector in the pseudo_z and pseudo_y and
        # surface_normal directions:
        component_t = np.multiply.outer(t_projection_out, pseudo_z)
        component_p = np.multiply.outer(p_projection_out, pseudo_y)
        component_n = np.multiply.outer((1 - t_projection_out**2 - p_projection_out**2) ** 0.5, forwards_normal)
        output_direction_vector = component_t + component_p + component_n
        return output_direction_vector

    def scatter_direction_exact(
        self,
        ray: Ray,
        intersection_point: Optional[np.ndarray] = None,
        forward_normal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def ABCD_matrix(self, cos_theta_incoming: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError


class ReflectiveSurface(PhysicalSurface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: Optional[MaterialProperties] = None,
        **kwargs,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            radius=radius,
            diameter=diameter,
            material_properties=material_properties,
            **kwargs,
        )

    def scatter_direction_exact(
        self, ray: Ray, intersection_point: Optional[np.ndarray] = None, forward_normal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Notice that this function does not reflect along the normal of the mirror but along the normal projection
        # of the ray on the mirror.
        _, forward_normal = self.enrich_intersection_geometries(
            ray, intersection_point=intersection_point, forward_normal=forward_normal
        )
        reflected_direction_vector = generalized_mirror_law(k_vector=ray.k_vector, n_forwards=forward_normal)
        return reflected_direction_vector


class RefractiveSurface(PhysicalSurface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        radius: float,
        n_1: float = 1,
        n_2: float = 1,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: Optional[MaterialProperties] = None,
        **kwargs,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            radius=radius,
            diameter=diameter,
            material_properties=material_properties,
            **kwargs,
        )
        self.n_1 = n_1
        self.n_2 = n_2

    def scatter_direction_exact(
        self, ray: Ray, intersection_point: Optional[np.ndarray] = None, forward_normal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # explainable derivation of the calculation in lab archives: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-33
        _, n_forwards = self.enrich_intersection_geometries(
            ray, intersection_point=intersection_point, forward_normal=forward_normal
        )
        refracted_direction_vector = generalized_snells_law(
            k_vector=ray.k_vector, n_forwards=n_forwards, n_1=self.n_1, n_2=self.n_2
        )
        return refracted_direction_vector


class AsphericSurface(Surface):
    def __init__(
        self,
        center: np.ndarray,
        outwards_normal: np.ndarray,
        polynomial_coefficients: Union[Polynomial, np.ndarray, List[float]],  # a0, a2, a4...
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.nan, **kwargs)
        self._center = center
        self.outwards_normal = normalize_vector(outwards_normal)
        self.name = name
        self.diameter = diameter
        self.material_properties = material_properties
        self.polynomial = (
            polynomial_coefficients
            if isinstance(polynomial_coefficients, Polynomial)
            else Polynomial(polynomial_coefficients)
        )
        self.thickness_center = self.polynomial((self.diameter / 2) ** 2)  # thickness at the center of the surface
        self.radius = 1 / (2 * self.polynomial.coef[1])

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        # For a sketch and a detalied explanation on the calculation, go to:
        # "Intersection with a cyllindrically symmetric surface with polynominal parameterization x\left(\rho\right)" in my research lyx file #TODO: convert to PDF

        # Flatten rays for independent solves
        origin_original_shape = ray.origin.shape[:-1]
        origin_flattened = ray.origin.reshape(-1, 3)
        k_vector_flattened = ray.k_vector.reshape(-1, 3)

        ray_origin_relative_to_center = (
            origin_flattened - self.center
        )  # (N, 3) points from the origin of the ray to the tip of the surface.
        cosine_theta_incidence_to_center_normal = k_vector_flattened @ self.outwards_normal

        t_1 = (
            ray_origin_relative_to_center @ self.inwards_normal / cosine_theta_incidence_to_center_normal
        )  # end at the plane that is thickness_center away from the center along the surface normal
        t_2 = (
            t_1 - self.thickness_center / cosine_theta_incidence_to_center_normal
        )  # Start from the plane that contains the center and is normal to the surface normal

        # When coming from the convex side, we might have t_min > t_max, so we need to swap them
        t_min = np.minimum(t_1, t_2) - 1e-4  # add a small margin to avoid numerical issues
        t_max = np.maximum(t_1, t_2) + 1e-4

        results = np.full((origin_flattened.shape[0],), np.nan)

        for i in range(origin_flattened.shape[0]):
            # Scalar functions
            def F_i(t):
                r_of_t = origin_flattened[i] + t * k_vector_flattened[i]
                equation_expression = self.defining_equation(r_of_t)
                return equation_expression

            try:
                t_hit = brentq(F_i, t_min[i], t_max[i], xtol=1e-12, rtol=1e-12)
            except ValueError:
                continue

            results[i] = t_hit

        # Reconstruct intersection points
        t = results.reshape(origin_original_shape)
        intersection = ray.parameterization(t)

        return intersection

    def relative_coordinates(self, r: np.ndarray) -> np.ndarray:
        # Convert a global coordinate to it's cylindrical coordinates relative to the surface's optical axis (rho, x)
        r_relative = r - self.center  # r.shape, vector pointing from the center of the surface to the point r
        r_relative_projected_on_n = (
            r_relative @ self.inwards_normal
        )  # r.shape[:-1], longitudinal position of r relative to the center plane along the inwards normal (bigger value means more inwards)
        r_relative_distance_from_center = np.sqrt(
            np.clip(np.sum(r_relative**2, -1) - r_relative_projected_on_n**2, a_min=0, a_max=np.inf)
        )  # r.shape[:-1]  # distance of r from optical axis
        return np.stack([r_relative_distance_from_center, r_relative_projected_on_n], axis=-1)  # \rho, x

    def defining_equation(self, r: np.ndarray) -> Union[np.ndarray, float]:
        # points on the surface satisfy this equation.
        # points on the concave side have positive values (they are "above" the polynomial curve as y-P(x) > 0) and vice versa.
        relative_coordinates = self.relative_coordinates(r)
        rho = relative_coordinates[..., 0]  # r.shape[:-1] distance of r from optical axis
        x = relative_coordinates[
            ..., 1
        ]  # r.shape[:-1], longitudinal position of r relative to the center plane along the inwards normal (bigger value means more inwards)
        polynomial_value = self.polynomial(rho**2)
        equation_expression = x - polynomial_value  # y - P(x)
        return equation_expression

    def normal_at_a_point(self, r: np.ndarray):
        relative_coordinates = self.relative_coordinates(r)
        rho = relative_coordinates[..., 0]
        dP_drho = (
            self.polynomial.deriv()(rho**2) * 2 * rho
        )  # r.shape[:-1]  # P is the polynomial of rho^2, so the derivative of P is dP/drho = dP/d(rho^2) * 2 * rho
        normal_vector_in_surface_coordinates = np.stack(
            [-dP_drho, np.ones_like(dP_drho)], axis=-1
        )  # r.shape[:-1, 2]  # normal vector in the (rho, x) coordinates
        normal_vector_in_surface_coordinates_normalized = normalize_vector(
            normal_vector_in_surface_coordinates
        )  # r.shape[:-1, 2]

        rho_vec = (r - self.center) - ((r - self.center) @ self.inwards_normal)[
            ..., np.newaxis
        ] * self.inwards_normal  # r.shape[:-1, 3]
        # rho_vec[np.linalg.norm(rho_vec, axis=-1) == 0, :] = self.inwards_normal  # It's either this or the True in the next line
        rho_hat = normalize_vector(rho_vec, ignore_null_vectors=True)  # r.shape[:-1, 3]
        normal = (
            normal_vector_in_surface_coordinates_normalized[..., 0, np.newaxis] * rho_hat
            + normal_vector_in_surface_coordinates_normalized[..., 1, np.newaxis] * self.inwards_normal
        )
        return normal

    @property
    def center(self):
        return self._center

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        raise NotImplementedError("No paraxial methods for aspherical surfaces")

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        # Take parameters and return points on the surface
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        # takes a point on the surface and returns the parameters
        raise NotImplementedError

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 2,
        plane: str = "xy",
        color: Optional[str] = None,
        diameter: float = 7.75e-3,
        fine_resolution=False,
        **kwargs,
    ):
        if plane != "xy" or self.outwards_normal[2] != 0:
            raise NotImplementedError("Plotting AsphericSurface is only implemented for the 'xy' plane.")
        if dim != 2:
            raise NotImplementedError("Plotting AsphericSurface is only implemented for 2D plots.")
        if fine_resolution:
            N_points = 10000
        else:
            N_points = 100
        if ax is None:
            fig, ax = plt.subplots()

        t_dummy = np.linspace(-self.diameter / 2, self.diameter / 2, N_points)

        transverse_direction = np.cross(self.outwards_normal, np.array([0, 0, 1]))
        longitudinal_direction = self.inwards_normal

        r = (
            self.center
            + transverse_direction * t_dummy[:, np.newaxis]
            + self.polynomial(t_dummy**2)[:, np.newaxis] * longitudinal_direction
        )
        ax.plot(r[:, 0], r[:, 1], color=color if color is not None else "blue", **kwargs)

        r_back_side = (
            self.center + self.inwards_normal * self.thickness_center + transverse_direction * t_dummy[:, np.newaxis]
        )
        # create kwargs without linestyle to avoid warning:
        kwargs.pop("linestyle", None)
        kwargs.pop("ls", None)
        ax.plot(
            r_back_side[:, 0], r_back_side[:, 1], linestyle="--", color=color if color is not None else "blue", **kwargs
        )


class AsphericRefractiveSurface(AsphericSurface, RefractiveSurface):
    def __init__(
        self,
        center: np.ndarray,
        outwards_normal: np.ndarray,
        polynomial_coefficients: Union[Polynomial, np.ndarray, List[float]],  # a0, a2, a4...
        n_1: float,
        n_2: float,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        curvature_sign: int = CurvatureSigns.concave,  # With respect to the incoming beam.
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        super().__init__(
            center=center,
            outwards_normal=outwards_normal,
            polynomial_coefficients=polynomial_coefficients,
            name=name,
            diameter=diameter,
            material_properties=material_properties,
            n_1=n_1,
            n_2=n_2,
            **kwargs,
        )
        self.curvature_sign = curvature_sign

    def ABCD_matrix(self, cos_theta_incoming: Optional[float] = None) -> np.ndarray:
        paraxial_approximation_surface = CurvedRefractiveSurface(radius=self.radius, outwards_normal=RIGHT, center=ORIGIN, n_1=self.n_1, n_2=self.n_2, curvature_sign=self.curvature_sign)
        return paraxial_approximation_surface.ABCD_matrix(cos_theta_incoming=cos_theta_incoming)

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError


class FlatSurface(Surface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.inf, **kwargs)
        if distance_from_origin is None and center is None:
            raise ValueError("Either distance_from_origin or center must be specified")
        if distance_from_origin is not None and center is not None:
            raise ValueError("Only one of distance_from_origin or center must be specified")
        if distance_from_origin is not None:
            self.distance_from_origin = distance_from_origin
            self.center_of_mirror_private = self.outwards_normal * distance_from_origin
        if center is not None:
            self.center_of_mirror_private = center
            self.distance_from_origin = center @ self.outwards_normal

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        surface_reference_point = self.outwards_normal * self.distance_from_origin
        ray_origin_to_surface_reference_point = surface_reference_point - ray.origin
        ray_origin_distance_from_surface = ray_origin_to_surface_reference_point @ self.outwards_normal
        cos_angle_between_ray_direction_and_plane_normal = ray.k_vector @ self.outwards_normal
        ray_length_to_surface = ray_origin_distance_from_surface / cos_angle_between_ray_direction_and_plane_normal
        intersection_point = ray.parameterization(ray_length_to_surface)
        return intersection_point

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        # Notes are available here: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-36
        cos_theta = self.outwards_normal @ ray.k_vector
        if cos_theta > 0:
            forwards_normal = self.outwards_normal
        else:
            forwards_normal = self.inwards_normal
            cos_theta = np.abs(cos_theta)
        sin_abs_theta = np.linalg.norm(np.cross(ray.k_vector, forwards_normal))
        theta = np.arcsin(sin_abs_theta)  # You might as - but wait, can't we use the arccos of the cos_theta already
        # calculated? the answer is no, because d/dx(cos) is 0 around 0 and d/dx(arccos) is infinite around 0, Which
        # leads to numerical instability when dealing with small angles.

        closest_point_in_plane_to_global_origin = self.distance_from_origin * self.outwards_normal  # v in notes

        displacement_in_plane = ray.origin - (forwards_normal @ ray.origin) * forwards_normal

        ray_origin_projected_onto_plane = (
            closest_point_in_plane_to_global_origin + displacement_in_plane
        )  # p_r in notes

        distance_between_rays_origin_and_plane = np.abs(
            self.distance_from_origin - (self.outwards_normal @ ray.origin)
        )  # h in notes

        vector_in_plane_in_k_n_plane = ray.k_vector - cos_theta * forwards_normal  # u in notes

        if np.linalg.norm(vector_in_plane_in_k_n_plane) < 1e-20:
            intersection_point = ray_origin_projected_onto_plane
        else:
            vector_in_plane_in_k_n_plane = normalize_vector(vector_in_plane_in_k_n_plane)
            intersection_point = (
                ray_origin_projected_onto_plane
                + theta * distance_between_rays_origin_and_plane * vector_in_plane_in_k_n_plane
            )
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

    def normal_at_a_point(self, point: np.ndarray) -> np.ndarray:
        outwards_normal_reshaped = np.broadcast_to(self.outwards_normal, point.shape).copy()
        return outwards_normal_reshaped


class FlatMirror(FlatSurface, ReflectiveSurface):

    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        diameter: Optional[float] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
            radius=np.inf,
            diameter=diameter,
        )

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 3,
        length=0.6,
        plane: str = "xy",
    ):
        return super().plot(ax, name, dim, length, plane)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # Assumes the ray is in the x-y plane, and the mirror is in the z-x plane
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

    @property
    def radius(self):
        return np.inf

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError


class FlatRefractiveSurface(FlatSurface, RefractiveSurface):

    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        n_1: float = 1,
        n_2: float = 1,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        diameter: Optional[float] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
            diameter=diameter,
        )
        self.n_1 = n_1
        self.n_2 = n_2

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # Note ! this code assumes the ray is in the x-y plane! Until it is fixed, the only perturbations in x,y,phi should be calculated!
        sin_theta_incoming = np.sqrt(1 - cos_theta_incoming**2)
        sin_theta_outgoing = (self.n_1 / self.n_2) * sin_theta_incoming
        cos_theta_outgoing = stable_sqrt(1 - sin_theta_outgoing**2)
        return np.array(
            [
                [1, 0, 0, 0],
                [0, self.n_1 / self.n_2, 0, 0],
                [0, 0, cos_theta_outgoing / cos_theta_incoming, 0],
                [0, 0, 0, (self.n_1 * cos_theta_incoming) / (self.n_2 * cos_theta_outgoing)],
            ]
        )


class IdealLens(FlatSurface, PhysicalSurface):
    def __init__(
        self,
        outwards_normal: np.ndarray,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        diameter: Optional[float] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
            diameter=diameter,
        )
        self.focal_length = focal_length

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 3,
        length=0.6,
        plane: str = "xy",
    ):
        return super().plot(ax, name, dim, length, plane)

    def get_parameterization(self, points: np.ndarray):
        return super().get_parameterization(points)

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        return super().parameterization(t, p)

    @property
    def center(self):
        return super().center

    def scatter_direction_exact(
        self,
        ray: Ray,
        intersection_point: Optional[np.ndarray] = None,
        forward_normal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        intersection_point, forward_normal = self.enrich_intersection_geometries(
            ray,
            intersection_point=intersection_point,
            forward_normal=forward_normal,
        )
        pseudo_z, pseudo_y = self.spanning_vectors()
        t, p = self.get_parameterization(intersection_point)  # Those are the coordinates of pseudo_z and pseudo_y
        t_projection, p_projection = ray.k_vector @ pseudo_z, ray.k_vector @ pseudo_y
        theta, phi = np.pi / 2 - np.arccos(t_projection), np.pi / 2 - np.arccos(p_projection)
        input_vector = np.array([t, theta, p, phi])
        if len(input_vector.shape) > 1:
            input_vector = np.swapaxes(input_vector, 0, 1)
        output_vector = self.ABCD_matrix(cos_theta_incoming=1) @ input_vector
        if len(input_vector.shape) > 1:
            output_vector = np.swapaxes(output_vector, 0, 1)
        t_projection_out, p_projection_out = np.cos(np.pi / 2 - output_vector[1, ...]), np.cos(
            np.pi / 2 - output_vector[3, ...]
        )
        # ABCD_MATRIX METHOD
        component_t = np.multiply.outer(t_projection_out, pseudo_z)
        component_p = np.multiply.outer(p_projection_out, pseudo_y)
        component_n = np.multiply.outer((1 - t_projection_out**2 - p_projection_out**2) ** 0.5, forward_normal)
        output_direction_vector = component_t + component_p + component_n

        return output_direction_vector

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        # THIS CURRENTLY DOES NOT HOLD FOR THE CASE WHERE THE RAY IS NOT PERPENDICULAR TO THE LENS!
        ABCD = np.array(
            [
                [1, 0, 0, 0],
                [-1 / self.focal_length, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1 / self.focal_length, 1],
            ]
        )
        return ABCD

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float):
        raise NotImplementedError


class CurvedSurface(Surface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of
        # the plate.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        curvature_sign: int = 1,
        # 1 for concave (where the ray is hitting the sphere from the inside) and -1 for convex
        # (where the ray is hitting the sphere from the outside). this is used to find the correct intersection
        # point of a ray with the surface
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=radius, **kwargs)
        self.curvature_sign = curvature_sign
        if origin is None and center is None:
            raise ValueError("Either origin or center must be provided.")
        elif origin is not None and center is not None:
            raise ValueError("Only one of origin and center must be provided.")
        elif origin is None:
            self.origin = center + radius * self.inwards_normal
        else:
            self.origin = origin

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        # The following expression is the result of calculation "Intersection of a parameterized line and a sphere"
        # in the research lyx file
        Delta = ray.origin - self.origin  # m_rays | 3
        Delta_squared = np.sum(Delta**2, axis=-1)  # m_rays
        Delta_projection_on_k = np.sum(Delta * ray.k_vector, axis=-1)  # m_rays
        with np.errstate(invalid="ignore"):
            length = -Delta_projection_on_k + self.curvature_sign * np.sqrt(
            Delta_projection_on_k**2 - Delta_squared + self.radius**2
        )
        intersection_point = ray.parameterization(length)
        return intersection_point

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        flat_surface = FlatSurface(center=self.center, outwards_normal=self.outwards_normal)
        intersection_point = flat_surface.find_intersection_with_ray_paraxial(ray)
        return intersection_point

    def parameterization(
        self,
        t: Union[np.ndarray, float],  # the length of arc to travel on the sphere from the center
        # of the mirror to the point of interest, in the direction "pseudo_z". pseudo_z is
        # described in the get_spanning_vectors method. it is analogous to theta / R in the
        # classical parameterization.
        p: Union[np.ndarray, float],  # The same as theta but in the direction of pseudo_y. It is analogous
        # to phi / R in the classical parameterization.
    ) -> np.ndarray:
        # This parameterization treats the sphere as if as the center of the mirror was on the x-axis.
        # The conceptual difference between this parameterization and the classical one of [sin(theta)cos(phi),
        # sin(theta)sin(phi), cos(theta)]] is that here there is barely any Jacobian determinant.
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        # Notice how the order of rotations matters. First we rotate around the z axis, then around the y-axis.
        # Doing it the other way around would give parameterization that is not aligned with the conventional theta, phi
        # parameterization. This is important for the get_parameterization method.
        rotation_matrix = rotation_matrix_around_n(pseudo_y, -t / self.radius) @ rotation_matrix_around_n(
            pseudo_z, p / self.radius
        )  # The minus sign is because of the
        # orientation of the pseudo_y axis.

        points = self.origin + self.radius * rotation_matrix @ self.outwards_normal
        return points

    def get_parameterization(self, points: np.ndarray):
        pseudo_y, pseudo_z = self.get_spanning_vectors()
        normalized_points = (points - self.origin) / self.radius
        p = np.arctan2(normalized_points @ pseudo_y, normalized_points @ self.outwards_normal) * self.radius
        # Notice that theta is like theta but instead of ranging in [0, pi] it ranges in [-pi/2, pi/2].
        t = np.arcsin(np.clip(normalized_points @ pseudo_z, -1, 1)) * self.radius
        return t, p

    @property
    def center(self):
        return self.origin + self.radius * self.outwards_normal

    def get_spanning_vectors(self):
        # For the case of the sphere with normal on the x-axis, those will be the y and z axis.
        # For the case of the sphere with normal on the y-axis, those will be the x and z axis.
        pseudo_y = np.cross(np.array([0, 0, 1]), self.inwards_normal)
        pseudo_z = np.cross(self.inwards_normal, pseudo_y)  # Should be approximately equal to \hat{z}, and exactly
        # equal if the outwards_normal is in the x-y plane.
        return pseudo_y, pseudo_z

    def normal_at_a_point(self, point: np.ndarray) -> np.ndarray:
        normal = point - self.origin
        normal = normalize_vector(normal)
        return normal

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        name: Optional[str] = None,
        dim: int = 2,
        plane: str = "xy",
        diameter: Optional[float] = 7.75e-3,
        fine_resolution: bool = False,
        **kwargs,
    ):
        if diameter is None:
            diameter = 0.6 * self.radius
        super().plot(ax, name, dim, diameter=diameter, plane=plane, fine_resolution=fine_resolution, **kwargs)


class CurvedMirror(CurvedSurface, ReflectiveSurface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: np.ndarray = None,  # Not the center of the sphere but the center of
        # the plate, where the beam should hit.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        curvature_sign: int = 1,
        name: Optional[str] = None,
        diameter: float = np.nan,
        thermal_properties: Optional[MaterialProperties] = None,
    ):

        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            radius=radius,
            center=center,
            origin=origin,
            diameter=diameter,
            curvature_sign=curvature_sign,
        )

    def scatter_direction_paraxial(self, ray: Ray) -> np.ndarray:
        # This is maybe wrong but does not matter too much because anyway they are not used for the central line finding
        # ATTENTION - THIS SHOULD NOT BE HERE FOR NON-STANDING WAVES CAVITIES - BUT i AM DEALING ONLY WITH THOSE...
        return self.scatter_direction_exact(ray)
        # intersection_point = self.find_intersection_with_ray(ray, paraxial=True)
        # return self.reflect_direction_exact(ray, intersection_point=intersection_point)

    def ABCD_matrix(self, cos_theta_incoming: float = None):
        # order of rows/columns elements is [theta, theta, phi, phi]
        # An approximation is done here (beyond the small angles' approximation) by assuming that the central line
        # lives in the x,y plane, such that the plane of incidence is the x,y plane (parameterized by phi and phi)
        # and the sagittal plane is its transverse (parameterized by theta and theta).
        # This is justified for small perturbations of a cavity whose central line actually lives in the x,y plane.
        # It is not really justified for bigger perturbations and should be corrected.
        # It should be corrected by first finding the real axes, # And then apply a rotation matrix to this matrix on
        # both sides.
        # ATTENTION - THIS SHOULD NOT BE HERE FOR NON-STANDING WAVES CAVITIES - BUT i AM DEALING ONLY WITH THOSE...
        cos_theta_incoming = 1
        ABCD = np.array(
            [
                [1, 0, 0, 0],
                [-2 * cos_theta_incoming / self.radius, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 2 / (self.radius * cos_theta_incoming), -1],
            ]
        )
        return ABCD

    def plot_2d(self, ax: Optional[plt.Axes] = None, name: Optional[str] = None):
        if ax is None:
            fig, ax = plt.subplots()
        d_theta = 0.3
        p = np.linspace(-d_theta, d_theta, 50)
        p_grey = np.linspace(d_theta, -d_theta + 2 * np.pi, 100)
        points = self.parameterization(0, p)
        grey_points = self.parameterization(0, p_grey)
        ax.plot(points[:, 0], points[:, 1], "b-")
        ax.plot(
            grey_points[:, 0],
            grey_points[:, 1],
            color=(0.81, 0.81, 0.81),
            linestyle="-.",
            linewidth=0.5,
            label=None,
        )
        ax.plot(self.origin[0], self.origin[1], "bo")

    def thermal_transformation(
        self,
        P_laser_power: float,
        w_spot_size: float,
        transform_mirror: bool = True,
        **kwargs,
    ):
        if not transform_mirror or np.isnan(w_spot_size):
            return self
        else:
            poisson_ratio_factor = (1 + self.material_properties.nu_poisson_ratio) / (
                1 - self.material_properties.nu_poisson_ratio
            )
            delta_T = (
                PHYSICAL_SIZES_DICT["c_mirror_radius_expansion"]
                * P_laser_power
                * self.material_properties.beta_surface_absorption
                / (self.material_properties.kappa_conductivity * w_spot_size)
            )
            delta_curvature = (
                -delta_T * self.material_properties.alpha_expansion * poisson_ratio_factor / w_spot_size
            )  # The minus is because we are cooling it down.
            # delta_z = delta_curvature * w_spot_size ** 2  # Technically the curvature is calculated based on this delta_z, but I skip it in the code and calculate the curvature directly.
            new_radius = (self.radius**-1 + delta_curvature) ** -1  # ARBITRARY - TAKING ONLY THE T AXIS
            self.material_properties.temperature = ROOM_TEMPERATURE - delta_T  # The delta_T is negative, and after
            # cooling the mirror goes to room temperature. Therefore, the temperature is when heated is the room
            # temperature minus the delta_T.

            new_thermal_properties = copy.copy(self.material_properties)
            new_thermal_properties.temperature = delta_T

            new_mirror = CurvedMirror(
                radius=new_radius,
                outwards_normal=self.outwards_normal,
                center=self.center,  # + (new_radius - self.radius) * self.inwards_normal,
                thermal_properties=new_thermal_properties,
            )
            return new_mirror


class CurvedRefractiveSurface(CurvedSurface, RefractiveSurface):
    def __init__(
        self,
        radius: float,
        outwards_normal: np.ndarray,  # Pointing from the origin of the sphere to the mirror's center.
        center: Optional[np.ndarray] = None,  # Not the center of the sphere but the center of the plate.
        origin: Optional[np.ndarray] = None,  # The center of the sphere.
        n_1: float = 1,
        n_2: float = 1.5,
        curvature_sign: int = 1,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        thickness: Optional[float] = 5e-4,
        diameter: Optional[float] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            radius=radius,
            center=center,
            origin=origin,
            curvature_sign=curvature_sign,
            diameter=diameter,
        )
        self.n_1 = n_1
        self.n_2 = n_2
        self.thickness = thickness

    def ABCD_matrix(self, cos_theta_incoming: float = None) -> np.ndarray:
        cos_theta_outgoing = np.sqrt(1 - (self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming**2))
        R_signed = self.radius * self.curvature_sign
        delta_n_e_out_of_plane = self.n_2 * cos_theta_outgoing - self.n_1 * cos_theta_incoming
        delta_n_e_in_plane = delta_n_e_out_of_plane / (cos_theta_incoming * cos_theta_outgoing)

        # See the comment in the ABCD_matrix method of the CurvedSurface class for an explanation of the approximation.
        ABCD = np.array(
            [
                [1, 0, 0, 0],  # theta
                [delta_n_e_out_of_plane / (R_signed * self.n_2), self.n_1 / self.n_2, 0, 0],  # theta
                [0, 0, cos_theta_outgoing / cos_theta_incoming, 0],  # phi
                [
                    0,
                    0,
                    delta_n_e_in_plane / (R_signed * self.n_2),
                    cos_theta_incoming * self.n_1 / (cos_theta_outgoing * self.n_2),
                ],
            ]
        )  # phi
        return ABCD

    def thermal_transformation(
        self,
        P_laser_power: float,
        w_spot_size: float,
        n_surface_transform_lens: bool = True,
        n_volumetric_transform_lens: bool = True,
        curvature_transform_lens: bool = True,
        change_lens_by_changing_n: bool = False,
        change_lens_by_changing_R: bool = True,
        z_transform_lens: bool = False,
        **kwargs,
    ):
        # This function follows the derivations from the file https://mynotebook.labarchives.com/doc/view/MTA3Ljl8MTA1ODU5NS84My9FbnRyeVBhcnQvMjE1NTkxNDI0fDI3My45?nb_id=MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ%3D%3D
        if np.isnan(w_spot_size):
            return self
        n_inside = np.max((self.n_1, self.n_2))
        delta_T_volumetric = (
            PHYSICAL_SIZES_DICT["c_lens_volumetric_absorption"]
            * self.material_properties.alpha_volume_absorption
            * P_laser_power
            / self.material_properties.kappa_conductivity
        )  # ARBITRARY - CHANGE THE DIMENSIONLESS CONSTANT
        delta_T_surface = (
            PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"]
            * self.material_properties.beta_surface_absorption
            * P_laser_power
            / (self.material_properties.kappa_conductivity * w_spot_size)
        )  # ARBITRARY - CHANGE THE DIMENSIONLESS CONSTANT
        delta_T = delta_T_volumetric + delta_T_surface
        self.material_properties.temperature = ROOM_TEMPERATURE - delta_T

        common_coefficient = (
            self.material_properties.beta_surface_absorption
            * P_laser_power
            / (self.material_properties.kappa_conductivity * w_spot_size**2)
        )
        delta_optical_length_curvature_n_surface = (
            -PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"] * common_coefficient * self.material_properties.dn_dT
        )
        delta_optical_length_curvature_n_volumetric = (
            -PHYSICAL_SIZES_DICT["c_lens_volumetric_absorption"]
            * self.material_properties.alpha_volume_absorption
            * P_laser_power
            * self.material_properties.dn_dT
            / self.material_properties.kappa_conductivity
            * (1 / self.radius + self.thickness / w_spot_size**2)
        )  # (1 / self.radius self.thickness / w_spot_size ** 2) The last parenthesis should be this but the 1/R is negligible.
        delta_optical_length_curvature_buldging = (
            -PHYSICAL_SIZES_DICT["c_lens_focal_length_expansion"]
            * common_coefficient
            * self.material_properties.alpha_expansion
            * n_inside
            * (1 + self.material_properties.nu_poisson_ratio)
            / (1 - self.material_properties.nu_poisson_ratio)
        )

        delta_optical_length_curvature = (
            delta_optical_length_curvature_n_surface * n_surface_transform_lens
            + delta_optical_length_curvature_n_volumetric * n_volumetric_transform_lens
            + delta_optical_length_curvature_buldging * curvature_transform_lens
        )

        if change_lens_by_changing_n:  # Equation (2) from the documentation in the link above
            radius_new = self.radius
            n_new = n_inside - delta_optical_length_curvature * self.radius

        elif change_lens_by_changing_R:  # Equation (3) from the documentation in the link above
            radius_new = n_inside * self.radius / (n_inside - delta_optical_length_curvature * self.radius)
            n_new = n_inside
        else:
            raise ValueError("at least change_lens_by_changing_n or change_lens_by_changing_R has to be True")

        if self.n_1 == 1:
            n_1 = 1
            n_2 = n_new
        else:
            n_1 = n_new
            n_2 = 1

        if z_transform_lens:
            # delta_z = 0
            # center_new = self.center + delta_z * self.outwards_normal
            raise NotImplementedError
        else:
            center_new = self.center

        new_thermal_properties = copy.copy(self.material_properties)
        new_thermal_properties.temperature = ROOM_TEMPERATURE

        return CurvedRefractiveSurface(
            radius=radius_new,
            outwards_normal=self.outwards_normal,
            center=center_new,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=self.curvature_sign,
            name=self.name,
            thermal_properties=new_thermal_properties,
            diameter=self.diameter,
        )


def generate_lens_from_params(params: OpticalElementParams):
    p = params
    # generates a convex-convex lens from the parameters
    center = np.array([p.x, p.y, p.z])
    forward_direction = unit_vector_of_angles(p.theta, p.phi)

    # Generate names according to the direction the lens is pointing to
    main_axis = np.argmax(np.abs(forward_direction))
    directions_nams = [["_left", "_right"], ["_down", "_up"], ["_back", "_front"]]
    suffixes = directions_nams[main_axis]
    if forward_direction[main_axis] < 0:
        suffixes = suffixes[::-1]
    if p.name is None:
        name = "Lens"
    else:
        name = p.name
    names = [name + suffix for suffix in suffixes]

    center_1 = center - (1 / 2) * p.T_c * forward_direction
    center_2 = center + (1 / 2) * p.T_c * forward_direction
    surface_1 = CurvedRefractiveSurface(
        radius=p.r_1,
        outwards_normal=-forward_direction,
        center=center_1,
        n_1=p.n_outside_or_before,
        n_2=p.n_inside_or_after,
        curvature_sign=-1,
        name=names[0],
        thermal_properties=p.material_properties,
        thickness=p.T_c / 2,
        diameter=p.diameter,
    )

    surface_2 = CurvedRefractiveSurface(
        radius=p.r_2,
        outwards_normal=forward_direction,
        center=center_2,
        n_1=p.n_inside_or_after,
        n_2=p.n_outside_or_before,
        curvature_sign=1,
        name=names[1],
        thermal_properties=p.material_properties,
        thickness=p.T_c / 2,
        diameter=p.diameter,
    )
    return surface_1, surface_2


def generate_aspheric_lens_from_params(params: OpticalElementParams):
    p = params
    # generates a convex-convex lens from the parameters
    center = np.array([p.x, p.y, p.z])
    forward_direction = unit_vector_of_angles(p.theta, p.phi)

    # Generate names according to the direction the lens is pointing to
    suffixes = [" - flat side", " - curved side"]
    if p.name is None:
        name = "Aspheric Lens"
    else:
        name = p.name
    names = [name + suffix for suffix in suffixes]

    center_1 = center
    center_2 = center + p.T_c * forward_direction
    surface_1 = FlatRefractiveSurface(
        outwards_normal=-forward_direction,
        center=center_1,
        n_1=p.n_outside_or_before,
        n_2=p.n_inside_or_after,
        name=names[0],
        thermal_properties=p.material_properties,
        diameter=p.diameter,
    )

    surface_2 = AsphericRefractiveSurface(
        polynomial_coefficients=p.polynomial_coefficients,
        outwards_normal=forward_direction,
        center=center_2,
        n_1=p.n_inside_or_after,
        n_2=p.n_outside_or_before,
        name=names[1],
        thermal_properties=p.material_properties,
        thickness=p.T_c / 2,
        diameter=p.diameter,
    )
    return surface_1, surface_2


def generate_aspheric_lens_params(
    f: float,
    T_c: float,
    n: float,
    forward_normal: np.ndarray,
    flat_faces_center: np.ndarray,
    diameter: float,
    polynomial_degree: int = 6,
    n_outside: float = 1.0,
    material_properties: Optional[MaterialProperties] = None,
    name: Optional[str] = None,
):
    p = LensParams(n=n, f=f, T_c=T_c)
    coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=polynomial_degree)
    params = OpticalElementParams(
        name=name,
        surface_type=SurfacesTypes.thick_aspheric_lens,
        x=flat_faces_center[0],
        y=flat_faces_center[1],
        z=flat_faces_center[2],
        theta=angles_of_unit_vector(forward_normal)[0],
        phi=angles_of_unit_vector(forward_normal)[1],
        r_1=np.inf,
        r_2=1/(2*coeffs[1]),
        curvature_sign=0,
        diameter=diameter,
        polynomial_coefficients=coeffs,
        T_c=T_c,
        n_inside_or_after=n,
        n_outside_or_before=n_outside,
        material_properties=material_properties,
    )
    return params


def convert_curved_refractive_surface_to_ideal_lens(surface: CurvedRefractiveSurface):
    focal_length = 1 / (surface.n_2 - surface.n_1) * surface.radius * (-1 * surface.curvature_sign)
    ideal_lens = IdealLens(
        outwards_normal=surface.outwards_normal,
        center=surface.center,
        focal_length=focal_length,
        name=surface.name,
        thermal_properties=surface.material_properties,
        diameter=surface.diameter,
    )

    flat_refractive_surface = FlatRefractiveSurface(
        outwards_normal=surface.outwards_normal,
        center=surface.center,
        n_1=surface.n_1,
        n_2=surface.n_2,
        name=surface.name + "_refractive_surface",
        thermal_properties=surface.material_properties,
        diameter=surface.diameter,
    )

    return ideal_lens, flat_refractive_surface


def generate_thick_ideal_lens_from_params(params: OpticalElementParams):
    surface_1, surface_4 = generate_lens_from_params(params)
    ideal_lens_1 = convert_curved_refractive_surface_to_ideal_lens(surface_1)
    ideal_lens_2 = convert_curved_refractive_surface_to_ideal_lens(surface_4)
    return ideal_lens_1, ideal_lens_2


class Arm:
    def __init__(
        self,
        surface_0: Surface,
        surface_1: Surface,
        central_line: Optional[Ray] = None,
        mode_parameters_on_surface_0: Optional[LocalModeParameters] = None,
        mode_parameters_on_surface_1: Optional[LocalModeParameters] = None,
        mode_principle_axes: Optional[np.ndarray] = None,
    ):
        if isinstance(surface_0, RefractiveSurface):
            self.n: float = surface_0.n_2
        elif isinstance(surface_1, RefractiveSurface):
            self.n: float = surface_1.n_1
        else:
            self.n: float = 1.0

        if mode_parameters_on_surface_0 is None:
            mode_parameters_on_surface_0: LocalModeParameters = LocalModeParameters(
                q=np.nan, lambda_0_laser=np.nan, n=self.n
            )
        if mode_parameters_on_surface_1 is None:
            mode_parameters_on_surface_1: LocalModeParameters = LocalModeParameters(
                q=np.nan, lambda_0_laser=np.nan, n=self.n
            )
        self.surface_0: Surface = surface_0
        self.surface_1: Surface = surface_1
        self.mode_parameters_on_surface_0: LocalModeParameters = mode_parameters_on_surface_0
        self.mode_parameters_on_surface_1: LocalModeParameters = mode_parameters_on_surface_1
        self.central_line: Ray = central_line
        self.mode_principle_axes: Optional[np.ndarray] = mode_principle_axes

        if isinstance(surface_0, CurvedRefractiveSurface) and isinstance(surface_1, CurvedRefractiveSurface):
            assert (
                surface_0.n_2 == surface_1.n_1
            ), "The refractive index according to first element is not the same as the refractive index according to the second element"

    def propagate_ray(self, ray: Ray, use_paraxial_ray_tracing: bool = False):
        ray.n = self.n
        if isinstance(self.surface_1, PhysicalSurface):
            # ATTENTION - THIS SHOULD NOT BE HERE FOR NON-STANDING WAVES CAVITIES - BUT I AM DEALING ONLY WITH THOSE...
            if isinstance(self.surface_1, CurvedMirror):
                use_paraxial_ray_tracing = False
            propagated_ray = self.surface_1.propagate_ray(ray, paraxial=use_paraxial_ray_tracing)
        else:
            new_position = self.surface_1.find_intersection_with_ray(ray, paraxial=use_paraxial_ray_tracing)
            ray.length = np.linalg.norm(new_position - ray.origin, axis=-1)
            propagated_ray = Ray(new_position, ray.k_vector, n=ray.n)

        return propagated_ray

    @property
    def lambda_0_laser(self):
        if self.mode_parameters_on_surface_0 is not None:
            return self.mode_parameters_on_surface_0.lambda_0_laser
        else:
            return None

    @property
    def ABCD_matrix_free_space(self):
        if self.central_line is None:
            raise ValueError("Central line not set")
        matrix = ABCD_free_space(self.central_line.length)
        return matrix

    @property
    def ABCD_matrix_reflection(self):
        if self.central_line is None:
            raise ValueError("Central line not set")
        cos_theta = np.abs(self.central_line.k_vector @ self.surface_1.outwards_normal)  # ABS because we want the
        # angle between the ray and the normal to be positive
        if isinstance(self.surface_1, PhysicalSurface):
            matrix = self.surface_1.ABCD_matrix(cos_theta)
        else:
            matrix = np.eye(4)
        return matrix

    @property
    def ABCD_matrix(self):
        matrix = self.ABCD_matrix_reflection @ self.ABCD_matrix_free_space
        return matrix

    def propagate_local_mode_parameters(self):
        if self.mode_parameters_on_surface_0 is None:
            raise ValueError("Mode parameters on surface 0 not set")
        self.mode_parameters_on_surface_1 = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_0, self.ABCD_matrix_free_space, n_1=self.n, n_2=self.n
        )
        next_mode_parameters = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_1,
            self.ABCD_matrix_reflection,
            n_1=self.surface_1.to_params.n_outside_or_before,
            n_2=self.surface_1.to_params.n_inside_or_after,
        )
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
    def valid_mode_inside(self):
        if np.all(self.mode_parameters_on_surface_0.z_R[0] > 0):
            return True
        else:
            return False

    @property
    def mode_parameters(self):
        if np.isnan(self.mode_parameters_on_surface_0.z_R[0]):
            return ModeParameters(
                center=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
                k_vector=np.array([np.nan, np.nan, np.nan]),
                w_0=np.array([np.nan, np.nan]),
                principle_axes=np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]),
                lambda_0_laser=self.lambda_0_laser,
                n=self.n,
            )
        self.mode_parameters_on_surface_0.z_R[0]
        center = (
            self.central_line.origin
            - self.mode_parameters_on_surface_0.z_minus_z_0[..., np.newaxis] / self.n * self.central_line.k_vector
        )
        mode_parameters = ModeParameters(
            center=center,
            k_vector=self.central_line.k_vector,
            w_0=self.mode_parameters_on_surface_0.w_0,
            principle_axes=self.mode_principle_axes,
            lambda_0_laser=self.lambda_0_laser,
            n=self.n,
        )
        return mode_parameters

    def local_mode_parameters_on_a_point(self, point: np.ndarray) -> LocalModeParameters:
        if self.central_line is None:
            raise ValueError("Central line not set")
        if np.isnan(self.mode_parameters_on_surface_0.q):
            return self.mode_parameters_on_surface_0

        point_plane_distance_from_surface_1 = (point - self.central_line.origin) @ self.central_line.k_vector
        propagation_ABCD = ABCD_free_space(point_plane_distance_from_surface_1)
        local_mode_parameters = propagate_local_mode_parameter_through_ABCD(
            self.mode_parameters_on_surface_0, propagation_ABCD, n_1=self.n, n_2=self.n
        )
        return local_mode_parameters

    @property
    def surfaces(self):
        return [self.surface_0, self.surface_1]

    @property
    def mode_parameters_on_surfaces(self):
        return [self.mode_parameters_on_surface_0, self.mode_parameters_on_surface_1]

    def calculate_incidence_angle(self, surface_index: int) -> float:

        return calculate_incidence_angle(
            surface=self.surfaces[surface_index],
            mode_parameters=self.mode_parameters,
        )

    def specs(self):
        list_of_data_frames = []
        for i in [0, 1]:
            local_mode_parameters = self.mode_parameters_on_surfaces[i]
            spot_size_on_surface = local_mode_parameters.spot_size[0]
            surface = self.surfaces[i]
            if isinstance(surface, CurvedRefractiveSurface):
                if i == 0 and surface.n_2 == 1 or i == 1 and surface.n_1 == 1:
                    angle_side = ""
                else:
                    angle_side = "_inside"
            else:
                angle_side = ""
            if local_mode_parameters.spot_size[0] != local_mode_parameters.spot_size[1]:
                warnings.warn("Not yep implemented for astigmatic systems! using the spot size for one arbitrary axis")

            angle_of_incidence_deg = calculate_incidence_angle(self.surfaces[i], self.mode_parameters)

            df = pd.DataFrame(
                {
                    "Element": [surface.name] * 4,
                    "Parameter": [
                        "Spot size diameter [m]",
                        "Minimal clear aperture diameter [m]",
                        "Temperature raise [K]",
                        f"Angle of incidence{angle_side} [deg]",
                    ],
                    "Value": [
                        spot_size_on_surface * 2,
                        spot_size_on_surface * 2 * 2.5,
                        surface.material_properties.temperature - ROOM_TEMPERATURE,
                        angle_of_incidence_deg,
                    ],
                },
            )

            list_of_data_frames.append(df)
        joined_df = pd.concat(list_of_data_frames)
        return joined_df

    @property
    def acquired_gouy_phase_per_axis(self):
        # The acquired gouy phase for the (0,0)'th mode. for any greater mode, (n,m) it will be:
        # (n+m+1) * acquired_gouy_phase_value
        if self.mode_parameters_on_surface_0 is None:
            return None
        if self.mode_parameters_on_surface_1 is None:
            self.propagate_local_mode_parameters()
        # The 1/2 factor is because it is done to the two components of the mode independently
        goy_phase_0 = (
            1 / 2 * np.arctan(self.mode_parameters_on_surface_0.z_minus_z_0 / self.mode_parameters_on_surface_0.z_R)
        )
        goy_phase_1 = (
            1 / 2 * np.arctan(self.mode_parameters_on_surface_1.z_minus_z_0 / self.mode_parameters_on_surface_1.z_R)
        )
        acquired_gouy_phase_value = -(goy_phase_1 - goy_phase_0)  # The minus is in the definition of the Gouy phase
        return acquired_gouy_phase_value

    @property
    def acquired_gouy_phase(self):
        acquired_gouy_phase_per_axis_values = self.acquired_gouy_phase_per_axis
        acquired_gouy_phase_value = np.sum(acquired_gouy_phase_per_axis_values)
        return acquired_gouy_phase_value

    @property
    def name(self):
        return nvl(self.surface_0.name, "unnamed_surface") + " -> " + nvl(self.surface_1.name, "unnamed_surface")

    @property
    def propagation_kernel(self):
        if isinstance(self.surface_0, FlatSurface):
            sign = np.sign((self.surface_1.center - self.surface_0.center) @ self.surface_0.outwards_normal)
            normal_function = lambda x: self.surface_0.outwards_normal * sign  # Add sign
        elif isinstance(self.surface_0, CurvedSurface):
            normal_function = lambda r: normal_to_a_sphere(
                r_surface=r, o_center=self.surface_0.origin, sign=-self.surface_0.curvature_sign
            )  # Add sign

        def propagation_kernel(r_source: np.ndarray, r_observer: np.ndarray, k: float):
            M = m_total(r_source=r_source, r_observer=r_observer, k=k, normal_function=normal_function, n_index=self.n)
            return M

        return propagation_kernel


def complete_orthonormal_basis(v: np.ndarray) -> np.ndarray:
    """
    Given a normalized 3D vector v, returns a 3x3 orthonormal basis matrix
    where the first column is v.
    If v is close to a standard basis vector, returns the regular basis.
    """
    v = np.asarray(v)
    v = v / np.linalg.norm(v)
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    tol = 1e-8

    if np.allclose(v, x, atol=tol):
        return np.vstack((z, y))
    elif np.allclose(v, y, atol=tol):
        return np.vstack((z, x))
    elif np.allclose(v, z, atol=tol):
        return np.vstack((x, y))
    else:
        # Find a vector not parallel to v
        if abs(v[0]) < 0.9:
            temp = x
        else:
            temp = y
        # Gram-Schmidt process
        v2 = temp - np.dot(temp, v) * v
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v, v2)
        v3 /= np.linalg.norm(v3)
        return np.vstack((v2, v3))


def simple_mode_propagator(
    surfaces: Optional[list] = None,
    arms: Optional[list] = None,
    local_mode_parameters_initial: LocalModeParameters = None,
    ray_initial: Ray = None,
    mode_parameters_initial: Optional[ModeParameters] = None,
    initial_mode_on_first_surface: bool = False,
):  # Calculate it manually in refactored version
    assert (surfaces is not None or arms is not None) and not (
        surfaces is not None and arms is not None
    ), "Either surfaces or arms must be provided, but not both."
    assert (local_mode_parameters_initial is not None or mode_parameters_initial is not None) and not (
        local_mode_parameters_initial is not None and mode_parameters_initial is not None
    ), "Either local_mode_parameters_initial or mode_parameters_initial must be provided, but not both."

    if mode_parameters_initial is not None:
        raise NotImplementedError("Not yet implemented for mode_parameters_initial")

    if surfaces is not None:
        arms = [
            Arm(
                surfaces[i],
                surfaces[i + 1],
            )
            for i in range(len(surfaces) - 1)
        ]
    last_step = arms[-1].surface_1.center - arms[-1].surface_0.center
    arms.append(
        Arm(
            surfaces[-1],
            FlatSurface(
                outwards_normal=last_step / np.linalg.norm(last_step),
                center=arms[-1].surface_1.center + 10 * last_step,
                name="dummy_surface_final",
            ),
        )
    )

    if not initial_mode_on_first_surface:
        dummy_plane = FlatSurface(
            outwards_normal=-ray_initial.k_vector, center=ray_initial.origin, name="dummy_initial_plane"
        )
        arms.insert(0, Arm(surface_0=dummy_plane, surface_1=arms[0].surface_0, central_line=ray_initial))

    central_line = ray_initial
    for arm in arms:
        arm.central_line = central_line
        arm.mode_principle_axes = complete_orthonormal_basis(central_line.k_vector)
        central_line = arm.propagate_ray(central_line)

    local_mode_parameters_current = local_mode_parameters_initial
    for arm in arms:
        arm.mode_parameters_on_surface_0 = local_mode_parameters_current
        local_mode_parameters_current = arm.propagate_local_mode_parameters()
    return arms


##############
class OpticalSystem:
    def __init__(
        self,
        physical_surfaces: List[PhysicalSurface],
        lambda_0_laser: Optional[float] = None,
        params: Optional[List[OpticalElementParams]] = None,
        names: Optional[List[str]] = None,
        power: Optional[float] = None,
        t_is_trivial: bool = True,
        p_is_trivial: bool = True,
        given_initial_central_line: Optional[Union[Ray, bool]] = None,
        given_initial_local_mode_parameters: Optional[LocalModeParameters] = None,
        use_paraxial_ray_tracing: bool = True,
    ):
        self.physical_surfaces = physical_surfaces
        self.arms: List[Arm] = [
            Arm(
                self.physical_surfaces[i],
                self.physical_surfaces[i+1],
            )
            for i in range(len(self.physical_surfaces) - 1)
        ]
        self.central_line_successfully_traced: Optional[bool] = None
        self.resonating_mode_successfully_traced: Optional[bool] = None
        self.lambda_0_laser: Optional[float] = lambda_0_laser
        self.params = params
        self.names_memory = names
        self.power = power
        self.p_is_trivial = p_is_trivial
        self.t_is_trivial = t_is_trivial
        self.use_paraxial_ray_tracing = use_paraxial_ray_tracing

        if given_initial_central_line is not None:
            if isinstance(given_initial_central_line, Ray):
                self.set_given_central_line(initial_ray=given_initial_central_line)
            elif given_initial_central_line is True:
                self.set_given_central_line(initial_ray=self.default_initial_ray)
        if given_initial_local_mode_parameters is not None:
            self.set_given_mode_parameters(
                local_mode_parameters_first_surface=given_initial_local_mode_parameters,
            )

    @staticmethod
    def from_params(
        params: Union[np.ndarray, List[OpticalElementParams]],
        **kwargs,
    ):
        if isinstance(params, np.ndarray):
            raise ValueError(
                "Cavity.from_params no longer supports np.ndarray input. Please provide a list of OpticalElementParams."
            )
            # params = [OpticalElementParams.from_array(params[i, :]) for i in range(len(params))]
        physical_surfaces = []
        for i, p in enumerate(params):
            if p.name is None:
                p.name = f"Surface_{i}"
            surface_temp = Surface.from_params(p)
            if isinstance(surface_temp, tuple):
                physical_surfaces.extend(surface_temp)
            else:
                physical_surfaces.append(surface_temp)
        cavity = Cavity(
            physical_surfaces,
            params=params,
            **kwargs,
        )
        return cavity

    @property
    def to_params(self) -> List[OpticalElementParams]:
        if self.params is None:
            params = [surface.to_params for surface in self.physical_surfaces]
        else:
            params = self.params
        return params

    @property
    def formatted_textual_params(self) -> str:
        if self.params is None:
            return "No parameters set for this cavity."
        textual_representation = "params = " + str(self.params).replace(
            "OpticalElementParams", "\n          OpticalElementParams"
        ).replace("))]", "))\n         ]")
        return textual_representation

    @property
    def to_array(self) -> np.ndarray:
        array = np.stack([param.to_array for param in self.to_params], axis=0)
        return array

    @property
    def id(self):
        hashed_str = int(md5(str(self.to_params).encode("utf-8")).hexdigest()[:5], 16)
        return hashed_str

    @property
    def central_line(self) -> Optional[List[Ray]]:
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
        elif len(self.ABCD_matrices) == 1:
            return self.ABCD_matrices[0]
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
        return [arm.surface_0 for arm in self.arms]

    @property
    def default_initial_k_vector(self) -> np.ndarray:
        if self.central_line is not None and self.central_line_successfully_traced:
            initial_k_vector = self.central_line[0].k_vector
        else:
            initial_k_vector = self.arms[0].surface_1.center - self.arms[0].surface_0.center
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
            initial_ray = Ray(origin=self.arms[0].surface_0.center, k_vector=initial_k_vector)
            return initial_ray

    @property
    def names(self):
        if self.names_memory is not None:
            return self.names_memory
        else:
            names_params = [p.name if p.name is not None else i for i, p in enumerate(self.to_params)]
            return names_params

    @property
    def roundtrip_power_losses(self):
        # if roundtrip_power_losses = 0.2 then every roundtrip 0.2 of the power is lost
        if self.central_line_successfully_traced is False:
            return None
        # losses = 0
        starting_power = 1
        for arm in self.arms:
            first_surface = arm.surface_0
            if isinstance(first_surface, (CurvedMirror, FlatMirror)):
                surface_unlost_portion = first_surface.material_properties.intensity_reflectivity
            elif isinstance(first_surface, (CurvedRefractiveSurface, IdealLens)):
                surface_unlost_portion = first_surface.material_properties.intensity_transmittance
            else:
                raise ValueError(f"Surface type {type(first_surface)} not implemented in this function")
            alpha = 0
            if hasattr(
                first_surface, "n_2"
            ):  # Do not include volumetric losses if the arms is made of air. this is a bad implementation, and the volumetric losses should be included in the arms properties.
                if first_surface.n_2 != 1:
                    alpha = first_surface.material_properties.alpha_volume_absorption
            volume_absorption_unlost_portion_log = alpha * arm.central_line.length
            volume_absorption_unlost_portion = np.exp(-volume_absorption_unlost_portion_log)
            if isinstance(first_surface, (CurvedMirror, FlatMirror)):
                starting_power *= surface_unlost_portion
            elif isinstance(first_surface, CurvedRefractiveSurface):
                starting_power *= surface_unlost_portion * volume_absorption_unlost_portion
            # losses += surface_coherent_loss + surface_absorption_loss + volume_absorption_loss_log
        return 1 - starting_power  # , losses

    @property
    def roundtrip_optical_length(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            optical_length = 0
            for arm in self.arms:
                if isinstance(arm.surface_0, CurvedRefractiveSurface):
                    optical_length += arm.central_line.length * arm.surface_0.n_2
                else:
                    optical_length += arm.central_line.length
        return optical_length

    @property
    def roundtrip_time(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.roundtrip_optical_length / C_LIGHT_SPEED

    @property
    def free_spectral_range(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return 1 / self.roundtrip_time

    @property
    def power_decay_rate(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.roundtrip_power_losses / self.roundtrip_time

    @property
    def amplitude_decay_rate(self):
        if self.central_line_successfully_traced is False:
            return None
        else:
            return self.power_decay_rate / 2

    @property
    def finesse(self):
        return np.pi * self.free_spectral_range / self.amplitude_decay_rate

    def propagate_ray(self, ray: Ray, n_arms: Optional[int] = None) -> List[Ray]:
        ray_history = [ray]
        n_arms = nvl(n_arms, len(self.arms))
        for i in range(n_arms):
            arm = self.arms[i % len(self.arms)]
            ray = arm.propagate_ray(ray, use_paraxial_ray_tracing=self.use_paraxial_ray_tracing)
            ray_history.append(ray)
        return ray_history

    def set_given_central_line(self, initial_ray: Ray):
        # This line is to save the central line in the ray history, so that it can be plotted later.
        central_line = self.propagate_ray(initial_ray)
        for i, arm in enumerate(self.arms):
            arm.central_line = central_line[i]

    def set_given_mode_parameters(
        self,
        local_mode_parameters_first_surface: Optional[LocalModeParameters] = None,
    ):
        # If there is a valid mode to start propagating, then propagate it through the cavity:
        local_mode_parameters_current = local_mode_parameters_first_surface
        for arm in self.arms:
            arm.mode_parameters_on_surface_0 = local_mode_parameters_current
            local_mode_parameters_current = arm.propagate_local_mode_parameters()
            arm.mode_principle_axes = self.principle_axes(arm.central_line.k_vector)


    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        # ATTENTION! THIS ASSUMES THAT ALL THE CENTRAL LINE arms ARE IN THE SAME PLANE.
        # I find the biggest psuedo z because if the first two k_vector are parallel, the cross product is zero and the
        # result of the cross product will be determined by arbitrary numerical errors.
        possible_pseudo_zs = [
            np.cross(self.central_line[0].k_vector, self.central_line[i].k_vector)
            for i in range(1, len(self.central_line))
        ]  # Points to the positive
        biggest_psuedo_z = possible_pseudo_zs[np.argmax([np.linalg.norm(pseudo_z) for pseudo_z in possible_pseudo_zs])]
        # biggest_psuedo_z = np.cross(self.central_line[0].k_vector, self.central_line[1].k_vector)
        if np.linalg.norm(biggest_psuedo_z) < 1e-14:
            pseudo_z = np.array([0, 0, 1])
        else:
            pseudo_z = normalize_vector(biggest_psuedo_z)
        pseudo_x = np.cross(pseudo_z, k_vector)
        principle_axes = np.stack([pseudo_z, pseudo_x], axis=-1).T  # [z_x, z_y, z_z], [x_x, x_y, x_z]
        return principle_axes

    def ray_of_initial_parameters(self, initial_parameters: np.ndarray):
        # Assumes initial_parameters is of the shape [..., 4] where the last axis of size for represents theta, theta,
        # (two numbers to represent the location and angle on the first surface) and theta, phi (two angles of the k_vector).
        k_vector_i = unit_vector_of_angles(theta=initial_parameters[..., 1], phi=initial_parameters[..., 3])
        origin_i = self.arms[0].surface_0.parameterization(t=initial_parameters[..., 0], p=initial_parameters[..., 2])
        input_ray = Ray(origin=origin_i, k_vector=k_vector_i)
        return input_ray  # input_ray.origin and input_ray.k_vector are of shape [..., 3] where the ... is the same as
        # the first axis of initial_parameters.

    def generate_spot_size_lines(self, dim=2, plane="xy"):
        list_of_spot_size_lines = []
        if not np.isnan(self.arms[0].mode_parameters.z_R[0]):
            for arm in self.arms:
                spot_size_lines_separated = generate_spot_size_lines(
                    arm.mode_parameters,
                    first_point=arm.central_line.origin,
                    last_point=arm.central_line.origin + arm.central_line.k_vector * arm.central_line.length,
                    principle_axes=arm.mode_principle_axes,
                    dim=dim,
                    plane=plane,
                )
                list_of_spot_size_lines.extend(spot_size_lines_separated)
        return list_of_spot_size_lines


    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        axis_span: Optional[Union[float, np.ndarray]] = None,
        camera_center: Union[float, int] = -1,
        dim: int = 2,
        laser_color: str = "r",
        plane: str = "xy",
        plot_mode_lines: bool = True,
        plot_central_line: bool = True,
        additional_rays: Optional[List[Ray]] = None,
        diameters: Optional[Union[float, np.ndarray]] = None,
        fine_resolution=False,
        **kwargs,
    ) -> plt.Axes:
        if axis_span is None:

            axes_range = np.array(
                [
                    np.max([m.center[0] for m in self.physical_surfaces])
                    - np.min([m.center[0] for m in self.physical_surfaces]),
                    np.max([m.center[1] for m in self.physical_surfaces])
                    - np.min([m.center[1] for m in self.physical_surfaces]),
                    np.max([m.center[2] for m in self.physical_surfaces])
                    - np.min([m.center[2] for m in self.physical_surfaces]),
                ]
            )

            if self.t_is_trivial and self.p_is_trivial and dim == 2:
                if (
                    not np.isnan(self.arms[0].mode_parameters.z_R[0])
                    and np.min(self.arms[0].mode_parameters_on_surface_0.z_R) > 0
                ):
                    maximal_spot_size = np.max([arm.mode_parameters_on_surface_0.spot_size[0] for arm in self.arms])
                    axis_span = np.array([axes_range[0], 6 * maximal_spot_size])
                else:
                    axis_span = np.array([axes_range[0], 0.01])
            else:
                axes_range[axes_range == 0] = 5e-3
                axis_span = axes_range
        else:
            axis_span = np.array([axis_span, axis_span, axis_span])

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            if dim == 3:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)

        if camera_center == -1:
            origin_camera = np.array(
                [
                    (
                        np.max([m.center[0] for m in self.physical_surfaces])
                        + np.min([m.center[0] for m in self.physical_surfaces])
                    )
                    / 2,
                    (
                        np.max([m.center[1] for m in self.physical_surfaces])
                        + np.min([m.center[1] for m in self.physical_surfaces])
                    )
                    / 2,
                    (
                        np.max([m.center[2] for m in self.physical_surfaces])
                        + np.min([m.center[2] for m in self.physical_surfaces])
                    )
                    / 2,
                ]
            )

        else:
            camera_center_int = int(np.floor(camera_center))
            if np.mod(camera_center, 1) == 0.5:
                origin_camera = (
                    self.arms[camera_center_int].surface_0.center + self.arms[camera_center_int].surface_1.center
                ) / 2
            else:
                origin_camera = self.surfaces[camera_center_int].center

        x_index, y_index = plane_name_to_xy_indices(plane)
        ax.set_xlim(
            origin_camera[x_index] - axis_span[0] * 0.55,
            origin_camera[x_index] + axis_span[0] * 0.55,
        )
        ax.set_ylim(
            origin_camera[y_index] - axis_span[1] * 0.55,
            origin_camera[y_index] + axis_span[1] * 0.55,
        )

        if self.central_line is not None and plot_central_line:
            for ray in self.central_line:
                ray.plot(
                    ax=ax,
                    dim=dim,
                    color=laser_color,
                    plane=plane,
                    linestyle="--",
                    alpha=0.8,
                )

        if additional_rays is not None:
            for i, ray in enumerate(additional_rays):
                ray.plot(ax=ax, dim=dim, plane=plane, linestyle="--", alpha=0.8, color="blue")

        if diameters is not None:
            if isinstance(diameters, float):
                diameters = np.ones(len(self.surfaces)) * diameters
        else:
            diameters = [
                (
                    element.diameter
                    if not np.isnan(element.diameter)
                    else element.radius if isinstance(element, CurvedSurface) else 7.75e-3
                )
                for element in self.physical_surfaces
            ]
        laser_color = laser_color if self.resonating_mode_successfully_traced is True else "grey"

        for i, surface in enumerate(self.surfaces):
            surface.plot(ax=ax, dim=dim, plane=plane, diameter=diameters[i], fine_resolution=fine_resolution, **kwargs)
            # If there is not information on the spot size of the element, plot it with default length:
            if (
                self.resonating_mode_successfully_traced
                and not np.any(np.isnan(self.arms[0].mode_parameters.z_R))
                and not np.any(self.arms[0].mode_parameters.z_R == 0)
            ) and plot_mode_lines:
                # If there is information on the spot size of the element, plot it with the spot size length*2.5:
                spot_size = self.arms[i].mode_parameters_on_surface_0.spot_size
                if plane == "xy":
                    spot_size = spot_size[1]
                else:
                    spot_size = spot_size[0]
                diameter = spot_size * 5
                surface.plot(
                    ax=ax,
                    dim=dim,
                    plane=plane,
                    diameter=diameter,
                    alpha=0.5,
                    linestyle="--",
                    color=laser_color,
                    fine_resolution=fine_resolution,
                )

        if self.lambda_0_laser is not None and plot_mode_lines and self.arms[0].central_line is not None:
            try:
                spot_size_lines = self.generate_spot_size_lines(dim=dim, plane=plane)

                for line in spot_size_lines:
                    if dim == 2:
                        ax.plot(
                            line[0, :],
                            line[1, :],
                            color=laser_color,
                            linestyle="--",
                            alpha=0.8,
                            linewidth=0.5,
                        )
                    else:
                        ax.plot(
                            line[0, :],
                            line[1, :],
                            line[2, :],
                            color=laser_color,
                            linestyle="--",
                            alpha=0.8,
                            linewidth=0.5,
                        )
            except (FloatingPointError, AttributeError):
                # print("Mode was not successfully found, mode lines not plotted.")
                pass
        ax.grid()
        if additional_rays is not None:
            ax.legend()
        return ax


    @property
    def total_acquired_gouy_phase(self):
        if np.isnan(self.arms[0].mode_parameters.z_R[0]) or self.arms[0].mode_parameters_on_surface_0.z_R[0] == 0:
            return None
        gouy_phases = [arm.acquired_gouy_phase for arm in self.arms]
        return sum(gouy_phases)

    def output_radius_of_curvature(self, initial_distance: float) -> float:
        # Currently assume 1d problem for simplicty, if required it can be expanded
        if isinstance(self.physical_surfaces[0], PhysicalSurface):  # Bad implementation. to correct it, I need to make sure that OpticalSystem can actually accept a surface that is not a PhysicalSurface, and allow for General surfaces in general. Also - better decompose the self.ABCD matrices to the propagation and reflection ABCD matrices.
            first_ABCD = self.physical_surfaces[0].ABCD_matrix(cos_theta_incoming=1)[:2, :2]
        else:
            first_ABCD = np.eye(2)
        ABCD = self.ABCD_round_trip[:2, :2] @ first_ABCD
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        R_out = -(A * initial_distance + B) / (C * initial_distance + D)
        return R_out

    def required_initial_distance_for_desired_output_radius_of_curvature(self, desired_R_out: float) -> float:
        ABCD = self.ABCD_round_trip[:2, :2] @ self.physical_surfaces[0].ABCD_matrix(cos_theta_incoming=1)[:2, :2]
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        initial_distance = -(B + D * desired_R_out) / (A + C * desired_R_out)
        return initial_distance

    def invert(self):
        inverted_physical_surfaces = []
        for surface in self.physical_surfaces[::-1]:
            inverted_surface = copy.deepcopy(surface)
            if isinstance(surface, RefractiveSurface):
                n_1, n_2 = inverted_surface.n_1, inverted_surface.n_2
                inverted_surface.n_1 = n_2
                inverted_surface.n_2 = n_1
            if isinstance(surface, (CurvedSurface, AsphericSurface)):
                inverted_surface.curvature_sign *= -1
            inverted_physical_surfaces.append(inverted_surface)

        if self.central_line is not None:
            origin_inverted = self.physical_surfaces[-1].find_intersection_with_ray(self.central_line[-1])
            k_vector_inverted = -self.central_line[-1].k_vector
            initial_ray_inverted = Ray(origin=origin_inverted, k_vector=k_vector_inverted)
        else:
            initial_ray_inverted = None

        inverted_system = OpticalSystem(physical_surfaces=inverted_physical_surfaces, lambda_0_laser=self.lambda_0_laser, given_initial_central_line=initial_ray_inverted)
        return inverted_system

##############


class Cavity(OpticalSystem):
    def __init__(
        self,
        physical_surfaces: List[PhysicalSurface],
        standing_wave: bool = False,
        lambda_0_laser: Optional[float] = None,
        params: Optional[List[OpticalElementParams]] = None,
        names: Optional[List[str]] = None,
        set_central_line: bool = True,
        set_mode_parameters: bool = True,
        set_initial_surface: bool = False,
        t_is_trivial: bool = False,
        p_is_trivial: bool = False,
        power: Optional[float] = None,
        initial_local_mode_parameters: Optional[LocalModeParameters] = None,
        initial_mode_parameters: Optional[ModeParameters] = None,
        use_brute_force_for_central_line: bool = False,  # remove it once we know it works
        debug_printing_level: int = 0,  # 0 for no prints, 1 for main prints, 2 for all prints
        use_paraxial_ray_tracing: bool = False,
    ):
        super().__init__(physical_surfaces=physical_surfaces,
                         lambda_0_laser=lambda_0_laser,
                         params=params,
                         names=names,
                         t_is_trivial=t_is_trivial,
                         p_is_trivial=p_is_trivial,
                         power=power,
                         use_paraxial_ray_tracing=use_paraxial_ray_tracing,
                         )

        self.standing_wave = standing_wave
        self.arms: List[Arm] = [
            Arm(
                self.physical_surfaces_ordered[i],
                self.physical_surfaces_ordered[np.mod(i + 1, len(self.physical_surfaces_ordered))],
            )
            for i in range(len(self.physical_surfaces_ordered))
        ]
        self.central_line_successfully_traced: Optional[bool] = None
        self.resonating_mode_successfully_traced: Optional[bool] = None
        self.use_brute_force_for_central_line = use_brute_force_for_central_line
        self.debug_printing_level = debug_printing_level
        self.use_paraxial_ray_tracing = use_paraxial_ray_tracing

        if set_central_line:
            self.set_central_line()
        if set_mode_parameters:
            self.set_mode_parameters(
                mode_parameters_first_arm=initial_mode_parameters,
                local_mode_parameters_first_surface=initial_local_mode_parameters,
            )
        if set_initial_surface:
            self.set_initial_surface()

    @property
    def physical_surfaces_ordered(self):
        if self.standing_wave:
            backwards_list = copy.deepcopy(self.physical_surfaces[-2:0:-1])
            for surface in backwards_list:
                # if isinstance(surface, (CurvedRefractiveSurface, AsphericRefractiveSurface)):
                #     surface.curvature_sign = -surface.curvature_sign
                #     n_1, n_2 = surface.n_1, surface.n_2
                #     surface.n_1 = n_2
                #     surface.n_2 = n_1
                if isinstance(surface, RefractiveSurface):
                    n_1, n_2 = surface.n_1, surface.n_2
                    surface.n_1 = n_2
                    surface.n_2 = n_1
                if isinstance(surface, (CurvedSurface, AsphericSurface)):
                    surface.curvature_sign *= -1


            return self.physical_surfaces + backwards_list
        else:
            return self.physical_surfaces

    @property
    def surfaces_ordered(self):
        return [arm.surface_0 for arm in self.arms]

    @property
    def surfaces(self):
        if self.standing_wave:
            return [arm.surface_0 for arm in self.arms[: len(self.arms) // 2 + 1]]
        else:
            return [arm.surface_0 for arm in self.arms]

    @property
    def perturbable_params_names(self):
        perturbable_params_names_list = params_to_perturbable_params_names(
            self.to_params, self.t_is_trivial and self.p_is_trivial
        )
        return perturbable_params_names_list


    def trace_ray_parametric(self, starting_position_and_angles: np.ndarray) -> Tuple[np.ndarray, List[Ray]]:
        # Like trace ray, but works as a function of the starting position and angles as parameters on the starting
        # surface, instead of the starting position and angles as a vector in 3D space.

        initial_ray = self.ray_of_initial_parameters(starting_position_and_angles)
        ray_history = self.propagate_ray(initial_ray)
        final_intersection_point = ray_history[-1].origin
        t_o, p_o = self.arms[0].surface_0.get_parameterization(final_intersection_point)  # Here it is the initial
        # surface on purpose: the final ray's origin should be on the initial surface, after one round trip.
        theta_o, phi_o = angles_of_unit_vector(ray_history[-1].k_vector)
        final_position_and_angles = np.stack([t_o, theta_o, p_o, phi_o], axis=-1)
        return final_position_and_angles, ray_history

    def f_roots(self, starting_position_and_angles: np.ndarray) -> np.ndarray:
        # The roots of this function are the initial parameters for the central line. (position x, y, angles theta, phi)
        # try:
        final_position_and_angles, _ = self.trace_ray_parametric(starting_position_and_angles / STRETCH_FACTOR)
        diff = np.zeros_like(starting_position_and_angles)
        diff[..., [0, 2]] = (
            final_position_and_angles[..., [0, 2]] - starting_position_and_angles[..., [0, 2]] / STRETCH_FACTOR
        )
        diff[..., [1, 3]] = angles_difference(
            starting_position_and_angles[..., [1, 3]] / STRETCH_FACTOR,
            final_position_and_angles[..., [1, 3]],
        )
        diff[np.isnan(diff)] = np.inf
        return diff * STRETCH_FACTOR

    def f_roots_standing_wave(self, angles: np.ndarray):
        # This function returns 0 also if the ray is pointing to the exact opposite direction of the central line
        # make sure it does not create problems.
        last_ray_index = len(self.physical_surfaces) - 2  # minus one for the first surface and -1 because of python's
        # 0 indexing.
        k_vector = unit_vector_of_angles(angles[0], angles[1])
        ray = Ray(self.physical_surfaces[0].origin, k_vector)
        ray_history = self.propagate_ray(ray)
        last_arms_ray = ray_history[last_ray_index]  # -2

        origins_plane = FlatSurface(
            outwards_normal=self.physical_surfaces[-1].outwards_normal, center=self.physical_surfaces[-1].origin
        )
        intersection_point = origins_plane.find_intersection_with_ray(last_arms_ray, paraxial=False)  # ATTEMPT
        t, p = origins_plane.get_parameterization(intersection_point)

        # Alternative syntax (that results in rotated parameterization)
        # d = np.cross(last_arms_ray.k_vector, rays_origin_to_surface_origin)  # This is a signed distance, which
        # transverse_spanning_vector_1, transverse_spanning_vector_2 = self.physical_surfaces[-1].spanning_vectors()  # -1
        # d_1 = d @ transverse_spanning_vector_1
        # d_2 = d @ transverse_spanning_vector_2
        # is good for the solver.
        # print(angles[1] - np.pi, phi)
        result_array = np.array([t, p])
        return result_array

    def find_central_line_solver(self):
        theta_initial_guess, phi_initial_guess = self.default_initial_angles
        initial_guess = np.array([0, theta_initial_guess, 0, phi_initial_guess]) * STRETCH_FACTOR

        if self.t_is_trivial and self.p_is_trivial:
            central_line_initial_parameters = initial_guess
        else:
            if self.t_is_trivial and not self.p_is_trivial:
                initial_guess_subspace = initial_guess[[2, 3]]
                f_roots_subspace = lambda x: self.f_roots(np.array([initial_guess[0], initial_guess[1], x[0], x[1]]))[
                    [2, 3]
                ]
                central_line_initial_parameters: np.ndarray = optimize.fsolve(f_roots_subspace, initial_guess_subspace)
                central_line_initial_parameters = np.concatenate(
                    (initial_guess[[0, 1]], central_line_initial_parameters)
                )
            elif not self.t_is_trivial and self.p_is_trivial:
                initial_guess_subspace = initial_guess[[0, 1]]
                f_roots_subspace = lambda x: self.f_roots(np.array([x[0], x[1], initial_guess[2], initial_guess[3]]))[
                    [0, 1]
                ]
                central_line_initial_parameters: np.ndarray = optimize.fsolve(f_roots_subspace, initial_guess_subspace)
                central_line_initial_parameters = np.concatenate(
                    (central_line_initial_parameters, initial_guess[[2, 3]])
                )
            else:
                central_line_initial_parameters: np.ndarray = optimize.fsolve(self.f_roots, initial_guess)
            # In the documentation it says optimize.fsolve returns a solution, together with some flags, and also this
            # is how pycharm suggests to use it. But in practice it returns only the solution, not sure why.

        root_error = np.linalg.norm(self.f_roots(central_line_initial_parameters))
        central_line_initial_parameters /= STRETCH_FACTOR

        # print(f"root_error: {root_error}")
        # print(f"diff: {self.f_roots(central_line_initial_parameters)}")

        central_line_successfully_traced = root_error < CENTRAL_LINE_TOLERANCE * STRETCH_FACTOR

        return central_line_initial_parameters, central_line_successfully_traced

    def find_central_line_brute_force(
        self,
        N_resolution: int = 11,
        range_limit: float = 1e-4,
        zoom_factor: float = 1.4,
        N_iterations: int = 50,
    ) -> Tuple[np.ndarray, bool]:
        theta_initial_guess, phi_initial_guess = self.default_initial_angles
        central_line_initial_parameters = np.array([0, theta_initial_guess, 0, phi_initial_guess])

        if self.t_is_trivial and self.p_is_trivial:
            return central_line_initial_parameters, True

        if self.debug_printing_level >= 2:
            fig, ax = plt.subplots(N_iterations, 3, figsize=(24, N_iterations * 3))

        for i in range(N_iterations):
            initial_parameters = generate_initial_parameters_grid(
                central_line_initial_parameters,
                range_limit,
                N_resolution,
                p_is_trivial=self.p_is_trivial,
                t_is_trivial=self.t_is_trivial,
            )
            diff = self.f_roots(initial_parameters)
            diff_norm = np.linalg.norm(diff, axis=-1)

            smallest_elements_index = np.unravel_index(np.argmin(diff_norm), diff.shape[:-1])
            central_line_initial_parameters = initial_parameters[smallest_elements_index]
            range_limit /= zoom_factor

            central_line_successfully_traced = diff_norm[smallest_elements_index] < CENTRAL_LINE_TOLERANCE
            if self.debug_printing_level >= 2:
                print(f"iteration {i}, error: {diff_norm[smallest_elements_index]}")
                print(f"iteration {i}, center: {central_line_initial_parameters}\n")
                print(f"iteration {i + 1}, range_limit: {range_limit:.3e}")
                ax[i, 0].imshow(diff_norm)
                # plt.colorbar()
                # Add a dot at the minimum:
                ax[i, 0].scatter(smallest_elements_index[1], smallest_elements_index[0], color="r")
                if self.p_is_trivial:
                    parameters_indices = [0, 1]
                else:
                    parameters_indices = [2, 3]
                diff_position = diff_norm[:, smallest_elements_index[1]]
                ax[i, 1].plot(initial_parameters[:, 0, parameters_indices[0]], diff_position)
                ax[i, 1].axvline(initial_parameters[smallest_elements_index[0], 0, parameters_indices[0]], color="r")
                ax[i, 1].set_title(f"{i}: position")

                diff_angle = diff_norm[smallest_elements_index[0], :]
                ax[i, 2].plot(initial_parameters[0, :, parameters_indices[1]], diff_angle)
                ax[i, 2].axvline(initial_parameters[0, smallest_elements_index[1], parameters_indices[1]], color="r")
                ax[i, 2].set_title(f"{i}: angle")

        if self.debug_printing_level >= 2:
            plt.show()
        if self.debug_printing_level >= 1:
            print(f"root_error: {diff_norm[smallest_elements_index]}")
            print(f"diff: {diff[smallest_elements_index]}")

        return central_line_initial_parameters, central_line_successfully_traced

    def find_central_line_standing_wave(self):
        # This function assumes the centers of the origins (sphere's center) of the first and last mirrors are withing
        # their arms, which will not be for the case of the astigmatic cavity with the extra mirror, but for now it should work.
        if not isinstance(self.physical_surfaces[0], CurvedMirror) and isinstance(
            self.physical_surfaces[-1], CurvedMirror
        ):
            warnings.warn(
                "For this method to work the first and last surfaces should be mirrors, using regular solver instead"
            )
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_solver()

        else:
            theta_default, phi_default = self.default_initial_angles
            if self.t_is_trivial and self.p_is_trivial:
                solution_angles = np.array([theta_default, phi_default])
                central_line_successfully_traced = True
            elif (
                not self.t_is_trivial and self.p_is_trivial
            ):  # This syntax is for the case when we know the perturbation is only in
                # one dimensions, and the function can be reduced to one dimension as well. however, I think that the choice
                # of the one dimensions on which to solve the problem will not be consistent for different orientations of the
                # cavity, and so it might cause problems in the future
                # has a non-consistent entry in the
                def f_reduced(theta):
                    z, y = self.f_roots_standing_wave(np.array([theta, phi_default]))
                    if np.isnan(z):
                        z = np.inf * np.sign(theta)
                    return z

                solution = optimize.root_scalar(
                    f_reduced,
                    x0=theta_default,
                    x1=theta_default + 1e-10,
                    xtol=1e-9,
                )
                solution_angles = np.array([solution.root, phi_default])
                central_line_successfully_traced = solution.converged
            elif self.t_is_trivial and not self.p_is_trivial:

                def f_reduced(phi):
                    z, y = self.f_roots_standing_wave(np.array([theta_default, phi]))
                    if np.isnan(y):
                        y = np.inf * np.sign(phi)
                    return y

                solution = optimize.root_scalar(
                    f_reduced,
                    x0=phi_default,
                    x1=phi_default + 1e-9,
                    xtol=1e-9,
                )  # x0=np.array([self.default_initial_angles[1]])
                # print(f"phi_solution = {solution.root}, y distance = {f_reduced(solution.root)}")
                solution_angles = np.array([theta_default, solution.root])
                central_line_successfully_traced = solution.converged
            else:
                solution = optimize.root(self.f_roots_standing_wave, x0=np.array([*self.default_initial_angles]))
                solution_angles = solution.x
                central_line_successfully_traced = solution.success
            solution_ray = Ray(self.physical_surfaces[0].origin, -unit_vector_of_angles(*solution_angles))  # minus sign
            # central_line_successfully_traced = solution.
            # on the angles because we are searching now the intersection between the ray and the surface behind it.

            # Retrive the parameterization of the ray with respect to the first surface:
            ray_origin_on_first_surface = self.physical_surfaces[0].find_intersection_with_ray(solution_ray)
            solution_parameterization = self.physical_surfaces[0].get_parameterization(ray_origin_on_first_surface)
            central_line_initial_parameters = np.array(
                [solution_parameterization[0], solution_angles[0], solution_parameterization[1], solution_angles[1]]
            )

        return central_line_initial_parameters, central_line_successfully_traced

    def set_central_line(self, **kwargs) -> Tuple[np.ndarray, bool]:
        if self.standing_wave:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_standing_wave()
        elif self.use_brute_force_for_central_line:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_brute_force(
                **kwargs
            )
        else:
            central_line_initial_parameters, central_line_successfully_traced = self.find_central_line_solver()

        if central_line_successfully_traced:
            self.central_line_successfully_traced = central_line_successfully_traced
            origin_solution = self.arms[0].surface_0.parameterization(
                central_line_initial_parameters[0], central_line_initial_parameters[2]
            )  # theta, phi
            k_vector_solution = unit_vector_of_angles(
                central_line_initial_parameters[1], central_line_initial_parameters[3]
            )  # theta, phi
            central_line = Ray(origin_solution, k_vector_solution)
            # This line is to save the central line in the ray history, so that it can be plotted later.
            central_line = self.propagate_ray(central_line)
            if self.standing_wave:
                # If it is a standing wave - set the backward trip to be identical to the forwards, but reversed:
                n_physical_arms = len(self.physical_surfaces) - 1
                for i, arm in enumerate(self.arms[0:n_physical_arms]):
                    arm.central_line = central_line[i]
                for i, arm in enumerate(self.arms[n_physical_arms:]):
                    origin = central_line[n_physical_arms - i - 1].parameterization(
                        central_line[n_physical_arms - i - 1].length
                    )
                    k_vector = -central_line[n_physical_arms - i - 1].k_vector
                    length = central_line[n_physical_arms - i - 1].length
                    arm.central_line = Ray(origin=origin, k_vector=k_vector, length=length)
            else:
                for i, arm in enumerate(self.arms):
                    arm.central_line = central_line[i]
            self.central_line_successfully_traced = central_line_successfully_traced
        else:
            self.central_line_successfully_traced = False
            return central_line_initial_parameters, self.central_line_successfully_traced

    def set_mode_parameters(
        self,
        mode_parameters_first_arm: Optional[ModeParameters] = None,
        local_mode_parameters_first_surface: Optional[LocalModeParameters] = None,
    ):
        # Sets the mode parameters sequentially in all arms of the cavity. tries to find a mode solution for the cavity,
        # if it fails, it will set the resonating_mode_successfully_traced to False, and will use the input
        # local_mode_parameters_first_surface instead.
        if self.central_line_successfully_traced is None:
            self.set_central_line()
        if self.central_line_successfully_traced is False:
            self.resonating_mode_successfully_traced = False
            return None

        local_mode_parameters_current = local_mode_parameters_of_round_trip_ABCD(
            round_trip_ABCD=self.ABCD_round_trip, lambda_0_laser=self.lambda_0_laser, n=self.arms[0].n
        )
        if (
            local_mode_parameters_current.z_R[0] == 0 or local_mode_parameters_current.z_R[1] == 0
        ):  # When there is no solution,
            # the z_R value comes out as zero.
            self.resonating_mode_successfully_traced = False
            if local_mode_parameters_first_surface is not None or mode_parameters_first_arm is not None:  # if there is
                # no wave solution, but the user gave an input wave to the cavity, then just propagate it throughout
                # the cavity, even though it is not a wave solution.
                if mode_parameters_first_arm is not None:
                    # If the user preferred to give ModeParameters instead of LocalModeParameters, then convert it to
                    # LocalModeParameters.
                    local_mode_parameters_first_surface = mode_parameters_first_arm.local_mode_parameters(
                        (self.arms[0].surface_0.center - mode_parameters_first_arm.center[0])
                        @ mode_parameters_first_arm.k_vector
                    )
                local_mode_parameters_current = local_mode_parameters_first_surface

        # If there is a valid mode to start propagating, then propagate it through the cavity:
        if local_mode_parameters_current.z_R[0] != 0 and local_mode_parameters_current.z_R[1] != 0:
            for arm in self.arms:
                arm.mode_parameters_on_surface_0 = local_mode_parameters_current
                local_mode_parameters_current = arm.propagate_local_mode_parameters()
                arm.mode_principle_axes = self.principle_axes(arm.central_line.k_vector)
            if self.resonating_mode_successfully_traced is not False:
                self.resonating_mode_successfully_traced = True
        else:
            for arm in self.arms:
                arm.mode_parameters_on_surface_0 = LocalModeParameters(
                    z_minus_z_0=np.array([np.nan, np.nan]),
                    z_R=np.array([np.nan, np.nan]),
                    lambda_0_laser=self.lambda_0_laser,
                )
                arm.mode_parameters_on_surface_1 = LocalModeParameters(
                    z_minus_z_0=np.array([np.nan, np.nan]),
                    z_R=np.array([np.nan, np.nan]),
                    lambda_0_laser=self.lambda_0_laser,
                )

    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        if self.central_line_successfully_traced is None:
            self.set_central_line()
        principle_axes = super().principle_axes(k_vector)
        return principle_axes

    def generate_spot_size_lines(self, dim=2, plane="xy"):
        if np.isnan(self.arms[0].mode_parameters.z_R[0]):
            self.set_mode_parameters()
        spot_size_lines = super().generate_spot_size_lines(dim=dim, plane=plane)
        return spot_size_lines

    def set_initial_surface(self) -> Optional[Surface]:
        # adds a virtual surface on the first arm that is perpendicular to the beam and centered between the first two
        # physical_surfaces.
        if not isinstance(self.arms[0].surface_0, PhysicalSurface):
            return self.arms[0].surface_0
        # gets a surface that sits between the first two physical_surfaces, centered and perpendicular to the central line.
        if self.central_line is None:
            final_position_and_angles, success = self.set_central_line()
            if not success:
                # warnings.warn("Could not find central line, so no initial surface could be set.")
                return None
        middle_point = (self.central_line[0].origin + self.central_line[1].origin) / 2
        initial_surface = FlatSurface(outwards_normal=-self.central_line[0].k_vector, center=middle_point)

        first_leg = self.arms[0]
        first_leg_first_sub_leg = Arm(first_leg.surface_0, initial_surface)
        first_leg_second_sub_leg = Arm(initial_surface, first_leg.surface_1)
        if self.standing_wave:
            last_leg = self.arms[-1]
            last_leg_first_sub_leg = Arm(last_leg.surface_0, initial_surface)
            last_leg_second_sub_leg = Arm(initial_surface, last_leg.surface_1)
            legs_list = (
                [first_leg_second_sub_leg]
                + self.arms[1:-1]
                + [
                    last_leg_first_sub_leg,
                    last_leg_second_sub_leg,
                    first_leg_first_sub_leg,
                ]
            )
        else:
            legs_list = [first_leg_second_sub_leg] + self.arms[1:] + [first_leg_first_sub_leg]
        self.arms = legs_list
        # Now, after you found the initial_surface, we can retrace the central line, but now let it out from the
        # initial surface, instead of the first mirror.
        self.set_central_line(override_existing=True)
        return initial_surface

    def ABCD_round_trip_matrix_numeric(
        self, central_line_initial_parameters: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if central_line_initial_parameters is None:
            central_line_initial_parameters, success = self.set_central_line()
            if not success:
                raise ValueError("Could not find central line")

        if isinstance(self.arms[0].surface_0, PhysicalSurface):
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


    def calculated_shifted_cavity_overlap_integral(
        self, perturbation_pointer: Union[PerturbationPointer, List[PerturbationPointer]]
    ) -> Tuple[np.ndarray]:
        # For a prturbation of more than one parameter, the first dimension of shift is the shift version, and the second dimension for the parameter index
        # For example, if shift = [[1e-6, 2e-6], [3e-6, 4e-6]], then the first perturbation is [1e-6, 2e-6] and the second is [3e-6, 4e-6].
        n_shifts = len(perturbation_pointer)
        overlaps = np.zeros(n_shifts, dtype=np.float64)
        for i in range(n_shifts):
            new_cavity = perturb_cavity(self, perturbation_pointer[i])
            try:
                overlap = calculate_cavities_overlap(cavity_1=self, cavity_2=new_cavity)
            except np.linalg.LinAlgError:
                continue
            overlaps[i] = np.abs(overlap)
        if n_shifts == 1:
            overlaps = overlaps[0]
        # DEBUG PLOT:
        # fig, ax = plt.subplots(2, 1, figsize=(16, 16))
        # plot_mirror_lens_mirror_cavity_analysis(new_cavity, add_unheated_cavity=False,
        #                                         auto_set_x=True, auto_set_y=True,
        #                                         diameters=[7.75e-3, 7.75e-3, 7.75e-3, 0.0254], ax=ax[0])
        #
        # spot_size_lines_original = self.generate_spot_size_lines(dim=2, plane='xy')
        # for line in spot_size_lines_original:
        #     ax[0].plot(line[0, :], line[1, :], color='green', linestyle='--', alpha=0.8, linewidth=0.5,
        #                label="perturbed_mode")
        # plot_2_cavity_perturbation_overlap(cavity=self, second_cavity=new_cavity, real_or_abs='abs', ax=ax[1])
        # if isinstance(perturbation_pointer, list):
        #     param_name_0 = perturbation_pointer[0].parameter_name
        #     parameter_value_0 = perturbation_pointer[0].perturbation_value
        # else:
        #     param_name_0 = perturbation_pointer.parameter_name
        #     parameter_value_0 = perturbation_pointer.perturbation_value
        # plt.suptitle(
        #     f"param_name_0={param_name_0}, {parameter_value_0=:.3e}\n")
        # fig.tight_layout()
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # plt.savefig(f"outputs/debugging/perturbation_overlap_{timestamp}.png")
        # plt.close(fig)
        # DEBUG PLOT END
        return overlaps

    def calculate_parameter_tolerance(
        self,
        perturbation_pointer: PerturbationPointer,
        initial_step: float = 1e-7,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
    ) -> float:

        if np.isnan(self.arms[0].mode_parameters.NA[0]):
            warnings.warn("cavity has no mode even before perturbation, returning nan.")
            return np.nan
        if perturbation_pointer.perturbation_value is None or isinstance(
            perturbation_pointer.perturbation_value, (float, int)
        ):

            def f(shift):
                resulting_overlap = self.calculated_shifted_cavity_overlap_integral(perturbation_pointer(shift))
                return resulting_overlap

            tolerance = functions_first_crossing_both_directions(
                f=f,
                initial_step=initial_step,
                crossing_value=overlap_threshold,
                accuracy=accuracy,
            )
            return tolerance
        else:
            overlap_series: np.ndarray = self.calculated_shifted_cavity_overlap_integral(perturbation_pointer)
            # Return the shift value where the overlap crosses the threshold which is closest to zero:
            overlap_series_calibrated = overlap_threshold - overlap_series
            product = overlap_series_calibrated[1:] * overlap_series_calibrated[:-1]
            sign_change_mask = np.logical_or(product < 0, np.isnan(product))
            sign_change_indices = np.nonzero(sign_change_mask)[0] + 1  # shift by 1 because we looked at b[1:]

            if sign_change_indices.size == 0:
                return None  # or np.nan, or raise, depending on what you want

            # 2) Among those indices, find the one where |a[i]| is minimal
            idx_best = sign_change_indices[
                np.argmin(np.abs(perturbation_pointer.perturbation_value[sign_change_indices]))
            ]

            # 3) Return the corresponding a[i]
            return perturbation_pointer.perturbation_value[idx_best]

    def generate_tolerance_dataframe(
        self,
        initial_step: float = 1e-7,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
        perturbable_params_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names
        tolerance_df = pd.DataFrame(index=self.names, columns=perturbable_params_names, dtype="float")
        # np.zeros((len(self.to_params), len(perturbable_params_names)))
        for element_index, element_name in (
            pbar_outer := tqdm(enumerate(tolerance_df.index), disable=self.debug_printing_level < 1)
        ):
            pbar_outer.set_description(f"  Tolerance Matrix - {element_name}")
            for param_name in (pbar := tqdm(tolerance_df.columns, disable=self.debug_printing_level < 1)):
                pbar.set_description(f"    Tolerance Matrix - {element_name} -  {param_name}")
                if (
                    self.to_params[element_index].surface_type == SurfacesTypes.thick_lens
                    and param_name in ["theta", "phi"]
                    and self.use_paraxial_ray_tracing
                ):  # Lens is invariant to small rotations under paraxial approx.
                    continue
                tolerance_df.loc[element_name, param_name] = self.calculate_parameter_tolerance(
                    perturbation_pointer=PerturbationPointer(element_index, param_name),
                    initial_step=initial_step,
                    overlap_threshold=overlap_threshold,
                    accuracy=accuracy,
                )
                if self.debug_printing_level >= 2:
                    print(
                        f"tolerance of {param_name} of {element_name}: {tolerance_df.loc[element_name, param_name]:.3e}"
                    )
        return tolerance_df

    def generate_overlap_series(
        self,
        shifts: Union[np.ndarray, float],  # Float is interpreted as linspace's limits,
        # np.ndarray means that the element_index'th parameter_index'th element of shifts is the linspace limits of
        # the element_index'th parameter_index'th parameter.
        shift_size: int = 50,
        perturbable_params_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names
        overlaps = np.zeros((len(self.to_params), len(perturbable_params_names), shift_size))
        for element_index in tqdm(
            range(len(self.to_params)), desc="Overlap Series - element_index", disable=self.debug_printing_level < 1
        ):
            for j, parameter_name in tqdm(
                enumerate(perturbable_params_names),
                desc="Overlap Series - parameter_index",
                disable=self.debug_printing_level < 1,
            ):
                if isinstance(shifts, (float, int)):
                    shift_series = np.linspace(-shifts, shifts, shift_size)
                else:
                    if np.isnan(shifts[element_index, j]):
                        shift_series = np.linspace(-1e-10, 1e-10, shift_size)
                    else:
                        shift_series = np.linspace(
                            -shifts[element_index, j],
                            shifts[element_index, j],
                            shift_size,
                        )
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if (
                    SurfacesTypes.has_refractive_index(self.to_params[element_index].surface_type)
                    or parameter_name != ParamsNames.n_inside_or_after
                ):
                    overlaps[element_index, j, :] = self.calculated_shifted_cavity_overlap_integral(
                        perturbation_pointer=PerturbationPointer(
                            element_index=element_index, parameter_name=parameter_name, perturbation_value=shift_series
                        )
                    )
        return overlaps

    def generate_overlaps_graphs(
        self,
        initial_step: float = 1e-6,
        overlap_threshold: float = 0.9,
        accuracy: float = 1e-3,
        arm_index_for_NA: int = 0,
        tolerance_dataframe: Optional[pd.DataFrame] = None,
        overlaps_series: Optional[np.ndarray] = None,
        names: Optional[List[str]] = None,
        ax: Optional[np.ndarray] = None,
        perturbable_params_names: Optional[List[str]] = None,
    ):
        if names is None:
            names = self.names

        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names

        if ax is None:
            fig, ax = plt.subplots(
                len(self.to_params),
                len(perturbable_params_names),
                figsize=(len(perturbable_params_names) * 5, len(self.to_params) * 2.1),
            )
        else:
            fig = ax.flatten()[0].get_figure()

        if tolerance_dataframe is None:
            tolerance_dataframe = self.generate_tolerance_dataframe(
                initial_step=initial_step, overlap_threshold=overlap_threshold, accuracy=accuracy
            )

        if overlaps_series is None:
            overlaps_series = self.generate_overlap_series(
                shifts=2 * np.abs(np.array(tolerance_dataframe)), shift_size=30
            )
        plt.suptitle(f"NA={self.arms[arm_index_for_NA].mode_parameters.NA[0]:.3e}")

        for i in range(len(self.to_params)):
            for j, parameter_name in enumerate(perturbable_params_names):
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if parameter_name == ParamsNames.n_inside_or_after and np.isnan(tolerance_dataframe.iloc[i, j]):
                    continue
                tolerance = tolerance_dataframe.iloc[i, j]
                if tolerance == 0 or np.isnan(tolerance):
                    tolerance = initial_step
                tolerance_abs = np.abs(tolerance)
                shifts = np.linspace(-2 * tolerance_abs, 2 * tolerance_abs, overlaps_series.shape[2])

                ax[i, j].plot(shifts, overlaps_series[i, j, :])

                title = f"{names[i]}, {parameter_name}, tolerance: {tolerance_abs:.2e}"
                ax[i, j].set_title(title)
                if i == len(self.to_params) - 1:
                    ax[i, j].set_xlabel("Shift")
                if j == 0:
                    ax[i, j].set_ylabel("Overlap")
                ax[i, j].axvline(tolerance, color="g", linestyle="--")
                ax[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax[i, j].axhline(
                    overlap_threshold,
                    color="r",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.5,
                )
                try:
                    min_value = np.nanmin(overlaps_series[i, j, :])
                    ax[i, j].set_ylim(1.1 * min_value - 0.1, 1.1 - 0.1 * min_value)
                except ValueError:
                    pass
        fig.tight_layout()
        return ax

    def thermal_transformation(self, **kwargs) -> Cavity:
        unheated_surfaces = []
        assert (
            self.power is not None
        ), "The power of the laser is not defined. It must be defined for thermal calculations"

        for i, surface in enumerate(self.physical_surfaces):
            # if isinstance(unheated_surface, CurvedRefractiveSurface):
            #     unheated_surface = surface
            # else:
            unheated_surface = surface.thermal_transformation(
                P_laser_power=-self.power,
                w_spot_size=self.arms[i].mode_parameters_on_surface_0.spot_size[0],
                **kwargs,
            )
            unheated_surfaces.append(unheated_surface)

        # After heating the lens is not necessarily symmetrical, and so we have to decompose it to two surfaces.
        if self.names[0] is None:
            names = None
        else:
            names = copy.copy(self.names)
            for i, surface_type in enumerate([p.surface_type for p in self.to_params]):
                if surface_type == SurfacesTypes.thick_lens:
                    names.insert(i + 1, names[i] + "_2")
                    names[i] = names[i] + "_1"

        unheated_cavity = Cavity(
            physical_surfaces=unheated_surfaces,
            standing_wave=self.standing_wave,
            lambda_0_laser=self.lambda_0_laser,
            names=names,
            set_central_line=True,
            set_mode_parameters=True,
            set_initial_surface=False,
            t_is_trivial=self.t_is_trivial,
            p_is_trivial=self.p_is_trivial,
            power=0,
        )

        return unheated_cavity

    def analyze_thermal_transformation(self, arm_index_for_NA: int) -> Tuple[dict, List[Cavity]]:
        N = 4
        boolean_array = np.eye(N).astype(bool)
        boolean_array = np.vstack((np.zeros((1, N), dtype=bool), np.ones((1, N), dtype=bool), boolean_array))
        cavities = []  # [self]
        NA_orgiginal = self.arms[arm_index_for_NA].mode_parameters.NA[0]
        NAs = np.zeros(N + 2)
        # NAs[0] = NA_orgiginal
        for i in range(N + 2):
            (
                curvature_transform_lens,
                n_surface_transform_lens,
                n_volumetric_transform_lens,
                transform_mirror,
            ) = boolean_array[i, :]
            unheated_cavity = self.thermal_transformation(
                curvature_transform_lens=curvature_transform_lens,
                n_surface_transform_lens=n_surface_transform_lens,
                n_volumetric_transform_lens=n_volumetric_transform_lens,
                transform_mirror=transform_mirror,
            )
            cavities.append(unheated_cavity)
            NAs[i] = unheated_cavity.arms[arm_index_for_NA].mode_parameters.NA[0]
        names_list = [
            "No transformation",
            "All Transformations",
            "Only lens curvature ",
            "Only lens n surface ",
            "Only lens n volumetric ",
            "Only lens z ",
            "Only mirror",
        ]
        results_dict = dict(zip(names_list, NA_orgiginal / NAs))
        return results_dict, cavities

    def specs(
        self,
        save_specs_name: Optional[str] = None,
        tolerance_dataframe: Union[np.ndarray, bool] = False,
        print_specs: bool = False,
        contracted: bool = True,
    ):
        elements_array = self.to_array.T.copy()
        elements_array = np.real(elements_array) + np.pi * np.imag(elements_array)
        df_elements = pd.DataFrame(
            elements_array,
            columns=self.names,
            index=list(PRETTY_INDICES_NAMES.values()),
        )

        df_elements_stacked = stack_df_for_print(df_elements)
        NAs_list = []
        lengths_list = []
        df_arms_list = []
        for i, arm in enumerate(self.arms):
            if (
                self.standing_wave and i >= len(self.arms) // 2
            ):  # If it is a standing wave cavity, print only half of the arms, as the second half is the same arms in reverse
                break
            NAs_list.append(arm.mode_parameters.NA[0])
            lengths_list.append(arm.central_line.length)
            df_arms_list.append(arm.specs())

        df_cavity = pd.DataFrame(
            {
                "Parameter": [
                    "Id",
                    "Finesse",
                    "Free spectral range",
                    "Roundtrip power losses",
                    "Power decay rate",  # 'Amplitude amplification factor'
                ]
                + [f"NA_{i}" for i in range(len(NAs_list))]
                + [f"length_{i}" for i in range(len(lengths_list))],
                "Value": [
                    self.id,
                    self.finesse,
                    self.free_spectral_range,
                    self.roundtrip_power_losses,
                    self.power_decay_rate,
                ]
                + NAs_list
                + lengths_list,
            }
        )
        df_cavity["Element"] = "Cavity"
        # df_cavity['Category'] = 'Cavity'
        # If it is a standing wave cavity, print only half of the arms, as the second half is the same arms in reverse
        df_arms = pd.concat(df_arms_list)

        if isinstance(tolerance_dataframe, bool):
            tolerance_dataframe = np.array(np.abs(self.generate_tolerance_dataframe()))
        if self.p_is_trivial and self.t_is_trivial:
            index = [
                "Tolerance - axial displacement",
                "Tolerance - transversal displacement",
                "Tolerance - tilt angle",
                "Tolerance - radius of Curvature",
                "Tolerance - refractive Index",
            ]
        else:
            index = [PRETTY_INDICES_NAMES[param_name] for param_name in self.perturbable_params_names]
        df_tolerance = pd.DataFrame(tolerance_dataframe.T, columns=self.names, index=index)
        df_tolerance_stacked = stack_df_for_print(df_tolerance)

        whole_df = pd.concat([df_elements_stacked, df_cavity, df_arms, df_tolerance_stacked])
        whole_df["Value"] = whole_df["Value"].apply(lambda x: signif(x, 6))
        whole_df.drop_duplicates(inplace=True)

        if contracted:
            whole_df = whole_df[
                ~whole_df["Parameter"].isin(
                    [
                        "Power decay rate",
                        "Roundtrip power losses",
                        "Free spectral range",
                        "Azimuthal angle [rads]",
                        "Curvature sign",
                        "Elevation angle [rads]",
                        "Poisson ratio",
                        "Surface type",
                        "x [m]",
                        "y [m]",
                        "z [m]",
                        "Angle of incidence_inside [deg]",
                    ]
                )
            ]
            whole_df = whole_df[whole_df["Value"] != 0]

        index = pd.MultiIndex.from_arrays([whole_df["Element"], whole_df["Parameter"]])
        whole_df.set_index(index, inplace=True)
        whole_df.drop(columns=["Parameter", "Element"], inplace=True)
        whole_df.sort_index(inplace=True)

        if save_specs_name is not None:
            whole_df.to_csv(
                f'data//cavities-specs//specs_{datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")}_{save_specs_name}.csv'
            )

        if print_specs:
            print(whole_df, end="\n\n")

        return whole_df

    @property
    def delta_f_frequency_transversal_modes(self):
        if (
            np.isnan(self.arms[0].mode_parameters.z_R[0])
            or self.arms[0].mode_parameters_on_surface_0.z_R[0] == 0
            or np.isnan(self.arms[0].mode_parameters_on_surface_0.z_R[0])
        ):
            return None
        if (
            np.abs(self.arms[0].mode_parameters_on_surface_0.z_R[0] - self.arms[0].mode_parameters_on_surface_0.z_R[1])
            < 1e-14
        ):
            # If there is no astigmatism
            delta_f = (
                -self.total_acquired_gouy_phase / (2 * np.pi) * self.free_spectral_range
            )  # Derivation is at https://mynotebook.labarchives.com/share/Free%2520Electron%2520Lab/MTU2LjB8MTA1ODU5NS8xMjAtMzMzL1RyZWVOb2RlLzI4NTE0OTAzODZ8Mzk2LjA=
            return delta_f
        else:
            raise NotImplementedError(
                "The calculation of the frequency difference between the transversal modes is not implemented for astigmatic cavities."
            )

    def plot_spectrum(
        self,
        modes_decay_rate: float = 2,
        width_over_fsr: float = 0.1,
        n_base_mode: int = 10,
        n_transversal_modes: int = 5,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ):
        fsr = self.free_spectral_range
        lorentzian_width = fsr * width_over_fsr
        main_mode_picks_position = np.arange(n_base_mode) * fsr
        transversal_modes_picks_positions = np.arange(n_transversal_modes) * self.delta_f_frequency_transversal_modes
        picks_positions = main_mode_picks_position[:, None] + transversal_modes_picks_positions[None, :]
        picks_amplitudes = np.ones_like(picks_positions)
        picks_amplitudes = picks_amplitudes * np.exp(-modes_decay_rate * np.arange(1, n_transversal_modes + 1))[None, :]

        x_dummy = np.linspace(transversal_modes_picks_positions[-1], fsr * n_base_mode, 1000)

        # Lorentzian Function
        def lorentzian(x, x0, gamma, A, y0):
            return A * gamma / (np.pi * ((x - x0) ** 2 + gamma**2)) + y0

        lorentzians = lorentzian(
            x_dummy[None, None, :], picks_positions[:, :, None], lorentzian_width, picks_amplitudes[:, :, None], 0
        )
        lorentzians = lorentzians.sum(axis=(0, 1))

        colors = ["blue", "orange", "green", "red", "purple"]

        def plot_lorentzians(ax, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes):
            ax.plot(x_dummy, lorentzians)
            y_limit = ax.get_ylim()
            for i in range(n_transversal_modes):
                ax.vlines(
                    picks_positions[:, i],
                    ymin=y_limit[0],
                    ymax=y_limit[1],
                    color=colors[i],
                    linestyle="--",
                    linewidth=0.75,
                    label=f"Mode {i + 1}",
                )
            ax.hlines(
                (y_limit[1] + y_limit[0]) / 2,
                picks_positions[-2, 0],
                picks_positions[-2, 1],
                color="black",
                linestyle="--",
                linewidth=0.75,
                label="Same longitudinal modes",
            )
            ax.set_xlim(x_dummy[0], x_dummy[-1])
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Amplitude [a.u.]")
            ax.legend()

        if fig is None or ax is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            ax1, ax2 = ax

        plot_lorentzians(ax1, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes)

        plot_lorentzians(ax2, x_dummy, lorentzians, picks_positions, colors, n_transversal_modes)
        ax2.set_xlim(x_dummy[-1], x_dummy[0])
        ax2.set_title("Lorentzian Function - Reverse Frequency")

        plt.tight_layout()


def generate_tolerance_of_NA(
    params_array: np.ndarray,
    parameter_index_for_NA_control: Tuple[int, int],
    arm_index_for_NA: int,
    parameter_values: np.ndarray,
    initial_step: float = 1e-6,
    overlap_threshold: float = 0.9,
    accuracy: float = 1e-3,
    lambda_0_laser: float = 1064e-9,
    standing_wave: bool = True,
    t_is_trivial: bool = False,
    p_is_trivial: bool = True,
    return_cavities: bool = False,
    debug_printing_level: int = 0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Cavity]]]:
    tolerance_matrix = np.zeros(
        (
            params_array.shape[0],
            params_to_perturbable_params_names(params_array, t_is_trivial and p_is_trivial),
            parameter_values.shape[0],
        )
    )
    NAs = np.zeros(parameter_values.shape[0])
    cavities = []
    for k, parameter_value in tqdm(
        enumerate(parameter_values), desc="tolerance_of_NA: parameter_value", disable=debug_printing_level < 1
    ):
        params_temp = params_array.copy()
        params_temp[parameter_index_for_NA_control] = parameter_value
        cavity = Cavity.from_params(
            params=params_temp,
            set_mode_parameters=True,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
            t_is_trivial=t_is_trivial,
            p_is_trivial=p_is_trivial,
            debug_printing_level=debug_printing_level,
        )
        if np.any(np.isnan(cavity.mode_parameters[arm_index_for_NA].NA)) or np.any(
            cavity.mode_parameters[arm_index_for_NA].NA == 0
        ):
            continue
        NAs[k] = cavity.mode_parameters[arm_index_for_NA].NA[0]  # ARBITRARY
        cavities.append(cavity)
        tolerance_matrix[:, :, k] = cavity.generate_tolerance_dataframe(
            initial_step=initial_step, overlap_threshold=overlap_threshold, accuracy=accuracy
        )
    if return_cavities:
        return NAs, tolerance_matrix, cavities
    else:
        return NAs, tolerance_matrix


def plot_tolerance_of_NA(
    params: Optional[np.ndarray] = None,
    parameter_index_for_NA_control: Optional[Tuple[int, int]] = None,
    arm_index_for_NA: Optional[int] = None,
    parameter_values: Optional[np.ndarray] = None,
    initial_step: Optional[float] = 1e-6,
    overlap_threshold: Optional[float] = 0.9,
    accuracy: Optional[float] = 1e-3,
    names: Optional[List[str]] = None,
    lambda_0_laser: Optional[float] = 1064e-9,
    standing_wave: Optional[bool] = True,
    t_is_trivial: bool = False,
    p_is_trivial: bool = True,
    NAs: Optional[np.ndarray] = None,
    tolerance_matrix: Optional[np.ndarray] = None,
):
    if tolerance_matrix is None:
        NAs, tolerance_matrix = generate_tolerance_of_NA(
            params_array=params,
            parameter_index_for_NA_control=parameter_index_for_NA_control,
            arm_index_for_NA=arm_index_for_NA,
            parameter_values=parameter_values,
            initial_step=initial_step,
            overlap_threshold=overlap_threshold,
            accuracy=accuracy,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
        )
    tolerance_matrix = np.abs(tolerance_matrix)
    number_of_params = len(params_to_perturbable_params_names(params, t_is_trivial and p_is_trivial))
    fig, ax = plt.subplots(
        tolerance_matrix.shape[0],
        number_of_params,
        figsize=(number_of_params * 5, tolerance_matrix.shape[0] * 2),
    )
    if names is None:
        names = [None for _ in range(params.shape[0])]
    for i in range(tolerance_matrix.shape[0]):
        for j in range(number_of_params):
            ax[i, j].plot(NAs, tolerance_matrix[i, j, :], color="g")
            title = f"{names[i]}, {INDICES_DICT_INVERSE[j]}"
            ax[i, j].set_title(title)
            if i == tolerance_matrix.shape[0] - 1:
                ax[i, j].set_xlabel("NA")
            if j == 0:
                ax[i, j].set_ylabel("Tolerance")
            ax[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax[i, j].set_yscale("log")
    fig.tight_layout()
    return fig, ax


def plot_tolerance_of_NA_same_plot(
    params: Optional[np.ndarray] = None,
    parameter_index_for_NA_control: Optional[Tuple[int, int]] = None,
    arm_index_for_NA: Optional[int] = None,
    parameter_values: Optional[np.ndarray] = None,
    initial_step: Optional[float] = 1e-6,
    overlap_threshold: Optional[float] = 0.9,
    accuracy: Optional[float] = 1e-3,
    names: Optional[List[str]] = None,
    lambda_0_laser: Optional[float] = 1064e-9,
    standing_wave: Optional[bool] = True,
    NAs: Optional[np.ndarray] = None,
    tolerance_matrix: Optional[np.ndarray] = None,
    ax: plt.Axes = None,
    t_and_p_are_trivial: bool = False,
):
    if tolerance_matrix is None:
        NAs, tolerance_matrix = generate_tolerance_of_NA(
            params_array=params,
            parameter_index_for_NA_control=parameter_index_for_NA_control,
            arm_index_for_NA=arm_index_for_NA,
            parameter_values=parameter_values,
            initial_step=initial_step,
            overlap_threshold=overlap_threshold,
            accuracy=accuracy,
            lambda_0_laser=lambda_0_laser,
            standing_wave=standing_wave,
        )
    tolerance_matrix = np.abs(tolerance_matrix)
    n_elements = tolerance_matrix.shape[0]
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    if names is None:
        names = [None for _ in range(n_elements)]

    j_ranges = [
        [INDICES_DICT["x"]],
        [INDICES_DICT["y"], INDICES_DICT["z"]],
        [INDICES_DICT["theta"], INDICES_DICT["phi"]],
        [INDICES_DICT["r_1"], INDICES_DICT["n_inside_or_after"]],
    ]
    titles = [
        "Axial Position",
        "Transverse Position",
        "Tilt Angles",
        "Radius and Index",
    ]

    if t_and_p_are_trivial:
        j_ranges[1].remove(INDICES_DICT["z"])
        j_ranges[2].remove(INDICES_DICT["theta"])

    for l, a in enumerate(ax.ravel()):
        for i in range(n_elements):
            for j in j_ranges[l]:
                # The condition inside is for the case it is a mirror and the parameter is n, and then we don't want
                # to draw it.
                if not (
                    j == INDICES_DICT["n_inside_or_after"]
                    and (np.isnan(tolerance_matrix[i, j, 0]) or tolerance_matrix[i, j, 0] == 0)
                ):
                    linewidth = 1 + 0.2 * (n_elements - i - 1)
                    print(linewidth)
                    a.plot(
                        NAs,
                        tolerance_matrix[i, j, :],
                        linewidth=linewidth,
                        label=f"{names[i]}, {INDICES_DICT_INVERSE[j]}",
                    )

        a.set_xlabel("NA")
        a.set_ylabel("Tolerance")
        # a.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        a.set_yscale("log")
        a.set_xscale("log")
        a.grid(True)
        a.set_title(titles[l])
        a.legend()
    return ax


def calculate_gaussian_parameters_on_surface(surface: FlatSurface, mode_parameters: ModeParameters):
    # Derivations to all this mathematical s**theta is the LabArchives: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-13
    intersection_point = surface.find_intersection_with_ray(mode_parameters.ray)
    intersection_point = intersection_point[0, :]
    z_minus_z_0 = np.linalg.norm(intersection_point - mode_parameters.center, axis=1)
    q_u, q_v = z_minus_z_0 + 1j * mode_parameters.z_R
    k = 2 * np.pi / mode_parameters.lambda_0_laser

    # Those are the vectors that define the mode and the surface: r_0 is the surface's center with respect to the mode's
    # center, t_hat and p_hat are the unit vectors that span the surface, k_hat is the mode's k vector,
    # u_hat and v_hat are the principle axes of the mode.
    r_0 = surface.center - mode_parameters.center[0, :]  # Technically there are two centers, but their difference is
    # only in the k_hat direction, which doesn'theta make a difference on the projection on the two principle axes of the
    # mode, and for the projection of the k_hat vector we anyway need to set an arbitrary 0, so we can just take the
    # first center.
    t_hat, p_hat = normalize_vector(surface.parameterization(1, 0) - surface.parameterization(0, 0)), normalize_vector(
        surface.parameterization(0, 1) - surface.parameterization(0, 0)
    )
    k_hat = mode_parameters.k_vector
    u_hat_v_hat = mode_parameters.principle_axes
    u_hat = u_hat_v_hat[0, :]
    v_hat = u_hat_v_hat[1, :]

    # The mode as a function of the surface's parameterization:
    # exp([theta,phi] @ A_2 @ [theta,phi] + b @ [theta,phi] + c

    A = (
        1j
        * k
        * np.array(
            [
                [
                    (t_hat @ u_hat) ** 2 / q_u + (t_hat @ v_hat) ** 2 / q_v,
                    (t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v,
                ],
                [
                    (t_hat @ u_hat) * (p_hat @ u_hat) / q_u + (t_hat @ v_hat) * (p_hat @ v_hat) / q_v,
                    (p_hat @ u_hat) ** 2 / q_u + (p_hat @ v_hat) ** 2 / q_v,
                ],
            ]
        )
    )
    b = (
        -1j
        * k
        * np.array(
            [
                (k_hat @ t_hat) + (r_0 @ u_hat) * (t_hat @ u_hat) / q_u + (r_0 @ v_hat) * (t_hat @ v_hat) / q_v,
                (k_hat @ p_hat) + (r_0 @ u_hat) * (p_hat @ u_hat) / q_u + (r_0 @ v_hat) * (p_hat @ v_hat) / q_v,
            ]
        )
    )
    c = -(1 / 2) * 1j * k * (2 * (k_hat @ r_0) + (r_0 @ u_hat) ** 2 / q_u + (r_0 @ v_hat) ** 2 / q_v)

    return A, b, c


def evaluate_cavities_modes_on_surface(cavity_1: Cavity, cavity_2: Cavity, arm_index: int = 0):
    # Chooses a plane on which to evaluate the modes, and calculate the gaussian coefficients of the modes on that plane
    # for both cavities.
    correct_modes = True
    for cavity in [cavity_1, cavity_2]:
        if np.isnan(cavity.arms[0].mode_parameters.z_R[0]):
            try:
                cavity.set_mode_parameters()
            except FloatingPointError:
                correct_modes = False
                break

    mode_parameters_1 = cavity_1.arms[arm_index].mode_parameters
    mode_parameters_2 = cavity_2.arms[arm_index].mode_parameters

    if mode_parameters_1 is None or mode_parameters_2 is None:
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
        return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes

    NAs = np.concatenate((mode_parameters_1.NA, mode_parameters_2.NA))
    if (
        cavity_1.central_line_successfully_traced is False
        or cavity_2.central_line_successfully_traced is False
        or correct_modes is False
        or np.any(np.isnan(NAs))
    ):
        A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
        return A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes

    # Note that the waist might be outside the arm, but even if it is, the mode is still valid.
    cavity_1_waist_pos = mode_parameters_1.center[0, :]  # we take the waist of the first transversal direction
    surface = FlatSurface(center=cavity_1_waist_pos, outwards_normal=mode_parameters_1.k_vector)
    try:
        A_1, b_1, c_1 = calculate_gaussian_parameters_on_surface(surface, mode_parameters_1)
        A_2, b_2, c_2 = calculate_gaussian_parameters_on_surface(surface, mode_parameters_2)
    except FloatingPointError:
        A_1, A_2, b_1, b_2, c_1, c_2, surface, correct_modes = 0, 0, 0, 0, 0, 0, 0, False
    return A_1, A_2, b_1, b_2, c_1, c_2, surface, correct_modes


def calculate_cavities_overlap(cavity_1: Cavity, cavity_2: Cavity, arm_index: int = 0) -> float:
    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_modes = evaluate_cavities_modes_on_surface(cavity_1, cavity_2, arm_index)
    if correct_modes is False:
        return np.nan
    else:
        return gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)


def evaluate_gaussian(A: np.ndarray, b: np.ndarray, c: complex, axis_span: float, N: int = 100):
    x = np.linspace(-axis_span, axis_span, N)
    y = np.linspace(-axis_span, axis_span, N)
    X, Y = np.meshgrid(x, y)
    R = np.stack([X, Y], axis=2)
    # mu = np.array([x_2, y_2])
    # R_shifted = R - mu[None, None, :]
    R_normed_squared = np.einsum("ijk,kl,ijl->ij", R, A, R)
    functions_values = safe_exponent(-(1 / 2) * R_normed_squared + np.einsum("k,ijk->ij", b, R) + c)
    return functions_values


def perturb_cavity(
    cavity: Cavity,
    perturbation_pointer: Union[PerturbationPointer, List[PerturbationPointer], Tuple[PerturbationPointer]],
    **kwargs,  # For the initialization of the new cavity
):
    new_params = copy.deepcopy(cavity.to_params)
    for perturbation_pointer_temp in perturbation_pointer:
        current_value = getattr(
            new_params[perturbation_pointer_temp.element_index], perturbation_pointer_temp.parameter_name
        )
        new_value = current_value + perturbation_pointer_temp.perturbation_value
        # Set the new value back to the attribute
        setattr(
            new_params[perturbation_pointer_temp.element_index], perturbation_pointer_temp.parameter_name, new_value
        )

    parameters_names = [p.parameter_name for p in perturbation_pointer]

    # If the original cavity was symmetrical in the theta axis or the phi axis, and the perturbation does not disturb this
    # symmetry, then the new cavity is also symmetrical in the theta axis or the phi axis:
    perturbance_in_z = [1 for i in parameters_names if i in [ParamsNames.z, ParamsNames.theta]]
    perturbance_in_y = [1 for i in parameters_names if i in [ParamsNames.y, ParamsNames.phi]]
    perturbance_in_z = bool(len(perturbance_in_z))
    perturbance_in_y = bool(len(perturbance_in_y))

    t_is_trivial = cavity.t_is_trivial and not perturbance_in_z
    p_is_trivial = cavity.p_is_trivial and not perturbance_in_y

    new_cavity = Cavity.from_params(
        params=new_params,
        standing_wave=cavity.standing_wave,
        lambda_0_laser=cavity.lambda_0_laser,
        t_is_trivial=t_is_trivial,
        p_is_trivial=p_is_trivial,
        power=cavity.power,
        use_brute_force_for_central_line=cavity.use_brute_force_for_central_line,
        debug_printing_level=cavity.debug_printing_level,
        use_paraxial_ray_tracing=cavity.use_paraxial_ray_tracing,
        **kwargs,
    )
    return new_cavity


def plot_gaussian_subplot(
    A: np.ndarray,
    b: np.ndarray,
    c: float,
    axis_span: float = 0.0005,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        fig, ax = plt.subplots()
    functions_values = evaluate_gaussian(A, b, c, axis_span)
    im = ax.imshow(np.real(functions_values))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    return fig, ax


def plot_2_gaussians_subplots(
    A_1: np.ndarray,
    A_2: np.ndarray,
    # mu_1: np.ndarray, mu_2: np.ndarray, # Seems like I don't need the mus.
    b_1: np.ndarray,
    b_2: np.ndarray,
    c_1: float,
    c_2: float,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    axis_span: float = 0.0005,
    title: Optional[str] = "",
):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    plot_gaussian_subplot(A_1, b_1, c_1, axis_span, fig, ax[0])
    plot_gaussian_subplot(A_2, b_2, c_2, axis_span, fig, ax[1])
    if title is not None:
        fig.suptitle(title)


def plot_2_gaussians_colors(
    A_1: np.ndarray,
    A_2: np.ndarray,
    b_1: np.ndarray,
    b_2: np.ndarray,
    c_1: float,
    c_2: float,
    ax: Optional[plt.Axes] = None,
    axis_span: Optional[float] = None,
    title: Optional[str] = "",
    real_or_abs: str = "abs",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if axis_span is None:
        A_1_diagonal, A_2_diagonal = np.diag(A_1), np.diag(A_2)
        axis_span = 8 * max(np.sqrt(2 / np.min(np.real(A_1_diagonal))), np.sqrt(2 / np.min(np.real(A_2_diagonal))))
    first_gaussian_values = evaluate_gaussian(A_1, b_1, c_1, axis_span)
    second_gaussian_values = evaluate_gaussian(A_2, b_2, c_2, axis_span)
    first_gaussian_values = first_gaussian_values / np.max(np.abs(first_gaussian_values))
    second_gaussian_values = second_gaussian_values / np.max(np.abs(second_gaussian_values))
    third_color_channel = np.zeros_like(first_gaussian_values)
    rgb_image = np.stack([first_gaussian_values, second_gaussian_values, third_color_channel], axis=2)
    if real_or_abs == "abs":
        rgb_image = np.clip(np.abs(rgb_image), 0, 1)
    elif real_or_abs == "real":
        rgb_image = np.real(rgb_image)
    elif real_or_abs == "abs_squared":
        rgb_image = np.clip(np.abs(rgb_image) ** 2, 0, 1)
    else:
        raise ValueError("real_or_abs must be 'abs', 'real' or 'abs_squared'")

    ax.imshow(rgb_image, extent=(-axis_span, axis_span, -axis_span, axis_span))
    ax.set_title(title)


def plot_2_cavity_perturbation_overlap(
    cavity: Cavity,
    perturbation_pointer: Optional[PerturbationPointer] = None,
    second_cavity: Cavity = None,
    arm_index: int = 0,
    ax: Optional[plt.Axes] = None,
    axis_span: Optional[float] = None,
    real_or_abs: str = "abs",
):
    if second_cavity is None:
        second_cavity = perturb_cavity(cavity, perturbation_pointer)

    A_1, A_2, b_1, b_2, c_1, c_2, P1, correct_mode = evaluate_cavities_modes_on_surface(
        cavity, second_cavity, arm_index=arm_index
    )
    if correct_mode:
        overlap = gaussians_overlap_integral(A_1, A_2, b_1, b_2, c_1, c_2)
        plot_2_gaussians_colors(
            A_1,
            A_2,
            b_1,
            b_2,
            c_1,
            c_2,
            ax=ax,
            axis_span=axis_span,
            title=f"Cavity perturbation overlap = {np.abs(overlap):.4f}",
            real_or_abs=real_or_abs,
        )


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
    k = 2 * np.pi / mode_parameters.lambda_0_laser
    integrand = -1j * k / 2 * (u_projection**2 / q_u + v_projection**2 / q_v + k_projection)
    gaussian = safe_exponent(integrand)
    return gaussian


def find_distance_to_first_crossing_positive_side(
    shifts: np.ndarray, overlaps: np.ndarray, crossing_value: float = 0.9
):
    overlaps_under_crossing = overlaps < crossing_value
    if np.any(overlaps_under_crossing):
        first_overlap_crossing = np.argmax(overlaps_under_crossing)
        if first_overlap_crossing == 0:
            crossing_shift = np.nan
        else:
            crossing_shift = interval_parameterization(
                shifts[first_overlap_crossing - 1],
                shifts[first_overlap_crossing],
                (crossing_value - overlaps[first_overlap_crossing - 1])
                / (overlaps[first_overlap_crossing] - overlaps[first_overlap_crossing - 1]),
            )
    else:
        crossing_shift = np.nan
    return crossing_shift


def find_distance_to_first_crossing(shifts: np.ndarray, overlaps: np.ndarray, crossing_value: float = 0.9):
    # Assumes shifts is ascending and that overlaps[i] is the overlap of shift[i]
    positive_shifts = shifts[shifts >= 0]
    negative_shifts = -shifts[shifts <= 0]
    positive_shifts_overlaps = overlaps[shifts >= 0]
    negative_shifts_overlaps = overlaps[shifts <= 0]
    negative_shifts_overlaps = negative_shifts_overlaps[::-1]
    negative_shifts = negative_shifts[::-1]
    crossing_positive_shift = find_distance_to_first_crossing_positive_side(
        positive_shifts, positive_shifts_overlaps, crossing_value=crossing_value
    )
    crossing_negative_shift = find_distance_to_first_crossing_positive_side(
        negative_shifts, negative_shifts_overlaps, crossing_value=crossing_value
    )
    if np.isnan(crossing_negative_shift):
        crossing_shift = crossing_positive_shift
    elif np.isnan(crossing_positive_shift):
        crossing_shift = -crossing_negative_shift
    elif crossing_negative_shift < crossing_positive_shift:
        crossing_shift = -crossing_negative_shift
    elif crossing_positive_shift <= crossing_negative_shift:
        crossing_shift = crossing_positive_shift
    else:
        raise ValueError("Debug me")
    return crossing_shift


def functions_first_crossing_both_directions(
    f: Callable,
    initial_step: float,
    crossing_value: float = 0.9,
    accuracy: float = 0.001,
) -> float:
    positive_step = functions_first_crossing(f, initial_step, crossing_value, accuracy)
    negative_step = functions_first_crossing(lambda x: f(-x), initial_step, crossing_value, accuracy)
    if positive_step < negative_step:
        return positive_step
    else:
        return -negative_step


def match_a_mirror_to_mode(
    mode: ModeParameters,
    material_properties: MaterialProperties,
    z: Optional[float] = None,
    R: Optional[float] = None,
    name: Optional[str] = None,
) -> Union[FlatMirror, CurvedMirror]:
    if z is None and R is None or (z is not None and R is not None):
        raise ValueError("You must provide either z or R, but not both, and not neither.")
    elif z is not None:
        if z == 0:
            mirror = FlatMirror(
                center=mode.center[0, :],
                outwards_normal=mode.k_vector,
                thermal_properties=material_properties,
                name=name,
            )
        else:
            R_z_inverse = np.abs(z / (z**2 + mode.z_R[0] ** 2))
            center = mode.center[0, :] + mode.k_vector * z
            outwards_normal = mode.k_vector * np.sign(z)
            mirror = CurvedMirror(
                center=center,
                outwards_normal=outwards_normal,
                radius=R_z_inverse**-1,
                thermal_properties=material_properties,
                name=name,
            )
    elif R is not None:
        center = mode.z_of_R(R, output_type=np.ndarray)
        outwards_normal = mode.k_vector * np.sign(R)
        if np.isclose(R, 0):
            mirror = FlatMirror(
                center=center,
                outwards_normal=outwards_normal,
                thermal_properties=material_properties,
                name=name,
            )
        else:
            mirror = CurvedMirror(
                center=center,
                outwards_normal=outwards_normal,
                radius=np.abs(R),
                thermal_properties=material_properties,
                name=name,
            )
    return mirror


def local_mode_2_of_lens_parameters(
    lens_parameters: np.ndarray, local_mode_1: LocalModeParameters
):  # les_parameters = [r, n, w]
    R, w, n = lens_parameters
    params = np.array([0, 0, 0, 0, R, n, w, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    surface_0, surface_2 = generate_lens_from_params(params)
    ABCD_first = surface_0.ABCD_matrix(cos_theta_incoming=1)
    ABCD_between = ABCD_free_space(w)
    ABCD_second = surface_2.ABCD_matrix(cos_theta_incoming=1)
    ABCD_total = ABCD_second @ ABCD_between @ ABCD_first
    propagated_mode = propagate_local_mode_parameter_through_ABCD(local_mode_1, ABCD_total, n_1=1, n_2=1)
    return propagated_mode


def match_a_lens_parameters_to_modes(
    local_mode_1: LocalModeParameters,
    local_mode_2: LocalModeParameters,
    fixed_n_lens: Optional[float] = None,
    fix_z_2: bool = False,
):
    def f_roots(lens_parameters: np.ndarray):
        if fixed_n_lens is not None:
            lens_parameters = np.array([lens_parameters[0], lens_parameters[1], fixed_n_lens])
        propagated_mode = local_mode_2_of_lens_parameters(lens_parameters, local_mode_1)
        q_error = propagated_mode.q[0] - local_mode_2.q[0]
        if not fix_z_2:  # if we don't fix z_2, then the error in z_2 is set to 0, regardless of the actual value.
            q_error = 1j * np.imag(q_error)

        if fixed_n_lens is not None:
            return np.array([np.real(q_error), np.imag(q_error)])
        else:
            return np.array([np.real(q_error), np.imag(q_error), 0])

    if fixed_n_lens is not None:
        lens_parameters = optimize.fsolve(f_roots, np.array([1e-2, 1e-3]))
        lens_parameters = np.array([lens_parameters[0], lens_parameters[1], fixed_n_lens])
    else:
        lens_parameters = optimize.fsolve(f_roots, np.array([1e-2, 1e-3, 1.6]))

    resulted_mode_2_parameters = local_mode_2_of_lens_parameters(lens_parameters, local_mode_1)
    return lens_parameters, resulted_mode_2_parameters


def compare_2_cylindrical_cavities(
    params_1: np.ndarray,
    params_2: np.ndarray,
    generate_tolerance_of_NA_dict: dict = {},
    cavities_names: Tuple[str] = ("cavity 1", "cavity 2"),
    elements_names: Tuple[str] = ("Long Arm Mirror", "Lens", "Short Arm Mirror"),
):
    NAs_1, tolerance_matrix_1 = generate_tolerance_of_NA(
        params_1, t_is_trivial=True, p_is_trivial=True, **generate_tolerance_of_NA_dict
    )
    NAs_2, tolerance_matrix_2 = generate_tolerance_of_NA(
        params_2, t_is_trivial=True, p_is_trivial=True, **generate_tolerance_of_NA_dict
    )
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plot_tolerance_of_NA_same_plot(
        params=params_1,
        names=[element_name + " " + cavities_names[0] for element_name in elements_names],
        NAs=NAs_1,
        tolerance_matrix=np.abs(tolerance_matrix_1),
        ax=ax,
        t_and_p_are_trivial=True,
    )
    plot_tolerance_of_NA_same_plot(
        params=params_2,
        names=[element_name + " " + cavities_names[1] for element_name in elements_names],
        NAs=NAs_1,
        tolerance_matrix=np.abs(tolerance_matrix_2),
        ax=ax,
        t_and_p_are_trivial=True,
    )
    return ax


def maximize_overlap(
    cavity: Cavity,
    perturbed_parameter_index: Tuple[int, int],
    perturbation_value: float,
    control_parameters_indices: Tuple[List[int], List[int]],
    print_progress: bool = False,
):
    perturbed_cavity = perturb_cavity(cavity, perturbed_parameter_index, perturbation_value)
    original_overlap = np.abs(calculate_cavities_overlap(cavity_1=cavity, cavity_2=perturbed_cavity))
    if print_progress:
        print("Original overlap:", original_overlap)
        I = 0

    def controlled_overlap(control_parameters_values: np.ndarray):
        corrected_cavity = perturb_cavity(
            perturbed_cavity, control_parameters_indices, control_parameters_values
        )  # * 1e-3
        overlap = calculate_cavities_overlap(cavity_1=cavity, cavity_2=corrected_cavity)
        overlap_abs_minus = np.nan_to_num(-np.abs(overlap), nan=2)
        if print_progress:
            nonlocal I
            I += 1
            print(
                "Iteration",
                I,
                "control_parameters_values",
                control_parameters_values,
                "overlap:",
                np.abs(overlap),
            )
        return overlap_abs_minus

    best_overlap = optimize.minimize(controlled_overlap, x0=np.zeros(len(control_parameters_indices[0])), tol=1e-6)
    # best_overlap.x *= 1e-3
    if print_progress:
        print("Number of iterations:", I)
    # best_overlap = optimize.fsolve(controlled_overlap, x0=np.zeros(len(control_parameters_indices[0])))

    return best_overlap, original_overlap


def find_minimal_width_for_spot_size_and_radius(radius, spot_size_radius, T_edge=1e-3, h_divided_by_spot_size=2.8):
    # relies on the derivation in figures/lens thickness calculation.jpg
    h = h_divided_by_spot_size * spot_size_radius
    try:
        dT_c = radius * (1 - np.sqrt(1 - h**2 / radius**2))
        minimal_T_c = 2 * dT_c + T_edge
        return minimal_T_c
    except FloatingPointError:
        warnings.warn(
            "The spot size radius is too large for the given radius, returning nan",
        )
        return np.nan


def calculate_incidence_angle(surface: Surface, mode_parameters: ModeParameters) -> float:
    # Calculates the incidence angle between the beam at the E=E_0e^-1 lateral_position (one spot size away from it's optical axis)
    # and the surface of an optical element
    if isinstance(surface, FlatSurface):
        return np.arccos(surface.outwards_normal @ mode_parameters.k_vector)

    surface_center_to_waist_position_vector = mode_parameters.center[0, :] - surface.center
    from_the_convex_side = np.sign(surface.outwards_normal @ surface_center_to_waist_position_vector)
    surface_to_waist_distance_signed = np.linalg.norm(surface_center_to_waist_position_vector) * from_the_convex_side

    angle_of_incidence = np.arcsin(
        ((surface.radius + surface_to_waist_distance_signed) * mode_parameters.NA[0]) / surface.radius
    )

    angle_of_incidence_deg = np.degrees(angle_of_incidence)
    return angle_of_incidence_deg


def generate_spot_size_lines(
    mode_parameters: ModeParameters,
    first_point: np.ndarray,
    last_point: np.ndarray,
    dim: int = 2,
    plane: str = "xy",
    principle_axes: Optional[np.ndarray] = None,
):
    if mode_parameters.principle_axes is not None and principle_axes is None:
        principle_axes = mode_parameters.principle_axes
    elif plane == "xy" and principle_axes is None:
        principle_axes = np.array([[0, 0, 1], [0, -1, 0]])
    central_line = Ray(
        origin=first_point, k_vector=mode_parameters.k_vector, length=np.linalg.norm(last_point - first_point)
    )
    t = np.linspace(0, central_line.length, 1000)  # 100 is always enough
    ray_points = central_line.parameterization(t=t)
    z_minus_z_0 = np.linalg.norm(ray_points[:, np.newaxis, :] - mode_parameters.center, axis=2)  # Before
    # the norm the size is 100 | 2 | 3 and after it is 100 | 2 (100 points for in_plane and out_of_plane
    # dimensions)
    sign = np.array([1, -1])
    spot_size_value = spot_size(z_minus_z_0, mode_parameters.z_R, mode_parameters.lambda_0_laser, mode_parameters.n)
    spot_size_lines = (
        ray_points[:, np.newaxis, np.newaxis, :]
        + spot_size_value[:, :, np.newaxis, np.newaxis]
        * principle_axes[np.newaxis, :, np.newaxis, :]
        * sign[np.newaxis, np.newaxis, :, np.newaxis]
    )  # The size is 100 (n_points) | 2 (axis, []) | 2 (sign, [1, -1]) | 3 (coordinate, [x,y,z])
    if dim == 2:
        if plane in ["xy", "yx"]:
            relevant_axis_index = 1
            relevant_diminsions = [0, 1]
        elif plane in ["xz", "zx"]:
            relevant_axis_index = 0
            relevant_diminsions = [0, 2]
        else:
            relevant_axis_index = 0
            relevant_diminsions = [1, 2]
        spot_size_lines = spot_size_lines[:, relevant_axis_index, :, relevant_diminsions]  # Drop the z axis,
        # and drop the lines of the transverse axis the size is:
        # 2 (selected spatial axes) | 100 (n_points) | 2 (sign, [1, -1]
        spot_size_lines_separated = [spot_size_lines[:, :, 0], spot_size_lines[:, :, 1]]
    else:
        spot_size_lines_separated = [
            spot_size_lines[:, 0, 0, :],
            spot_size_lines[:, 0, 1, :],
            spot_size_lines[:, 1, 0, :],
            spot_size_lines[:, 1, 1, :],
        ]  # Each
        # element is a  100 | 3 array.

    return spot_size_lines_separated


def find_equal_angles_surface(
    mode_before_lens: ModeParameters,
    surface_0: CurvedRefractiveSurface,
    T_edge: float = 1e-3,
    h: float = 3.875e-3,
) -> CurvedRefractiveSurface:
    mode_parameters_just_before_surface_0 = mode_before_lens.local_mode_parameters(
        np.linalg.norm(surface_0.center - mode_before_lens.center[0])
    )
    first_angle_of_incidence = calculate_incidence_angle(
        surface=surface_0,
        mode_parameters=mode_before_lens,
    )
    dT_c_0 = dT_c_of_a_lens(R=surface_0.radius, h=h)
    mode_parameters_right_after_surface_0 = propagate_local_mode_parameter_through_ABCD(
        mode_parameters_just_before_surface_0,
        surface_0.ABCD_matrix(cos_theta_incoming=1),
        n_1=surface_0.n_1,
        n_2=surface_0.n_2,
    )

    def match_surface_to_radius(R_1: float) -> CurvedRefractiveSurface:
        T_c = dT_c_0 + T_edge + dT_c_of_a_lens(R=R_1, h=h)
        center_1 = surface_0.center + surface_0.inwards_normal * T_c
        second_surface = CurvedRefractiveSurface(
            radius=R_1,
            outwards_normal=-surface_0.outwards_normal,
            center=center_1,
            n_1=surface_0.n_2,
            n_2=1,
            curvature_sign=-1 * surface_0.curvature_sign,
            thickness=T_c / 2,
        )
        return second_surface

    def f_for_root(R_1: np.ndarray) -> float:  # ARBITRARY - ASSUMES CONVEX LENS
        # R_1 = R_1[0]
        second_surface = match_surface_to_radius(R_1)
        arm = Arm(
            surface_0=surface_0,
            surface_1=second_surface,
            central_line=Ray(
                origin=surface_0.center,
                k_vector=normalize_vector(second_surface.center - surface_0.center),
                length=np.linalg.norm(second_surface.center - surface_0.center),
            ),
            mode_parameters_on_surface_0=mode_parameters_right_after_surface_0,
        )
        local_mode_parameters_right_after_surface_2 = arm.propagate_local_mode_parameters()
        mode_parameters_after_surface_2 = local_mode_parameters_right_after_surface_2.to_mode_parameters(
            location_of_local_mode_parameter=arm.central_line.parameterization(t=arm.central_line.length),
            k_vector=arm.central_line.k_vector,
            # ARBITRARY - ASSUMES CENTREAL LINE IS PERPENDICULAR TO SURFACE_2 - SHOULD BE CHANGED TO
            # THE NEXT CENTRAL LINE, AFTER REFRACTION
        )
        second_angle_of_incidence = calculate_incidence_angle(
            surface=second_surface,
            mode_parameters=mode_parameters_after_surface_2,
        )
        diff = first_angle_of_incidence - second_angle_of_incidence
        return diff

    R_1 = optimize.brentq(f=f_for_root, a=h, b=1000 * surface_0.radius)  # surface_0.radius

    second_surface = match_surface_to_radius(R_1)

    return second_surface


def find_required_value_for_desired_change(
    # This is the best function in the world <3
    cavity_generator: Callable,  # Takes a float as input and returns a cavity
    desired_parameter: Callable,  # Takes a cavity as input and returns a float
    # (NA of some arm, length of some arm, radius of curvature, etc.)
    desired_value: float,  # Desired value to end up with for the parameter
    solver: Callable = optimize.fsolve,
    print_progress=False,
    **kwargs,  # Additional arguments for the solver
) -> Tuple[Cavity, float]:
    def f_root(input_parameter_value: Union[float, np.ndarray]):
        if print_progress:
            print(f"input_parameter_value: {input_parameter_value}")
        if isinstance(input_parameter_value, np.ndarray):
            input_parameter_value = input_parameter_value[0]
        perturbed_cavity = cavity_generator(input_parameter_value)
        output_parameter_value = desired_parameter(perturbed_cavity)
        diff = output_parameter_value - desired_value
        if np.isnan(diff):
            diff = -np.abs(input_parameter_value) * 1e10
        if print_progress:
            print(f"output_parameter_value: {output_parameter_value:.3e}, diff: {diff:.3e}")

        return diff

    input_parameter_value = solver(f_root, **kwargs)
    cavity = cavity_generator(input_parameter_value[0])
    return cavity, input_parameter_value[0]


def find_required_perturbation_for_desired_change(
    cavity: Cavity,
    perturbation_pointer: PerturbationPointer,
    desired_parameter: Callable,
    desired_value: float,
    solver: Callable = optimize.fsolve,
    print_progress=False,
    **kwargs,
) -> Tuple[Cavity, float]:
    def cavity_generator(perturbation_value: float):
        return perturb_cavity(cavity, perturbation_pointer=perturbation_pointer(perturbation_value))

    cavity, perturbation_value = find_required_value_for_desired_change(
        cavity_generator=cavity_generator,
        desired_parameter=desired_parameter,
        desired_value=desired_value,
        solver=solver,
        print_progress=print_progress,
        **kwargs,
    )

    return cavity, perturbation_value


def mirror_lens_mirror_generator_with_unconcentricity(unconcentricity: float, base_params: list[OpticalElementParams]):
    # unconcentricity is the unconcentricity of the long arm
    base_params[0].x = -base_params[0].r_1
    base_params[0].y = 0
    base_params[0].z = 0
    n = base_params[1].n_inside_or_after
    R_1 = base_params[1].r_1
    R_2 = base_params[1].r_2
    lens = generate_lens_from_params(base_params[1])
    lens_left_center = lens[0].center
    T_c = base_params[1].T_c
    f = focal_length_of_lens(R_1, R_2, n, T_c)
    h_2 = f * (n - 1) * T_c / (R_1 * n)
    h_1 = f * (n - 1) * T_c / (R_2 * n)
    d_1 = np.linalg.norm(lens_left_center)
    lens_right_center = lens[1].center
    d_2 = (1 / f - 1 / (d_1 + h_1)) ** -1 - h_2
    # d_2_alternative = d2 = (
    #             d_1*R_2*(n*(R_1 - T_c) + T_c) + R_1*R_2*T_c
    #         ) / (
    #             d_1*(n - 1)*(n*(R_1 + R_2 - T_c) + T_c) - R_1*(n*(R_2 - T_c) + T_c)
    #         )  # This was extracted using ABCD matrices (in research file, 'Image of a point using a thick lens')
    # # And leads to the same result
    right_mirror_coc = lens_right_center + np.array([d_2 - unconcentricity, 0, 0])
    base_params[2].x = right_mirror_coc[0] + base_params[2].r_1
    base_params[2].y = right_mirror_coc[1]
    base_params[2].z = right_mirror_coc[2]
    return Cavity.from_params(
        params=base_params,
        standing_wave=True,
        lambda_0_laser=LAMBDA_0_LASER,
        set_central_line=True,
        set_mode_parameters=False,
        set_initial_surface=False,
        t_is_trivial=True,
        p_is_trivial=True,
        use_paraxial_ray_tracing=False,
        debug_printing_level=1,
    )


def mirror_lens_mirror_cavity_generator(
    NA_left: float = 0.1,
    waist_to_lens: Optional[float] = None,
    h: float = 3.875e-3,
    R_left: float = 5e-3,
    R_right: float = 5e-3,
    T_c: float = 5e-3,
    T_edge: float = 1e-3,
    lens_fixed_properties="sapphire",
    mirrors_fixed_properties="ULE",
    R_small_mirror: Optional[float] = 5e-3,
    waist_to_left_mirror: Optional[float] = None,
    lambda_0_laser=1064e-9,
    set_h_instead_of_w: bool = True,
    collimation_mode: str = "symmetric arm",  # 'symmetric arm' or 'on waist', if both R and L of long arm are given
    # this value is ignored. if non is given, then the assert will raise an error.
    big_mirror_radius: Optional[float] = 2e-1,
    right_arm_length: Optional[float] = None,
    set_R_right_to_equalize_angles: bool = True,
    set_R_right_to_R_left: bool = False,
    set_R_right_to_collimate: bool = False,
    set_R_left_to_collimate: bool = True,
    **kwargs,
):
    # This function receives many parameters that can define a cavity of mirror-lens-mirror and creates a Cavity object
    # out of them.

    assert (R_small_mirror is not None or waist_to_left_mirror is not None) and not (
        R_small_mirror is not None and waist_to_left_mirror is not None
    ), "Either R_small_mirror or waist_to_left_mirror must be set, but not both"

    assert not (
        set_R_left_to_collimate and set_R_right_to_collimate
    ), "Too many solutions: can't set automatically both R_left to collimate and R_right to collimate"

    smart_collimation = set_R_right_to_collimate or set_R_left_to_collimate

    assert (
        not (set_R_right_to_collimate + set_R_right_to_equalize_angles + set_R_right_to_R_left) > 1
    ), "Too many constraints on R_right, must choose either set_R_right_to_collimate or set_R_right_to_equalize_angles or set_R_right_to_R_left"
    assert not (
        smart_collimation and big_mirror_radius is None and right_arm_length is None
    ), "Collimation is not well defined without the big mirror radius or the right arm length, one of them must be set"

    assert not (
        smart_collimation and collimation_mode == "on waist" and right_arm_length is None
    ), "Collimation on waist is not well defined without the right arm length, it must be set"

    if smart_collimation:
        if set_R_left_to_collimate:

            def cavity_generator(R_left_):
                cavity = mirror_lens_mirror_cavity_generator(
                    NA_left=NA_left,
                    waist_to_lens=waist_to_lens,
                    h=h,
                    R_left=R_left_,
                    R_right=R_right,
                    T_c=T_c,
                    T_edge=T_edge,
                    lens_fixed_properties=lens_fixed_properties,
                    mirrors_fixed_properties=mirrors_fixed_properties,
                    waist_to_left_mirror=waist_to_left_mirror,
                    R_small_mirror=R_small_mirror,
                    lambda_0_laser=lambda_0_laser,
                    set_h_instead_of_w=set_h_instead_of_w,
                    collimation_mode=collimation_mode,
                    big_mirror_radius=big_mirror_radius,
                    right_arm_length=right_arm_length,
                    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                    set_R_right_to_R_left=set_R_right_to_R_left,
                    set_R_right_to_collimate=False,
                    set_R_left_to_collimate=False,
                    **kwargs,
                )
                return cavity

            x0 = R_left
        else:

            def cavity_generator(R_right_):
                cavity = mirror_lens_mirror_cavity_generator(
                    NA_left=NA_left,
                    waist_to_lens=waist_to_lens,
                    h=h,
                    R_left=R_left,
                    R_right=R_right_,
                    T_c=T_c,
                    T_edge=T_edge,
                    lens_fixed_properties=lens_fixed_properties,
                    mirrors_fixed_properties=mirrors_fixed_properties,
                    waist_to_left_mirror=waist_to_left_mirror,
                    R_small_mirror=R_small_mirror,
                    lambda_0_laser=lambda_0_laser,
                    set_h_instead_of_w=set_h_instead_of_w,
                    collimation_mode=collimation_mode,
                    big_mirror_radius=big_mirror_radius,
                    right_arm_length=right_arm_length,
                    set_R_right_to_equalize_angles=set_R_right_to_equalize_angles,
                    set_R_right_to_R_left=set_R_right_to_R_left,
                    set_R_right_to_collimate=False,
                    set_R_left_to_collimate=False,
                    **kwargs,
                )
                return cavity

            x0 = R_right

        if right_arm_length is not None and big_mirror_radius is None:
            desired_waist_position = right_arm_length / 2
        elif right_arm_length is None and big_mirror_radius is not None:
            desired_waist_position = big_mirror_radius
        elif right_arm_length is not None and big_mirror_radius is not None:
            desired_waist_position = right_arm_length - big_mirror_radius

        cavity, _ = find_required_value_for_desired_change(
            cavity_generator=cavity_generator,
            # Takes a float as input and returns a cavity
            desired_parameter=lambda cavity: 1 / cavity.arms[2].mode_parameters_on_surfaces[0].z_minus_z_0[0],
            desired_value=-1 / (desired_waist_position),  # We work with the inverse to be stable with collimated beams
            x0=x0,
        )
        return cavity

    if set_R_right_to_R_left:
        R_right = R_left

    mirrors_material_properties = convert_material_to_mirror_or_lens(
        PHYSICAL_SIZES_DICT[f"material_properties_{mirrors_fixed_properties}"], "mirror"
    )
    lens_material_properties = convert_material_to_mirror_or_lens(
        PHYSICAL_SIZES_DICT[f"material_properties_{lens_fixed_properties}"], "lens"
    )

    # Generate left arm's mirror:
    w_0_left = lambda_0_laser / (np.pi * NA_left)
    x_left_waist = 0
    mode_left_center = np.array([x_left_waist, 0, 0])

    mode_left_k_vector = np.array([1, 0, 0])
    mode_left = ModeParameters(
        center=np.stack([mode_left_center, mode_left_center], axis=0),
        k_vector=mode_left_k_vector,
        w_0=np.array([w_0_left, w_0_left]),
        principle_axes=np.array([[0, 0, 1], [0, 1, 0]]),
        lambda_0_laser=lambda_0_laser,
    )
    if waist_to_left_mirror is not None:
        if waist_to_lens is None:
            x_left = x_left_waist - waist_to_left_mirror
            x_lens_left = x_left_waist + waist_to_left_mirror
        else:
            x_left = x_left_waist - waist_to_left_mirror
            x_lens_left = x_left_waist + waist_to_lens
        mirror_left = match_a_mirror_to_mode(
            mode=mode_left, z=x_left - mode_left.center[0, 0], material_properties=mirrors_material_properties
        )
    elif R_small_mirror is not None:
        mirror_left = match_a_mirror_to_mode(
            mode=mode_left, R=-R_small_mirror, material_properties=mirrors_material_properties
        )
        if waist_to_lens is not None:
            x_lens_left = x_left_waist + waist_to_lens
        else:
            x_lens_left = mirror_left.center[0, 0] + np.linalg.norm(mode_left.center[0, :] - mirror_left.center[0, :])

    else:
        raise ValueError("this line should not be reachable due to insert in the beginning")

    # Generate lens:
    # if lens_material_properties_override:
    (
        n,
        alpha_lens,
        beta_lens,
        kappa_lens,
        dn_dT_lens,
        nu_lens,
        alpha_absorption_lens,
        intensity_reflectivity,
        intensity_transmittance,
        temperature,
    ) = lens_material_properties.to_array
    surface_left = CurvedRefractiveSurface(
        center=np.array([x_lens_left, 0, 0]),
        radius=R_left,
        outwards_normal=np.array([-1, 0, 0]),
        n_1=1,
        n_2=n,
        curvature_sign=-1,
        name="lens_left",
        thermal_properties=lens_material_properties,
    )
    if set_R_right_to_equalize_angles:
        surface_right = find_equal_angles_surface(
            mode_before_lens=mode_left,
            surface_0=surface_left,
            T_edge=T_edge,
            h=h,
        )
        T_c = np.linalg.norm(surface_right.center - surface_left.center)
    else:
        if set_h_instead_of_w:
            # In case the user wants to set the height and the edge thickness of the lens instead of the thickness, then
            # the thickness is calculated using tha radii and the edge thickness.
            assert (
                R_left
            ), f"transverse radius of lens ({h:.2e}), can not be bigger than left radius of curvature ({R_left:.2e})"
            assert (
                R_right
            ), f"transverse radius of lens ({h:.2e}), can not be bigger than right radius of curvature ({R_right:.2e})"
            dT_c_left = R_left * (1 - np.sqrt(1 - h**2 / R_left**2))
            dT_c_right = R_right * (1 - np.sqrt(1 - h**2 / R_right**2))
            T_c = T_edge + dT_c_left + dT_c_right

        x_2_right = x_lens_left + T_c

        surface_right = CurvedRefractiveSurface(
            center=np.array([x_2_right, 0, 0]),
            radius=R_right,
            outwards_normal=np.array([1, 0, 0]),
            n_1=n,
            n_2=1,
            curvature_sign=1,
            name="lens_right",
            thermal_properties=lens_material_properties,
        )

    mode_parameters_just_before_surface_left = mode_left.local_mode_parameters(
        np.linalg.norm(surface_left.center - mode_left.center[0])
    )

    mode_parameters_right_after_surface_left = propagate_local_mode_parameter_through_ABCD(
        mode_parameters_just_before_surface_left,
        surface_left.ABCD_matrix(cos_theta_incoming=1),
        n_1=1,
        n_2=n,
    )

    arm_lens = Arm(
        surface_0=surface_left,
        surface_1=surface_right,
        central_line=Ray(
            origin=surface_left.center,
            k_vector=normalize_vector(surface_right.center - surface_left.center),
            length=np.linalg.norm(surface_right.center - surface_left.center),
        ),
        mode_parameters_on_surface_0=mode_parameters_right_after_surface_left,
    )
    mode_parameters_right_after_surface_right = arm_lens.propagate_local_mode_parameters()

    mode_right = mode_parameters_right_after_surface_right.to_mode_parameters(
        location_of_local_mode_parameter=surface_right.center,
        k_vector=np.array([1, 0, 0]),
    )
    z_minus_z_0_right_surface = mode_parameters_right_after_surface_right.z_minus_z_0[0]

    if collimation_mode == "on waist":
        mirror_right = match_a_mirror_to_mode(
            mode=mode_right,
            R=0,
            material_properties=mirrors_material_properties,
        )
    elif big_mirror_radius is not None and right_arm_length is not None:
        warnings.warn("setting both big_mirror_radius and right_arm_length, ignoring NA set")
        center = surface_right.center + mode_right.k_vector * right_arm_length
        outwards_normal = mode_right.k_vector  # not convex compatible currently
        R = big_mirror_radius
        mirror_right = CurvedMirror(
            center=center,
            outwards_normal=outwards_normal,
            radius=R,
            thermal_properties=mirrors_material_properties,
        )
    elif big_mirror_radius is not None and right_arm_length is None:
        mirror_right = match_a_mirror_to_mode(
            mode=mode_right, R=big_mirror_radius, material_properties=mirrors_material_properties
        )
    elif big_mirror_radius is None and right_arm_length is not None:
        z_minus_z_0_right_mirror = z_minus_z_0_right_surface + right_arm_length
        mirror_right = match_a_mirror_to_mode(
            mode=mode_right, z=z_minus_z_0_right_mirror, material_properties=mirrors_material_properties
        )
    else:
        if z_minus_z_0_right_surface > 0:  # It can not be symmetric if the mode is past the waist already at the lens,
            # in such a case, we make it not symmetric.
            z_minus_z_0_right_mirror = (
                z_minus_z_0_right_surface + right_arm_length + 1 / z_minus_z_0_right_surface
            )  # This 1 / z_minus_z_0_right_surface is here to make the mirror further as the NA grows larger.
        else:
            z_minus_z_0_right_mirror = -z_minus_z_0_right_surface
        mirror_right = match_a_mirror_to_mode(
            mode=mode_right, z=z_minus_z_0_right_mirror, material_properties=mirrors_material_properties
        )

    mirror_left_params = mirror_left.to_params
    mirror_left_params.name = "Small Mirror"
    mirror_left.material_properties = mirrors_material_properties
    mirror_right_params = mirror_right.to_params
    mirror_right_params.name = "Big Mirror"
    mirror_right.material_properties = mirrors_material_properties

    params_lens = surface_right.to_params
    params_lens.x = (surface_left.center[0] + surface_right.center[0]) / 2
    params_lens.r_1 = surface_left.radius
    params_lens.r_2 = surface_right.radius
    params_lens.T_c = T_c
    params_lens.n_inside_or_after = n
    params_lens.n_outside_or_before = 1
    params_lens.surface_type = SurfacesTypes.thick_lens
    params_lens.name = "Lens"
    params_lens.material_properties = lens_material_properties

    params = [mirror_left_params, params_lens, mirror_right_params]

    cavity = Cavity.from_params(
        params,
        lambda_0_laser=lambda_0_laser,
        standing_wave=True,
        p_is_trivial=True,
        t_is_trivial=True,
        set_mode_parameters=True,
        initial_mode_parameters=mode_left,
        **kwargs,
    )

    return cavity


def fabry_perot_generator(
    radii: Union[Tuple[float, float], float],
    NA: Optional[float] = None,
    unconcentricity: Optional[float] = None,
    lambda_0_laser=LAMBDA_0_LASER,
    **kwargs,
):
    if isinstance(radii, float):
        radii = (radii, radii)
    if NA is not None:
        w_0 = w_0_of_NA(NA=NA, lambda_laser=lambda_0_laser)
        mode_0 = ModeParameters(
            center=np.array([0, 0, 0]),
            k_vector=np.array([1, 0, 0]),
            lambda_0_laser=LAMBDA_0_LASER,
            w_0=np.array([w_0, w_0]),
            n=1,
            principle_axes=np.array([[1, 0, 0], [0, 1, 0]]),
        )
        mirror_1 = match_a_mirror_to_mode(
            mode=mode_0, material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"], R=radii[0]
        )
        mirror_2 = match_a_mirror_to_mode(
            mode=mode_0, material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"], R=-radii[1]
        )
    elif unconcentricity is not None:
        mirror_1 = CurvedMirror(
            origin=np.array([0, 0, 0]),
            outwards_normal=np.array([-1, 0, 0]),
            radius=radii[0],
            name="Left Mirror",
        )
        mirror_2 = CurvedMirror(
            origin=np.array([-unconcentricity, 0, 0]),
            outwards_normal=np.array([1, 0, 0]),
            radius=radii[1],
            name="Right Mirror",
        )
    else:
        raise ValueError("Either NA or unconcentricity must be provided.")
    return Cavity(
        physical_surfaces=[mirror_1, mirror_2],
        lambda_0_laser=lambda_0_laser,
        t_is_trivial=True,
        p_is_trivial=True,
        standing_wave=True,
        **kwargs,
    )


def reverse_elements_order_of_mirror_lens_mirror(params: Union[np.ndarray, List[OpticalElementParams]]) -> np.ndarray:
    if isinstance(params, np.ndarray):
        # swap first and third rows of params:
        new_params = params.copy()
        new_params[[0, 2]] = new_params[[2, 0]]
        new_params[1, [4, 5]] = new_params[1, [5, 4]]
        new_params[1, 3] += 1j
    else:
        new_params = [params[2], params[1], params[0]]
        new_params[1].r_1, new_params[1].r_2 = new_params[1].r_2, new_params[1].r_1
        new_params[1].phi += np.pi
    return new_params


def generate_mirror_lens_mirror_cavity_textual_summary(
    cavity: Cavity,
    CA: float = 5e-3,
    h: float = 3.875e-3,
    T_edge=1e-3,
    minimal_h_divided_by_spot_size: float = 2.5,
    set_h_instead_of_w: bool = True,
    left_mirror_is_first=True,
):
    if left_mirror_is_first:
        small_mirror_index = 0
        lens_short_arm_surface_index = 1
        lens_long_arm_surface_index = 2
        big_mirror_index = 3
        lens_short_arm_surface_in_arm_index = 0
        lens_long_arm_surface_in_arm_index = 1
        short_arm_index = 0
        long_arm_index = 2
    else:
        small_mirror_index = 3
        lens_short_arm_surface_index = 2
        lens_long_arm_surface_index = 1
        big_mirror_index = 0
        lens_short_arm_surface_in_arm_index = 1
        lens_long_arm_surface_in_arm_index = 0
        short_arm_index = 2
        long_arm_index = 0

    R_short_side = cavity.surfaces_ordered[lens_short_arm_surface_index].radius
    R_long_side = cavity.surfaces_ordered[lens_long_arm_surface_index].radius
    # try: # I should add a non-existent mode and initialize it with np.nan when there is no mode instead of this solution.
    valid_mode = cavity.resonating_mode_successfully_traced
    if valid_mode:
        spot_size_lens_long_side = (
            cavity.arms[1].mode_parameters_on_surfaces[lens_long_arm_surface_in_arm_index].spot_size[0]
        )
        spot_size_lens_short_side = (
            cavity.arms[1].mode_parameters_on_surfaces[lens_short_arm_surface_in_arm_index].spot_size[0]
        )
        waist_to_lens_short_arm = np.abs(
            cavity.surfaces_ordered[lens_short_arm_surface_index].center[0]
            - cavity.mode_parameters[short_arm_index].center[0, 0]
        )
        angle_right = cavity.arms[long_arm_index].calculate_incidence_angle(
            surface_index=lens_short_arm_surface_in_arm_index
        )
        angle_left = cavity.arms[short_arm_index].calculate_incidence_angle(
            surface_index=lens_long_arm_surface_in_arm_index
        )
        spot_size_left_mirror = (
            cavity.arms[short_arm_index].mode_parameters_on_surfaces[lens_short_arm_surface_in_arm_index].spot_size[0]
        )
        spot_size_right_mirror = (
            cavity.arms[long_arm_index].mode_parameters_on_surfaces[lens_long_arm_surface_in_arm_index].spot_size[0]
        )
        short_arm_NA = cavity.arms[short_arm_index].mode_parameters.NA[0]
        long_arm_NA = cavity.arms[long_arm_index].mode_parameters.NA[0]
        long_arm_length = np.linalg.norm(
            cavity.surfaces_ordered[big_mirror_index].center
            - cavity.surfaces_ordered[lens_long_arm_surface_index].center
        )
        waist_to_lens_long_arm = np.abs(
            cavity.mode_parameters[long_arm_index].center[0, 0]
            - cavity.surfaces_ordered[lens_long_arm_surface_index].center[0]
        )
        short_arm_length = np.linalg.norm(
            cavity.surfaces_ordered[lens_short_arm_surface_index].center
            - cavity.surfaces_ordered[small_mirror_index].center
        )
    else:
        spot_size_lens_long_side = np.nan
        spot_size_lens_short_side = np.nan
        waist_to_lens_short_arm = np.nan
        angle_right = np.nan
        angle_left = np.nan
        spot_size_left_mirror = np.nan
        spot_size_right_mirror = np.nan
        short_arm_NA = np.nan
        long_arm_NA = np.nan
        long_arm_length = np.linalg.norm(
            cavity.surfaces_ordered[big_mirror_index].center
            - cavity.surfaces_ordered[lens_long_arm_surface_index].center
        )
        waist_to_lens_long_arm = np.nan
        short_arm_length = np.linalg.norm(
            cavity.surfaces_ordered[lens_short_arm_surface_index].center
            - cavity.surfaces_ordered[small_mirror_index].center
        )

    CA_divided_by_2spot_size = CA / (2 * spot_size_lens_long_side)

    # except AttributeError:
    #     spot_size_lens_right = np.nan
    #     CA_divided_by_2spot_size = np.nan
    #     waist_to_lens_short_arm = np.nan
    #     angle_right = np.nan
    #     angle_left = np.nan
    #     spot_size_left_mirror = np.nan
    #     spot_size_right_mirror = np.nan

    T_c = np.linalg.norm(cavity.surfaces_ordered[2].center - cavity.surfaces_ordered[1].center)

    R_left_mirror = cavity.surfaces_ordered[small_mirror_index].radius

    minimal_width_lens = find_minimal_width_for_spot_size_and_radius(
        radius=R_short_side,
        spot_size_radius=spot_size_lens_long_side,
        T_edge=T_edge,
        h_divided_by_spot_size=minimal_h_divided_by_spot_size,
    )
    geometric_feasibility = True
    if set_h_instead_of_w:
        if CA_divided_by_2spot_size < 2.5:
            geometric_feasibility = False
        lens_specs_string = (
            f"R_small = {R_short_side:.3e},  R_big = {R_long_side:.3e},  D = {2 * h:.3e},  T_edge = {T_edge:.2e},  T_c = {T_c:.3e},  CA={CA:.2e}\n"
            f"spot size (long) (2w) = {2 * spot_size_lens_long_side:.3e}, spot size (short) (2w) = {2 * spot_size_lens_short_side:.3e},   CA / 2w_spot_size = {CA_divided_by_2spot_size:.3e}, lens is wide enough = {geometric_feasibility},   {angle_left=:.2f},   {angle_right=:.2f}"
        )
    else:
        if T_c < minimal_width_lens:
            geometric_feasibility = False
        minimal_CA_lens = minimal_h_divided_by_spot_size * spot_size_lens_long_side
        lens_specs_string = f"R_lens = {R_short_side:.3e},  T_c = {T_c:.3e},  minimal_w_lens = {minimal_width_lens:.2e},  minimal_CA_lens={minimal_CA_lens:.3e},  lens is thick enough = {geometric_feasibility}"

    if cavity.resonating_mode_successfully_traced is True:
        mode_flag = ""
    else:
        mode_flag = "**INVALID MODE**"
    textual_summary = (
        f"{mode_flag} Short Arm: NA = {short_arm_NA:.3e},  length = {short_arm_length:.3e} [m], waist to lens = {waist_to_lens_short_arm:.3e} {mode_flag}\n"
        f"Long Arm NA = {long_arm_NA:.3e},  length = {long_arm_length:.3e} [m], waist to lens = {waist_to_lens_long_arm:.3e}\n"
        f"{lens_specs_string}\n"
        f"R small mirror = {R_left_mirror:.3e}, spot diameter small mirror = {2 * spot_size_left_mirror:.2e}, R big mirror = {cavity.surfaces_ordered[3].radius:.3e}, spot diameter big mirror = {2 * spot_size_right_mirror:.2e}"
    )

    return textual_summary


def plot_mirror_lens_mirror_cavity_analysis(
    cavity: Cavity,
    auto_set_x: bool = True,
    x_span: float = 4e-1,
    auto_set_y: bool = True,
    y_span: float = 8e-3,
    camera_center: Union[int, float] = 1,
    minimal_h_divided_by_spot_size: float = 2.5,
    T_edge=1e-3,
    CA: float = 5e-3,
    h: float = 3.875e-3,
    set_h_instead_of_w: bool = True,
    ax: Optional[plt.Axes] = None,
    add_unheated_cavity: Union[bool, Cavity] = False,
    left_mirror_is_first=True,
    **kwargs,
):
    if left_mirror_is_first is False:
        raise NotImplementedError
    # Assumes: surfaces[0] is the left mirror, surfaces[1] is the lens_left side, surfaces[2] is the lens_right side,
    # surfaces[3] is the right mirror.
    R_left = cavity.surfaces_ordered[1].radius
    T_c = np.linalg.norm(cavity.surfaces_ordered[2].center - cavity.surfaces_ordered[1].center)
    x_left_mirror = cavity.surfaces_ordered[0].center[0]

    assert ax is None or add_unheated_cavity is False, "Can't add unheated cavity when ax is given"

    if add_unheated_cavity:
        fig, ax = plt.subplots(2, 1, figsize=(16, 12))
    elif ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax = [ax]
    else:
        ax = [ax]
    cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[0], **kwargs)

    textual_summary = generate_mirror_lens_mirror_cavity_textual_summary(
        cavity,
        CA=CA,
        h=h,
        T_edge=T_edge,
        minimal_h_divided_by_spot_size=minimal_h_divided_by_spot_size,
        set_h_instead_of_w=set_h_instead_of_w,
    )

    ax[0].set_title(textual_summary)

    if auto_set_x:
        # cavity_length = cavity.surfaces[3].center[0] - cavity.surfaces[0].center[0]
        # ax[0].set_xlim(cavity.surfaces[0].center[0] - 0.01 * cavity_length, cavity.surfaces[3].center[0] + 0.01 * cavity_length)
        direction = np.sign(cavity.surfaces_ordered[3].center[0] - x_left_mirror)
        x_limits = [x_left_mirror - 0.01 * direction, cavity.surfaces_ordered[3].center[0] + 0.01 * direction]
        min_x = min(x_limits)
        max_x = max(x_limits)
        ax[0].set_xlim(min_x, max_x)
    if auto_set_y:
        y_lim = maximal_lens_height(R_left, T_c) * 1.1
    else:
        y_lim = y_span
    ax[0].set_ylim(-y_lim, y_lim)
    if not add_unheated_cavity is False:  # if add_unheated_cavity is either True or a Cavity object
        if isinstance(add_unheated_cavity, bool):
            unheated_cavity = cavity.thermal_transformation()
        else:
            unheated_cavity = add_unheated_cavity
        unheated_cavity.plot(axis_span=x_span, camera_center=camera_center, ax=ax[1])
        try:
            spot_size_lens_long_arm_side = cavity.arms[1].mode_parameters_on_surfaces[1].spot_size[0]
            short_arm_NA_unheated_cavity = unheated_cavity.arms[0].mode_parameters.NA[0]
        except AttributeError:
            spot_size_lens_long_arm_side = "Non existent"
            short_arm_NA_unheated_cavity = "Non existent"
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].set_title(
            f"unheated_cavity, short arm NA={short_arm_NA_unheated_cavity:.2e}, Left mirror 2*spot size = {2 * spot_size_lens_long_arm_side:.2e}"
        )
    plt.subplots_adjust(hspace=0.35)
    plt.gcf().tight_layout()

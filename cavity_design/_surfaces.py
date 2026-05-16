from __future__ import annotations

import copy
import warnings
from typing import Optional, Union, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq

from ._utils import (
    MaterialProperties,
    OpticalSurfaceParams,
    SurfacesTypes,
    CurvatureSigns,
    PHYSICAL_SIZES_DICT,
    ROOM_TEMPERATURE,
    normalize_vector,
    unit_vector_of_angles,
    angles_of_unit_vector,
    rotation_matrix_around_n,
    generalized_mirror_law,
    generalized_snells_law,
    stable_sqrt,
    cos_without_trailing_epsilon,
    nvl,
    LensParams,
    solve_aspheric_profile,
    RIGHT,
    ORIGIN,
    LEFT
)
from ._rays import Ray


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
        # Pointing outwards towards the convex side
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

    def propagate_ray(self, ray: Ray, paraxial: bool = False) -> Ray:
        # Physical surfaces override this function to also change the ray's k_vector.
        intersection_point = self.find_intersection_with_ray(ray, paraxial=paraxial)
        length = np.linalg.norm(intersection_point - ray.origin, axis=-1)
        propagated_ray = Ray(origin=intersection_point, k_vector=ray.k_vector, length=length, n=ray.n)
        return propagated_ray

    def parameterization(self, t: Union[np.ndarray, float], p: Union[np.ndarray, float]) -> np.ndarray:
        # Take parameters and return points on the surface
        raise NotImplementedError

    def get_parameterization(self, points: np.ndarray):
        # takes a point on the surface and returns the parameters
        raise NotImplementedError

    def ABCD_matrix(self, cos_theta_incoming=1):
        # Will be overriden by physical surfaces that actually affect the rays/modes.
        return np.eye(4)

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
    def from_params(params: OpticalSurfaceParams, name: Optional[str] = None):
        p = params  # Just for brevity in the code
        center = np.array([p.x, p.y, p.z])
        outwards_normal = unit_vector_of_angles(p.theta, p.phi)
        if p.surface_type == SurfacesTypes.curved_mirror:  # Mirror
            surface = CurvedMirror(
                radius=p.radius,
                outwards_normal=outwards_normal,
                center=center,
                curvature_sign=p.curvature_sign,
                name=p.name,
                diameter=p.diameter,
                material_properties=p.material_properties,
            )
        elif p.surface_type == SurfacesTypes.curved_refractive_surface:  # Refractive surface (one side of a lens)
            surface = CurvedRefractiveSurface(
                radius=p.radius,
                outwards_normal=outwards_normal,
                center=center,
                n_1=p.n_outside_or_before,
                n_2=p.n_inside_or_after,
                curvature_sign=p.curvature_sign,
                name=p.name,
                material_properties=p.material_properties,
                thickness=p.T_c,
                diameter=p.diameter,
            )
        elif p.surface_type == SurfacesTypes.ideal_lens:  # Ideal lens
            surface = IdealLens(
                outwards_normal=outwards_normal,
                center=center,
                focal_length=p.radius,
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
                curvature_sign=p.curvature_sign,
            )
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
    def to_params(self) -> OpticalSurfaceParams:
        x, y, z = self.center
        if isinstance(self, IdealLens):
            radius = self.focal_length
        elif isinstance(self, CurvedSurface):
            radius = self.radius
        else:
            radius = 0
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
            radius = self.radius
        else:
            raise ValueError(f"Unknown surface type {type(self)}")
        if isinstance(self, AsphericSurface):
            polynomial_coefficients = self.polynomial.coef
        else:
            polynomial_coefficients = None
        if self.material_properties is None:
            self.material_properties = MaterialProperties()

        params = OpticalSurfaceParams(
            name=self.name,
            surface_type=surface_type,
            x=x,
            y=y,
            z=z,
            theta=theta,
            phi=phi,
            radius=radius,
            curvature_sign=curvature_sign,
            T_c=np.nan,
            n_inside_or_after=n_2,
            n_outside_or_before=n_1,
            diameter=self.diameter,
            material_properties=self.material_properties,
            polynomial_coefficients=polynomial_coefficients,
        )
        return params

    @property
    def inverse(self):
        inverted_surface = copy.deepcopy(self)
        if isinstance(self, RefractiveSurface):
            n_1, n_2 = self.n_1, self.n_2
            inverted_surface.n_1 = n_2
            inverted_surface.n_2 = n_1
        if isinstance(self, (CurvedSurface, AsphericSurface)) and not isinstance(self, ReflectiveSurface):
            inverted_surface.curvature_sign *= -1
        return inverted_surface


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
        n_output = getattr(self, "n_2", ray.n)
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

    def ABCD_matrix(self, cos_theta_incoming: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
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
        curvature_sign: int = CurvatureSigns.concave,  # With respect to the incoming beam.
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.nan, **kwargs)
        self._center = center
        self.outwards_normal = normalize_vector(outwards_normal)
        self.curvature_sign = curvature_sign
        self.name = name
        self.diameter = diameter
        self.material_properties = material_properties
        self.polynomial = (
            polynomial_coefficients
            if isinstance(polynomial_coefficients, Polynomial)
            else Polynomial(polynomial_coefficients)
        )
        assert (
            self.polynomial.coef[1] >= 0
        ), "Negative curvature in polynomial, Currently the direction of the curvature should be encoded in the outwards normal direction and not in the polynomial coefficients, so the coefficient of the quadratic term should be positive. This might be relaxed in the future if needed."
        self.thickness_center = self.polynomial((self.diameter / 2) ** 2)  # thickness at the center of the surface
        if self.polynomial.coef[1] == 0:
            self.radius = np.inf
        else:
            self.radius = 1 / (2 * self.polynomial.coef[1])

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        # For a sketch and a detailed explanation on the calculation, go to:
        # "Intersection with a cyllindrically symmetric surface with polynominal parameterization x\left(\rho\right)" in my research lyx file

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

    def normal_at_a_point(self, point: np.ndarray):
        relative_coordinates = self.relative_coordinates(point)
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

        rho_vec = (point - self.center) - ((point - self.center) @ self.inwards_normal)[
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
        paraxial_surface = CurvedSurface(
            radius=self.radius,
            outwards_normal=self.outwards_normal,
            center=self.center,
            curvature_sign=self.curvature_sign,
        )
        intersection_point = paraxial_surface.find_intersection_with_ray_paraxial(ray)
        return intersection_point

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
    ) -> plt.Axes:
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
        return ax


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
            curvature_sign=curvature_sign,
            **kwargs,
        )
        self.curvature_sign = curvature_sign

    def __str__(self):
        return f"AsphericRefractiveSurface(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, polynomial_coefficients={self.polynomial.coef}, n_1={self.n_1}, n_2={self.n_2}, curvature_sign={self.curvature_sign})"

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
        paraxial_approximation_surface = CurvedRefractiveSurface(
            radius=self.radius,
            outwards_normal=RIGHT,
            center=ORIGIN,
            n_1=self.n_1,
            n_2=self.n_2,
            curvature_sign=self.curvature_sign,
        )
        return paraxial_approximation_surface.ABCD_matrix(cos_theta_incoming=cos_theta_incoming)

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError

    @staticmethod
    def pseudo_spherical(
        radius: float,
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
        base_polynomial_coefficients = np.array(
            [
                0,
                1 / (2 * radius),
                1 / (8 * radius**3),
                1 / (16 * radius**5),
                5 / (128 * radius**7),
                7 / (256 * radius**9),
            ]
        )
        # pad polynomial_coefficients to be the same length as base_polynomial_coefficients, if it is longer, trim it ad ad a warning:
        if isinstance(polynomial_coefficients, Polynomial):
            polynomial_coefficients_array = polynomial_coefficients.coef
        else:
            polynomial_coefficients_array = np.array(polynomial_coefficients)
        if len(polynomial_coefficients_array) > len(base_polynomial_coefficients):
            warnings.warn("Polynomial coefficients are longer than the base polynomial coefficients, trimming them.")
            polynomial_coefficients_array = polynomial_coefficients_array[: len(base_polynomial_coefficients)]
        elif len(polynomial_coefficients_array) < len(base_polynomial_coefficients):
            polynomial_coefficients_array = np.pad(
                polynomial_coefficients_array,
                (0, len(base_polynomial_coefficients) - len(polynomial_coefficients_array)),
                mode="constant",
            )
        final_polynomial_coefficients = base_polynomial_coefficients + polynomial_coefficients_array
        return AsphericRefractiveSurface(
            center=center,
            outwards_normal=outwards_normal,
            polynomial_coefficients=final_polynomial_coefficients,
            name=name,
            diameter=diameter,
            material_properties=material_properties,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=curvature_sign,
            **kwargs,
        )


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
        if ray.k_vector.ndim > 1:
            raise NotImplementedError(
                "function is not yet implemented for multiple rays, consider using non-paraxial ray tracing"
            )
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
    def __str__(self):
        return f"FlatMirror(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal})"

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

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
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

    def __str__(self):
        return f"FlatRefractiveSurface(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, n_1={self.n_1}, n_2={self.n_2})"

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
        # Note \! this code assumes the ray is in the x\-y plane\! Until it is fixed, the only perturbations in x,y,phi should be calculated\!
        cos_theta_incoming = np.asarray(cos_theta_incoming)
        sin_theta_incoming = np.sqrt(1 - cos_theta_incoming**2)
        sin_theta_outgoing = (self.n_1 / self.n_2) * sin_theta_incoming
        cos_theta_outgoing = stable_sqrt(1 - sin_theta_outgoing**2)
        mat = np.zeros(cos_theta_incoming.shape + (4, 4), dtype=cos_theta_outgoing.dtype)
        mat[..., 0, 0] = 1
        mat[..., 1, 1] = self.n_1 / self.n_2
        mat[..., 2, 2] = cos_theta_outgoing / cos_theta_incoming
        mat[..., 3, 3] = (self.n_1 * cos_theta_incoming) / (self.n_2 * cos_theta_outgoing)
        return mat


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

    def __str__(self):
        return f"IdealLens(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, focal_length={self.focal_length})"

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

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
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
        curvature_sign: int = -1,
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
            length = -Delta_projection_on_k - self.curvature_sign * np.sqrt(
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
        curvature_sign: int = -1,
        name: Optional[str] = None,
        diameter: float = np.nan,
        material_properties: Optional[MaterialProperties] = None,
    ):

        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=material_properties,
            radius=radius,
            center=center,
            origin=origin,
            diameter=diameter,
            curvature_sign=curvature_sign,
        )

    def __str__(self):
        return f"CurvedMirror(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, radius={self.radius})"

    def scatter_direction_paraxial(self, ray: Ray) -> np.ndarray:
        # This is maybe wrong but does not matter too much because anyway they are not used for the central line finding
        # ATTENTION - THIS SHOULD NOT BE HERE FOR NON-STANDING WAVES CAVITIES - BUT i AM DEALING ONLY WITH THOSE...
        return self.scatter_direction_exact(ray)
        # intersection_point = self.find_intersection_with_ray(ray, paraxial=True)
        # return self.reflect_direction_exact(ray, intersection_point=intersection_point)

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None):
        # order of rows/columns elements is [out-of-plane, out-of-plane, in-plane, in-plane]
        # ATTENTION - THE NEXT PARAGRAPHS IS PROBABLY NO LONGER VALID
        # An approximation is done here (beyond the small angles' approximation) by assuming that the central line
        # lives in the x,y plane, such that the plane of incidence is the x,y plane (parameterized by phi and phi)
        # and the sagittal plane is its transverse (parameterized by theta and theta).
        # This is justified for small perturbations of a cavity whose central line actually lives in the x,y plane.
        # It is not really justified for bigger perturbations and should be corrected.
        # It should be corrected by first finding the real axes, # And then apply a rotation matrix to this matrix on
        # both sides.
        if cos_theta_incoming is None:
            cos_theta_incoming = 1.0

        cos_theta_incoming = np.asarray(cos_theta_incoming)
        # ATTENTION - THIS SHOULD NOT BE HERE FOR NON-STANDING WAVES CAVITIES - BUT I AM DEALING ONLY WITH THOSE...
        cos_theta_incoming = np.ones_like(cos_theta_incoming)

        ABCD = np.zeros((*cos_theta_incoming.shape, 4, 4), dtype=float)
        ABCD[..., 0, 0] = 1
        ABCD[..., 1, 0] = -2 * cos_theta_incoming / self.radius
        ABCD[..., 1, 1] = 1
        ABCD[..., 2, 2] = (
            -1
        )  # Minus due to axis inversion (moving a bit to the left in plane before incidence results in moving a bit to the right after reflection)
        ABCD[..., 3, 2] = 2 / (self.radius * cos_theta_incoming)
        ABCD[..., 3, 3] = -1
        return ABCD

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
                center=self.center,
                material_properties=new_thermal_properties,
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
        curvature_sign: int = -1,
        name: Optional[str] = None,
        material_properties: Optional[MaterialProperties] = None,
        thickness: Optional[float] = 5e-4,
        diameter: Optional[float] = None,
    ):
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=material_properties,
            radius=radius,
            center=center,
            origin=origin,
            curvature_sign=curvature_sign,
            diameter=diameter,
        )
        self.n_1 = n_1
        self.n_2 = n_2
        self.thickness = thickness

    def __str__(self):
        return f"CurvedRefractiveSurface(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, radius={self.radius}, n_1={self.n_1}, n_2={self.n_2})"

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
        cos_theta_incoming = np.asarray(cos_theta_incoming)
        cos_theta_outgoing = np.sqrt(1 - (self.n_1 / self.n_2) ** 2 * (1 - cos_theta_incoming**2))
        R_signed = self.radius * self.curvature_sign
        delta_n_e_out_of_plane = self.n_1 * cos_theta_incoming - self.n_2 * cos_theta_outgoing
        delta_n_e_in_plane = delta_n_e_out_of_plane / (cos_theta_incoming * cos_theta_outgoing)

        ABCD = np.zeros((*cos_theta_incoming.shape, 4, 4), dtype=float)
        ABCD[..., 0, 0] = 1
        ABCD[..., 1, 0] = delta_n_e_out_of_plane / (R_signed * self.n_2)
        ABCD[..., 1, 1] = self.n_1 / self.n_2
        ABCD[..., 2, 2] = cos_theta_outgoing / cos_theta_incoming
        ABCD[..., 3, 2] = delta_n_e_in_plane / (R_signed * self.n_2)
        ABCD[..., 3, 3] = cos_theta_incoming * self.n_1 / (cos_theta_outgoing * self.n_2)
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
            material_properties=new_thermal_properties,
            diameter=self.diameter,
        )

def generate_aspheric_lens_params(
    back_focal_length: float,
    T_c: float,
    n: float,
    forward_normal: np.ndarray,
    flat_faces_center: np.ndarray,
    diameter: float,
    polynomial_degree: int = 6,
    n_outside: float = 1.0,
    material_properties: Optional[MaterialProperties] = None,
    name: Optional[str] = None,
) -> List[OpticalSurfaceParams]:
    if name is None:
        name = "Aspheric Lens"
    p = LensParams(n=n, f=back_focal_length, T_c=T_c)
    coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=polynomial_degree)
    theta, phi = angles_of_unit_vector(forward_normal)
    curved_center = flat_faces_center + T_c * forward_normal
    flat_params = OpticalSurfaceParams(
        name=name + " - flat side",
        surface_type=SurfacesTypes.flat_refractive_surface,
        x=flat_faces_center[0],
        y=flat_faces_center[1],
        z=flat_faces_center[2],
        theta=theta + np.pi,
        phi=phi,
        radius=0,
        curvature_sign=0,
        diameter=diameter,
        polynomial_coefficients=None,
        T_c=np.nan,
        n_inside_or_after=n,
        n_outside_or_before=n_outside,
        material_properties=material_properties,
    )
    curved_params = OpticalSurfaceParams(
        name=name + " - curved side",
        surface_type=SurfacesTypes.aspheric_surface,
        x=curved_center[0],
        y=curved_center[1],
        z=curved_center[2],
        theta=theta,
        phi=phi,
        radius=1 / (2 * coeffs[1]),
        curvature_sign=CurvatureSigns.concave,
        diameter=diameter,
        polynomial_coefficients=coeffs,
        T_c=T_c / 2,
        n_inside_or_after=n_outside,
        n_outside_or_before=n,
        material_properties=material_properties,
    )
    return [flat_params, curved_params]


def convert_curved_refractive_surface_to_ideal_lens(surface: CurvedRefractiveSurface):
    focal_length = 1 / (surface.n_2 - surface.n_1) * surface.radius * (surface.curvature_sign)
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



LASER_OPTIK_MIRROR = CurvedMirror(
    radius=5e-3,
    diameter=7.75e-3,
    outwards_normal=LEFT,
    origin=ORIGIN,
    name="Laser Optik Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)
COASTLINE_20CM_MIRROR = CurvedMirror(
    radius=20e-2,
    diameter=25.4e-3,
    outwards_normal=RIGHT,
    origin=ORIGIN,
    name="Coastline 20cm Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)
COASTLINE_50CM_MIRROR = CurvedMirror(
    radius=50e-2,
    diameter=25.4e-3,
    outwards_normal=RIGHT,
    origin=ORIGIN,
    name="Coastline 50cm Mirror",
    material_properties=PHYSICAL_SIZES_DICT["material_properties_fused_silica"],
)
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
    PARAMS_DEPRECATION_MESSAGE,
    SurfacesTypes,
    CurvatureSigns,
    PHYSICAL_SIZES_DICT,
    ROOM_TEMPERATURE,
    init_repr,
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
    LEFT,
)
from ._rays import Ray

# Positions (centers/origins) are normally real, but an imaginary component is allowed to encode a *relative* position
# (a shift from the previous element, resolved later). Anything with a significant imaginary part (or a nan) is
# considered "not well-defined". POSITION_TINY is the threshold below which an imaginary/real component is treated as
# numerical noise (no physical size in the system is below 1e-12).
POSITION_TINY = 1e-16


def _to_position_array(value: Optional[np.ndarray]) -> np.ndarray:
    # Keep a genuinely-complex (relative) position as complex; otherwise return a clean real float array.
    # None (an undefined, "floating" position) is stored as the conventional size-3 nan array.
    if value is None:
        return np.full(3, np.nan)
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        if np.all(np.abs(arr.imag) <= POSITION_TINY):
            return np.real(arr).astype(float)
        return arr.astype(complex)
    return arr.astype(float)


def _position_is_well_defined(vec: np.ndarray) -> bool:
    arr = np.asarray(vec)
    if np.any(np.isnan(arr)):
        return False
    if np.iscomplexobj(arr) and np.any(np.abs(arr.imag) > POSITION_TINY):
        return False
    return True


class Surface:
    def __init__(
        self,
        outwards_normal: Optional[np.ndarray] = None,
        radius: float = np.nan,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        self.outwards_normal = outwards_normal  # Goes through the (nan-safe, normalizing) setter below.
        self.name = name
        self.radius = radius
        self.diameter = diameter
        self.material_properties = material_properties

    @property
    def outwards_normal(self) -> np.ndarray:
        return self._outwards_normal

    @outwards_normal.setter
    def outwards_normal(self, value: Optional[np.ndarray]):
        # An undefined normal is stored as a size-3 nan array (to preserve structure). A well defined normal
        # is normalized. nan components are passed through untouched (no normalization).
        if value is None:
            self._outwards_normal = np.full(3, np.nan)
        else:
            value = np.asarray(value, dtype=float)
            if np.any(np.isnan(value)):
                self._outwards_normal = value
            else:
                self._outwards_normal = normalize_vector(value)

    @property
    def center(self):
        raise NotImplementedError

    @center.setter
    def center(self, value: np.ndarray):
        raise NotImplementedError

    @property
    def inwards_normal(self):
        return -self.outwards_normal

    @inwards_normal.setter
    def inwards_normal(self, value: Optional[np.ndarray]):
        self.outwards_normal = None if value is None else -np.asarray(value, dtype=float)

    @property
    def positions_defined(self) -> bool:
        # True only when both the orientation and the location of the surface are fully specified: no nans and no
        # unresolved (significantly imaginary) relative positions.
        return _position_is_well_defined(self.outwards_normal) and _position_is_well_defined(self.center)

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
            if isinstance(self, SphericalRefractiveSurface):
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
        warnings.warn(PARAMS_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        return Surface._from_params(params, name=name)

    @staticmethod
    def _from_params(params: OpticalSurfaceParams, name: Optional[str] = None):
        # Non-warning implementation, for internal use by the deprecated params-based entry points.
        p = params  # Just for brevity in the code
        center = np.array([p.x, p.y, p.z])
        outwards_normal = unit_vector_of_angles(p.theta, p.phi)
        if p.surface_type == SurfacesTypes.curved_mirror:  # Mirror
            surface = SphericalMirror(
                radius=p.radius,
                outwards_normal=outwards_normal,
                center=center,
                curvature_sign=p.curvature_sign,
                name=p.name,
                diameter=p.diameter,
                material_properties=p.material_properties,
            )
        elif p.surface_type == SurfacesTypes.curved_refractive_surface:  # Refractive surface (one side of a lens)
            surface = SphericalRefractiveSurface(
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
        warnings.warn(PARAMS_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        return self._to_params

    @property
    def _to_params(self) -> OpticalSurfaceParams:
        # Non-warning implementation, for internal use by the deprecated params-based entry points.
        x, y, z = self.center
        if isinstance(self, IdealLens):
            radius = self.focal_length
        elif isinstance(self, SphericalSurface):
            radius = self.radius
        else:
            radius = 0
        theta, phi = angles_of_unit_vector(self.outwards_normal)
        n_1 = 1
        n_2 = 1
        if isinstance(self, SphericalMirror):
            surface_type = SurfacesTypes.curved_mirror
            curvature_sign = self.curvature_sign
        elif isinstance(self, SphericalRefractiveSurface):
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

        with warnings.catch_warnings():  # The public entry points already warned; don't warn again from in here.
            warnings.simplefilter("ignore", DeprecationWarning)
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
        if isinstance(self, (SphericalSurface, AsphericSurface)) and not isinstance(self, ReflectiveSurface):
            inverted_surface.curvature_sign *= -1
        return inverted_surface

    @property
    def init_syntax(self) -> str:
        """The Python expression that reconstructs this surface, at full precision.

        Eval-able in a ``from cavity_design import *`` namespace; this is the surface's part of the textual
        representation of an optical system (see ``OpticalSystem.init_syntax``)."""
        class_name = type(self).__name__
        keyword_arguments = [("name", self.name)]
        if isinstance(self, CartesianOval):
            # n_1/n_2 define the *shape* of an oval, so they are emitted for the bare geometry class too.
            keyword_arguments += [
                ("center", self.center),
                ("outwards_normal", self.outwards_normal),
                ("E_1", self.E_1),
                ("E_2", self.E_2),
                ("n_1", self.n_1),
                ("n_2", self.n_2),
                ("curvature_sign", self.curvature_sign),
                ("diameter", self.diameter),
                ("material_properties", self.material_properties),
            ]
        elif isinstance(self, AsphericSurface):
            keyword_arguments += [
                ("center", self.center),
                ("outwards_normal", self.outwards_normal),
                ("polynomial_coefficients", self.polynomial.coef),
                ("curvature_sign", self.curvature_sign),
            ]
            if isinstance(self, AsphericRefractiveSurface):
                keyword_arguments += [("n_1", self.n_1), ("n_2", self.n_2)]
            keyword_arguments += [
                ("diameter", self.diameter),
                ("material_properties", self.material_properties),
            ]
        elif isinstance(self, SphericalSurface):
            keyword_arguments += [
                ("radius", self.radius),
                ("outwards_normal", self.outwards_normal),
                ("center", self.center),
                ("curvature_sign", self.curvature_sign),
            ]
            if isinstance(self, SphericalRefractiveSurface):
                keyword_arguments += [
                    ("n_1", self.n_1),
                    ("n_2", self.n_2),
                    ("thickness", self.thickness),
                ]
            keyword_arguments += [
                ("diameter", self.diameter),
                ("material_properties", self.material_properties),
            ]
        elif isinstance(self, FlatSurface):
            keyword_arguments += [
                ("outwards_normal", self.outwards_normal),
                ("center", self.center),
            ]
            if isinstance(self, FlatRefractiveSurface):
                keyword_arguments += [("n_1", self.n_1), ("n_2", self.n_2)]
            if isinstance(self, IdealLens):
                keyword_arguments += [("focal_length", self.focal_length)]
            # The three concrete flat subclasses name the material argument 'thermal_properties'; the bare
            # FlatSurface forwards **kwargs to Surface, whose name is 'material_properties'.
            material_key = (
                "thermal_properties"
                if isinstance(self, (FlatMirror, FlatRefractiveSurface, IdealLens))
                else "material_properties"
            )
            keyword_arguments += [
                ("diameter", self.diameter),
                (material_key, self.material_properties),
            ]
        else:
            raise NotImplementedError(f"init_syntax is not implemented for surfaces of type {class_name}")
        parts = [f"{key}={init_repr(value)}" for key, value in keyword_arguments if value is not None]
        return f"{class_name}({', '.join(parts)})"

    def to_position(self, position: np.ndarray) -> "Surface":
        """Return a copy of this surface with its center placed at ``position`` (orientation unchanged).

        Non-mutating and chainable: ``element.to_position(p).to_orientation(n)``. Works also when this surface's
        position is still undefined (nan)."""
        new_surface = copy.deepcopy(self)
        new_surface.center = _to_position_array(position)
        return new_surface

    def to_orientation(self, outwards_normal: np.ndarray) -> "Surface":
        """Return a copy of this surface rotated about its center so its outwards normal equals ``outwards_normal``.

        The center (the vertex, for a curved surface) is preserved; derived quantities (e.g. the sphere origin)
        are recomputed accordingly. Non-mutating and chainable."""
        new_surface = copy.deepcopy(self)
        old_center = np.array(new_surface.center)
        new_surface.outwards_normal = np.asarray(outwards_normal, dtype=float)
        new_surface.center = old_center
        return new_surface


def surfaces_are_equivalent(surface_1: Surface, surface_2: Surface, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """True when the two surfaces are the same optical surface up to numerical noise.

    Compares the concrete type, the name, the pose (center and outwards normal, nan-tolerant) and the intrinsic
    optical attributes (radius, curvature sign, refractive indices, focal length, thickness, diameter, polynomial
    coefficients and material properties)."""
    if type(surface_1) is not type(surface_2):
        return False
    if getattr(surface_1, "name", None) != getattr(surface_2, "name", None):
        return False

    def values_close(value_1, value_2) -> bool:
        if value_1 is None or value_2 is None:
            return value_1 is None and value_2 is None
        array_1, array_2 = np.asarray(value_1, dtype=complex), np.asarray(value_2, dtype=complex)
        if array_1.shape != array_2.shape:
            return False
        return bool(np.allclose(array_1, array_2, rtol=rtol, atol=atol, equal_nan=True))

    if not values_close(surface_1.center, surface_2.center):
        return False
    if not values_close(surface_1.outwards_normal, surface_2.outwards_normal):
        return False
    for attribute in (
        "radius",
        "curvature_sign",
        "n_1",
        "n_2",
        "E_1",
        "E_2",
        "focal_length",
        "thickness",
        "diameter",
    ):
        if not values_close(getattr(surface_1, attribute, None), getattr(surface_2, attribute, None)):
            return False
    polynomial_1 = getattr(surface_1, "polynomial", None)
    polynomial_2 = getattr(surface_2, "polynomial", None)
    if (polynomial_1 is None) != (polynomial_2 is None):
        return False
    if polynomial_1 is not None and not values_close(polynomial_1.coef, polynomial_2.coef):
        return False
    # MaterialProperties is a dataclass whose auto-generated __eq__ fails on nan fields; its repr is stable and
    # full-precision, so string equality is the robust comparison.
    return repr(surface_1.material_properties) == repr(surface_2.material_properties)


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
        self,
        ray: Ray,
        forward_normal: Optional[np.ndarray] = None,
        paraxial: bool = False,
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
        self,
        ray: Ray,
        intersection_point: Optional[np.ndarray] = None,
        forward_normal: Optional[np.ndarray] = None,
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
        self,
        ray: Ray,
        intersection_point: Optional[np.ndarray] = None,
        forward_normal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # explainable derivation of the calculation in lab archives: https://mynotebook.labarchives.com/MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ==/page/11290221-33
        _, n_forwards = self.enrich_intersection_geometries(
            ray, intersection_point=intersection_point, forward_normal=forward_normal
        )
        refracted_direction_vector = generalized_snells_law(
            k_vector=ray.k_vector, n_forwards=n_forwards, n_1=self.n_1, n_2=self.n_2
        )
        return refracted_direction_vector


def _first_not_none(*values):
    """The first argument that is not None. Used to let an explicit argument win over one taken from a source
    surface, which in turn wins over a plain default."""
    for value in values:
        if value is not None:
            return value
    return None


def _combine_polynomial_corrections(
    base_polynomial_coefficients: np.ndarray,
    polynomial_coefficients: Optional[Union[Polynomial, np.ndarray, List[float]]],
) -> np.ndarray:
    """Add sag corrections (a0, a2, a4...) on top of a base sag profile, padding or trimming them to its length."""
    base_polynomial_coefficients = np.asarray(base_polynomial_coefficients, dtype=float)
    if polynomial_coefficients is None:  # No correction - the base profile as it is.
        polynomial_coefficients = np.zeros(1)
    if isinstance(polynomial_coefficients, Polynomial):
        corrections = np.asarray(polynomial_coefficients.coef, dtype=float)
    else:
        corrections = np.asarray(polynomial_coefficients, dtype=float)
    if len(corrections) > len(base_polynomial_coefficients):
        warnings.warn("Polynomial coefficients are longer than the base polynomial coefficients, trimming them.")
        corrections = corrections[: len(base_polynomial_coefficients)]
    elif len(corrections) < len(base_polynomial_coefficients):
        corrections = np.pad(
            corrections,
            (0, len(base_polynomial_coefficients) - len(corrections)),
            mode="constant",
        )
    return base_polynomial_coefficients + corrections


class AsphericSurface(Surface):
    def __init__(
        self,
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        polynomial_coefficients: Optional[Union[Polynomial, np.ndarray, List[float]]] = None,  # a0, a2, a4...
        curvature_sign: int = CurvatureSigns.concave,  # With respect to the incoming beam.
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        # polynomial_coefficients defaults to None only so that center can default to None (a floating surface,
        # like the other Surface types allow); the shape itself is not optional.
        if polynomial_coefficients is None:
            raise TypeError("polynomial_coefficients must be provided for an aspheric surface")
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.nan, **kwargs)
        self._center = _to_position_array(center)
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
            np.clip(
                np.sum(r_relative**2, -1) - r_relative_projected_on_n**2,
                a_min=0,
                a_max=np.inf,
            )
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

    @center.setter
    def center(self, value: np.ndarray):
        self._center = _to_position_array(value)

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        paraxial_surface = SphericalSurface(
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
            r_back_side[:, 0],
            r_back_side[:, 1],
            linestyle="--",
            color=color if color is not None else "blue",
            **kwargs,
        )
        return ax


class AsphericRefractiveSurface(AsphericSurface, RefractiveSurface):
    def __init__(
        self,
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        polynomial_coefficients: Optional[Union[Polynomial, np.ndarray, List[float]]] = None,  # a0, a2, a4...
        n_1: float = 1,
        n_2: float = 1,
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
        paraxial_approximation_surface = SphericalRefractiveSurface(
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
        radius: Optional[Union[float, "SphericalSurface"]] = None,  # A SphericalSurface: take its geometry and glass.
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        polynomial_coefficients: Optional[
            Union[Polynomial, np.ndarray, List[float]]
        ] = None,  # a0, a2, a4... corrections to the spherical profile
        n_1: Optional[float] = None,  # Defaults to 1, or to the source surface's value when one is given.
        n_2: Optional[float] = None,  # Defaults to 1, or to the source surface's value when one is given.
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        curvature_sign: Optional[int] = None,  # With respect to the incoming beam. Defaults to concave.
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        """The asphere whose profile is the Taylor expansion of a sphere, with optional corrections on top.

        ``radius`` is either the vertex radius itself, or a whole :class:`SphericalSurface` (typically a
        :class:`SphericalRefractiveSurface`) to copy - its radius, pose, curvature sign, glass, aperture, name and
        material are then used as the defaults. Any argument passed explicitly still wins over the source surface's
        own value, so a single surface can be re-placed or re-indexed on the way through."""
        source_surface = radius if isinstance(radius, SphericalSurface) else None
        if source_surface is not None:
            radius = source_surface.radius
            center = _first_not_none(center, source_surface.center)
            outwards_normal = _first_not_none(outwards_normal, source_surface.outwards_normal)
            name = _first_not_none(name, source_surface.name)
            diameter = _first_not_none(diameter, source_surface.diameter)
            material_properties = _first_not_none(material_properties, source_surface.material_properties)
        if radius is None:
            raise TypeError("pseudo_spherical needs either a radius or a SphericalSurface to expand.")
        # A plain SphericalSurface (a mirror, say) carries no refractive indices, hence the getattr rather than a
        # direct attribute access.
        n_1 = _first_not_none(n_1, getattr(source_surface, "n_1", None), 1)
        n_2 = _first_not_none(n_2, getattr(source_surface, "n_2", None), 1)
        curvature_sign = _first_not_none(
            curvature_sign, getattr(source_surface, "curvature_sign", None), CurvatureSigns.concave
        )

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
        final_polynomial_coefficients = _combine_polynomial_corrections(
            base_polynomial_coefficients, polynomial_coefficients
        )
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

    @staticmethod
    def pseudo_cartesian_oval(
        oval: Optional["CartesianOval"] = None,  # A CartesianOval to imitate; or give its parameters below instead.
        degree: int = 10,  # Highest power of rho kept in the profile: a0, a2, ... a_degree.
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        E_1: Optional[float] = None,  # Signed object distance from the vertex. > 0: real object.
        E_2: Optional[float] = None,  # Signed image distance from the vertex. > 0: real image.
        polynomial_coefficients: Optional[
            Union[Polynomial, np.ndarray, List[float]]
        ] = None,  # a0, a2, a4... corrections to the oval profile
        n_1: Optional[float] = None,
        n_2: Optional[float] = None,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        curvature_sign: Optional[int] = None,  # Derived from the optics; only needed for a flat (afocal) vertex.
        material_properties: MaterialProperties = None,
        expansion_method: str = "taylor",  # or "fit"; see CartesianOval.sag_polynomial_coefficients.
        **kwargs,
    ):
        """The polynomial asphere that imitates a Cartesian oval, with optional corrections on top.

        The counterpart of :meth:`pseudo_spherical`: where that one expands a sphere, this one expands the exact
        aplanatic surface of a conjugate pair. Pass either a whole :class:`CartesianOval` as ``oval``, or the
        parameters one would be built from (``E_1``, ``E_2``, ``center``, ``outwards_normal``, ``n_1``, ``n_2`` ...);
        arguments given explicitly override the corresponding values of ``oval``.

        The point of going through an asphere at all is that the profile then becomes *editable*: unlike the oval,
        whose shape is pinned by its two foci, the returned surface accepts arbitrary ``polynomial_coefficients`` on
        top of the expansion - which is how a perfect-imaging profile is used as the starting point of a design that
        is then tuned by hand.

        ``curvature_sign`` behaves differently here than in :meth:`pseudo_spherical`: the illumination direction of
        an oval is fixed by which of its foci is the object, so it is derived from ``E_1``, ``E_2``, ``n_1`` and
        ``n_2``, and only has to be stated for the degenerate afocal surface whose vertex is flat. Note that
        ``degree`` bounds the correction list too - coefficients beyond it are trimmed."""
        center = _first_not_none(center, getattr(oval, "center", None))
        outwards_normal = _first_not_none(outwards_normal, getattr(oval, "outwards_normal", None))
        E_1 = _first_not_none(E_1, getattr(oval, "E_1", None))
        E_2 = _first_not_none(E_2, getattr(oval, "E_2", None))
        name = _first_not_none(name, getattr(oval, "name", None))
        diameter = _first_not_none(diameter, getattr(oval, "diameter", None))
        material_properties = _first_not_none(material_properties, getattr(oval, "material_properties", None))
        n_1 = _first_not_none(n_1, getattr(oval, "n_1", None), 1.0)
        n_2 = _first_not_none(n_2, getattr(oval, "n_2", None), 1.5)

        # Rebuilt rather than used as passed, so that overriding any single parameter re-derives the vertex radius
        # and the curvature sign along with it - and so that all the validation lives in one place.
        expanded_oval = CartesianOval(
            center=center,
            outwards_normal=outwards_normal,
            E_1=E_1,
            E_2=E_2,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=curvature_sign,
            name=name,
            diameter=diameter,
            material_properties=material_properties,
        )
        base_polynomial_coefficients = expanded_oval.sag_polynomial_coefficients(degree=degree, method=expansion_method)
        final_polynomial_coefficients = _combine_polynomial_corrections(
            base_polynomial_coefficients, polynomial_coefficients
        )
        return AsphericRefractiveSurface(
            center=center,
            outwards_normal=outwards_normal,
            polynomial_coefficients=final_polynomial_coefficients,
            name=name,
            diameter=diameter,
            material_properties=material_properties,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=expanded_oval.curvature_sign,
            **kwargs,
        )


def signed_vertex_radius_of_a_cartesian_oval(n_1: float, n_2: float, E_1: float, E_2: float) -> float:
    """The vertex radius of curvature of a Cartesian oval, signed along the propagation direction.

    Positive means the center of curvature lies downstream of the vertex. Obtained by differentiating the
    defining equation at the vertex, and identical to the textbook paraxial refraction formula
    ``n_2/s_2 - n_1/s_1 = (n_2 - n_1)/R`` with ``s_1 = -E_1`` and ``s_2 = +E_2``. Depends only on these four
    scalars, not on the pose, so it can be evaluated before the surface is oriented."""
    optical_power = n_1 / E_1 + n_2 / E_2
    if optical_power == 0:
        return np.inf
    return (n_2 - n_1) / optical_power


def _truncated_series_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Product of two power series, truncated back to the length of the first."""
    return np.convolve(first, second)[: len(first)]


def _truncated_series_square_root(series: np.ndarray) -> np.ndarray:
    """Square root of a power series with a positive constant term, truncated to the same length.

    From ``root * root == series`` order by order: the constant terms give ``root_0 = sqrt(series_0)``, and every
    higher order is one division by ``2 * root_0`` once the lower orders are known."""
    root = np.zeros_like(series)
    root[0] = np.sqrt(series[0])
    for k in range(1, len(series)):
        root[k] = (series[k] - np.dot(root[1:k], root[1:k][::-1])) / (2 * root[0])
    return root


def cartesian_oval_longitudinal_expansion(
    n_1: float, n_2: float, E_1: float, E_2: float, n_coefficients: int
) -> np.ndarray:
    """The power series of a Cartesian oval in rho**2, expanded about its vertex.

    Returns ``c_0 = 0, c_1, ... c_{n_coefficients-1}``, such that a point of the oval at transverse distance ``rho``
    from the optical axis lies ``sum_k c_k * rho**(2k)`` downstream of the vertex, along the propagation direction.
    Depends only on these four scalars, not on the pose. The leading term ``c_1 = 1 / (2 * signed_vertex_radius)``
    reproduces the matching sphere, so the expansion starts exactly where
    :meth:`AsphericRefractiveSurface.pseudo_spherical` does and departs from it at fourth order.

    Each coefficient is solved from the defining equation ``n_1*s_1*L_1 + n_2*s_2*L_2 = C`` in turn: perturbing
    ``c_k`` by ``delta`` moves the equation by ``(n_1 - n_2) * delta`` at order rho**(2k) and not at all below it,
    so one division per order suffices. The squared (quartic) form of the oval is deliberately avoided here - its
    derivative vanishes when ``C == 0``, which is exactly the aplanatic case where the oval degenerates into a
    sphere, whereas ``n_1 - n_2`` is non-zero for every oval that exists at all."""
    coefficients = np.zeros(n_coefficients)
    if n_coefficients < 2:
        return coefficients
    n_1_signed, n_2_signed = n_1 * np.sign(E_1), n_2 * np.sign(E_2)
    C = n_1 * E_1 + n_2 * E_2
    rho_squared = np.zeros(n_coefficients)
    rho_squared[1] = 1.0  # The expansion variable itself, as a series.
    for k in range(1, n_coefficients):
        # The two focus distances as series, with the coefficient currently being solved for still zero - so the
        # residual at order k is precisely the part contributed by the orders already fixed.
        to_focus_1, to_focus_2 = coefficients.copy(), coefficients.copy()
        to_focus_1[0] += E_1
        to_focus_2[0] -= E_2
        L_1 = _truncated_series_square_root(_truncated_series_product(to_focus_1, to_focus_1) + rho_squared)
        L_2 = _truncated_series_square_root(_truncated_series_product(to_focus_2, to_focus_2) + rho_squared)
        residual = n_1_signed * L_1 + n_2_signed * L_2
        residual[0] -= C
        coefficients[k] = -residual[k] / (n_1 - n_2)
    return coefficients


class CartesianOval(Surface):
    """The exact rotationally symmetric surface that images one conjugate pair perfectly.

    A Cartesian oval is the locus on which the optical path from an object focus to an image focus is
    stationary, so *every* ray leaving ``focus_1`` is refracted exactly through ``focus_2`` - no spherical
    aberration at all, unlike the polynomial fit of an :class:`AsphericSurface`.

    Pose and curvature follow the same conventions as the other curved surfaces:

    * ``center`` - the vertex, the point where the optical axis meets the surface.
    * ``outwards_normal`` - unit vector pointing towards the *convex* side.
    * ``radius`` - the vertex radius of curvature, as a non-negative magnitude.
    * ``origin`` - the center of curvature, ``center - outwards_normal * radius``.

    Since ``origin`` lies on the inwards side, the surface bulges towards ``outwards_normal`` and its sag is
    measured along ``inwards_normal``, exactly as for :class:`AsphericSurface`.

    The two focal distances are signed, measured from the vertex along the propagation direction:

    * ``E_1 > 0`` is a real object - the incoming beam diverges from ``focus_1``, which lies behind the
      surface. ``E_1 < 0`` is a virtual object - the incoming beam converges towards a point ahead of it.
    * ``E_2 > 0`` is a real image - the outgoing beam converges to ``focus_2``, which lies ahead of the
      surface. ``E_2 < 0`` is a virtual image - the outgoing beam diverges from a point behind it.

    Requiring both beams to travel forwards pins the sign of each optical path term to the sign of its own
    focal distance, so the defining equation needs no separate branch flag::

        n_1*sign(E_1)*|r - focus_1| + n_2*sign(E_2)*|r - focus_2| = C,      C = n_1*E_1 + n_2*E_2

    Eliminating the square roots turns this into the quartic Cartesian oval polynomial of
    https://en.wikipedia.org/wiki/Cartesian_oval . The un-squared form above is the one used throughout,
    because it selects a single branch of the oval and its gradient is well conditioned everywhere except
    at the foci themselves.

    Unlike an asphere, ``curvature_sign`` is *not* a free choice here. An asphere's sag direction and its
    illumination direction are independent, but an oval's illumination direction is fixed by which focus is
    the object and which is the image, so ``curvature_sign == sign(signed_vertex_radius)``. It is still
    accepted as an argument, for symmetry with the sibling classes, and validated.
    """

    NEWTON_ITERATIONS = 6
    NEWTON_RELATIVE_TOLERANCE = 1e-12  # of the optical path constant C; far below any optical tolerance.

    def __init__(
        self,
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        E_1: Optional[float] = None,  # Signed object distance from the vertex. > 0: real object.
        E_2: Optional[float] = None,  # Signed image distance from the vertex. > 0: real image.
        n_1: float = 1.0,
        n_2: float = 1.5,
        curvature_sign: Optional[int] = None,  # Derived from the optics; validated if given. See the class docstring.
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        # E_1/E_2 default to None only so that center can default to None (a floating surface, like the other
        # Surface types allow); the shape itself is not optional.
        if E_1 is None or E_2 is None:
            raise TypeError("E_1 and E_2 must be provided for a Cartesian oval")
        if E_1 == 0 or E_2 == 0:
            raise ValueError("E_1 and E_2 must be non-zero: a focus sitting on the vertex is degenerate.")
        if n_1 == n_2:
            raise ValueError(
                "n_1 and n_2 must differ: a Cartesian oval between equal refractive indices is degenerate."
            )
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.nan, n_1=n_1, n_2=n_2, **kwargs)
        self._center = _to_position_array(center)
        self.name = name
        self.diameter = diameter
        self.material_properties = material_properties
        # Unlike an asphere, the *shape* of an oval depends on the refractive indices, so a bare CartesianOval -
        # which does not inherit RefractiveSurface and therefore never reaches its __init__ - has to store them too.
        self.n_1 = n_1
        self.n_2 = n_2
        self.E_1 = E_1
        self.E_2 = E_2

        signed_radius = signed_vertex_radius_of_a_cartesian_oval(n_1=n_1, n_2=n_2, E_1=E_1, E_2=E_2)
        self.radius = np.abs(signed_radius)
        if np.isfinite(signed_radius):
            derived_curvature_sign = int(np.sign(signed_radius))
            if curvature_sign is not None and int(curvature_sign) != derived_curvature_sign:
                raise ValueError(
                    f"curvature_sign={curvature_sign} contradicts the optics: with n_1={n_1}, n_2={n_2}, E_1={E_1} and "
                    f"E_2={E_2} the signed vertex radius is {signed_radius}, which forces "
                    f"curvature_sign={derived_curvature_sign}. Either drop the argument and let it be derived, or "
                    f"flip outwards_normal if the surface is meant to face the other way."
                )
            self.curvature_sign = derived_curvature_sign
        else:
            # Afocal: the vertex is locally flat, so sign(signed_radius) says nothing about which side the light
            # arrives from and the caller has to state it.
            if curvature_sign is None or int(curvature_sign) == CurvatureSigns.flat:
                raise ValueError(
                    f"n_1/E_1 + n_2/E_2 == 0, so the vertex of this Cartesian oval is flat and its illumination "
                    f"direction cannot be derived. Pass curvature_sign=CurvatureSigns.convex (the light arrives from "
                    f"the outwards_normal side) or CurvatureSigns.concave explicitly."
                )
            self.curvature_sign = int(curvature_sign)

        self.C = n_1 * E_1 + n_2 * E_2  # The constant optical path difference that defines the surface.
        self.thickness_center = self.local_sag(self.diameter / 2)  # sag at the edge of the clear aperture

    # ---------------------------------------------------------------- geometry, all derived from the pose

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value: np.ndarray):
        self._center = _to_position_array(value)

    @property
    def propagation_direction(self) -> np.ndarray:
        """The direction the light travels through this surface.

        ``curvature_sign`` is taken with respect to the incoming ray, so a convex surface (+1) is one the light
        reaches from the ``outwards_normal`` side, i.e. travelling along ``inwards_normal``."""
        return -self.curvature_sign * self.outwards_normal

    @property
    def origin(self) -> np.ndarray:
        """The center of curvature of the osculating sphere at the vertex."""
        return self.center - self.radius * self.outwards_normal

    @property
    def focus_1(self) -> np.ndarray:
        """The object focus. The incoming beam diverges from it when ``E_1 > 0``."""
        return self.center - self.E_1 * self.propagation_direction

    @property
    def focus_2(self) -> np.ndarray:
        """The image focus. The outgoing beam converges to it when ``E_2 > 0``."""
        return self.center + self.E_2 * self.propagation_direction

    @property
    def signed_indices(self) -> Tuple[float, float]:
        """``(n_1*sign(E_1), n_2*sign(E_2))`` - the weights of the two optical path terms."""
        return self.n_1 * np.sign(self.E_1), self.n_2 * np.sign(self.E_2)

    def defining_equation(self, r: np.ndarray) -> Union[np.ndarray, float]:
        """Points on the surface satisfy ``defining_equation(r) == 0``.

        This is the optical path residual ``n_1*s_1*L_1 + n_2*s_2*L_2 - C``. Unlike
        ``AsphericSurface.defining_equation`` its sign carries no concave/convex meaning; it exists to be
        driven to zero by the root finders below."""
        n_1_signed, n_2_signed = self.signed_indices
        L_1 = np.linalg.norm(r - self.focus_1, axis=-1)
        L_2 = np.linalg.norm(r - self.focus_2, axis=-1)
        return n_1_signed * L_1 + n_2_signed * L_2 - self.C

    def normal_at_a_point(self, point: np.ndarray) -> np.ndarray:
        n_1_signed, n_2_signed = self.signed_indices
        d_1 = point - self.focus_1
        d_2 = point - self.focus_2
        L_1 = np.linalg.norm(d_1, axis=-1)[..., np.newaxis]
        L_2 = np.linalg.norm(d_2, axis=-1)[..., np.newaxis]
        # The gradient of the defining equation, which is normal to its level set. Its overall sign is irrelevant:
        # forward_normal_at_a_point re-signs it against the ray's k_vector before Snell's law is applied.
        gradient = n_1_signed * d_1 / L_1 + n_2_signed * d_2 / L_2
        return normalize_vector(gradient)

    def local_sag(self, rho: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """The sag at transverse distance ``rho`` from the optical axis, measured along ``inwards_normal``.

        Purely local geometry, so this works on a floating surface whose center is still undefined. A ``rho`` the
        surface never reaches - an oval closes on itself, so it has a widest point - comes back as nan, the same
        way a ray that misses does."""
        rho = np.asarray(rho, dtype=float)
        n_1_signed, n_2_signed = self.signed_indices
        signed_radius = self.radius * self.curvature_sign
        # Longitudinal coordinate measured from the vertex along the propagation direction, seeded with the
        # parabola of the matching sphere.
        with np.errstate(invalid="ignore", divide="ignore"):
            xi = rho**2 / (2 * signed_radius)
            for _ in range(self.NEWTON_ITERATIONS):
                L_1 = np.sqrt((xi + self.E_1) ** 2 + rho**2)
                L_2 = np.sqrt((xi - self.E_2) ** 2 + rho**2)
                f = n_1_signed * L_1 + n_2_signed * L_2 - self.C
                f_prime = n_1_signed * (xi + self.E_1) / L_1 + n_2_signed * (xi - self.E_2) / L_2
                xi = xi - f / f_prime
            # Past the widest point of the oval there is no solution at all, and the iteration wanders off to an
            # arbitrary number instead of failing. Checked, rather than trusted, on the same terms as the ray solver.
            residual = (
                n_1_signed * np.sqrt((xi + self.E_1) ** 2 + rho**2)
                + n_2_signed * np.sqrt((xi - self.E_2) ** 2 + rho**2)
                - self.C
            )
            converged = np.abs(residual) <= self.NEWTON_RELATIVE_TOLERANCE * max(abs(self.C), 1.0)
            xi = np.where(converged, xi, np.nan)
        # propagation_direction == curvature_sign * inwards_normal, so the sag along inwards_normal is
        # curvature_sign * xi - non-negative, because radius == signed_radius * curvature_sign is non-negative.
        return self.curvature_sign * xi

    def sag_polynomial_coefficients(
        self, degree: int = 10, method: str = "taylor", rho_max: Optional[float] = None
    ) -> np.ndarray:
        """This oval as the sag coefficients ``a_0, a_2, ... a_degree`` of a polynomial in rho**2.

        These are exactly what an :class:`AsphericSurface` takes, in the same convention - a non-negative sag
        measured along ``inwards_normal`` - so they turn the oval into the polynomial asphere that best imitates
        it. :meth:`AsphericRefractiveSurface.pseudo_cartesian_oval` wraps this into a finished surface.

        ``method="taylor"`` expands about the vertex, matching what
        :meth:`AsphericRefractiveSurface.pseudo_spherical` does for a sphere: exact on axis, with the whole error
        pushed to the edge of the aperture. ``method="fit"`` least-squares fits the sag out to ``rho_max``
        (by default the aperture radius) instead, spreading the residual over the aperture - usually the better
        surface at a given degree, at the price of no longer being the local expansion.

        Worth knowing before trusting a high degree: the Taylor series has a finite radius of convergence, set by
        the oval's own geometry and *not* by its vertex radius - for a steeply curved oval it can be well inside
        the clear aperture, and raising ``degree`` then stops helping. Compare against :meth:`local_sag`, which is
        exact everywhere, and fall back to ``method="fit"`` when the expansion stalls.

        Pose-independent either way, so it also works on a floating oval whose center is still undefined."""
        if degree < 2 or degree % 2 != 0:
            raise ValueError(f"degree must be an even power of rho of at least 2 (2, 4, 6 ...), got {degree}.")
        n_coefficients = degree // 2 + 1  # a_0 through a_degree.
        if method == "taylor":
            longitudinal_coefficients = cartesian_oval_longitudinal_expansion(
                n_1=self.n_1, n_2=self.n_2, E_1=self.E_1, E_2=self.E_2, n_coefficients=n_coefficients
            )
            # Same conversion as at the end of local_sag: propagation_direction == curvature_sign * inwards_normal,
            # so the sag along inwards_normal is curvature_sign times the longitudinal coordinate.
            return self.curvature_sign * longitudinal_coefficients
        if method == "fit":
            rho_max = self.diameter / 2 if rho_max is None else rho_max
            rho = np.linspace(0, rho_max, 512)
            # Fitted in the normalized variable (rho/rho_max)**2, because the raw powers of rho**2 span twenty
            # orders of magnitude over a millimetric aperture and the design matrix would be hopeless.
            normalized = (rho / rho_max) ** 2
            design_matrix = np.stack([normalized**k for k in range(1, n_coefficients)], axis=-1)
            fitted_coefficients = np.linalg.lstsq(design_matrix, self.local_sag(rho), rcond=None)[0]
            coefficients = np.zeros(n_coefficients)
            # a_0 is held at zero rather than fitted, so that the vertex of the asphere stays on the oval's center.
            coefficients[1:] = fitted_coefficients / rho_max ** (2 * np.arange(1, n_coefficients))
            return coefficients
        raise ValueError(f"Unknown expansion method {method!r}; expected 'taylor' or 'fit'.")

    # ---------------------------------------------------------------- ray tracing

    def _seed_length(self, ray: Ray) -> np.ndarray:
        """An analytic first guess of the ray length to the surface, from the sphere that matches it at the vertex."""
        if np.isfinite(self.radius):
            # SphericalSurface places its origin at center + radius*inwards_normal, which is exactly this surface's
            # origin, and uses curvature_sign to pick the near/far root - the same root the oval solution sits near.
            matching_sphere = SphericalSurface(
                radius=self.radius,
                outwards_normal=self.outwards_normal,
                center=self.center,
                curvature_sign=self.curvature_sign,
            )
            seed_point = matching_sphere.find_intersection_with_ray_exact(ray)
            length = np.sum((seed_point - ray.origin) * ray.k_vector, axis=-1)
        else:
            length = np.full(ray.origin.shape[:-1], np.nan)
        # Rays that miss the matching sphere altogether still deserve an attempt, and so does the flat-vertex case:
        # fall back to the tangent plane at the vertex.
        with np.errstate(invalid="ignore", divide="ignore"):
            tangent_plane_length = np.sum((self.center - ray.origin) * self.outwards_normal, axis=-1) / np.sum(
                ray.k_vector * self.outwards_normal, axis=-1
            )
        return np.where(np.isnan(length), tangent_plane_length, length)

    def find_intersection_with_ray_exact(self, ray: Ray) -> np.ndarray:
        # Two stages: the analytic seed above, then a vectorized Newton-Raphson on the exact optical path residual
        # f(t) = n_1*s_1*L_1(t) + n_2*s_2*L_2(t) - C, whose derivative along the ray is the projection of k_vector
        # onto the two focus-to-point directions. The seed is close enough that a handful of iterations reach
        # machine precision, so unlike AsphericSurface this needs no per-ray Python loop.
        n_1_signed, n_2_signed = self.signed_indices
        focus_1, focus_2 = self.focus_1, self.focus_2
        length = self._seed_length(ray)

        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(self.NEWTON_ITERATIONS):
                r = ray.parameterization(length)
                d_1, d_2 = r - focus_1, r - focus_2
                L_1 = np.linalg.norm(d_1, axis=-1)
                L_2 = np.linalg.norm(d_2, axis=-1)
                f = n_1_signed * L_1 + n_2_signed * L_2 - self.C
                f_prime = (
                    n_1_signed * np.sum(d_1 * ray.k_vector, axis=-1) / L_1
                    + n_2_signed * np.sum(d_2 * ray.k_vector, axis=-1) / L_2
                )
                length = length - f / f_prime

            # A ray counts as hitting the surface only if it converged and landed inside the clear aperture. The
            # aperture test matters more here than for AsphericSurface, whose bracketed solve is bounded by
            # construction while an unbounded Newton step is not.
            r = ray.parameterization(length)
            converged = np.abs(self.defining_equation(r)) <= self.NEWTON_RELATIVE_TOLERANCE * max(abs(self.C), 1.0)
            inside_aperture = self.radial_distance_from_axis(r) <= self.diameter / 2
            length = np.where(converged & inside_aperture, length, np.nan)
        return ray.parameterization(length)

    def radial_distance_from_axis(self, r: np.ndarray) -> np.ndarray:
        """Distance of ``r`` from the optical axis of this surface."""
        r_relative = r - self.center
        longitudinal = np.sum(r_relative * self.outwards_normal, axis=-1)
        return np.sqrt(np.clip(np.sum(r_relative**2, axis=-1) - longitudinal**2, a_min=0, a_max=np.inf))

    def find_intersection_with_ray_paraxial(self, ray: Ray) -> np.ndarray:
        paraxial_surface = SphericalSurface(
            radius=self.radius,
            outwards_normal=self.outwards_normal,
            center=self.center,
            curvature_sign=self.curvature_sign,
        )
        intersection_point = paraxial_surface.find_intersection_with_ray_paraxial(ray)
        return intersection_point

    # ---------------------------------------------------------------- the rest of the Surface interface

    @property
    def inverse(self):
        # Reversing the light reverses the propagation direction, so the two foci swap roles without moving:
        # focus_1 of the inverse is focus_2 of the original and vice versa, and the defining equation comes out
        # literally unchanged. The signed vertex radius negates, so curvature_sign flips on its own while radius
        # and origin - which are properties of the shape, not of the illumination - stay put.
        return type(self)(
            center=self.center,
            outwards_normal=self.outwards_normal,
            E_1=self.E_2,
            E_2=self.E_1,
            n_1=self.n_2,
            n_2=self.n_1,
            curvature_sign=-self.curvature_sign,
            name=self.name,
            diameter=self.diameter,
            material_properties=self.material_properties,
        )

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
            raise NotImplementedError("Plotting CartesianOval is only implemented for the 'xy' plane.")
        if dim != 2:
            raise NotImplementedError("Plotting CartesianOval is only implemented for 2D plots.")
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
            + self.local_sag(np.abs(t_dummy))[:, np.newaxis] * longitudinal_direction
        )
        ax.plot(r[:, 0], r[:, 1], color=color if color is not None else "blue", **kwargs)

        r_back_side = (
            self.center + self.inwards_normal * self.thickness_center + transverse_direction * t_dummy[:, np.newaxis]
        )
        # create kwargs without linestyle to avoid warning:
        kwargs.pop("linestyle", None)
        kwargs.pop("ls", None)
        ax.plot(
            r_back_side[:, 0],
            r_back_side[:, 1],
            linestyle="--",
            color=color if color is not None else "blue",
            **kwargs,
        )
        return ax


class RefractiveCartesianOval(CartesianOval, RefractiveSurface):
    def __init__(
        self,
        center: Optional[np.ndarray] = None,  # None: floating (undefined); imaginary: relative to the previous element.
        outwards_normal: Optional[np.ndarray] = None,
        E_1: Optional[float] = None,  # Signed object distance from the vertex. > 0: real object.
        E_2: Optional[float] = None,  # Signed image distance from the vertex. > 0: real image.
        n_1: float = 1.0,
        n_2: float = 1.5,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        curvature_sign: Optional[int] = None,  # Derived from the optics; validated if given.
        material_properties: MaterialProperties = None,
        **kwargs,
    ):
        super().__init__(
            center=center,
            outwards_normal=outwards_normal,
            E_1=E_1,
            E_2=E_2,
            name=name,
            diameter=diameter,
            material_properties=material_properties,
            n_1=n_1,
            n_2=n_2,
            curvature_sign=curvature_sign,
            **kwargs,
        )

    def __str__(self):
        return f"RefractiveCartesianOval(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, E_1={self.E_1}, E_2={self.E_2}, n_1={self.n_1}, n_2={self.n_2}, curvature_sign={self.curvature_sign})"

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
        paraxial_approximation_surface = SphericalRefractiveSurface(
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


class FlatSurface(Surface):
    def __init__(
        self,
        outwards_normal: Optional[np.ndarray] = None,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(outwards_normal=outwards_normal, name=name, radius=np.inf, **kwargs)
        if distance_from_origin is not None and center is not None:
            raise ValueError("Only one of distance_from_origin or center must be specified")
        # An undefined position is stored as a size-3 nan center. distance_from_origin is a derived property.
        if distance_from_origin is not None:
            self.center_of_mirror_private = self.outwards_normal * distance_from_origin
        elif center is not None:
            self.center_of_mirror_private = _to_position_array(center)
        else:
            self.center_of_mirror_private = np.full(3, np.nan)

    @property
    def distance_from_origin(self):
        # Signed distance of the plane from the global origin along the outwards normal. Derived from the center so
        # it always stays consistent when either the center or the normal is updated through their setters.
        return self.center_of_mirror_private @ self.outwards_normal

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

    @center.setter
    def center(self, value: np.ndarray):
        # distance_from_origin is derived from this, so nothing else to update.
        self.center_of_mirror_private = _to_position_array(value)

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
        outwards_normal: Optional[np.ndarray] = None,
        distance_from_origin: Optional[float] = None,
        center: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        thermal_properties: Optional[MaterialProperties] = None,
        diameter: Optional[float] = None,
    ):
        # Note: radius is not forwarded here — FlatSurface.__init__ already fixes it to np.inf. Passing it again
        # would collide ("multiple values for keyword argument 'radius'").
        super().__init__(
            outwards_normal=outwards_normal,
            name=name,
            material_properties=thermal_properties,
            distance_from_origin=distance_from_origin,
            center=center,
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

    def ABCD_matrix(self, cos_theta_incoming: Union[float, np.ndarray] = None) -> np.ndarray:
        # Assumes the ray is in the x-y plane, and the mirror is in the z-x plane
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])

    # radius is a plain attribute set to np.inf by FlatSurface.__init__ (no read-only property, so the base
    # `self.radius = radius` assignment does not fail).

    def thermal_transformation(self, P_laser_power: float, w_spot_size: float, **kwargs):
        raise NotImplementedError


class FlatRefractiveSurface(FlatSurface, RefractiveSurface):

    def __init__(
        self,
        outwards_normal: Optional[np.ndarray] = None,
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
        outwards_normal: Optional[np.ndarray] = None,
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


class SphericalSurface(Surface):
    def __init__(
        self,
        radius: float = np.nan,
        outwards_normal: Optional[np.ndarray] = None,  # Pointing from the origin of the sphere to the mirror's center.
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
        # An undefined position is stored as a size-3 nan origin. The origin is the center of the sphere; it is derived
        # from the surface center (the vertex) via the inwards normal when a center is given instead.
        if origin is not None and center is not None:
            raise ValueError("Only one of origin and center must be provided.")
        elif center is not None:
            self.origin = _to_position_array(center) + radius * self.inwards_normal
        elif origin is not None:
            self.origin = _to_position_array(origin)
        else:
            self.origin = np.full(3, np.nan)

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

    @center.setter
    def center(self, value: np.ndarray):
        self.origin = _to_position_array(value) + self.radius * self.inwards_normal

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value: float):
        # Changing the radius keeps the surface center (the vertex) fixed and moves the sphere origin accordingly,
        # matching the convention used when rebuilding a surface from its params (where x/y/z is the vertex).
        if hasattr(self, "origin") and not np.any(np.isnan(self.center)):
            center = self.center
            self._radius = value
            self.origin = center + value * self.inwards_normal
        else:
            self._radius = value

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
        diameter = nvl(nvl(diameter, self.diameter), 0.6 * self.radius)
        super().plot(
            ax,
            name,
            dim,
            diameter=diameter,
            plane=plane,
            fine_resolution=fine_resolution,
            **kwargs,
        )


class SphericalMirror(SphericalSurface, ReflectiveSurface):
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
        return f"SphericalMirror(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, radius={self.radius})"

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

            new_mirror = SphericalMirror(
                radius=new_radius,
                outwards_normal=self.outwards_normal,
                center=self.center,
                material_properties=new_thermal_properties,
            )
            return new_mirror


class SphericalRefractiveSurface(SphericalSurface, RefractiveSurface):
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
        return f"SphericalRefractiveSurface(name={self.name}, center={self.center}, outwards_normal={self.outwards_normal}, radius={self.radius}, n_1={self.n_1}, n_2={self.n_2})"

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

        return SphericalRefractiveSurface(
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
    flat_faces_center: Optional[np.ndarray],
    diameter: float,
    polynomial_degree: int = 6,
    n_outside: float = 1.0,
    material_properties: Optional[MaterialProperties] = None,
    name: Optional[str] = None,
) -> List[OpticalSurfaceParams]:
    warnings.warn(
        "generate_aspheric_lens_params is deprecated; use generate_aspheric_lens, which returns live surfaces. "
        + PARAMS_DEPRECATION_MESSAGE,
        DeprecationWarning,
        stacklevel=2,
    )
    if name is None:
        name = "Aspheric Lens"
    p = LensParams(n=n, f=back_focal_length, T_c=T_c)
    coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=polynomial_degree)
    theta, phi = angles_of_unit_vector(forward_normal)
    if flat_faces_center is None or np.any(np.isnan(flat_faces_center)):
        flat_faces_center = np.array([np.nan, np.nan, np.nan])
        curved_center = T_c * forward_normal * 1j
    else:
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


def generate_aspheric_lens(
    back_focal_length: float,
    T_c: float,
    n: float,
    forward_normal: np.ndarray,
    flat_faces_center: Optional[np.ndarray],
    diameter: float,
    polynomial_degree: int = 6,
    n_outside: float = 1.0,
    material_properties: Optional[MaterialProperties] = None,
    name: Optional[str] = None,
) -> List[Surface]:
    """Generate the two live surfaces of a plano-convex aspheric lens (flat side first, along ``forward_normal``).

    Object-based replacement of the deprecated ``generate_aspheric_lens_params``. When ``flat_faces_center`` is
    None (or contains nans), the flat face is left undefined and the curved face's center is encoded as a relative
    (imaginary) offset of T_c, to be resolved once the lens is placed."""
    if name is None:
        name = "Aspheric Lens"
    p = LensParams(n=n, f=back_focal_length, T_c=T_c)
    coeffs = solve_aspheric_profile(p, y_max=diameter / 2, degree=polynomial_degree)
    forward_normal = normalize_vector(np.asarray(forward_normal, dtype=float))
    if flat_faces_center is None or np.any(np.isnan(flat_faces_center)):
        flat_faces_center = np.full(3, np.nan)
        curved_center = T_c * forward_normal * 1j
    else:
        flat_faces_center = np.asarray(flat_faces_center, dtype=float)
        curved_center = flat_faces_center + T_c * forward_normal
    flat_surface = FlatRefractiveSurface(
        outwards_normal=-forward_normal,
        center=flat_faces_center,
        n_1=n_outside,
        n_2=n,
        name=name + " - flat side",
        thermal_properties=material_properties,
        diameter=diameter,
    )
    curved_surface = AsphericRefractiveSurface(
        center=curved_center,
        outwards_normal=forward_normal,
        polynomial_coefficients=coeffs,
        n_1=n,
        n_2=n_outside,
        curvature_sign=CurvatureSigns.concave,
        name=name + " - curved side",
        diameter=diameter,
        material_properties=material_properties,
    )
    return [flat_surface, curved_surface]


def convert_curved_refractive_surface_to_ideal_lens(surface: SphericalRefractiveSurface):
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

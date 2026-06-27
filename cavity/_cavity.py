from __future__ import annotations

import copy
import warnings
from hashlib import md5
from datetime import datetime
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from utils import (
    MaterialProperties,
    OpticalElementParams,
    SurfacesTypes,
    ParamsNames,
    PerturbationPointer,
    CENTRAL_LINE_TOLERANCE,
    STRETCH_FACTOR,
    ROOM_TEMPERATURE,
    PRETTY_INDICES_NAMES,
    INDICES_DICT,
    nvl,
    normalize_vector,
    unit_vector_of_angles,
    angles_of_unit_vector,
    angles_difference,
    generate_initial_parameters_grid,
    ABCD_free_space,
    stack_df_for_print,
    signif,
    functions_first_crossing,
    plane_name_to_xy_indices,
    normal_to_a_sphere,
    m_total,
    C_LIGHT_SPEED,
    spot_size,
)
from ._modes import (
    LocalModeParameters,
    ModeParameters,
    propagate_local_mode_parameter_through_ABCD,
    local_mode_parameters_of_round_trip_ABCD,
)
from ._rays import Ray, RaySequence
from ._surfaces import (
    Surface,
    PhysicalSurface,
    ReflectiveSurface,
    RefractiveSurface,
    CurvedSurface,
    CurvedMirror,
    CurvedRefractiveSurface,
    FlatSurface,
    FlatMirror,
    FlatRefractiveSurface,
    IdealLens,
    AsphericSurface,
    AsphericRefractiveSurface,
)


# ---------------------------------------------------------------------------
# Helper functions used by Arm, OpticalSystem, and Cavity
# ---------------------------------------------------------------------------


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


def calculate_incidence_angle(surface: Surface, mode_parameters: ModeParameters) -> float:
    # old code is in git before commit 68ee9d05bff74c20b80f76af9896fcea89948562 and derivation of old code is here:
    # https://mynotebook.labarchives.com/doc/view/ODcuMTAwMDAwMDAwMDAwMDF8MTA1ODU5NS82Ny9FbnRyeVBhcnQvMTg4MjAyOTYyOXwyMjEuMQ==?nb_id=MTM3NjE3My41fDEwNTg1OTUvMTA1ODU5NS9Ob3RlYm9vay8zMjQzMzA0MzY1fDM0OTMzNjMuNQ%3D%3D
    # Derivation for this syntax is in the research file under subsection "Angle of incidence of marginal ray and a spherical surface"
    # And a demonstration that it works is here: https://www.desmos.com/calculator/qk6yradzbn
    if isinstance(surface, FlatSurface):
        return np.arccos(surface.outwards_normal @ mode_parameters.k_vector)
    if isinstance(surface, AsphericSurface):
        raise NotImplementedError("Calculate incidence angle for aspheric surfaces is not implemented yet.")
    optical_axis = surface.outwards_normal
    surface_center_to_waist_position_vector = mode_parameters.center[0, :] - surface.center
    from_the_convex_side = np.sign(surface.outwards_normal @ surface_center_to_waist_position_vector)
    w_0, z_R, R, z_0, z_s = (
        mode_parameters.w_0[0],
        mode_parameters.z_R[0],
        surface.radius,
        mode_parameters.center[0, :] @ optical_axis,
        surface.origin @ optical_axis,
    )
    # --- Quadratic coefficients ---
    A = (w_0**2 / z_R**2) + 1.0
    B = -2.0 * ((w_0**2 / z_R**2) * z_0 + z_s)
    C = w_0**2 + (w_0**2 / z_R**2) * z_0**2 + z_s**2 - R**2

    # --- Discriminant ---
    discriminant = B**2 - 4.0 * A * C

    if discriminant < 0:
        raise ValueError("No real intersection between Gaussian envelope and sphere.")

    sqrt_disc = np.sqrt(discriminant)

    # --- Solutions z1, z2 ---
    z2 = (-B + sqrt_disc) / (2.0 * A)
    # z1 = (-B - sqrt_disc) / (2.0 * A)
    z_star = z2
    y_star = np.sqrt(R**2 - (z_star - z_s) ** 2)

    # ---------------------------------------------------------
    # Choose which intersection you want:
    # ---------------------------------------------------------

    # --- Beam radius at intersection ---

    # --- Derivative w'(z) ---
    w_prime = w_0 * (z_star - z_0) / (z_R**2 * np.sqrt(1.0 + ((z_star - z_0) ** 2) / z_R**2))

    # --- Ray direction vector (not normalized) ---
    v = np.array([1.0, w_prime])

    # --- Surface normal vector (not normalized) ---
    n = np.array([z_star - z_s, y_star])

    # --- Angle between them ---
    cross_mag = np.abs(v[0] * n[1] - v[1] * n[0])
    gamma = np.arcsin(cross_mag / (np.linalg.norm(v) * np.linalg.norm(n)))
    angle_of_incidence_deg = np.degrees(gamma)

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
        ]  # Each element is a  100 | 3 array.

    return spot_size_lines_separated


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------


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

    def propagate_local_mode_parameters(self, local_mode_parameters_on_surface_0: Optional[LocalModeParameters]):
        mode_parameters_on_surface_1 = propagate_local_mode_parameter_through_ABCD(
            local_mode_parameters_on_surface_0, self.ABCD_matrix_free_space, n_2=self.n
        )
        mode_parameters_after_surface_1 = propagate_local_mode_parameter_through_ABCD(
            mode_parameters_on_surface_1, self.ABCD_matrix_reflection, n_2=self.surface_1.to_params.n_inside_or_after
        )
        return mode_parameters_on_surface_1, mode_parameters_after_surface_1

    def set_local_mode_parameters(self) -> LocalModeParameters:
        if self.mode_parameters_on_surface_0 is None:
            raise ValueError("Mode parameters on surface 0 not set")
        mode_parameters_on_surface_1, mode_parameters_after_surface_1 = self.propagate_local_mode_parameters(
            self.mode_parameters_on_surface_0
        )
        self.mode_parameters_on_surface_1 = mode_parameters_on_surface_1
        return mode_parameters_after_surface_1

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
        center = (
            self.central_line.origin
            - self.mode_parameters_on_surface_0.z_minus_z_0[..., np.newaxis] * self.central_line.k_vector
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
            self.mode_parameters_on_surface_0, propagation_ABCD, n_2=self.n
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
            self.set_local_mode_parameters()
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
                r_surface=r, o_center=self.surface_0.origin, sign=self.surface_0.curvature_sign
            )  # Add sign

        def propagation_kernel(r_source: np.ndarray, r_observer: np.ndarray, k: float):
            M = m_total(r_source=r_source, r_observer=r_observer, k=k, normal_function=normal_function, n_index=self.n)
            return M

        return propagation_kernel


# ---------------------------------------------------------------------------
# complete_orthonormal_basis
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# simple_mode_propagator
# ---------------------------------------------------------------------------


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
        local_mode_parameters_current = arm.set_local_mode_parameters()
    return arms


# ---------------------------------------------------------------------------
# OpticalSystem
# ---------------------------------------------------------------------------


class OpticalSystem:
    def __init__(
        self,
        surfaces: list[Surface],
        lambda_0_laser: Optional[float] = None,
        params: Optional[list[OpticalElementParams]] = None,
        power: Optional[float] = None,
        t_is_trivial: bool = True,
        p_is_trivial: bool = True,
        given_initial_central_line: Optional[Union[Ray, bool]] = True,
        given_initial_local_mode_parameters: Optional[LocalModeParameters] = None,
        use_paraxial_ray_tracing: bool = True,
    ):
        self.arms: list[Arm] = [
            Arm(
                surfaces[i],
                surfaces[i + 1],
            )
            for i in range(len(surfaces) - 1)
        ]
        self.central_line_successfully_traced: Optional[bool] = None
        self.resonating_mode_successfully_traced: Optional[bool] = None
        self.lambda_0_laser: Optional[float] = lambda_0_laser
        self.params = params
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
                local_mode_parameters_on_first_surface=given_initial_local_mode_parameters,
            )

    @property
    def surfaces(self):
        surfaces = [arm.surface_0 for arm in self.arms] + [self.arms[-1].surface_1]
        return surfaces

    @property
    def physical_surfaces(self):
        physical_surfaces = [surface for surface in self.surfaces if isinstance(surface, PhysicalSurface)]
        return physical_surfaces

    @staticmethod
    def params_to_surfaces(
        params: Union[np.ndarray, list[OpticalElementParams]],
    ):
        if isinstance(params, np.ndarray):
            raise ValueError(
                "Cavity.from_params no longer supports np.ndarray input. Please provide a list of OpticalElementParams."
            )
            # params = [OpticalElementParams.from_array(params[i, :]) for i in range(len(params))]
        surfaces = []
        for i, p in enumerate(params):
            if p.name is None:
                p.name = f"Surface_{i}"
            surface_temp = Surface.from_params(p)
            if isinstance(surface_temp, tuple):
                surfaces.extend(surface_temp)
            else:
                surfaces.append(surface_temp)
        return surfaces

    @staticmethod
    def from_params(params: Union[np.ndarray, list[OpticalElementParams]], **kwargs):
        surfaces = OpticalSystem.params_to_surfaces(params)
        optical_system = OpticalSystem(
            surfaces,
            params=params,
            **kwargs,
        )
        return optical_system

    @property
    def to_params(self) -> list[OpticalElementParams]:
        if self.params is None:
            params = [surface.to_params for surface in self.surfaces]
        else:
            params = self.params
        return params

    @property  # TODO change it to be the __str__ function without breaking existing code
    def formatted_textual_params(self) -> str:
        if self.to_params is None:
            return "No parameters set for this cavity."
        textual_representation = "params = " + str(self.to_params).replace(
            "OpticalElementParams", "\n          OpticalElementParams"
        ).replace("))]", "))\n         ]")
        return textual_representation

    def __str__(self):
        return self.formatted_textual_params

    @property
    def to_array(self) -> np.ndarray:
        array = np.stack([param.to_array for param in self.to_params], axis=0)
        return array

    @property
    def id(self):
        hashed_str = int(md5(str(self.to_params).encode("utf-8")).hexdigest()[:5], 16)
        return hashed_str

    @property
    def central_line(self) -> Optional[list[Ray]]:
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
    def default_initial_k_vector(self) -> np.ndarray:
        if self.central_line is not None and self.central_line_successfully_traced:
            initial_k_vector = self.central_line[0].k_vector
        else:
            initial_k_vector = self.arms[0].surface_1.center - self.arms[0].surface_0.center
            initial_k_vector = normalize_vector(initial_k_vector)
        return initial_k_vector

    @property
    def default_initial_angles(self) -> tuple[float, float]:
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
        names = [p.name if p.name is not None else f"{i}: {p.surface_type}" for i, p in enumerate(self.to_params)]
        return names

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

    def propagate_ray(
        self, ray: Ray, n_arms: Optional[int] = None, propagate_with_first_surface_first: bool = False
    ) -> RaySequence:
        ray_history = [ray]
        n_arms = nvl(n_arms, len(self.arms))

        if propagate_with_first_surface_first:
            # For cavities the ray usually starts from first surface and so the first propagation is by the second surface.
            # For other optical systems, the ray starts before the first surface and so the first propagation is by the first surface.
            ray = self.arms[0].surface_0.propagate_ray(ray, paraxial=self.use_paraxial_ray_tracing)
            ray_history.append(ray)

        for i in range(n_arms):
            arm = self.arms[i % len(self.arms)]
            ray = arm.propagate_ray(ray, use_paraxial_ray_tracing=self.use_paraxial_ray_tracing)
            ray_history.append(ray)

        ray_sequence = RaySequence(ray_history)
        return ray_sequence

    def propagate_mode_parameters(
        self,
        local_mode_parameters_on_first_surface: Optional[LocalModeParameters] = None,
        mode_parameters: Optional[ModeParameters] = None,
        n_arms: Optional[int] = None,
        propagate_with_first_surface_first: bool = False,
    ):
        n_arms = nvl(n_arms, len(self.arms))
        local_mode_parameters_history = []

        if local_mode_parameters_on_first_surface is not None:
            local_mode_parameters_current = local_mode_parameters_on_first_surface
            local_mode_parameters_history.append(local_mode_parameters_current)
        else:
            if propagate_with_first_surface_first:
                local_mode_parameters_before_first_surface = mode_parameters.local_mode_parameters_at_a_point(
                    p=self.surfaces[0].center
                )
                local_mode_parameters_current = propagate_local_mode_parameter_through_ABCD(
                    local_mode_parameters=local_mode_parameters_before_first_surface,
                    ABCD=self.surfaces[0].ABCD_matrix(cos_theta_incoming=1),
                    n_2=self.arms[0].n,
                )
                local_mode_parameters_history.extend(
                    [local_mode_parameters_before_first_surface, local_mode_parameters_current]
                )
            else:
                local_mode_parameters_current = mode_parameters.local_mode_parameters_at_a_point(
                    p=self.surfaces[0].center
                )
                local_mode_parameters_history.append(local_mode_parameters_current)

        for i in range(n_arms):
            arm = self.arms[i % len(self.arms)]
            mode_parameters_on_next_surface, local_mode_parameters_current = arm.propagate_local_mode_parameters(
                local_mode_parameters_current
            )
            local_mode_parameters_history.extend([mode_parameters_on_next_surface, local_mode_parameters_current])
        return local_mode_parameters_history

    def set_given_central_line(self, initial_ray: Ray):
        # This line is to save the central line in the ray history, so that it can be plotted later.
        central_line = self.propagate_ray(initial_ray)
        for i, arm in enumerate(self.arms):
            arm.central_line = central_line[i]

    def set_given_mode_parameters(
        self,
        local_mode_parameters_on_first_surface: Optional[LocalModeParameters] = None,
        mode_parameters: Optional[ModeParameters] = None,
        propagate_with_first_surface_first: bool = False,
    ):
        # If there is a valid mode to start propagating, then propagate it through the cavity:
        mode_parameters_history = self.propagate_mode_parameters(
            local_mode_parameters_on_first_surface=local_mode_parameters_on_first_surface,
            mode_parameters=mode_parameters,
            n_arms=None,
            propagate_with_first_surface_first=propagate_with_first_surface_first,
        )
        if propagate_with_first_surface_first:
            mode_parameters_history = mode_parameters_history[
                1:
            ]  # Remove the first one which is before the first surface
        for i, arm in enumerate(self.arms):
            arm.mode_parameters_on_surface_0 = mode_parameters_history[2 * i]
            arm.mode_parameters_on_surface_1 = mode_parameters_history[2 * i + 1]

    def principle_axes(self, k_vector: np.ndarray):
        # Returns two vectors that are orthogonal to k_vector and each other, one lives in the central line plane,
        # the other is perpendicular to the central line plane.
        # ATTENTION! THIS ASSUMES THAT ALL THE CENTRAL LINE arms ARE IN THE SAME PLANE.
        # I find the biggest psuedo z because if the first two k_vector are parallel, the cross product is zero and the
        # result of the cross product will be determined by arbitrary numerical errors.
        possible_pseudo_zs = [
            np.cross(self.central_line[0].k_vector, self.central_line[i].k_vector)
            for i in range(0, len(self.central_line))
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
        additional_rays: Optional[list[Ray]] = None,
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

    def output_radius_of_curvature(
        self, initial_distance: Optional[float] = None, source_position: Optional[np.ndarray] = None
    ) -> float:
        # Currently assume 1d problem for simplicty (only the :2, :2 elements for the first dimension are used), if required it can be expanded
        if initial_distance is None and source_position is not None:
            initial_distance = (self.surfaces[0].center - source_position) @ self.central_line[
                0
            ].k_vector  # ATTENTION, THIS MIGHT NOT BE THE OPTICAL AXIS OUTSIDE THE FIRST ARM
        first_ABCD = self.surfaces[0].ABCD_matrix(cos_theta_incoming=1)[:2, :2]
        ABCD = self.ABCD_round_trip[:2, :2] @ first_ABCD
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        R_out = -(A * initial_distance + B) / (C * initial_distance + D)
        return R_out

    def required_initial_distance_for_desired_output_radius_of_curvature(self, desired_R_out: float) -> float:
        first_ABCD = self.surfaces[0].ABCD_matrix(cos_theta_incoming=1)[:2, :2]
        ABCD = self.ABCD_round_trip[:2, :2] @ first_ABCD
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        initial_distance = -(B + D * desired_R_out) / (A + C * desired_R_out)
        return initial_distance

    def invert(self) -> OpticalSystem:
        inverted_surfaces = []
        for surface in self.physical_surfaces[::-1]:
            inverted_surface = copy.deepcopy(surface)
            if isinstance(surface, RefractiveSurface):
                n_1, n_2 = inverted_surface.n_1, inverted_surface.n_2
                inverted_surface.n_1 = n_2
                inverted_surface.n_2 = n_1
            if isinstance(surface, (CurvedSurface, AsphericSurface)):
                inverted_surface.curvature_sign *= -1
            inverted_surfaces.append(inverted_surface)

        if self.central_line is not None:
            origin_inverted = self.physical_surfaces[-1].find_intersection_with_ray(self.central_line[-1])
            k_vector_inverted = -self.central_line[-1].k_vector
            initial_ray_inverted = Ray(origin=origin_inverted, k_vector=k_vector_inverted)
        else:
            initial_ray_inverted = None

        inverted_system = OpticalSystem(
            surfaces=inverted_surfaces,
            lambda_0_laser=self.lambda_0_laser,
            given_initial_central_line=initial_ray_inverted,
            use_paraxial_ray_tracing=self.use_paraxial_ray_tracing,
        )
        return inverted_system


# ---------------------------------------------------------------------------
# Cavity
# ---------------------------------------------------------------------------


class Cavity(OpticalSystem):
    def __init__(
        self,
        surfaces: list[Surface],
        standing_wave: bool = False,
        lambda_0_laser: Optional[float] = None,
        params: Optional[list[OpticalElementParams]] = None,
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
        ordered_surfaces = self._order_surfaces_for_initialization(surfaces, standing_wave=standing_wave)

        super().__init__(
            surfaces=ordered_surfaces,
            lambda_0_laser=lambda_0_laser,
            params=params,
            power=power,
            t_is_trivial=t_is_trivial,
            p_is_trivial=p_is_trivial,
            use_paraxial_ray_tracing=use_paraxial_ray_tracing,
        )

        self.standing_wave = standing_wave
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

    @staticmethod
    def from_params(params: Union[np.ndarray, list[OpticalElementParams]], **kwargs):
        surfaces = OpticalSystem.params_to_surfaces(params)
        cavity = Cavity(
            surfaces,
            params=params,
            **kwargs,
        )
        return cavity

    @staticmethod
    def _order_surfaces_for_initialization(surfaces: list[Surface], standing_wave) -> list[Surface]:
        # Suppose the surfaces are A, B, C, D:
        if standing_wave:
            backwards_list = surfaces[-2::-1]  # not including last  #  C -> B -> A
            backwards_list_inverted = [surface.inverse for surface in backwards_list]  # C^-1 -> B^-1 -> A^-1
            ordered_list = surfaces + backwards_list_inverted  # A -> B -> C -> D -> C^-1 -> B^-1 -> A^-1
        else:
            ordered_list = surfaces + [surfaces[0]]  # A -> B -> C -> D -> A
        return ordered_list

    @property
    def surfaces_ordered(self):
        # For standing wave: A -> B -> C -> D -> C^-1 -> B^-1 -> A^-1
        # For ring cavity: A -> B -> C -> D -> A
        return [arm.surface_0 for arm in self.arms] + [self.arms[-1].surface_1]

    @property
    def physical_surfaces_ordered(self):
        physical_surfaces_ordered_list = [
            surface for surface in self.surfaces_ordered if isinstance(surface, PhysicalSurface)
        ]
        return physical_surfaces_ordered_list

    @property
    def surfaces(self):
        # Overrides OpticalSystem.surfaces
        if self.standing_wave:
            # A -> B -> C -> D
            return [arm.surface_0 for arm in self.arms[: len(self.arms) // 2 + 1]]
        else:
            # A -> B -> C -> D
            return [arm.surface_0 for arm in self.arms]

    @property
    def perturbable_params_names(self):
        from .__init__ import params_to_perturbable_params_names
        perturbable_params_names_list = params_to_perturbable_params_names(
            self.to_params, self.t_is_trivial and self.p_is_trivial
        )
        return perturbable_params_names_list

    def trace_ray_parametric(self, starting_position_and_angles: np.ndarray) -> tuple[np.ndarray, list[Ray]]:
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
    ) -> tuple[np.ndarray, bool]:
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

    def set_central_line(self, **kwargs) -> tuple[np.ndarray, bool]:
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
                local_mode_parameters_current = arm.set_local_mode_parameters()
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
        self, perturbation_pointer: Union[PerturbationPointer, list[PerturbationPointer]]
    ) -> tuple[np.ndarray]:
        # For a prturbation of more than one parameter, the first dimension of shift is the shift version, and the second dimension for the parameter index
        # For example, if shift = [[1e-6, 2e-6], [3e-6, 4e-6]], then the first perturbation is [1e-6, 2e-6] and the second is [3e-6, 4e-6].
        # Deferred import to avoid circular dependency with _analysis
        from ._analysis import perturb_cavity, calculate_cavities_overlap

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
        perturbable_params_names: Optional[list[str]] = None,
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
        shift_numel: int = 50,
        perturbable_params_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        if perturbable_params_names is None:
            perturbable_params_names = self.perturbable_params_names
        overlaps = np.zeros((len(self.to_params), len(perturbable_params_names), shift_numel))
        for element_index in tqdm(
            range(len(self.to_params)), desc="Overlap Series - element_index", disable=self.debug_printing_level < 1
        ):
            for j, parameter_name in tqdm(
                enumerate(perturbable_params_names),
                desc="Overlap Series - parameter_index",
                disable=self.debug_printing_level < 1,
            ):
                if isinstance(shifts, (float, int)):
                    shift_series = np.linspace(-shifts, shifts, shift_numel)
                else:
                    if np.isnan(shifts[element_index, j]):
                        shift_series = np.linspace(-1e-10, 1e-10, shift_numel)
                    else:
                        shift_series = np.linspace(
                            -shifts[element_index, j],
                            shifts[element_index, j],
                            shift_numel,
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
        names: Optional[list[str]] = None,
        ax: Optional[np.ndarray] = None,
        perturbable_params_names: Optional[list[str]] = None,
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
                shifts=2 * np.abs(np.array(tolerance_dataframe)), shift_numel=30
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
                    if np.isnan(min_value):
                        min_value = 0
                    ax[i, j].set_ylim(min_value - 0.01, 1.01)
                except ValueError:
                    pass
        fig.tight_layout()
        return ax

    def thermal_transformation(self, **kwargs) -> Cavity:
        unheated_surfaces = []
        assert (
            self.power is not None
        ), "The power of the laser is not defined. It must be defined for thermal calculations"

        for i, surface in enumerate(self.surfaces):
            if isinstance(surface, PhysicalSurface):
                unheated_surface = surface.thermal_transformation(
                    P_laser_power=-self.power,
                    w_spot_size=self.arms[i].mode_parameters_on_surface_0.spot_size[0],
                    **kwargs,
                )
            else:
                unheated_surface = surface
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
            surfaces=unheated_surfaces,
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

    def analyze_thermal_transformation(self, arm_index_for_NA: int) -> tuple[dict, list[Cavity]]:
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

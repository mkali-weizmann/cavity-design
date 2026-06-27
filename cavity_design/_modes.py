from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from ._surfaces import CurvedMirror, FlatMirror
from ._utils import (
    normalize_vector,
    spot_size,
    w_0_of_z_R,
    z_R_of_w_0,
    R_of_q,
    NA_of_z_R,
    decompose_ABCD_matrix,
    z_R_of_NA,
)
from ._rays import Ray


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

    def __repr__(self) -> str:
        return (
            f"LocalModeParameters(z_minus_z_0={self.z_minus_z_0}, "
            f"z_R={self.z_R}, spot_size={self.spot_size}), NA={self.NA}"
        )

    @property
    def NA(self):
        return NA_of_z_R(z_R=self.z_R, lambda_0_laser=self.lambda_0_laser)

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

    def to_mode_parameters(self, location_of_local_mode_parameter: np.ndarray, k_vector: np.ndarray) -> ModeParameters:
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
            n=self.n,
        )

    @property
    def spot_size(self):
        if np.any(self.z_R == 0):
            w_z = np.array([np.nan, np.nan])
        else:
            w_z = spot_size(z=self.z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)  # spot_size
        return w_z

    @property
    def radius_of_curvature(self):
        if np.any(self.z_R == 0):
            R = np.array([np.nan, np.nan])
        else:
            R = R_of_q(self.q)
        return R


@dataclass
class ModeParameters:
    # I have once spent a few hours unifying LocalModeParameters and ModeParameters into one class and at the end
    # saw it only makes the code more cumbersome and less readable, so I rolled back.
    center: (
        np.ndarray
    )  # First dimension is theta or phi (the two transversal axes of the mode), second dimension is x, y, z
    k_vector: np.ndarray
    lambda_0_laser: Optional[float]
    w_0: Optional[np.ndarray] = None
    z_R: Optional[np.ndarray] = None
    principle_axes: Optional[np.ndarray] = None  # First dimension is theta or phi, second dimension is x, y, z
    n: float = 1  # refractive index

    def __post_init__(self):

        if self.w_0 is None and self.z_R is not None:
            self.w_0 = w_0_of_z_R(z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)
        elif self.z_R is None and self.w_0 is not None:
            self.z_R = z_R_of_w_0(w_0=self.w_0, lambda_laser=self.lambda_laser)
        elif self.w_0 is None and self.z_R is None:
            raise ValueError("Either w_0 or z_R must be provided")

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
                return NA_of_z_R(z_R=self.z_R, lambda_0_laser=self.lambda_0_laser)

    def local_mode_parameters(self, z_minus_z_0):
        return LocalModeParameters(z_minus_z_0=z_minus_z_0, z_R=self.z_R, lambda_0_laser=self.lambda_0_laser, n=self.n)

    def local_mode_parameters_at_a_point(self, p: Union[float, np.ndarray]) -> LocalModeParameters:
        if isinstance(p, np.ndarray):
            z_minus_z_0 = (p - self.center) @ self.k_vector
        else:
            z_minus_z_0 = p
        return self.local_mode_parameters(z_minus_z_0=z_minus_z_0)

    def R_of_z(self, p: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(p, np.ndarray):
            z_minus_z_0 = (p - self.center) @ self.k_vector
        else:
            z_minus_z_0 = np.asarray(p)

        R_z = np.where(z_minus_z_0 == 0, np.inf, (z_minus_z_0**2 + self.z_R**2) / z_minus_z_0)
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

    def invert_direction(self):
        # good for standing waves, where the mode go both ways:
        inverted_direction_mode = ModeParameters(
            center=self.center,
            k_vector=-self.k_vector,
            lambda_0_laser=self.lambda_0_laser,
            w_0=self.w_0,
            principle_axes=self.principle_axes,
            n=self.n,
        )
        return inverted_direction_mode

    def plot(self, first_point: np.ndarray, last_point: np.ndarray, plane="xy", dim=2, ax=None, **kwargs):
        spot_size_lines = generate_spot_size_lines(
            mode_parameters=self,
            first_point=first_point,
            last_point=last_point,
            dim=2,
            plane=plane,
            principle_axes=self.principle_axes,
        )
        if ax is None:
            fig, ax = plt.subplots()
        for line in spot_size_lines:
            ax.plot(line[0, :], line[1, :], **kwargs)
        return ax


def propagate_local_mode_parameter_through_ABCD(
    local_mode_parameters: LocalModeParameters, ABCD: np.ndarray, n_2: float = 1
) -> LocalModeParameters:
    A, B, C, D = decompose_ABCD_matrix(ABCD)
    # q_new = n_2 * (A * local_mode_parameters.q / n_1 + B) / (C * local_mode_parameters.q / n_1 + D)  # Siegman's convention
    q_new = (A * local_mode_parameters.q + B) / (C * local_mode_parameters.q + D)  # Wikipedia's convention
    return LocalModeParameters(q=q_new, lambda_0_laser=local_mode_parameters.lambda_0_laser, n=n_2)


def local_mode_parameters_of_round_trip_ABCD(
    round_trip_ABCD: np.ndarray,
    n: float,  # refractive_index at the beginning of the roundtrip
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
        origin=first_point, k_vector=mode_parameters.k_vector, length=float(np.linalg.norm(last_point - first_point))
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


def match_a_mirror_to_mode(
    mode: ModeParameters,
    z: Optional[float] = None,
    R: Optional[float] = None,
    **mirror_kwargs,
) -> Union[FlatMirror, CurvedMirror]:
    # Derivation is in section "Matching a mode to a mirror:" in my research file.
    if z is None and R is None or (z is not None and R is not None):
        raise ValueError("You must provide either z or R, but not both, and not neither.")
    elif z is not None:
        if z == 0:
            mirror = FlatMirror(
                center=mode.center[0, :],
                outwards_normal=mode.k_vector,
                **mirror_kwargs,
            )
        else:
            R_z_inverse = np.abs(z / (z**2 + mode.z_R[0] ** 2))
            center = mode.center[0, :] + mode.k_vector * z
            outwards_normal = mode.k_vector * np.sign(z)
            mirror = CurvedMirror(
                radius=R_z_inverse**-1,
                outwards_normal=outwards_normal,
                center=center,
                **mirror_kwargs,
            )
    elif R is not None:
        center = mode.z_of_R(R, output_type=np.ndarray)
        outwards_normal = mode.k_vector * np.sign(R)
        if np.isclose(R, 0):
            mirror = FlatMirror(
                center=center,
                outwards_normal=outwards_normal,
                **mirror_kwargs,
            )
        else:
            mirror = CurvedMirror(
                radius=np.abs(R),
                outwards_normal=outwards_normal,
                center=center,
                **mirror_kwargs,
            )
    else:
        raise ValueError("Debug me")
    return mirror


def match_a_local_mode_to_mirror(
    mirror: CurvedMirror,
    lambda_0_laser: float,
    NA: Optional[float] = None,
    z_R: Optional[float] = None,
    n=1,
    mode_going_away_from_mirror: bool = True,
) -> LocalModeParameters:
    # The function might assume the mirror is concave, but it is fine, since they are always concave.
    if z_R is None and NA is not None and lambda_0_laser is not None:
        z_R = z_R_of_NA(NA, lambda_laser=lambda_0_laser / n)
    elif z_R is None and (NA is None or lambda_0_laser is None):
        raise ValueError("You must provide either z_R or NA and lambda_0_laser, but not neither.")

    if mode_going_away_from_mirror:
        position_sign = -1
    else:
        position_sign = 1
    z_minus_z_0 = position_sign * (mirror.radius + np.sqrt(mirror.radius**2 - 4 * z_R**2)) / 2
    q = z_minus_z_0 + 1j * z_R
    local_mode_parameters = LocalModeParameters(q=np.array([q, q]), lambda_0_laser=lambda_0_laser, n=n)

    return local_mode_parameters


def match_a_mode_to_mirror(
    mirror: CurvedMirror,
    lambda_0_laser: float,
    NA: Optional[float] = None,
    z_R: Optional[float] = None,
    n=1,
    mode_going_away_from_mirror: bool = True,
) -> ModeParameters:
    local_mode_parameters = match_a_local_mode_to_mirror(
        mirror=mirror,
        lambda_0_laser=lambda_0_laser,
        NA=NA,
        z_R=z_R,
        n=n,
        mode_going_away_from_mirror=mode_going_away_from_mirror,
    )
    if mode_going_away_from_mirror:
        k_vector = mirror.inwards_normal
    else:
        k_vector = mirror.outwards_normal
    mode_parameters = local_mode_parameters.to_mode_parameters(
        location_of_local_mode_parameter=mirror.center, k_vector=k_vector
    )  # ASSUMEs CONCAVE MIRRORS, WHICH IS ALWAYS THE CASE
    return mode_parameters

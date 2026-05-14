from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ._utils import (
    normalize_vector,
    spot_size,
    w_0_of_z_R,
    z_R_of_w_0,
    R_of_q,
    NA_of_z_R, decompose_ABCD_matrix
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

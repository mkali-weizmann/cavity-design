from __future__ import annotations

from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np

from ._utils import plane_name_to_xy_indices, normalize_vector

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
            length = np.ones(origin.shape[:-1]) * length
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
        else:
            length = np.where(
                np.isinf(self.length) | np.isnan(self.length), 1, self.length
            )  # If length is inf, we take it to be 0 for plotting purposes
        # else:
        #     length = np.ones_like(self.origin[..., 0])
        ray_origin_reshaped = self.origin.reshape(-1, 3)
        ray_k_vector_reshaped = self.k_vector.reshape(-1, 3)
        lengths_reshaped = length.reshape(-1)

        label = kwargs.get("label", None)
        if isinstance(label, str) or label is None:
            kwargs = dict(kwargs)
            kwargs.pop("label", None)

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
                    label=label if isinstance(label, str) and i == 0 else "_nolegend_",
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
                    label=label if isinstance(label, str) and i == 0 else "_nolegend_",
                    **kwargs,
                )
                for i in range(ray_origin_reshaped.shape[0])
            ]

        return ax


class RaySequence:
    # A Ray which is assumed to have the first index as the "ray step", and the n attribute has the same length as the number of ray steps. For example, if we have a ray sequence of 3 steps, and each step has 10 rays, then the origin and k_vector attributes will be of the shape [3, 10, 3], and the n attribute will be of the shape [3].
    def __init__(
        self,
        rays: Optional[List[Ray]] = None,
        origin: Optional[np.ndarray] = None,
        k_vector: Optional[np.ndarray] = None,
        length: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
    ):
        if rays is not None:
            assert (
                origin is None and k_vector is None and length is None and n is None
            ), "If rays is given, origin, k_vector, length and n should not be given"
            origin = np.stack([ray.origin for ray in rays], axis=0)  # [n_steps, m_rays..., 3]
            k_vector = np.stack([ray.k_vector for ray in rays], axis=0)  # [n_steps, m_rays..., 3]
            n = np.array([ray.n for ray in rays])  # [n_steps]
            length = np.stack([ray.length for ray in rays], axis=0)  # [n_steps, m_rays...]
            # length[np.isnan(length)] = np.inf  # Last rays are often infinite  # Why was it here? consider changing it back

        self.origin = origin  # [n_steps, m_rays..., 3]
        self.k_vector = k_vector  # [n_steps, m_rays..., 3]
        self.length = length  # [n_steps, m_rays...]
        self.n = n  # [n_steps]

    def __getitem__(self, key) -> Union[Ray, RaySequence]:
        # If key is a tuple and the second-from-last element is a slice, this case is not implemented.
        origin = self.origin[key]
        k_vector = self.k_vector[key]
        if isinstance(key, int) or isinstance(key, slice):
            length = self.length[key]
            n = self.n[key]
        elif isinstance(key, tuple):
            length = self.length[key[: len(self.length.shape)]]
            n = self.n[key[0]]
        else:
            raise NotImplementedError(f"Key type not supported")

        if isinstance(key, int) or isinstance(key, tuple) and isinstance(key[0], int):
            return Ray(origin=origin, k_vector=k_vector, length=length, n=n)
        else:
            return RaySequence(origin=origin, k_vector=k_vector, length=length, n=n)

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
            length_before_t = (
                0 if first_step_before_t == 0 else relevant_lengths_array[first_step_before_t - 1, *full_index]
            )
            remaining_t = t - length_before_t
            point_at_t = (
                self.origin[first_step_before_t, *full_index]
                + remaining_t * self.k_vector[first_step_before_t, *full_index] / relevant_n[first_step_before_t]
            )
            output_points[*full_index, :] = point_at_t
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

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        dim=2,
        plane: str = "xy",
        length: Union[np.ndarray, float] = np.nan,
        colors: Optional[Union[List[str], str]] = None,
        labels: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        if colors is None:
            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            try:
                default_colors = prop_cycle.by_key().get("color")
            except Exception:
                default_colors = None
            if not default_colors:
                default_colors = [f"C{i}" for i in range(10)]
        elif isinstance(colors, str):
            default_colors = [colors]
        else:
            default_colors = colors
        colors = [default_colors[i % len(default_colors)] for i in range(len(self))]

        labels_list = [None] * len(self)
        if isinstance(labels, str):
            labels_list[0] = labels
        elif labels is not None:
            for i in range(min(len(self), len(labels))):
                labels_list[i] = labels[i]

        for i in range(len(self)):
            self[i].plot(ax=ax, dim=dim, plane=plane, length=length, color=colors[i], label=labels_list[i], **kwargs)

    def __len__(self) -> int:
        return self.origin.shape[0]

    @property
    def remove_escaped_rays(self):
        # Removes rays that did not intersect any surface, and therefore have nan in their length. last step always
        # has nan length, so it is not taken into account when checking which rays escaped and which didn't.
        missed_any_surface = np.any(np.isnan(self.length[:-1]), axis=0)
        had_total_internal_reflection = np.any(np.isnan(self.k_vector), axis=(0, -1))
        rays_that_escaped = np.bitwise_or(missed_any_surface, had_total_internal_reflection)
        rays_indices_that_didnt_escape = np.bitwise_not(rays_that_escaped)
        ray_sequence_filtered = self[:, rays_indices_that_didnt_escape]
        return ray_sequence_filtered
